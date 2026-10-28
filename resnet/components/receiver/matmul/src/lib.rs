use clap::Parser;
use std::io::{BufRead, BufReader, Write, Read};
use std::net::{TcpListener, TcpStream, Shutdown};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use std::env;
use std::collections::HashMap;
use std::sync::Mutex;

type WitData = bindings::planner::matmulworld::plan::Data;

mod bindings {
    use super::Component;
    wit_bindgen::generate!({
        generate_all,
        additional_derives: [
            serde::Deserialize,
            serde::Serialize,
        ],
    });
    export!(Component);
}

struct Component;

#[derive(Deserialize)]
struct RoutingTable {
    table: std::collections::HashMap<u8, String>,
    node_scores: std::collections::HashMap<u8, f32>,
    assignments: HashMap<u8, Vec<u8>>,
}

// MatMul NÃO é comutativo (diferente de Add/Mul), então o join-buffer não
// pode simplesmente guardar "o primeiro que chegou" — precisa saber qual
// ramo é lhs e qual é rhs. lhs_step/rhs_step (resolvidos pelo
// onnx_to_dfg.py a partir do producer real de node.input[0]/node.input[1])
// dizem isso. Chave externa = request_id (processo atende requests
// concorrentes), chave interna = origin_step (from_step de quem mandou).
static PENDING_BUFFERS: Mutex<Option<HashMap<u32, HashMap<u32, Vec<f32>>>>> = Mutex::new(None);

// Caso lhs/rhs constante (ex: matriz de projeção fixa) — cache do peso
// carregado do disco, igual ao CONST_CACHE do Mul/WEIGHTS_CACHE do Gemm.
static CONST_CACHE: Mutex<Option<HashMap<String, Vec<f32>>>> = Mutex::new(None);

#[derive(Deserialize, Serialize, Debug, Clone, Default)]
struct MatmulParams {
    // -1 = dimensão desconhecida no value_info do ONNX (onnx_to_dfg.py usa
    // o mesmo fallback do Split/Transpose) — convertido pra 1 na chamada.
    #[serde(default)]
    lhs_shape: Vec<i64>,
    #[serde(default)]
    rhs_shape: Vec<i64>,
    #[serde(default)]
    lhs_step: Option<u32>,
    #[serde(default)]
    rhs_step: Option<u32>,
    #[serde(default)]
    lhs_const_path: String,
    #[serde(default)]
    rhs_const_path: String,
}

#[derive(Deserialize, Debug, Clone)]
struct StepNode {
    step: u32,
    #[serde(default)]
    name: String,
    kernel_type: u8,
    #[serde(default)]
    params: MatmulParams,
    #[serde(default)]
    next_steps: Vec<u32>,
}

#[derive(Deserialize, Debug)]
struct ModelGraph {
    model: String,
    steps: Vec<StepNode>,
}

#[derive(Deserialize)]
struct ModelEntry {
    #[serde(default)]
    name: String,
    dfg_path: String,
    #[serde(default)]
    kernels: Vec<u8>,
}

// Lê o registro central de modelos (components/models.json) UMA VEZ no boot
// e carrega o ModelGraph de cada model_id cadastrado — substitui o antigo
// hardcoded "resnet18DFG.json" (Problema 2 / ver claude/project-state.md).
fn load_all_model_graphs() -> HashMap<u32, ModelGraph> {
    let registry_file = std::fs::File::open("models.json")
        .expect("Erro ao abrir models.json (registro de modelos)");
    let registry: HashMap<String, ModelEntry> = serde_json::from_reader(registry_file)
        .expect("Erro no parser do models.json");

    let mut graphs: HashMap<u32, ModelGraph> = HashMap::new();
    for (id_str, entry) in registry.iter() {
        let id: u32 = match id_str.parse() {
            Ok(id) => id,
            Err(_) => {
                eprintln!("[AVISO] chave de modelo invalida em models.json: {}", id_str);
                continue;
            }
        };
        let graph_filename = std::path::Path::new(&entry.dfg_path)
        .file_name()
        .unwrap_or_else(|| panic!("dfg_path {:?} sem nome de arquivo valido", entry.dfg_path));
    let graph_path = std::path::Path::new(graph_filename);
        let graph_file = std::fs::File::open(&graph_path)
            .unwrap_or_else(|_| panic!("Erro ao abrir {:?} (dfg_path do model_id {})", graph_path, id));
        let graph: ModelGraph = serde_json::from_reader(graph_file)
            .unwrap_or_else(|_| panic!("Erro no parser do grafo {:?}", graph_path));
        graphs.insert(id, graph);
    }
    graphs
}

fn load_const(path: &str) -> Vec<f32> {
    let mut lock = CONST_CACHE.lock().unwrap();
    let cache = lock.get_or_insert_with(HashMap::new);

    if let Some(cached) = cache.get(path) {
        return cached.clone();
    }

    let bytes = std::fs::read(path)
        .unwrap_or_else(|_| panic!("Erro ao abrir peso constante do MatMul: {}", path));
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    cache.insert(path.to_string(), floats.clone());
    floats
}

fn handle_client(
    mut stream: TcpStream,
    initialized: &mut i32,
    graphs: &HashMap<u32, ModelGraph>,
    routing: &RoutingTable,
) -> std::io::Result<()> {

    let start_total = Instant::now();

    let my_id: u8 = 17; //#### kernel_type do MatMul

    let mut len_buf = [0u8; 4];

    loop {
        let start_receive = Instant::now();
        if let Err(e) = stream.read_exact(&mut len_buf) {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                break;
            }
            return Err(e);
        }

        let len = u32::from_be_bytes(len_buf) as usize;
        println!("Recebendo payload de {} bytes...", len);

        let mut buffer = vec![0u8; len];
        stream.read_exact(&mut buffer)?;
        let _ = stream.shutdown(Shutdown::Both);
        let net_in_us = start_receive.elapsed().as_micros() as u64; // [T]

        let t_de = Instant::now(); // [T]
        let input_img: WitData = rmp_serde::from_slice(&buffer).expect("Failed to deserialize MessagePack response");
        let graph: &ModelGraph = graphs.get(&input_img.model_id).unwrap_or_else(|| {
            panic!("model_id {} nao encontrado no registro de grafos (models.json)", input_img.model_id)
        });
        let deserialize_us = t_de.elapsed().as_micros() as u64; // [T]

        let receiveduration = start_receive.elapsed().as_millis();

        let start_exec = Instant::now();
        *initialized += 1;
        let call_idx = *initialized;
        let ts_unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let current_step = input_img.current_kernel;
        let current_node = graph.steps.iter().find(|s| s.step == current_step)
            .unwrap_or_else(|| panic!("Step {} não encontrado no model_graph.json", current_step));

        let p = &current_node.params;
        let graph_load_us: u64 = 0; // [T]
        let req_id = input_img.request;
        let origin_step = input_img.from_step;
        let in_len = input_img.input.len();

        let t_join = Instant::now(); // [T]
        let (tensor_a, tensor_b): (Vec<f32>, Vec<f32>) = if !p.lhs_const_path.is_empty() {
            // lhs constante (ex: projeção fixa) — rhs é o ramo dinâmico.
            (load_const(&p.lhs_const_path), input_img.input.clone())
        } else if !p.rhs_const_path.is_empty() {
            // rhs constante — lhs é o ramo dinâmico.
            (input_img.input.clone(), load_const(&p.rhs_const_path))
        } else {
            let lhs_step = p.lhs_step.unwrap_or_else(|| panic!("Step {}: MatMul sem params.lhs_step — grafo desatualizado?", current_step));
            let rhs_step = p.rhs_step.unwrap_or_else(|| panic!("Step {}: MatMul sem params.rhs_step — grafo desatualizado?", current_step));

            let ready = {
                let mut lock = PENDING_BUFFERS.lock().unwrap();
                let by_request = lock.get_or_insert_with(HashMap::new);
                let slots = by_request.entry(req_id).or_insert_with(HashMap::new);
                slots.insert(origin_step, input_img.input.clone());
                slots.contains_key(&lhs_step) && slots.contains_key(&rhs_step)
            };

            if !ready {
                println!("Request {}: ramo from_step={} guardado no MatMul (lhs_step={}, rhs_step={}). Aguardando o resto...", req_id, origin_step, lhs_step, rhs_step);
                continue;
            }

            let mut lock = PENDING_BUFFERS.lock().unwrap();
            let by_request = lock.get_or_insert_with(HashMap::new);
            let mut slots = by_request.remove(&req_id).unwrap();
            let lhs_vec = slots.remove(&lhs_step).unwrap();
            let rhs_vec = slots.remove(&rhs_step).unwrap();
            (lhs_vec, rhs_vec)
        };
        let join_or_const_load_us = t_join.elapsed().as_micros() as u64; // [T]

        println!("Request {}: MatMul pronto pra executar (len_lhs={}, len_rhs={}).", req_id, tensor_a.len(), tensor_b.len());

        let lhs_shape: Vec<u32> = p.lhs_shape.iter().map(|&d| if d > 0 { d as u32 } else { 1 }).collect();
        let rhs_shape: Vec<u32> = p.rhs_shape.iter().map(|&d| if d > 0 { d as u32 } else { 1 }).collect();

        let t_call = Instant::now(); // [T]
        let mut status = bindings::planner::matmulworld::plan::matmul(&tensor_a, &tensor_b, &lhs_shape, &rhs_shape, &input_img);
        let kernel_call_us = t_call.elapsed().as_micros() as u64; // [T]

        let execduration = start_exec.elapsed().as_millis();
        let routing_load_us: u64 = 0; // [T]

        let current_score = routing.node_scores.get(&1).cloned().unwrap_or(0.0);

        let start_send = Instant::now();

        let t_clone_out = Instant::now(); // [T]
        let computed_output = status.output.clone();
        let out_len = computed_output.len();
        status.output = vec![];
        let clone_output_us = t_clone_out.elapsed().as_micros() as u64; // [T]

        let mut clone_in_send_us: u64 = 0; // [T]
        let mut connect_us: u64 = 0; // [T]
        let mut serialize_us: u64 = 0; // [T]
        let mut write_us: u64 = 0; // [T]
        let mut ack_us: u64 = 0; // [T]
        let mut sent_bytes: usize = 0; // [T]

        if current_node.next_steps.is_empty() {
            println!("Último step da rede. Retornando para o cliente em {}...", status.reply_to);
            let t = Instant::now(); // [T]
            status.input = computed_output.clone();
            clone_in_send_us += t.elapsed().as_micros() as u64; // [T]

            let t = Instant::now(); // [T]
            let mut next_stream = TcpStream::connect(&status.reply_to)?;
            connect_us += t.elapsed().as_micros() as u64; // [T]

            let t = Instant::now(); // [T]
            let response_payload = rmp_serde::to_vec(&status).expect("MSGPACK Serialization failed");
            serialize_us += t.elapsed().as_micros() as u64; // [T]
            let resp_len = response_payload.len() as u32;
            sent_bytes += response_payload.len(); // [T]

            let t = Instant::now(); // [T]
            next_stream.write_all(&resp_len.to_be_bytes())?;
            next_stream.write_all(&response_payload)?;
            next_stream.flush()?;
            write_us += t.elapsed().as_micros() as u64; // [T]
        } else {
            for &next_step in &current_node.next_steps {
                let next_node = graph.steps.iter().find(|s| s.step == next_step)
                    .unwrap_or_else(|| panic!("Próximo Step {} não encontrado no grafo", next_step));

                let target_address = routing.table.get(&next_node.kernel_type)
                    .cloned()
                    .unwrap_or_else(|| panic!("Tipo de Kernel {} não encontrado na routing_table", next_node.kernel_type));

                println!("Despachando Step {} -> Step {} (Kernel Tipo {}) em {}...", current_step, next_step, next_node.kernel_type, target_address);

                let t = Instant::now(); // [T]
                let mut next_status = status.clone();
                next_status.current_kernel = next_step;
                next_status.from_step = current_step;
                next_status.input = computed_output.clone();
                clone_in_send_us += t.elapsed().as_micros() as u64; // [T]

                let t = Instant::now(); // [T]
                let mut next_stream = TcpStream::connect(&target_address)?;
                connect_us += t.elapsed().as_micros() as u64; // [T]

                let t = Instant::now(); // [T]
                let response_payload = rmp_serde::to_vec(&next_status).expect("MSGPACK Serialization failed");
                serialize_us += t.elapsed().as_micros() as u64; // [T]
                let resp_len = response_payload.len() as u32;
                sent_bytes += response_payload.len(); // [T]

                let t = Instant::now(); // [T]
                next_stream.write_all(&resp_len.to_be_bytes())?;
                next_stream.write_all(&response_payload)?;
                next_stream.flush()?;
                write_us += t.elapsed().as_micros() as u64; // [T]

                let t = Instant::now(); // [T]
                let mut ack_buf = [0u8; 1];
                let _ = next_stream.read(&mut ack_buf);
                ack_us += t.elapsed().as_micros() as u64; // [T]
            }
        }

        let sendduration = start_send.elapsed().as_millis();
        let send_total_us = start_send.elapsed().as_micros() as u64; // [T]

        let totalduration = start_total.elapsed().as_millis();
        let total_us = start_total.elapsed().as_micros() as u64; // [T]

        println!(
            "METRIC_DATA: req={} step={} layer='{}' next={:?} in_len={} out_len={} exec_ms={} total_ms={}",
            req_id, current_step, current_node.name, current_node.next_steps, in_len, out_len, execduration, totalduration
        );

        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("receiver_logs.jsonl")
        {
            let log_linha = serde_json::json!({
                "request": req_id,
                "kernel_id": my_id,
                "step": current_step,
                "layer_name": current_node.name,
                "next_steps": current_node.next_steps,
                "input_len": in_len,
                "output_len": out_len,
                "total_receiver_ms": totalduration,
                "receive_ms": receiveduration,
                "exec_ms": execduration,
                "send_ms": sendduration,
                "ts_unix_ms": ts_unix_ms,
                "call_idx": call_idx,
                "listen": format!("{}:{}", env::args().nth(2).unwrap_or_default(), env::args().nth(1).unwrap_or_default()),
                "kernel_type": current_node.kernel_type,
                "params": serde_json::to_value(p).unwrap_or(serde_json::Value::Null),
                "payload_in_bytes": len,
                "payload_out_bytes": sent_bytes,
                "timing_us": {
                    "net_in": net_in_us,
                    "deserialize": deserialize_us,
                    "graph_load": graph_load_us,
                    "join_or_const_load": join_or_const_load_us,
                    "kernel_call": kernel_call_us,
                    "routing_load": routing_load_us,
                    "clone_output": clone_output_us,
                    "clone_in_send": clone_in_send_us,
                    "connect": connect_us,
                    "serialize": serialize_us,
                    "write": write_us,
                    "ack_wait": ack_us,
                    "send_total": send_total_us,
                    "total": total_us
                }
            });

            if let Ok(texto) = serde_json::to_string(&log_linha) {
                let _ = writeln!(file, "{}", texto);
            }
        }

        break;
    }

    println!("Conexão finalizada com o cliente.");
    Ok(())
}

impl bindings::exports::wasi::cli::run::Guest for Component {
    fn run() -> Result<(), ()> {

        let args: Vec<String> = env::args().collect();
        let port = args.get(1).map(|s| s.as_str()).unwrap_or("8097");
        let ip = args.get(2).map(|s| s.as_str()).unwrap_or("0.0.0.0");
        let bind_addr = format!("{}:{}", ip, port);
        let listener = TcpListener::bind(&bind_addr).expect(&format!("Não conseguiu abrir a porta {}", port));

        let t_boot = Instant::now();
        let graphs: HashMap<u32, ModelGraph> = load_all_model_graphs();

        let routing_file = std::fs::File::open("routing_table.json")
            .expect("Erro ao abrir routing_table.json");
        let routing: RoutingTable = serde_json::from_reader(routing_file)
            .expect("Erro no parser do routing_table.json");
        println!(
            "Grafo e routing table carregados em {}ms ({} steps)",
            t_boot.elapsed().as_millis(),
            graph.steps.len()
        );

        println!("Listening on {}", bind_addr);

        let mut init = 0;
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    println!("entrei!!");

                    if let Err(e) = handle_client(stream, &mut init, &graphs, &routing) {
                        eprintln!("Error handling client: {:?}", e);
                    }
                }
                Err(e) => {
                    eprintln!("Connection failed: {:?}", e);
                }
            }
            println!("Waiting for a new request...");
        }
        Ok(())
    }
}
