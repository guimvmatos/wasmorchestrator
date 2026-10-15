use clap::Parser;
use std::io::{BufRead, BufReader, Write, Read};
use std::net::{TcpListener, TcpStream, Shutdown};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use std::env;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

type WitData = bindings::planner::concatworld::plan::Data;

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

// Join-buffer do Concat. Diferente do Add/Mul (2 ramos, comutativos, chave
// só por request_id): aqui a ORDEM importa (Concat não é comutativo) e o
// número de ramos é variável (2, 3, 4+ — ver C2f/SPPF do YOLO). Por isso:
// - chave (request_id, step): requests concorrentes e Concats diferentes
//   no mesmo grafo não se cruzam;
// - valor é from_step -> fila (VecDeque, não um único Vec<f32>): o MESMO
//   from_step pode aparecer 2x em input_order — ex. C2f, onde as duas
//   metades de um Split alimentam o mesmo Concat. Um HashMap<u32, Vec<f32>>
//   simples perderia a 1ª fatia quando a 2ª chegasse com a mesma chave
//   (from_step idêntico pras duas, porque o Split marca from_step = seu
//   próprio step pra TODAS as rotas, não por output_index). Com fila,
//   cada chegada empilha no fim e a remontagem final consome em FIFO por
//   from_step — ordem que corresponde à ordem crescente de output_index
//   com que o Split despachou as rotas.
static PENDING_CONCAT: Mutex<Option<HashMap<(u32, u32), HashMap<u32, VecDeque<Vec<f32>>>>>> = Mutex::new(None);

// Espelha o formato emitido pelo onnx_to_dfg.py pro kernel_type 14 (Concat):
// {"const": bool, "step": Option<u32>, "shape": [i64, ...]}.
#[derive(Deserialize, Serialize, Debug, Clone, Default)]
struct InputOrderEntry {
    #[serde(default, rename = "const")]
    is_const: bool,
    #[serde(default)]
    step: Option<u32>,
    #[serde(default)]
    shape: Vec<i64>,
}

#[derive(Deserialize, Serialize, Debug, Clone, Default)]
struct ConcatParams {
    #[serde(default)]
    axis: u32,
    #[serde(default)]
    input_order: Vec<InputOrderEntry>,
}

#[derive(Deserialize, Debug, Clone)]
struct StepNode {
    step: u32,
    #[serde(default)]
    name: String,
    kernel_type: u8,
    #[serde(default)]
    params: ConcatParams,
    #[serde(default)]
    next_steps: Vec<u32>,
}

#[derive(Deserialize, Debug)]
struct ModelGraph {
    model: String,
    steps: Vec<StepNode>,
}

fn handle_client(
    mut stream: TcpStream,
    initialized: &mut i32,
    graph: &ModelGraph,
    routing: &RoutingTable,
) -> std::io::Result<()> {

    let start_total = Instant::now(); // TEMP

    let MY_ID: u8 = 14; //#### kernel_type do Concat

    let mut len_buf = [0u8; 4];

    loop {
        let start_receive = Instant::now(); // TEMP
        if let Err(e) = stream.read_exact(&mut len_buf) {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                break; // Cliente desconectou normalmente
            }
            return Err(e); // Erro real de rede
        }

        let len = u32::from_be_bytes(len_buf) as usize;
        println!("Recebendo payload de {} bytes...", len);

        let mut buffer = vec![0u8; len];
        stream.read_exact(&mut buffer)?;
        let _ = stream.shutdown(Shutdown::Both); //em teste
        let net_in_us = start_receive.elapsed().as_micros() as u64; // [T]

        let t_de = Instant::now(); // [T]
        let input_img: WitData = rmp_serde::from_slice(&buffer).expect("Failed to deserialize MessagePack response");
        let deserialize_us = t_de.elapsed().as_micros() as u64; // [T]

        let receiveduration = start_receive.elapsed().as_millis();

        let start_exec = Instant::now();
        *initialized += 1; // [T] contador de chamadas do processo
        let call_idx = *initialized; // [T]
        let ts_unix_ms = std::time::SystemTime::now() // [T]
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let current_step = input_img.current_kernel;
        let request_id = input_img.request;
        let from_step = input_img.from_step;
        let in_len = input_img.input.len();

        let current_node = graph.steps.iter().find(|s| s.step == current_step)
            .unwrap_or_else(|| panic!("Step {} não encontrado no model_graph.json", current_step));

        let p = &current_node.params;
        let graph_load_us: u64 = 0; // [T] carregado uma vez em run(), não por request

        let input_order = &p.input_order;
        if input_order.is_empty() {
            panic!("Step {} (Concat) sem input_order nos params — grafo desatualizado?", current_step);
        }
        if input_order.iter().all(|e| e.is_const || e.step != Some(from_step)) {
            panic!(
                "Concat step {}: chegou mensagem de from_step={} que não consta em input_order {:?}",
                current_step, from_step, input_order
            );
        }

        // 1. Join dos N ramos (igual em espírito ao Add/Mul, mas por
        // from_step em vez de "primeiro que chegar", porque a ordem importa).
        let t_join = Instant::now(); // [T]
        let ready_inputs: Option<HashMap<u32, VecDeque<Vec<f32>>>> = {
            let mut lock = PENDING_CONCAT.lock().unwrap();
            let outer = lock.get_or_insert_with(HashMap::new);
            let key = (request_id, current_step);
            let slot_map = outer.entry(key).or_insert_with(HashMap::new);
            slot_map.entry(from_step).or_insert_with(VecDeque::new).push_back(input_img.input.clone());

            let expected = input_order.iter().filter(|e| !e.is_const).count();
            let filled: usize = slot_map.values().map(|v| v.len()).sum();
            if filled >= expected {
                outer.remove(&key)
            } else {
                None
            }
        };
        let join_us = t_join.elapsed().as_micros() as u64; // [T]

        let mut arrived = match ready_inputs {
            Some(map) => map,
            None => {
                println!(
                    "Request {}: ramo from_step={} guardado no Concat step {}. Aguardando o resto...",
                    request_id, from_step, current_step
                );
                break;
            }
        };

        println!("Request {}: todos os ramos do Concat step {} chegaram! Concatenando...", request_id, current_step);

        // Entradas constantes (initializer do ONNX, não outro step) não
        // chegam pela rede — teriam que ser pré-carregadas no boot a partir
        // de um const_path, igual ao Mul faz pro caso tensor x constante.
        // Nenhum dos modelos vistos até agora (ResNet, YOLO) usa isso em
        // Concat, então não implementei: falha explícita em vez de
        // concatenar lixo.
        if input_order.iter().any(|e| e.is_const) {
            panic!("Concat step {}: entrada constante no input_order ainda não é suportada por este receiver", current_step);
        }

        let inputs: Vec<Vec<f32>> = input_order.iter()
            .map(|e| {
                let src_step = e.step.unwrap_or_else(|| panic!("Concat step {}: entrada sem step de origem", current_step));
                arrived.get_mut(&src_step).and_then(|dq| dq.pop_front())
                    .unwrap_or_else(|| panic!("Concat step {}: faltou o ramo do step {}", current_step, src_step))
            })
            .collect();
        let shapes: Vec<Vec<u32>> = input_order.iter()
            .map(|e| e.shape.iter().map(|&d| if d > 0 { d as u32 } else { 1 }).collect())
            .collect();
        let axis = p.axis;

        // 2. Executa o kernel síncrono
        let t_call = Instant::now(); // [T]
        let mut status = bindings::planner::concatworld::plan::concat(&inputs, &shapes, axis, &input_img);
        let kernel_call_us = t_call.elapsed().as_micros() as u64; // [T]

        let execduration = start_exec.elapsed().as_millis();
        let routing_load_us: u64 = 0; // [T] carregado uma vez em run(), não por request

        let current_score = routing.node_scores.get(&1).cloned().unwrap_or(0.0); //#### TODO colocar numero do NÓ aqui

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
                next_status.from_step = current_step; // marca ESTE step como origem pro próximo receiver
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
            request_id, current_step, current_node.name, current_node.next_steps, in_len, out_len, execduration, totalduration
        );

        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("receiver_logs.jsonl")
        {
            let log_linha = serde_json::json!({
                "request": request_id,
                "kernel_id": MY_ID,
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
                    "join": join_us,
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
        let port = args.get(1).map(|s| s.as_str()).unwrap_or("8094");
        let ip = args.get(2).map(|s| s.as_str()).unwrap_or("0.0.0.0");
        let bind_addr = format!("{}:{}", ip, port);
        let listener = TcpListener::bind(&bind_addr).expect(&format!("Não conseguiu abrir a porta {}", port));

        let t_boot = Instant::now();
        let graph_file = std::fs::File::open("resnet18DFG.json")
            .expect("Erro ao abrir resnet18DFG.json");
        let graph: ModelGraph = serde_json::from_reader(graph_file)
            .expect("Erro no parser do resnet18DFG.json");

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

                    if let Err(e) = handle_client(stream, &mut init, &graph, &routing) {
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
