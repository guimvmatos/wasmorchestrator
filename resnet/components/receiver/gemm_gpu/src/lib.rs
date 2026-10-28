use clap::Parser;
use std::io::{BufRead, BufReader, Write, Read};
use std::net::{TcpListener, TcpStream, Shutdown};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use std::env;
use std::collections::HashMap;
use std::cell::RefCell;
use std::sync::Arc;

type WitData = bindings::planner::gemmworld::plan::Data;

thread_local! {
    // Cache de pesos/bias já lidos do disco, chaveado pelo path do arquivo.
    // Sem isso, toda chamada relia o .bin inteiro do zero (até ~900ms pro
    // arquivo de 411MB da dense0), mesmo que já tivesse sido lido antes
    // nesse mesmo processo. O mesmo processo atende várias camadas
    // (dense0/dense1/dense2), cada uma com seu próprio arquivo de peso.
    static WEIGHTS_CACHE: RefCell<HashMap<String, Arc<Vec<f32>>>> = RefCell::new(HashMap::new());

    // Controla quais weight_id (= weight_path) já foram enviados ao
    // kernel GPU via gemm_load_weights. Depois da primeira vez, chamadas
    // seguintes pulam o marshalling do Vec<f32> inteiro pela fronteira
    // WIT — confirmado ~81ms pra 411MB, proporcional ao tamanho (ver
    // GEMM_TIMING vs kernel_call registrado anteriormente).
    static GPU_LOADED: RefCell<std::collections::HashSet<String>> = RefCell::new(std::collections::HashSet::new());
}

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

#[derive(Deserialize, Serialize, Debug, Clone, Default)]
struct GemmParams {
    #[serde(default)]
    weight_path: String,
    #[serde(default)]
    bias_path: String,
    #[serde(default)]
    in_features: u32,
    #[serde(default)]
    out_features: u32,
}

#[derive(Deserialize, Debug, Clone)]
struct StepNode {
    step: u32,
    #[serde(default)]
    name: String,
    kernel_type: u8,
    #[serde(default)]
    params: GemmParams,
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

fn load_bin(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("Erro ao carregar {}: {:?}", path, e));
    unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4).to_vec()
    }
}

// Lê do cache se o path já foi carregado antes neste processo; senão lê do
// disco com load_bin e guarda no cache para as próximas chamadas.
fn load_bin_cached(path: &str) -> Arc<Vec<f32>> {
    WEIGHTS_CACHE.with(|cache| {
        if let Some(existing) = cache.borrow().get(path) {
            return existing.clone();
        }
        let data = Arc::new(load_bin(path));
        cache.borrow_mut().insert(path.to_string(), data.clone());
        data
    })
}

fn handle_client(
    mut stream: TcpStream,
    initialized: &mut i32,
    graphs: &HashMap<u32, ModelGraph>,
    routing: &RoutingTable,
) -> std::io::Result<()> {

    let start_total = Instant::now(); // TEMP

    let MY_ID: u8 = 8; //#### TODO colocar o numero da funcao aqui...

    let mut len_buf = [0u8; 4];

    loop {
        let start_receive = Instant::now(); // TEMP
        if let Err(e) = stream.read_exact(&mut len_buf) {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                break; // Cliente desconectou normalmente
            }
            return Err(e); // Erro real de rede
        }

        
        // 2. Converter os bytes para o tamanho real da imagem
        let len = u32::from_be_bytes(len_buf) as usize;
        println!("Recebendo payload de {} bytes...", len);

        let mut buffer = vec![0u8; len];
        stream.read_exact(&mut buffer)?;
                let _ = stream.shutdown(Shutdown::Both); //em teste
        let net_in_us = start_receive.elapsed().as_micros() as u64; // [T]

        let t_de = Instant::now(); // [T]
        let mut input_img: WitData = rmp_serde::from_slice(&buffer).expect("Failed to deserialize MessagePack response");
        let graph: &ModelGraph = graphs.get(&input_img.model_id).unwrap_or_else(|| {
            panic!("model_id {} nao encontrado no registro de grafos (models.json)", input_img.model_id)
        });
        let deserialize_us = t_de.elapsed().as_micros() as u64; // [T]

        let receiveduration = start_receive.elapsed().as_millis();

        let start_exec = Instant::now();
        *initialized += 1; // [T] contador de chamadas do processo (1 = primeira = contexto GPU frio)
        let call_idx = *initialized; // [T]
        let ts_unix_ms = std::time::SystemTime::now() // [T]
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // 1. OBTÉM OS PARÂMETROS DO STEP ATUAL (grafo já carregado uma vez em run())
        let current_step = input_img.current_kernel;
        let current_node = graph.steps.iter().find(|s| s.step == current_step)
            .unwrap_or_else(|| panic!("Step {} não encontrado no grafo do modelo pedido", current_step));

        let p = &current_node.params;
        let graph_load_us: u64 = 0; // [T] carregado uma vez em run(), não por request

        // 2. Aloca o buffer de saída (1000 logits)
        input_img.output = vec![0.0f32; p.out_features as usize];

        // 3. Carrega os pesos e o bias do disco (cache local, path -> dados)
        let t_w = Instant::now(); // [T]
        let weights = if !p.weight_path.is_empty() {
            load_bin_cached(&p.weight_path)
        } else {
            Arc::new(Vec::new())
        };
        let weights_load_us = t_w.elapsed().as_micros() as u64; // [T] perto de 0 após a 1ª chamada p/ este path

        let t_b = Instant::now(); // [T]
        let bias = if !p.bias_path.is_empty() {
            load_bin_cached(&p.bias_path)
        } else {
            Arc::new(Vec::new())
        };
        let bias_load_us = t_b.elapsed().as_micros() as u64; // [T] idem

        // 3b. weight_id = weight_path (único por camada). Na primeira vez
        // que o kernel GPU vê este weight_id, manda os floats e ele cacheia
        // os buffers GPU; nas chamadas seguintes, só manda o identificador —
        // sem reenviar o Vec<f32> pela fronteira WIT.
        let weight_id = p.weight_path.clone();
        let t_gpu_load = Instant::now(); // [T]
        let gpu_load_triggered = GPU_LOADED.with(|loaded| { // [T]
            if loaded.borrow().contains(&weight_id) {
                false
            } else {
                bindings::planner::gemmworld::plan::gemm_load_weights(
                    &weight_id,
                    weights.as_slice(),
                    bias.as_slice(),
                );
                loaded.borrow_mut().insert(weight_id.clone());
                true
            }
        });
        let gpu_load_us = t_gpu_load.elapsed().as_micros() as u64; // [T] só não-zero na 1ª chamada p/ este weight_id

        // 4. Executa o GEMM (sem reenviar pesos — já cacheados no kernel)
        let t_call = Instant::now(); // [T]
        let mut status = bindings::planner::gemmworld::plan::gemm(
            &weight_id,
            p.in_features,
            p.out_features,
            &input_img,
        );
        let kernel_call_us = t_call.elapsed().as_micros() as u64; // [T]

        let execduration = start_exec.elapsed().as_millis();

        let routing_load_us: u64 = 0; // [T] carregado uma vez em run(), não por request

        // Captura o score deste nó (estou assumindo que este nó físico é o ID 1)
        let current_score = routing.node_scores.get(&1).cloned().unwrap_or(0.0); //#### TODO colocar numero do NÓ aqui deopis do get(&x).cloned... X deve ser o onumero do nó

        let start_send = Instant::now();

        let t_clone_out = Instant::now(); // [T]
        let computed_output = status.output.clone();
        let out_len = computed_output.len(); // <--- Salva o tamanho aqui
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
            status.input = computed_output.clone(); // <--- Adiciona .clone()
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
            input_img.request, current_step, current_node.name, current_node.next_steps, input_img.input.len(), out_len, execduration, totalduration
        );

        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("receiver_logs.jsonl") 
        {
            let log_linha = serde_json::json!({
                "request": input_img.request,
                "kernel_id": MY_ID,
                "step": current_step,
                "layer_name": current_node.name,
                "next_steps": current_node.next_steps,
                "input_len": input_img.input.len(),
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
                "weights_bytes": weights.len() * 4,
                "bias_bytes": bias.len() * 4,
                "gpu_load_triggered": gpu_load_triggered,
                "timing_us": {
                    "net_in": net_in_us,
                    "deserialize": deserialize_us,
                    "graph_load": graph_load_us,
                    "weights_load": weights_load_us,
                    "bias_load": bias_load_us,
                    "gpu_load": gpu_load_us,
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
        let port = args.get(1).map(|s| s.as_str()).unwrap_or("8088"); //fallback.. se nao passar por argumento ele vai coloar essa
        let ip = args.get(2).map(|s| s.as_str()).unwrap_or("0.0.0.0");
        let bind_addr = format!("{}:{}", ip, port);
        let listener = TcpListener::bind(&bind_addr).expect(&format!("Não conseguiu abrir a porta {}", port));

        // Carrega grafo e routing table uma vez, no boot do processo.
        // Antes: reabertos e reparseados a cada request (~200ms/req medidos
        // no graph_load). Os dois arquivos não mudam durante a vida do
        // processo, então recarregar por request era custo puro.
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
                    //call function to handle client
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