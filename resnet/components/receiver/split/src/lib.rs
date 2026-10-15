use clap::Parser;
use std::io::{BufRead, BufReader, Write, Read};
use std::net::{TcpListener, TcpStream, Shutdown};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use std::env;
use std::collections::HashMap;

type WitData = bindings::planner::splitworld::plan::Data;

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

// Espelha o formato emitido pelo onnx_to_dfg.py pro kernel_type 13 (Split).
#[derive(Deserialize, Serialize, Debug, Clone, Default)]
struct SplitParams {
    #[serde(default)]
    axis: u32,
    #[serde(default)]
    input_shape: Vec<i64>,
    // None = split igual (opset 18 "num_outputs", sem tamanhos explícitos) —
    // o receiver divide input_shape[axis] por num_outputs.
    #[serde(default)]
    split: Option<Vec<u32>>,
    #[serde(default)]
    num_outputs: u32,
}

// Preenchido pelo onnx_to_dfg.py só pra steps com >1 saída nomeada (Split é
// sempre um deles). Diz, por saída (output_index 0-based na ordem do
// node.output do ONNX), qual step consome ela — pode haver mais de uma
// entrada com o mesmo output_index se a mesma fatia alimenta 2 consumidores.
#[derive(Deserialize, Debug, Clone)]
struct OutputRoute {
    to_step: u32,
    output_index: u32,
}

#[derive(Deserialize, Debug, Clone)]
struct StepNode {
    step: u32,
    #[serde(default)]
    name: String,
    kernel_type: u8,
    #[serde(default)]
    params: SplitParams,
    #[serde(default)]
    #[allow(dead_code)] // mantido só por paridade com o schema; o roteamento real usa output_routes
    next_steps: Vec<u32>,
    #[serde(default)]
    output_routes: Vec<OutputRoute>,
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

    let MY_ID: u8 = 13; //#### kernel_type do Split

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
        let in_len = input_img.input.len();

        let current_node = graph.steps.iter().find(|s| s.step == current_step)
            .unwrap_or_else(|| panic!("Step {} não encontrado no model_graph.json", current_step));

        if current_node.output_routes.is_empty() {
            panic!("Step {} (Split) sem output_routes — grafo desatualizado?", current_step);
        }

        let p = &current_node.params;
        let graph_load_us: u64 = 0; // [T] carregado uma vez em run(), não por request

        let shape: Vec<u32> = p.input_shape.iter()
            .map(|&d| if d > 0 { d as u32 } else { 1 })
            .collect();
        let axis = p.axis;

        // Resolve tamanhos: explícitos (p.split) ou split igual por num_outputs.
        let sizes: Vec<u32> = match &p.split {
            Some(s) => s.clone(),
            None => {
                let n = p.num_outputs.max(1);
                let total = *shape.get(axis as usize).unwrap_or(&0);
                assert_eq!(
                    total % n, 0,
                    "Split step {}: shape[axis]={} não divide igualmente em num_outputs={}",
                    current_step, total, n
                );
                vec![total / n; n as usize]
            }
        };

        let t_call = Instant::now(); // [T]
        let outputs = bindings::planner::splitworld::plan::split(&input_img.input, &shape, axis, &sizes, &input_img);
        let kernel_call_us = t_call.elapsed().as_micros() as u64; // [T]

        let execduration = start_exec.elapsed().as_millis();
        let routing_load_us: u64 = 0; // [T] carregado uma vez em run(), não por request

        let current_score = routing.node_scores.get(&1).cloned().unwrap_or(0.0); //#### TODO colocar numero do NÓ aqui

        let start_send = Instant::now();
        let out_len: usize = outputs.iter().map(|o| o.len()).sum();

        let mut connect_us: u64 = 0; // [T]
        let mut serialize_us: u64 = 0; // [T]
        let mut write_us: u64 = 0; // [T]
        let mut ack_us: u64 = 0; // [T]
        let mut sent_bytes: usize = 0; // [T]

        // Split não tem o caso "último step da rede" — ele sempre alimenta
        // algo (senão não faria sentido existir no grafo). Cada saída vai
        // pro seu(s) destino(s) em output_routes, não num loop uniforme
        // sobre next_steps, porque destinos diferentes recebem FATIAS
        // diferentes do mesmo step.
        for route in &current_node.output_routes {
            let slice = outputs.get(route.output_index as usize).unwrap_or_else(|| {
                panic!(
                    "Split step {}: output_routes aponta output_index {} mas o kernel só devolveu {} fatias",
                    current_step, route.output_index, outputs.len()
                )
            });

            let next_node = graph.steps.iter().find(|s| s.step == route.to_step)
                .unwrap_or_else(|| panic!("Próximo Step {} não encontrado no grafo", route.to_step));

            let target_address = routing.table.get(&next_node.kernel_type)
                .cloned()
                .unwrap_or_else(|| panic!("Tipo de Kernel {} não encontrado na routing_table", next_node.kernel_type));

            println!(
                "Despachando Step {} -> Step {} (Kernel Tipo {}, saída {}) em {}...",
                current_step, route.to_step, next_node.kernel_type, route.output_index, target_address
            );

            let mut next_status = input_img.clone();
            next_status.current_kernel = route.to_step;
            next_status.from_step = current_step; // pra quando o destino for um Concat
            next_status.input = slice.clone();
            next_status.output = vec![];

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

        let sendduration = start_send.elapsed().as_millis();
        let send_total_us = start_send.elapsed().as_micros() as u64; // [T]

        let totalduration = start_total.elapsed().as_millis();
        let total_us = start_total.elapsed().as_micros() as u64; // [T]

        println!(
            "METRIC_DATA: req={} step={} layer='{}' routes={:?} in_len={} out_len={} exec_ms={} total_ms={}",
            request_id, current_step, current_node.name, current_node.output_routes, in_len, out_len, execduration, totalduration
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
                "output_routes": current_node.output_routes.iter().map(|r| (r.to_step, r.output_index)).collect::<Vec<_>>(),
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
                    "kernel_call": kernel_call_us,
                    "routing_load": routing_load_us,
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
        let port = args.get(1).map(|s| s.as_str()).unwrap_or("8093");
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
