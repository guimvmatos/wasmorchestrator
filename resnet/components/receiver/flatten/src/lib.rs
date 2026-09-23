use clap::Parser;
use std::io::{BufRead, BufReader, Write, Read};
use std::net::{TcpListener, TcpStream, Shutdown};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use std::env;
use std::collections::HashMap;

type WitData = bindings::planner::flattenworld::plan::Data;

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

#[derive(Deserialize, Debug, Clone, Default)]
struct flattenParams {}

#[derive(Deserialize, Debug, Clone)]
struct StepNode {
    step: u32,
    #[serde(default)]
    name: String,
    kernel_type: u8,
    #[serde(default)]
    params: flattenParams,
    #[serde(default)]
    next_steps: Vec<u32>,
}

#[derive(Deserialize, Debug)]
struct ModelGraph {
    model: String,
    steps: Vec<StepNode>,
}

fn handle_client(mut stream: TcpStream, initialized: &mut i32) -> std::io::Result<()> {

    let start_total = Instant::now(); // TEMP

    let MY_ID: u8 = 7; //#### TODO colocar o numero da funcao aqui...

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

        let mut input_img: WitData = rmp_serde::from_slice(&buffer).expect("Failed to deserialize MessagePack response");

        let receiveduration = start_receive.elapsed().as_millis();

        let start_exec = Instant::now();

        // 1. CARREGA O GRAFO E OBTÉM OS PARÂMETROS DO STEP ATUAL
        let graph_file = std::fs::File::open("resnet18DFG.json").expect("Erro ao abrir resnet18DFG.json");
let graph: ModelGraph = serde_json::from_reader(graph_file).expect("Erro no parser do resnet18DFG.json");

        let current_step = input_img.current_kernel;
        let current_node = graph.steps.iter().find(|s| s.step == current_step)
            .unwrap_or_else(|| panic!("Step {} não encontrado no model_graph.json", current_step));

        let _p = &current_node.params;

        // 2. Aloca a saída (512 canais de tamanho 1x1)
        input_img.output = vec![0.0f32; input_img.input.len()];

        // 3. Executa o kernel C do flatten diretamente
        let mut status = bindings::planner::flattenworld::plan::flatten(
            &input_img.input,
            &input_img,
        );

        let execduration = start_exec.elapsed().as_millis();

        let file = std::fs::File::open("routing_table.json").expect("Erro ao abrir JSON");
        let routing: RoutingTable = serde_json::from_reader(file).expect("Erro no JSON");

        // Captura o score deste nó (estou assumindo que este nó físico é o ID 1)
        let current_score = routing.node_scores.get(&1).cloned().unwrap_or(0.0); //#### TODO colocar numero do NÓ aqui deopis do get(&x).cloned... X deve ser o onumero do nó

        let start_send = Instant::now();

        let computed_output = status.output.clone();
        let out_len = computed_output.len(); // <--- Salva o tamanho aqui
        status.output = vec![];

        if current_node.next_steps.is_empty() {
            println!("Último step da rede. Retornando para o cliente em {}...", status.reply_to);
            status.input = computed_output.clone(); // <--- Adiciona .clone()
            let mut next_stream = TcpStream::connect(&status.reply_to)?;
            let response_payload = rmp_serde::to_vec(&status).expect("MSGPACK Serialization failed");
            let resp_len = response_payload.len() as u32;
            next_stream.write_all(&resp_len.to_be_bytes())?;
            next_stream.write_all(&response_payload)?;
            next_stream.flush()?;
        } else {
            for &next_step in &current_node.next_steps {
                let next_node = graph.steps.iter().find(|s| s.step == next_step)
                    .unwrap_or_else(|| panic!("Próximo Step {} não encontrado no grafo", next_step));

                let target_address = routing.table.get(&next_node.kernel_type)
                    .cloned()
                    .unwrap_or_else(|| panic!("Tipo de Kernel {} não encontrado na routing_table", next_node.kernel_type));

                println!("Despachando Step {} -> Step {} (Kernel Tipo {}) em {}...", current_step, next_step, next_node.kernel_type, target_address);

                let mut next_status = status.clone();
                next_status.current_kernel = next_step;
                next_status.input = computed_output.clone();

                let mut next_stream = TcpStream::connect(&target_address)?;
                let response_payload = rmp_serde::to_vec(&next_status).expect("MSGPACK Serialization failed");
                let resp_len = response_payload.len() as u32;
                next_stream.write_all(&resp_len.to_be_bytes())?;
                next_stream.write_all(&response_payload)?;
                next_stream.flush()?;

                let mut ack_buf = [0u8; 1];
                let _ = next_stream.read(&mut ack_buf);
            }
        }
        
        let sendduration = start_send.elapsed().as_millis();

        let totalduration = start_total.elapsed().as_millis();

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
                "send_ms": sendduration
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
        let port = args.get(1).map(|s| s.as_str()).unwrap_or("8087"); //fallback.. se nao passar por argumento ele vai coloar essa
        let ip = args.get(2).map(|s| s.as_str()).unwrap_or("0.0.0.0");
        let bind_addr = format!("{}:{}", ip, port);
        let listener = TcpListener::bind(&bind_addr).expect(&format!("Não conseguiu abrir a porta {}", port));

        println!("Listening on {}", bind_addr);

        let mut init = 0;
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    //call function to handle client
                    println!("entrei!!");

                    if let Err(e) = handle_client(stream, &mut init) {
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