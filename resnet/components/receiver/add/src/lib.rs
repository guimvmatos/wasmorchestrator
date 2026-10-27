use clap::Parser;
use std::io::{BufRead, BufReader, Write, Read};
use std::net::{TcpListener, TcpStream, Shutdown};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use std::env;
use std::collections::HashMap;

type WitData = bindings::planner::addworld::plan::Data;

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

use std::sync::Mutex;

// Caso tensor x tensor: buffer de junção para sincronizar os dois ramos por request_id
static PENDING_BUFFERS: Mutex<Option<HashMap<u32, Vec<f32>>>> = Mutex::new(None);

// Caso tensor x constante (ex: "/model.23/Add_1" no decode do YOLO, que soma
// com o tensor anchor_points fixo — dist2bbox: x2y2 = anchor_points + rb):
// cache do peso carregado do disco, igual ao Mul (kernel 11).
static CONST_CACHE: Mutex<Option<HashMap<String, Vec<f32>>>> = Mutex::new(None);

#[derive(Deserialize, Serialize, Debug, Clone, Default)]
struct AddParams {
    #[serde(default)]
    const_path: String,
    #[serde(default)]
    const_shape: Vec<u32>,
}

#[derive(Deserialize, Debug, Clone)]
struct StepNode {
    step: u32,
    #[serde(default)]
    name: String,
    kernel_type: u8,
    #[serde(default)]
    params: AddParams,
    #[serde(default)]
    next_steps: Vec<u32>,
}

#[derive(Deserialize, Debug)]
struct ModelGraph {
    model: String,
    steps: Vec<StepNode>,
}

fn load_const(path: &str) -> Vec<f32> {
    let mut lock = CONST_CACHE.lock().unwrap();
    let cache = lock.get_or_insert_with(HashMap::new);

    if let Some(cached) = cache.get(path) {
        return cached.clone();
    }

    let bytes = std::fs::read(path)
        .unwrap_or_else(|_| panic!("Erro ao abrir peso constante do Add: {}", path));
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    cache.insert(path.to_string(), floats.clone());
    floats
}

fn handle_client(mut stream: TcpStream, initialized: &mut i32) -> std::io::Result<()> {

    let start_total = Instant::now(); // TEMP

    let MY_ID: u8 = 5; //#### TODO colocar o numero da funcao aqui...

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

        let req_id = input_img.request;

        // Carrega o grafo ANTES de decidir join vs. constante — precisamos
        // de params.const_path (quando presente) pra saber se este step tem
        // um lado constante (ex: anchor_points do decode do YOLO) ou se é o
        // caso de sempre (resíduo de bloco conv, dois ramos dinâmicos).
        let graph_file = std::fs::File::open("resnet18DFG.json").expect("Erro ao abrir resnet18DFG.json");
        let graph: ModelGraph = serde_json::from_reader(graph_file).expect("Erro no parser do resnet18DFG.json");

        let current_step = input_img.current_kernel;
        let current_node = graph.steps.iter().find(|s| s.step == current_step)
            .unwrap_or_else(|| panic!("Step {} não encontrado no resnet18DFG.json", current_step));
        let p = &current_node.params;

        // 1. Decide o caso: constante (params.const_path preenchido) ou
        // tensor x tensor — join-buffer dos 2 ramos por request_id (padrão
        // de sempre, usado por TODO Add residual em ResNet/VGG/YOLO).
        let (tensor_a, tensor_b): (Vec<f32>, Vec<f32>) = if !p.const_path.is_empty() {
            let const_vec = load_const(&p.const_path);
            (input_img.input.clone(), const_vec)
        } else {
            let first_tensor = {
                let mut lock = PENDING_BUFFERS.lock().unwrap();
                let map = lock.get_or_insert_with(HashMap::new);

                if let Some(other_tensor) = map.remove(&req_id) {
                    Some(other_tensor) // Segundo ramo chegou!
                } else {
                    map.insert(req_id, input_img.input.clone()); // Primeiro ramo: guarda no buffer
                    None
                }
            };

            match first_tensor {
                Some(a) => (a, input_img.input.clone()),
                None => {
                    println!("Request {}: 1º ramo recebido no Add. Aguardando 2º ramo...", req_id);
                    break;
                }
            }
        };

        println!("Request {}: Add pronto pra executar (len_a={}, len_b={}).", req_id, tensor_a.len(), tensor_b.len());

        // 2. Executa a soma C (A + B)
        let mut status = bindings::planner::addworld::plan::add(
            &tensor_a,
            &tensor_b,
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
                next_status.from_step = current_step; // marca ESTE step como origem pro próximo receiver
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
        let port = args.get(1).map(|s| s.as_str()).unwrap_or("8085"); //fallback.. se nao passar por argumento ele vai coloar essa
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