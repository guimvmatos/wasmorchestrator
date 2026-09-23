use std::collections::{HashMap, VecDeque};
use std::env;
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::time::Instant;
use serde::Deserialize;

type WitData = bindings::planner::convworld::plan::Data;

fn load_weights_file(path: &str) -> Vec<f32> {
    if path.is_empty() {
        return Vec::new();
    }
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("Erro ao carregar pesos de {}: {:?}", path, e));
    unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr() as *const f32,
            bytes.len() / 4,
        ).to_vec()
    }
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

#[derive(Deserialize, Debug, Clone, Default)]
struct LayerParams {
    #[serde(default)]
    weight_path: String,
    #[serde(default)]
    bias_path: String,
    #[serde(default)]
    scale_path: String,
    #[serde(default)]
    beta_path: String,
    #[serde(default)]
    mean_path: String,
    #[serde(default)]
    var_path: String,
    #[serde(default)]
    epsilon: f32,
    #[serde(default)]
    channels: u32,
    #[serde(default)]
    cin: u32,
    #[serde(default)]
    cout: u32,
    #[serde(default)]
    h_in: u32,
    #[serde(default)]
    w_in: u32,
    #[serde(default)]
    kh: u32,
    #[serde(default)]
    kw: u32,
    #[serde(default)]
    stride: u32,
    #[serde(default)]
    padding: u32,
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
    params: LayerParams,
    #[serde(default)]
    next_steps: Vec<u32>,
}

#[derive(Deserialize, Debug)]
struct ModelGraph {
    model: String,
    steps: Vec<StepNode>,
}

fn handle_client(mut stream: TcpStream, _initialized: &mut i32) -> std::io::Result<()> {
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

        let input_img: WitData = rmp_serde::from_slice(&buffer).expect("Failed to deserialize MessagePack response");
        let _receiveduration = start_receive.elapsed().as_millis();

        let graph_file = std::fs::File::open("resnet18DFG.json").expect("Erro ao abrir resnet18DFG.json");
        let graph: ModelGraph = serde_json::from_reader(graph_file).expect("Erro no parser do resnet18DFG.json");

        let mut queue: VecDeque<(u32, Vec<f32>)> = VecDeque::new();
        let mut add_branches: HashMap<u32, Vec<f32>> = HashMap::new();

        queue.push_back((input_img.current_kernel, input_img.input.clone()));

        let mut final_output: Vec<f32> = Vec::new();
        let reply_to = input_img.reply_to.clone();
        let request_id = input_img.request;

        while let Some((current_step, current_input)) = queue.pop_front() {
            let current_node = graph.steps.iter().find(|s| s.step == current_step)
                .unwrap_or_else(|| panic!("Step {} não encontrado no grafo", current_step));

            println!("==> Executando Step {} | Camada: '{}' | Tipo: {}", current_step, current_node.name, current_node.kernel_type);

            let p = &current_node.params;
            let start_step_exec = Instant::now();
            let computed_output: Vec<f32>;

            match current_node.kernel_type {
                1 => {
                    let out_h = ((p.h_in + 2 * p.padding - p.kh) / p.stride) + 1;
                    let out_w = ((p.w_in + 2 * p.padding - p.kw) / p.stride) + 1;
                    let output_size = (p.cout * out_h * out_w) as usize;
                    let weights = load_weights_file(&p.weight_path);
                    let data = bindings::planner::convworld::plan::Data {
                        input: current_input,
                        output: vec![0.0f32; output_size],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                    };
                    let res = bindings::planner::convworld::plan::conv(
                        p.cout, p.cin, p.h_in, p.w_in, p.kh, p.kw, p.stride, p.padding,
                        &weights, &data,
                    );
                    computed_output = res.output;
                },
                2 => {
                    let scale = load_weights_file(&p.scale_path);
                    let beta  = load_weights_file(&p.beta_path);
                    let mean  = load_weights_file(&p.mean_path);
                    let var   = load_weights_file(&p.var_path);
                    let mut data = bindings::planner::bnworld::plan::Data {
                        input: current_input.clone(),
                        output: vec![0.0f32; current_input.len()],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                    };
                    let res = bindings::planner::bnworld::plan::bn(
                        p.channels, p.h_in, p.w_in, p.epsilon,
                        &scale, &beta, &mean, &var, &mut data,
                    );
                    computed_output = res.output;
                },
                3 => {
                    let mut data = bindings::planner::reluworld::plan::Data {
                        input: current_input.clone(),
                        output: vec![0.0f32; current_input.len()],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                    };
                    let res = bindings::planner::reluworld::plan::relu(&mut data);
                    computed_output = res.output;
                },
                4 => {
                    let out_h = ((p.h_in + 2 * p.padding - p.kh) / p.stride) + 1;
                    let out_w = ((p.w_in + 2 * p.padding - p.kw) / p.stride) + 1;
                    let output_size = (p.channels * out_h * out_w) as usize;
                    let mut data = bindings::planner::maxpoolworld::plan::Data {
                        input: current_input.clone(),
                        output: vec![0.0f32; output_size],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                    };
                    let res = bindings::planner::maxpoolworld::plan::maxpool(
                        p.channels, p.h_in, p.w_in, p.kh, p.kw, p.stride, p.padding, &mut data,
                    );
                    computed_output = res.output;
                },
                5 => {
                    if let Some(first_branch) = add_branches.remove(&current_step) {
                        println!("Step {}: 2º ramo recebido! Somando buffers de tamanhos {} e {}...", current_step, first_branch.len(), current_input.len());
                        let data = bindings::planner::addworld::plan::Data {
                            input: current_input.clone(),
                            output: vec![],
                            reply_to: reply_to.clone(),
                            current_kernel: current_step,
                            request: request_id,
                            kernels: vec![],
                        };
                        let res = bindings::planner::addworld::plan::add(&first_branch, &current_input, &data);
                        computed_output = res.output;
                    } else {
                        println!("Step {}: 1º ramo guardado no buffer. Aguardando outro ramo...", current_step);
                        add_branches.insert(current_step, current_input);
                        continue;
                    }
                },
                6 => {
                    let mut data = bindings::planner::gapworld::plan::Data {
                        input: current_input.clone(),
                        output: vec![],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                    };
                    let res = bindings::planner::gapworld::plan::gap(&current_input, &mut data);
                    computed_output = res.output;
                },
                7 => {
                    let data = bindings::planner::flattenworld::plan::Data {
                        input: current_input.clone(),
                        output: vec![],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                    };
                    let res = bindings::planner::flattenworld::plan::flatten(&current_input, &data);
                    computed_output = res.output;
                },
                8 => {
                    let weights = load_weights_file(&p.weight_path);
                    let bias    = load_weights_file(&p.bias_path);
                    let data = bindings::planner::gemmworld::plan::Data {
                        input: current_input,
                        output: vec![0.0f32; p.out_features as usize],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                    };
                    let res = bindings::planner::gemmworld::plan::gemm(
                        p.in_features, p.out_features, &weights, &bias, &data,
                    );
                    computed_output = res.output;
                },
                9 => {
                    let data = bindings::planner::dropoutworld::plan::Data {
                        input: current_input.clone(),
                        output: vec![],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                    };
                    let res = bindings::planner::dropoutworld::plan::dropout(&current_input, &data);
                    computed_output = res.output;
                },
                _ => panic!("Kernel type {} não suportado nesta partição de nó", current_node.kernel_type),
            }

            let step_duration = start_step_exec.elapsed().as_millis();

            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("receiver_logs.jsonl") {
                let log_linha = serde_json::json!({
                    "request": request_id,
                    "kernel_id": current_node.kernel_type,
                    "step": current_step,
                    "layer_name": current_node.name,
                    "next_steps": current_node.next_steps,
                    "exec_ms": step_duration
                });
                let _ = writeln!(file, "{}", log_linha);
            }

            if current_node.next_steps.is_empty() {
                println!("Fim da rede alcançado no Step {}! Saída com {} elementos.", current_step, computed_output.len());
                final_output = computed_output;
                break;
            } else {
                for &nxt in &current_node.next_steps {
                    queue.push_back((nxt, computed_output.clone()));
                }
            }
        }

        let final_data = WitData {
            input: final_output,
            output: vec![],
            reply_to: reply_to.clone(),
            current_kernel: 0,
            request: request_id,
            kernels: vec![],
        };

        let mut next_stream = TcpStream::connect(&reply_to)?;
        let response_payload = rmp_serde::to_vec(&final_data).expect("Erro ao serializar resposta");
        let resp_len = response_payload.len() as u32;
        next_stream.write_all(&resp_len.to_be_bytes())?;
        next_stream.write_all(&response_payload)?;
        next_stream.flush()?;

        break;
    }

    println!("Conexão finalizada com o cliente.");
    Ok(())
}

impl bindings::exports::wasi::cli::run::Guest for Component {
    fn run() -> Result<(), ()> {
        let args: Vec<String> = env::args().collect();
        let port = args.get(1).map(|s| s.as_str()).unwrap_or("8080");
        let ip = args.get(2).map(|s| s.as_str()).unwrap_or("0.0.0.0");
        let bind_addr = format!("{}:{}", ip, port);
        let listener = TcpListener::bind(&bind_addr).expect(&format!("Não conseguiu abrir a porta {}", port));

        println!("Listening on {}", bind_addr);

        let mut init = 0;
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    println!("Nova conexão recebida!");
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
