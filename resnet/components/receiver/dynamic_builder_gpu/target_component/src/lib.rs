use std::collections::{HashMap, VecDeque};
use std::env;
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::time::Instant;
use std::cell::RefCell;
use std::sync::Arc;
use serde::Deserialize;

const LOCAL_KERNELS: &[u8] = &[9, 10, 11];


type WitData = bindings::planner::dropoutworld::plan::Data;

thread_local! {
    // Cache de pesos/bias já lidos do disco, chaveado pelo path do arquivo.
    // Conv e Gemm costumam ser chamados repetidas vezes com o mesmo
    // weight_path (mesma camada, requests diferentes) dentro do mesmo
    // processo — sem isso, cada chamada relia o .bin inteiro do zero.
    static WEIGHTS_CACHE: RefCell<HashMap<String, Arc<Vec<f32>>>> = RefCell::new(HashMap::new());
}

fn load_weights_file_cached(path: &str) -> Arc<Vec<f32>> {
    if path.is_empty() {
        return Arc::new(Vec::new());
    }
    WEIGHTS_CACHE.with(|cache| {
        if let Some(existing) = cache.borrow().get(path) {
            return existing.clone();
        }
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("Erro ao carregar pesos de {}: {:?}", path, e));
        let data = Arc::new(unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4).to_vec()
        });
        cache.borrow_mut().insert(path.to_string(), data.clone());
        data
    })
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
struct OutputRoute {
    to_step: u32,
    output_index: u32,
}

// Espelha o formato emitido pelo onnx_to_dfg.py pros params de Concat
// (kernel_type 14): {"const": bool, "step": Option<u32>, "shape": [i64, ...]}.
#[derive(Deserialize, Debug, Clone, Default)]
struct InputOrderEntry {
    #[serde(default, rename = "const")]
    is_const: bool,
    #[serde(default)]
    step: Option<u32>,
    #[serde(default)]
    shape: Vec<i64>,
}

#[derive(Deserialize, Debug, Clone, Default)]
struct LayerParams {
    #[serde(default)]
    weight_path: String,
    #[serde(default)]
    bias_path: String,
    #[serde(default)]
    groups: u32,
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
    #[serde(default)]
    const_path: String,
    #[serde(default)]
    const_shape: Vec<u32>,
    // --- Split (13) ---
    #[serde(default)]
    axis: u32,
    #[serde(default)]
    input_shape: Vec<i64>,
    // None = split igual (opset 18 "num_outputs", sem tamanhos explícitos).
    #[serde(default)]
    split: Option<Vec<u32>>,
    #[serde(default)]
    num_outputs: u32,
    // --- Concat (14) ---
    #[serde(default)]
    input_order: Vec<InputOrderEntry>,
    // --- Resize (18) --- channels/h_in/w_in já existem acima, reusados.
    #[serde(default)]
    scale_h: f32,
    #[serde(default)]
    scale_w: f32,
    #[serde(default)]
    mode: String,
    // --- Transpose (16) --- input_shape já existe acima, reusado (Split).
    #[serde(default)]
    perm: Vec<u32>,
    // --- MatMul (17) --- -1 = dimensão desconhecida (mesmo fallback do
    // Split/Transpose, convertido pra 1 na chamada). lhs_step/rhs_step =
    // step que produziu node.input[0]/node.input[1] (None quando aquele
    // lado é constante — ver lhs_const_path/rhs_const_path).
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
    // --- Slice (19) --- input_shape já existe acima, reusado (Split/
    // Transpose/MatMul). starts/ends/steps em i64 (podem ter sentinelas
    // enormes do ONNX, ex: INT64_MAX pra "até o fim"), clampados pra i32 na
    // hora de chamar o kernel (ver rust_match do 19). axes é Vec<u32>
    // (plural, distinto do `axis` singular do Split/Softmax) — já vem
    // normalizado (não-negativo) pelo onnx_to_dfg.py.
    #[serde(default)]
    starts: Vec<i64>,
    #[serde(default)]
    ends: Vec<i64>,
    #[serde(default)]
    axes: Vec<u32>,
    #[serde(default)]
    steps: Vec<i64>,
    // --- Sub/Div (20/21) --- const_path/const_shape já existem acima
    // (reusados do Mul). const_is_first = true quando node.input[0] é a
    // constante (ex: anchor_points - pred, não pred - anchor_points) —
    // Sub/Div não são comutativos como Mul, então a ordem importa.
    #[serde(default)]
    const_is_first: bool,
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
    // Preenchido pelo onnx_to_dfg.py só pra steps com >1 saída roteada
    // individualmente (Split é sempre um deles). Usado em vez de next_steps
    // genérico quando presente.
    #[serde(default)]
    output_routes: Vec<OutputRoute>,
}

#[derive(Deserialize, Debug)]
struct ModelGraph {
    model: String,
    steps: Vec<StepNode>,
}

// Kernels não presentes neste binário caem no braço `_` do match abaixo e
// são despachados via TCP para o endereço indicado na routing_table para
// aquele kernel_type. Kernels presentes rodam direto, sem rede.
fn handle_client(
    mut stream: TcpStream,
    initialized: &mut i32,
    graph: &ModelGraph,
    routing: &RoutingTable,
) -> std::io::Result<()> {
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
        let deserialize_us = t_de.elapsed().as_micros() as u64; // [T]

        *initialized += 1; // [T] contador de requisições atendidas por este processo
        let call_idx = *initialized; // [T]
        let ts_unix_ms = std::time::SystemTime::now() // [T]
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Terceiro elemento da tupla = step de origem (from_step), propagado
        // igual ao `next_status.from_step = current_step` dos receivers isolados.
        let mut queue: VecDeque<(u32, Vec<f32>, u32)> = VecDeque::new();
        let mut add_branches: HashMap<u32, Vec<f32>> = HashMap::new();
        let mut mul_branches: HashMap<u32, Vec<f32>> = HashMap::new();
        // Concat (14): chave externa = step do Concat; chave interna =
        // from_step; valor = fila (não Vec simples) porque o MESMO from_step
        // pode aparecer 2x em input_order (duas metades do mesmo Split).
        let mut concat_branches: HashMap<u32, HashMap<u32, VecDeque<Vec<f32>>>> = HashMap::new();
        // MatMul (17): chave externa = step do MatMul; chave interna =
        // from_step (lhs_step ou rhs_step) -> buffer. Não usa VecDeque como
        // o Concat porque lhs_step == rhs_step (mesma origem alimentando os
        // dois lados) não é suportado por este dynamic_builder (ver panic
        // no rust_match do kernel 17) — um slot por origem basta.
        let mut matmul_branches: HashMap<u32, HashMap<u32, Vec<f32>>> = HashMap::new();
        // Sub (20) / Div (21): mesmo mecanismo de join-buffer do Mul — só
        // usado no caso raro em que NENHUM lado é constante (ver nota no
        // rust_match dos kernels 20/21 e no onnx_to_dfg.py).
        let mut sub_branches: HashMap<u32, Vec<f32>> = HashMap::new();
        let mut div_branches: HashMap<u32, Vec<f32>> = HashMap::new();

        queue.push_back((input_img.current_kernel, input_img.input.clone(), input_img.from_step));

        let mut final_output: Vec<f32> = Vec::new();
        let mut final_from_step: u32 = 0;
        let mut reached_final = false;
        let reply_to = input_img.reply_to.clone();
        let request_id = input_img.request;

        while let Some((current_step, current_input, origin_step)) = queue.pop_front() {
            let current_node = graph.steps.iter().find(|s| s.step == current_step)
                .unwrap_or_else(|| panic!("Step {} não encontrado no grafo", current_step));

            println!("==> Executando Step {} | Camada: '{}' | Tipo: {}", current_step, current_node.name, current_node.kernel_type);

            let p = &current_node.params;
            let computed_output: Vec<f32>;
            let input_len = current_input.len(); // [T]

            // [T] Timing no nível de STEP INTEIRO (inclui carregar pesos +
            // executar o kernel), não decomposto como nos receivers isolados
            // (que separam weights_load de kernel_call). Decompor exigiria
            // reescrever os 9 rust_match do catálogo KERNELS individualmente.
            let t_step = Instant::now(); // [T]

            match current_node.kernel_type {
                9 => {
                    let data = bindings::planner::dropoutworld::plan::Data {
                        input: current_input.clone(),
                        output: vec![],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                        from_step: origin_step,
                    };
                    let res = bindings::planner::dropoutworld::plan::dropout(&current_input, &data);
                    computed_output = res.output;
                },
                10 => {
                    let data = bindings::planner::sigmoidworld::plan::Data {
                        input: current_input.clone(),
                        output: vec![0.0f32; current_input.len()],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                        from_step: origin_step,
                    };
                    let res = bindings::planner::sigmoidworld::plan::sigmoid(&data);
                    computed_output = res.output;
                },
                11 => {
                    if !p.const_path.is_empty() {
                        let const_vec = load_weights_file_cached(&p.const_path);
                        let data = bindings::planner::mulworld::plan::Data {
                            input: current_input.clone(),
                            output: vec![],
                            reply_to: reply_to.clone(),
                            current_kernel: current_step,
                            request: request_id,
                            kernels: vec![],
                            from_step: origin_step,
                        };
                        let res = bindings::planner::mulworld::plan::mul(&current_input, const_vec.as_slice(), &data);
                        computed_output = res.output;
                    } else if let Some(first_branch) = mul_branches.remove(&current_step) {
                        println!("Step {}: 2º ramo recebido! Multiplicando buffers de tamanhos {} e {}...", current_step, first_branch.len(), current_input.len());
                        let data = bindings::planner::mulworld::plan::Data {
                            input: current_input.clone(),
                            output: vec![],
                            reply_to: reply_to.clone(),
                            current_kernel: current_step,
                            request: request_id,
                            kernels: vec![],
                            from_step: origin_step,
                        };
                        let res = bindings::planner::mulworld::plan::mul(&first_branch, &current_input, &data);
                        computed_output = res.output;
                    } else {
                        println!("Step {}: 1º ramo guardado no buffer. Aguardando outro ramo...", current_step);
                        mul_branches.insert(current_step, current_input);
                        continue;
                    }
                },
                _ => {
                    // Kernel não incluído neste binário: despacha por rede.
                    let t_route = Instant::now(); // [T]
                    let target_address = routing.table.get(&current_node.kernel_type)
                        .cloned()
                        .unwrap_or_else(|| panic!(
                            "Kernel tipo {} não está neste nó nem na routing_table",
                            current_node.kernel_type
                        ));

                    println!(
                        "Step {} (tipo {}) não está neste nó — despachando para {}...",
                        current_step, current_node.kernel_type, target_address
                    );

                    let payload_data = WitData {
                        input: current_input,
                        output: vec![],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                        from_step: origin_step,
                    };

                    let t_ser = Instant::now(); // [T]
                    let response_payload = rmp_serde::to_vec(&payload_data).expect("Erro ao serializar payload externo");
                    let serialize_us = t_ser.elapsed().as_micros() as u64; // [T]
                    let sent_bytes = response_payload.len(); // [T]

                    let t_conn = Instant::now(); // [T]
                    let mut next_stream = TcpStream::connect(&target_address)?;
                    let connect_us = t_conn.elapsed().as_micros() as u64; // [T]

                    let resp_len = response_payload.len() as u32;
                    let t_write = Instant::now(); // [T]
                    next_stream.write_all(&resp_len.to_be_bytes())?;
                    next_stream.write_all(&response_payload)?;
                    next_stream.flush()?;
                    let write_us = t_write.elapsed().as_micros() as u64; // [T]

                    let t_ack = Instant::now(); // [T]
                    let mut ack_buf = [0u8; 1];
                    let _ = next_stream.read(&mut ack_buf);
                    let ack_wait_us = t_ack.elapsed().as_micros() as u64; // [T]

                    let step_total_us = t_step.elapsed().as_micros() as u64; // [T]

                    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("receiver_logs.jsonl") { // [T]
                        let log_linha = serde_json::json!({
                            "request": request_id,
                            "kernel_id": current_node.kernel_type,
                            "kernel_type": current_node.kernel_type,
                            "step": current_step,
                            "layer_name": current_node.name,
                            "next_steps": current_node.next_steps,
                            "call_idx": call_idx,
                            "ts_unix_ms": ts_unix_ms,
                            "dispatch": "network",
                            "input_len": input_len,
                            "sent_bytes": sent_bytes,
                            "timing_us": {
                                "net_in": net_in_us,
                                "deserialize": deserialize_us,
                                "routing_load": t_route.elapsed().as_micros() as u64,
                                "connect": connect_us,
                                "serialize": serialize_us,
                                "write": write_us,
                                "ack_wait": ack_wait_us,
                                "step_total": step_total_us
                            }
                        });
                        let _ = writeln!(file, "{}", log_linha);
                    } // [T]

                    continue;
                },
            }

            let step_total_us = t_step.elapsed().as_micros() as u64; // [T]
            let output_len = computed_output.len(); // [T]
            let is_local = LOCAL_KERNELS.contains(&current_node.kernel_type); // [T]

            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("receiver_logs.jsonl") {
                let log_linha = serde_json::json!({
                    "request": request_id,
                    "kernel_id": current_node.kernel_type,
                    "kernel_type": current_node.kernel_type,
                    "step": current_step,
                    "layer_name": current_node.name,
                    "next_steps": current_node.next_steps,
                    "call_idx": call_idx, // [T]
                    "ts_unix_ms": ts_unix_ms, // [T]
                    "dispatch": if is_local { "local" } else { "network" }, // [T]
                    "input_len": input_len, // [T]
                    "output_len": output_len, // [T]
                    "timing_us": { // [T]
                        "net_in": net_in_us,
                        "deserialize": deserialize_us,
                        "kernel_call": step_total_us
                    }
                });
                let _ = writeln!(file, "{}", log_linha);
            }

            if current_node.next_steps.is_empty() {
                println!("Fim da rede alcançado localmente no Step {}! Saída com {} elementos.", current_step, computed_output.len());
                final_output = computed_output;
                final_from_step = current_step;
                reached_final = true;
                break;
            } else {
                for &nxt in &current_node.next_steps {
                    queue.push_back((nxt, computed_output.clone(), current_step));
                }
            }
        }

        // Só devolve pro cliente se a rede terminou DENTRO deste nó. Se tudo
        // que restava foi despachado por rede, quem responde é o nó remoto.
        if reached_final {
            let final_data = WitData {
                input: final_output,
                output: vec![],
                reply_to: reply_to.clone(),
                current_kernel: 0,
                request: request_id,
                kernels: vec![],
                from_step: final_from_step,
            };

            let mut next_stream = TcpStream::connect(&reply_to)?;
            let response_payload = rmp_serde::to_vec(&final_data).expect("Erro ao serializar resposta");
            let resp_len = response_payload.len() as u32;
            next_stream.write_all(&resp_len.to_be_bytes())?;
            next_stream.write_all(&response_payload)?;
            next_stream.flush()?;
        }

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

        // Grafo e routing table carregados uma vez, no boot do processo.
        let graph_file = std::fs::File::open("resnet18DFG.json").expect("Erro ao abrir resnet18DFG.json");
        let graph: ModelGraph = serde_json::from_reader(graph_file).expect("Erro no parser do resnet18DFG.json");

        let routing_file = std::fs::File::open("routing_table.json").expect("Erro ao abrir routing_table.json");
        let routing: RoutingTable = serde_json::from_reader(routing_file).expect("Erro no parser do routing_table.json");

        println!("Listening on {}", bind_addr);

        let mut init = 0;
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    println!("Nova conexão recebida!");
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
