#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

# 1. CATÁLOGO CENTRALIZADO DOS KERNELS
# Cada kernel tem uma ou duas variantes ("cpu" sempre; "gpu" só pra quem tem
# versão WebGPU: conv, bn, maxpool, gap, gemm). Conv/BN/MaxPool/Gemm têm a
# MESMA assinatura WIT nas duas variantes, então usam o mesmo rust_match —
# só GAP muda de assinatura entre CPU (a, result_container) e GPU (channels, img).

KERNELS = {
    1: {
        "name": "conv",
        "wit_import": "    import planner:convworld/plan;",
        "rust_match": """                1 => {
                    let out_h = ((p.h_in + 2 * p.padding - p.kh) / p.stride) + 1;
                    let out_w = ((p.w_in + 2 * p.padding - p.kw) / p.stride) + 1;
                    let output_size = (p.cout * out_h * out_w) as usize;
                    let weights = load_weights_file_cached(&p.weight_path);
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
                        weights.as_slice(), &data,
                    );
                    computed_output = res.output;
                },""",
        "variants": {
            "cpu": {
                "wkg_override": '"planner:convworld" = { path = "../../../kernel/conv/" }',
                "wasm_path": "../../../kernel/conv/convworld_component.wasm",
            },
            "gpu": {
                "wkg_override": '"planner:convworld" = { path = "../../../kernel/conv_gpu/wit" }',
                "wasm_path": "../../../kernel/conv_gpu/target/wasm32-wasip2/release/conv_gpu.wasm",
            },
        },
    },
    2: {
        "name": "bn",
        "wit_import": "    import planner:bnworld/plan;",
        "rust_match": """                2 => {
                    let scale = load_weights_file_cached(&p.scale_path);
                    let beta  = load_weights_file_cached(&p.beta_path);
                    let mean  = load_weights_file_cached(&p.mean_path);
                    let var   = load_weights_file_cached(&p.var_path);
                    let data = bindings::planner::bnworld::plan::Data {
                        input: current_input.clone(),
                        output: vec![0.0f32; current_input.len()],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                    };
                    let res = bindings::planner::bnworld::plan::bn(
                        p.channels, p.h_in, p.w_in, p.epsilon,
                        scale.as_slice(), beta.as_slice(), mean.as_slice(), var.as_slice(), &data,
                    );
                    computed_output = res.output;
                },""",
        "variants": {
            "cpu": {
                "wkg_override": '"planner:bnworld" = { path = "../../../kernel/bn/" }',
                "wasm_path": "../../../kernel/bn/bnworld_component.wasm",
            },
            "gpu": {
                "wkg_override": '"planner:bnworld" = { path = "../../../kernel/bn_gpu/wit" }',
                "wasm_path": "../../../kernel/bn_gpu/target/wasm32-wasip2/release/bn_gpu.wasm",
            },
        },
    },
    3: {
        "name": "relu",
        "wit_import": "    import planner:reluworld/plan;",
        "rust_match": """                3 => {
                    let data = bindings::planner::reluworld::plan::Data {
                        input: current_input.clone(),
                        output: vec![0.0f32; current_input.len()],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                    };
                    let res = bindings::planner::reluworld::plan::relu(&data);
                    computed_output = res.output;
                },""",
        "variants": {
            "cpu": {
                "wkg_override": '"planner:reluworld" = { path = "../../../kernel/relu/" }',
                "wasm_path": "../../../kernel/relu/reluworld_component.wasm",
            },
        },
    },
    4: {
        "name": "maxpool",
        "wit_import": "    import planner:maxpoolworld/plan;",
        "rust_match": """                4 => {
                    let out_h = ((p.h_in + 2 * p.padding - p.kh) / p.stride) + 1;
                    let out_w = ((p.w_in + 2 * p.padding - p.kw) / p.stride) + 1;
                    let output_size = (p.channels * out_h * out_w) as usize;
                    let data = bindings::planner::maxpoolworld::plan::Data {
                        input: current_input.clone(),
                        output: vec![0.0f32; output_size],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                    };
                    let res = bindings::planner::maxpoolworld::plan::maxpool(
                        p.channels, p.h_in, p.w_in, p.kh, p.kw, p.stride, p.padding, &data,
                    );
                    computed_output = res.output;
                },""",
        "variants": {
            "cpu": {
                "wkg_override": '"planner:maxpoolworld" = { path = "../../../kernel/maxpool/" }',
                "wasm_path": "../../../kernel/maxpool/maxpoolworld_component.wasm",
            },
            "gpu": {
                "wkg_override": '"planner:maxpoolworld" = { path = "../../../kernel/maxpool_gpu/wit" }',
                "wasm_path": "../../../kernel/maxpool_gpu/target/wasm32-wasip2/release/maxpool_gpu.wasm",
            },
        },
    },
    5: {
        "name": "add",
        "wit_import": "    import planner:addworld/plan;",
        "rust_match": """                5 => {
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
                },""",
        "variants": {
            "cpu": {
                "wkg_override": '"planner:addworld" = { path = "../../../kernel/add/" }',
                "wasm_path": "../../../kernel/add/addworld_component.wasm",
            },
        },
    },
    6: {
        "name": "gap",
        "wit_import": "    import planner:gapworld/plan;",
        # GAP muda de assinatura entre CPU e GPU — único kernel com dois rust_match.
        "rust_match_variants": {
            "cpu": """                6 => {
                    let data = bindings::planner::gapworld::plan::Data {
                        input: current_input.clone(),
                        output: vec![],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                    };
                    let res = bindings::planner::gapworld::plan::gap(&current_input, &data);
                    computed_output = res.output;
                },""",
            "gpu": """                6 => {
                    let data = bindings::planner::gapworld::plan::Data {
                        input: current_input.clone(),
                        output: vec![0.0f32; p.channels as usize],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                    };
                    let res = bindings::planner::gapworld::plan::gap(p.channels, &data);
                    computed_output = res.output;
                },""",
        },
        "variants": {
            "cpu": {
                "wkg_override": '"planner:gapworld" = { path = "../../../kernel/gap/" }',
                "wasm_path": "../../../kernel/gap/gapworld_component.wasm",
            },
            "gpu": {
                "wkg_override": '"planner:gapworld" = { path = "../../../kernel/gap_gpu/wit" }',
                "wasm_path": "../../../kernel/gap_gpu/target/wasm32-wasip2/release/gap_gpu.wasm",
            },
        },
    },
    7: {
        "name": "flatten",
        "wit_import": "    import planner:flattenworld/plan;",
        "rust_match": """                7 => {
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
                },""",
        "variants": {
            "cpu": {
                "wkg_override": '"planner:flattenworld" = { path = "../../../kernel/flatten/" }',
                "wasm_path": "../../../kernel/flatten/flattenworld_component.wasm",
            },
        },
    },
    8: {
        "name": "gemm",
        "wit_import": "    import planner:gemmworld/plan;",
        # CPU e GPU têm assinaturas WIT diferentes agora: a variante GPU
        # cacheia pesos no próprio kernel (gemm_load_weights, chamado só na
        # 1ª vez por weight_id), eliminando reenvio de até 411MB por chamada
        # pela fronteira do Component Model — confirmado ~90-97% de redução
        # no kernel_call do Gemm isolado.
        "rust_match_variants": {
            "cpu": """                8 => {
                    let weights = load_weights_file_cached(&p.weight_path);
                    let bias    = load_weights_file_cached(&p.bias_path);
                    let data = bindings::planner::gemmworld::plan::Data {
                        input: current_input,
                        output: vec![0.0f32; p.out_features as usize],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                    };
                    let res = bindings::planner::gemmworld::plan::gemm(
                        p.in_features, p.out_features, weights.as_slice(), bias.as_slice(), &data,
                    );
                    computed_output = res.output;
                },""",
            "gpu": """                8 => {
                    let weight_id = p.weight_path.clone();
                    if !GEMM_GPU_LOADED.with(|loaded| loaded.borrow().contains(&weight_id)) {
                        let weights = load_weights_file_cached(&p.weight_path);
                        let bias = load_weights_file_cached(&p.bias_path);
                        bindings::planner::gemmworld::plan::gemm_load_weights(
                            &weight_id, weights.as_slice(), bias.as_slice(),
                        );
                        GEMM_GPU_LOADED.with(|loaded| loaded.borrow_mut().insert(weight_id.clone()));
                    }
                    let data = bindings::planner::gemmworld::plan::Data {
                        input: current_input,
                        output: vec![0.0f32; p.out_features as usize],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                    };
                    let res = bindings::planner::gemmworld::plan::gemm(
                        &weight_id, p.in_features, p.out_features, &data,
                    );
                    computed_output = res.output;
                },""",
        },
        "variants": {
            "cpu": {
                "wkg_override": '"planner:gemmworld" = { path = "../../../kernel/gemm/" }',
                "wasm_path": "../../../kernel/gemm/gemmworld_component.wasm",
            },
            "gpu": {
                "wkg_override": '"planner:gemmworld" = { path = "../../../kernel/gemm_gpu/wit" }',
                "wasm_path": "../../../kernel/gemm_gpu/target/wasm32-wasip2/release/gemm_gpu.wasm",
            },
        },
    },
    9: {
        "name": "dropout",
        "wit_import": "    import planner:dropoutworld/plan;",
        "rust_match": """                9 => {
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
                },""",
        "variants": {
            "cpu": {
                "wkg_override": '"planner:dropoutworld" = { path = "../../../kernel/dropout/" }',
                "wasm_path": "../../../kernel/dropout/dropoutworld_component.wasm",
            },
        },
    },
}

WASI_GFX_OVERRIDE = (
    '"wasi-gfx-runtime:wasi-webgpu" = '
    '{ path = "../../../../../wasi-gfx-runtime/crates/wasi-webgpu-wasmtime/wit/" }'
)


def generate_local_kernels_const(active_kernel_ids):
    # Conhecido em build time: quais kernel_type estão embutidos NESTE wasm.
    # Usado só pra rotular cada linha do log como "local" ou "network" —
    # não influencia o despacho em si (isso já é decidido pelo match/_ =>).
    ids = ", ".join(str(k) for k in active_kernel_ids)
    return f"const LOCAL_KERNELS: &[u8] = &[{ids}];"


def get_rust_match(kernel_id, variant):
    meta = KERNELS[kernel_id]
    if "rust_match_variants" in meta:
        return meta["rust_match_variants"][variant]
    return meta["rust_match"]


# 2. GERADORES DE CONTEÚDO DOS ARQUIVOS
def generate_world_wit(active_kernel_ids):
    # O world.wit NUNCA menciona wasi:webgpu, mesmo com kernels GPU
    # selecionados — o wac propaga esse import automaticamente a partir do
    # .wasm plugado, não precisa ser declarado aqui.
    imports = "\n".join([KERNELS[k]["wit_import"] for k in active_kernel_ids])
    return f"""package example:server;

world runner {{
{imports}
    export wasi:cli/run@0.2.7;
}}
"""


def generate_wkg_toml(active_kernel_ids, variant_map):
    lines = []
    for k in active_kernel_ids:
        variant = variant_map.get(k, "cpu")
        lines.append(KERNELS[k]["variants"][variant]["wkg_override"])
    if any(v == "gpu" for v in variant_map.values()):
        lines.append(WASI_GFX_OVERRIDE)
    overrides = "\n".join(lines)
    return f"""[overrides]
{overrides}
"""


def generate_lib_rs(active_kernel_ids, variant_map):
    # active_kernel_ids também define LOCAL_KERNELS, usado só para rotular
    # o log (dispatch local/network) — não afeta o roteamento em si.
    first_kernel_mod = "convworld" if 1 in active_kernel_ids else None
    if first_kernel_mod is None:
        # WitData precisa de um módulo real; usa o primeiro kernel ativo.
        first_id = active_kernel_ids[0]
        first_kernel_mod = f"{KERNELS[first_id]['name']}world"

    match_arms = "\n".join(
        get_rust_match(k, variant_map.get(k, "cpu")) for k in active_kernel_ids
    )

    local_kernels_const = generate_local_kernels_const(active_kernel_ids)
    # Só declara o estado de controle do Gemm GPU se ele de fato estiver
    # presente neste nó — evita código morto/não-usado nos outros casos.
    gemm_gpu_state = ""
    if 8 in active_kernel_ids and variant_map.get(8) == "gpu":
        gemm_gpu_state = (
            "thread_local! {\n"
            "    // Controla quais weight_id (= weight_path) já foram enviados\n"
            "    // ao kernel GPU via gemm_load_weights neste processo. Evita\n"
            "    // reenviar o Vec<f32> inteiro pela fronteira WIT em chamadas\n"
            "    // repetidas da mesma camada.\n"
            "    static GEMM_GPU_LOADED: RefCell<std::collections::HashSet<String>> = RefCell::new(std::collections::HashSet::new());\n"
            "}"
        )
    return f"""use std::collections::{{HashMap, VecDeque}};
use std::env;
use std::io::{{Read, Write}};
use std::net::{{Shutdown, TcpListener, TcpStream}};
use std::time::Instant;
use std::cell::RefCell;
use std::sync::Arc;
use serde::Deserialize;

{local_kernels_const}
{gemm_gpu_state}

type WitData = bindings::planner::{first_kernel_mod}::plan::Data;

thread_local! {{
    // Cache de pesos/bias já lidos do disco, chaveado pelo path do arquivo.
    // Conv e Gemm costumam ser chamados repetidas vezes com o mesmo
    // weight_path (mesma camada, requests diferentes) dentro do mesmo
    // processo — sem isso, cada chamada relia o .bin inteiro do zero.
    static WEIGHTS_CACHE: RefCell<HashMap<String, Arc<Vec<f32>>>> = RefCell::new(HashMap::new());
}}

fn load_weights_file_cached(path: &str) -> Arc<Vec<f32>> {{
    if path.is_empty() {{
        return Arc::new(Vec::new());
    }}
    WEIGHTS_CACHE.with(|cache| {{
        if let Some(existing) = cache.borrow().get(path) {{
            return existing.clone();
        }}
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("Erro ao carregar pesos de {{}}: {{:?}}", path, e));
        let data = Arc::new(unsafe {{
            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4).to_vec()
        }});
        cache.borrow_mut().insert(path.to_string(), data.clone());
        data
    }})
}}

mod bindings {{
    use super::Component;
    wit_bindgen::generate!({{
        generate_all,
        additional_derives: [
            serde::Deserialize,
            serde::Serialize,
        ],
    }});
    export!(Component);
}}

struct Component;

#[derive(Deserialize)]
struct RoutingTable {{
    table: std::collections::HashMap<u8, String>,
    node_scores: std::collections::HashMap<u8, f32>,
    assignments: HashMap<u8, Vec<u8>>,
}}

#[derive(Deserialize, Debug, Clone, Default)]
struct LayerParams {{
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
}}

#[derive(Deserialize, Debug, Clone)]
struct StepNode {{
    step: u32,
    #[serde(default)]
    name: String,
    kernel_type: u8,
    #[serde(default)]
    params: LayerParams,
    #[serde(default)]
    next_steps: Vec<u32>,
}}

#[derive(Deserialize, Debug)]
struct ModelGraph {{
    model: String,
    steps: Vec<StepNode>,
}}

// Kernels não presentes neste binário caem no braço `_` do match abaixo e
// são despachados via TCP para o endereço indicado na routing_table para
// aquele kernel_type. Kernels presentes rodam direto, sem rede.
fn handle_client(
    mut stream: TcpStream,
    initialized: &mut i32,
    graph: &ModelGraph,
    routing: &RoutingTable,
) -> std::io::Result<()> {{
    let mut len_buf = [0u8; 4];

    loop {{
        let start_receive = Instant::now();
        if let Err(e) = stream.read_exact(&mut len_buf) {{
            if e.kind() == std::io::ErrorKind::UnexpectedEof {{
                break;
            }}
            return Err(e);
        }}

        let len = u32::from_be_bytes(len_buf) as usize;
        println!("Recebendo payload de {{}} bytes...", len);

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

        let mut queue: VecDeque<(u32, Vec<f32>)> = VecDeque::new();
        let mut add_branches: HashMap<u32, Vec<f32>> = HashMap::new();

        queue.push_back((input_img.current_kernel, input_img.input.clone()));

        let mut final_output: Vec<f32> = Vec::new();
        let mut reached_final = false;
        let reply_to = input_img.reply_to.clone();
        let request_id = input_img.request;

        while let Some((current_step, current_input)) = queue.pop_front() {{
            let current_node = graph.steps.iter().find(|s| s.step == current_step)
                .unwrap_or_else(|| panic!("Step {{}} não encontrado no grafo", current_step));

            println!("==> Executando Step {{}} | Camada: '{{}}' | Tipo: {{}}", current_step, current_node.name, current_node.kernel_type);

            let p = &current_node.params;
            let computed_output: Vec<f32>;
            let input_len = current_input.len(); // [T]

            // [T] Timing no nível de STEP INTEIRO (inclui carregar pesos +
            // executar o kernel), não decomposto como nos receivers isolados
            // (que separam weights_load de kernel_call). Decompor exigiria
            // reescrever os 9 rust_match do catálogo KERNELS individualmente.
            let t_step = Instant::now(); // [T]

            match current_node.kernel_type {{
{match_arms}
                _ => {{
                    // Kernel não incluído neste binário: despacha por rede.
                    let t_route = Instant::now(); // [T]
                    let target_address = routing.table.get(&current_node.kernel_type)
                        .cloned()
                        .unwrap_or_else(|| panic!(
                            "Kernel tipo {{}} não está neste nó nem na routing_table",
                            current_node.kernel_type
                        ));

                    println!(
                        "Step {{}} (tipo {{}}) não está neste nó — despachando para {{}}...",
                        current_step, current_node.kernel_type, target_address
                    );

                    let payload_data = WitData {{
                        input: current_input,
                        output: vec![],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                    }};

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

                    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("receiver_logs.jsonl") {{ // [T]
                        let log_linha = serde_json::json!({{
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
                            "timing_us": {{
                                "net_in": net_in_us,
                                "deserialize": deserialize_us,
                                "routing_load": t_route.elapsed().as_micros() as u64,
                                "connect": connect_us,
                                "serialize": serialize_us,
                                "write": write_us,
                                "ack_wait": ack_wait_us,
                                "step_total": step_total_us
                            }}
                        }});
                        let _ = writeln!(file, "{{}}", log_linha);
                    }} // [T]

                    continue;
                }},
            }}

            let step_total_us = t_step.elapsed().as_micros() as u64; // [T]
            let output_len = computed_output.len(); // [T]
            let is_local = LOCAL_KERNELS.contains(&current_node.kernel_type); // [T]

            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("receiver_logs.jsonl") {{
                let log_linha = serde_json::json!({{
                    "request": request_id,
                    "kernel_id": current_node.kernel_type,
                    "kernel_type": current_node.kernel_type,
                    "step": current_step,
                    "layer_name": current_node.name,
                    "next_steps": current_node.next_steps,
                    "call_idx": call_idx, // [T]
                    "ts_unix_ms": ts_unix_ms, // [T]
                    "dispatch": if is_local {{ "local" }} else {{ "network" }}, // [T]
                    "input_len": input_len, // [T]
                    "output_len": output_len, // [T]
                    "timing_us": {{ // [T]
                        "net_in": net_in_us,
                        "deserialize": deserialize_us,
                        "kernel_call": step_total_us
                    }}
                }});
                let _ = writeln!(file, "{{}}", log_linha);
            }}

            if current_node.next_steps.is_empty() {{
                println!("Fim da rede alcançado localmente no Step {{}}! Saída com {{}} elementos.", current_step, computed_output.len());
                final_output = computed_output;
                reached_final = true;
                break;
            }} else {{
                for &nxt in &current_node.next_steps {{
                    queue.push_back((nxt, computed_output.clone()));
                }}
            }}
        }}

        // Só devolve pro cliente se a rede terminou DENTRO deste nó. Se tudo
        // que restava foi despachado por rede, quem responde é o nó remoto.
        if reached_final {{
            let final_data = WitData {{
                input: final_output,
                output: vec![],
                reply_to: reply_to.clone(),
                current_kernel: 0,
                request: request_id,
                kernels: vec![],
            }};

            let mut next_stream = TcpStream::connect(&reply_to)?;
            let response_payload = rmp_serde::to_vec(&final_data).expect("Erro ao serializar resposta");
            let resp_len = response_payload.len() as u32;
            next_stream.write_all(&resp_len.to_be_bytes())?;
            next_stream.write_all(&response_payload)?;
            next_stream.flush()?;
        }}

        break;
    }}

    println!("Conexão finalizada com o cliente.");
    Ok(())
}}

impl bindings::exports::wasi::cli::run::Guest for Component {{
    fn run() -> Result<(), ()> {{
        let args: Vec<String> = env::args().collect();
        let port = args.get(1).map(|s| s.as_str()).unwrap_or("8080");
        let ip = args.get(2).map(|s| s.as_str()).unwrap_or("0.0.0.0");
        let bind_addr = format!("{{}}:{{}}", ip, port);
        let listener = TcpListener::bind(&bind_addr).expect(&format!("Não conseguiu abrir a porta {{}}", port));

        // Grafo e routing table carregados uma vez, no boot do processo.
        let graph_file = std::fs::File::open("resnet18DFG.json").expect("Erro ao abrir resnet18DFG.json");
        let graph: ModelGraph = serde_json::from_reader(graph_file).expect("Erro no parser do resnet18DFG.json");

        let routing_file = std::fs::File::open("routing_table.json").expect("Erro ao abrir routing_table.json");
        let routing: RoutingTable = serde_json::from_reader(routing_file).expect("Erro no parser do routing_table.json");

        println!("Listening on {{}}", bind_addr);

        let mut init = 0;
        for stream in listener.incoming() {{
            match stream {{
                Ok(stream) => {{
                    println!("Nova conexão recebida!");
                    if let Err(e) = handle_client(stream, &mut init, &graph, &routing) {{
                        eprintln!("Error handling client: {{:?}}", e);
                    }}
                }}
                Err(e) => {{
                    eprintln!("Connection failed: {{:?}}", e);
                }}
            }}
            println!("Waiting for a new request...");
        }}
        Ok(())
    }}
}}
"""


# 3. PIPELINE DE EXECUÇÃO
def main():
    parser = argparse.ArgumentParser(description="Compositor Dinâmico de Componentes WASM (CPU + GPU)")
    parser.add_argument("--kernels", type=str, required=True,
                         help="Lista de kernels a incluir NESTE binário, separados por vírgula (ex: 1,2,3,4,5)")
    parser.add_argument("--gpu", type=str, default="",
                         help="Subconjunto de --kernels que deve usar a variante GPU (ex: 1,2,4)")
    parser.add_argument("--component-dir", type=str, default=".", help="Diretório base do crate do receiver")
    parser.add_argument("--output", type=str, default="custom_node.wasm", help="Nome do arquivo WASM final composto")
    args = parser.parse_args()

    kernel_ids = [int(k.strip()) for k in args.kernels.split(",") if k.strip()]
    gpu_ids = [int(k.strip()) for k in args.gpu.split(",") if k.strip()]

    for k in kernel_ids:
        if k not in KERNELS:
            print(f"[ERRO] Kernel ID {k} inválido. Disponíveis: 1 a 9.")
            sys.exit(1)

    for k in gpu_ids:
        if k not in kernel_ids:
            print(f"[ERRO] --gpu contém {k}, que não está em --kernels.")
            sys.exit(1)
        if "gpu" not in KERNELS[k]["variants"]:
            supported = [kid for kid, meta in KERNELS.items() if "gpu" in meta["variants"]]
            print(f"[ERRO] Kernel {k} ({KERNELS[k]['name']}) não tem variante GPU. "
                  f"Kernels com GPU disponível: {supported}")
            sys.exit(1)

    variant_map = {k: ("gpu" if k in gpu_ids else "cpu") for k in kernel_ids}

    print(f"[*] Kernels incluídos neste nó: {kernel_ids}")
    print(f"[*] Variantes: {variant_map}")
    print(f"[*] Kernels NÃO incluídos serão despachados por rede via routing_table.json em runtime.")

    comp_dir = os.path.abspath(args.component_dir)
    wit_dir = os.path.join(comp_dir, "wit")
    src_dir = os.path.join(comp_dir, "src")
    os.makedirs(wit_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)

    with open(os.path.join(wit_dir, "world.wit"), "w") as f:
        f.write(generate_world_wit(kernel_ids))
    print(" -> wit/world.wit gerado.")

    with open(os.path.join(comp_dir, "wkg.toml"), "w") as f:
        f.write(generate_wkg_toml(kernel_ids, variant_map))
    print(" -> wkg.toml gerado.")

    with open(os.path.join(src_dir, "lib.rs"), "w") as f:
        f.write(generate_lib_rs(kernel_ids, variant_map))
    print(" -> src/lib.rs gerado.")

    print("[*] Atualizando dependências WIT com wkg...")
    res_wkg = subprocess.run(["wkg", "wit", "fetch"], cwd=comp_dir)
    if res_wkg.returncode != 0:
        print("[AVISO] wkg wit fetch falhou — confira os overrides em wkg.toml antes de compilar.")

    print("[*] Compilando componente receiver com Cargo...")
    cmd_cargo = ["cargo", "build", "--target", "wasm32-wasip2", "--release"]
    res = subprocess.run(cmd_cargo, cwd=comp_dir)
    if res.returncode != 0:
        print("[ERRO] Falha ao compilar o receiver em Rust.")
        sys.exit(1)

    release_dir = os.path.join(comp_dir, "target/wasm32-wasip2/release")
    crate_name = os.path.basename(comp_dir.rstrip("/"))
    current_wasm = os.path.join(release_dir, f"{crate_name}.wasm")

    if not os.path.exists(current_wasm):
        wasms = [os.path.join(release_dir, f) for f in os.listdir(release_dir)
                 if f.endswith(".wasm") and not f.startswith("temp_")]
        if wasms:
            current_wasm = wasms[0]
        else:
            print(f"[ERRO] Binário .wasm não encontrado em {release_dir}")
            sys.exit(1)

    print(f"[*] Artefato base localizado: {current_wasm}")
    print("[*] Compondo kernels via WAC...")

    temp_wasm = current_wasm
    temp_files = []

    for idx, k in enumerate(kernel_ids):
        variant = variant_map[k]
        k_meta = KERNELS[k]["variants"][variant]
        k_wasm_path = os.path.normpath(os.path.join(comp_dir, k_meta["wasm_path"]))

        if not os.path.exists(k_wasm_path):
            print(f"[ERRO] Binário do kernel {KERNELS[k]['name']} ({variant}) não encontrado em: {k_wasm_path}")
            sys.exit(1)

        if idx == len(kernel_ids) - 1:
            out_step = os.path.abspath(args.output)
        else:
            out_step = os.path.join(comp_dir, f"temp_plug_{idx}.wasm")
            temp_files.append(out_step)

        print(f" -> Plugando Kernel {k} ({KERNELS[k]['name']}, {variant})...")
        cmd_wac = ["wac", "plug", temp_wasm, "--plug", k_wasm_path, "-o", out_step]
        res = subprocess.run(cmd_wac)
        if res.returncode != 0:
            print(f"[ERRO] Falha ao executar wac plug no kernel {k}")
            sys.exit(1)

        temp_wasm = out_step

    for f in temp_files:
        if os.path.exists(f) and f != os.path.abspath(args.output):
            try:
                os.remove(f)
            except OSError:
                pass

    print(f"\n[SUCESSO] Componente final gerado com sucesso: {args.output}")


if __name__ == "__main__":
    main()