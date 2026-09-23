#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

# 1. CATÁLOGO CENTRALIZADO DOS KERNELS
KERNELS = {
    1: {
        "name": "conv",
        "wit_import": "    import planner:convworld/plan;",
        "wkg_override": '"planner:convworld" = { path = "../../../kernel/conv/" }',
        "module_name": "convworld",
        "c_wasm": "../../../kernel/conv/convworld_component.wasm",
        "rust_match": """                1 => {
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
                },"""
    },
    2: {
        "name": "bn",
        "wit_import": "    import planner:bnworld/plan;",
        "wkg_override": '"planner:bnworld"  = { path = "../../../kernel/bn/" }',
        "module_name": "bnworld",
        "c_wasm": "../../../kernel/bn/bnworld_component.wasm",
        "rust_match": """                2 => {
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
                },"""
    },
    3: {
        "name": "relu",
        "wit_import": "    import planner:reluworld/plan;",
        "wkg_override": '"planner:reluworld"  = { path = "../../../kernel/relu/" }',
        "module_name": "reluworld",
        "c_wasm": "../../../kernel/relu/reluworld_component.wasm",
        "rust_match": """                3 => {
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
                },"""
    },
    4: {
        "name": "maxpool",
        "wit_import": "    import planner:maxpoolworld/plan;",
        "wkg_override": '"planner:maxpoolworld"  = { path = "../../../kernel/maxpool/" }',
        "module_name": "maxpoolworld",
        "c_wasm": "../../../kernel/maxpool/maxpoolworld_component.wasm",
        "rust_match": """                4 => {
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
                },"""
    },
    5: {
        "name": "add",
        "wit_import": "    import planner:addworld/plan;",
        "wkg_override": '"planner:addworld"  = { path = "../../../kernel/add/" }',
        "module_name": "addworld",
        "c_wasm": "../../../kernel/add/addworld_component.wasm",
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
                },"""
    },
    6: {
        "name": "gap",
        "wit_import": "    import planner:gapworld/plan;",
        "wkg_override": '"planner:gapworld"  = { path = "../../../kernel/gap/" }',
        "module_name": "gapworld",
        "c_wasm": "../../../kernel/gap/gapworld_component.wasm",
        "rust_match": """                6 => {
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
                },"""
    },
    7: {
        "name": "flatten",
        "wit_import": "    import planner:flattenworld/plan;",
        "wkg_override": '"planner:flattenworld"  = { path = "../../../kernel/flatten/" }',
        "module_name": "flattenworld",
        "c_wasm": "../../../kernel/flatten/flattenworld_component.wasm",
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
                },"""
    },
    8: {
        "name": "gemm",
        "wit_import": "    import planner:gemmworld/plan;",
        "wkg_override": '"planner:gemmworld"  = { path = "../../../kernel/gemm/" }',
        "module_name": "gemmworld",
        "c_wasm": "../../../kernel/gemm/gemmworld_component.wasm",
        "rust_match": """                8 => {
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
                },"""
    },
    9: {
        "name": "dropout",
        "wit_import": "    import planner:dropoutworld/plan;",
        "wkg_override": '"planner:dropoutworld" = { path = "../../../kernel/dropout/" }',
        "module_name": "dropoutworld",
        "c_wasm": "../../../kernel/dropout/dropoutworld_component.wasm",
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
                },"""
    }
}

# 2. GERADORES DE CONTEÚDO DOS ARQUIVOS
def generate_world_wit(active_kernel_ids):
    imports = "\n".join([KERNELS[k]["wit_import"] for k in active_kernel_ids])
    return f"""package example:server;

world runner {{
{imports}
    export wasi:cli/run@0.2.7;
}}
"""

def generate_wkg_toml(active_kernel_ids):
    overrides = "\n".join([KERNELS[k]["wkg_override"] for k in active_kernel_ids])
    return f"""[overrides]
{overrides}
"""

def generate_lib_rs(active_kernel_ids):
    first_kernel_mod = KERNELS[active_kernel_ids[0]]["module_name"]
    match_arms = "\n".join([KERNELS[k]["rust_match"] for k in active_kernel_ids])

    return f"""use std::collections::{{HashMap, VecDeque}};
use std::env;
use std::io::{{Read, Write}};
use std::net::{{Shutdown, TcpListener, TcpStream}};
use std::time::Instant;
use serde::Deserialize;

type WitData = bindings::planner::{first_kernel_mod}::plan::Data;

fn load_weights_file(path: &str) -> Vec<f32> {{
    if path.is_empty() {{
        return Vec::new();
    }}
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("Erro ao carregar pesos de {{}}: {{:?}}", path, e));
    unsafe {{
        std::slice::from_raw_parts(
            bytes.as_ptr() as *const f32,
            bytes.len() / 4,
        ).to_vec()
    }}
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

fn handle_client(mut stream: TcpStream, _initialized: &mut i32) -> std::io::Result<()> {{
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

        while let Some((current_step, current_input)) = queue.pop_front() {{
            let current_node = graph.steps.iter().find(|s| s.step == current_step)
                .unwrap_or_else(|| panic!("Step {{}} não encontrado no grafo", current_step));

            println!("==> Executando Step {{}} | Camada: '{{}}' | Tipo: {{}}", current_step, current_node.name, current_node.kernel_type);

            let p = &current_node.params;
            let start_step_exec = Instant::now();
            let computed_output: Vec<f32>;

            match current_node.kernel_type {{
{match_arms}
                _ => panic!("Kernel type {{}} não suportado nesta partição de nó", current_node.kernel_type),
            }}

            let step_duration = start_step_exec.elapsed().as_millis();

            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("receiver_logs.jsonl") {{
                let log_linha = serde_json::json!({{
                    "request": request_id,
                    "kernel_id": current_node.kernel_type,
                    "step": current_step,
                    "layer_name": current_node.name,
                    "next_steps": current_node.next_steps,
                    "exec_ms": step_duration
                }});
                let _ = writeln!(file, "{{}}", log_linha);
            }}

            if current_node.next_steps.is_empty() {{
                println!("Fim da rede alcançado no Step {{}}! Saída com {{}} elementos.", current_step, computed_output.len());
                final_output = computed_output;
                break;
            }} else {{
                for &nxt in &current_node.next_steps {{
                    queue.push_back((nxt, computed_output.clone()));
                }}
            }}
        }}

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

        println!("Listening on {{}}", bind_addr);

        let mut init = 0;
        for stream in listener.incoming() {{
            match stream {{
                Ok(stream) => {{
                    println!("Nova conexão recebida!");
                    if let Err(e) = handle_client(stream, &mut init) {{
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
    parser = argparse.ArgumentParser(description="Compilador e Compositor Dinâmico de Componentes WASM")
    parser.add_argument("--kernels", type=str, required=True, help="Lista de kernels separados por vírgula (ex: 2,7 ou 1,2,3,4,5,6,7,8)")
    parser.add_argument("--component-dir", type=str, default=".", help="Diretório base do crate do receiver")
    parser.add_argument("--output", type=str, default="custom_node.wasm", help="Nome do arquivo WASM final composto")
    args = parser.parse_args()

    kernel_ids = [int(k.strip()) for k in args.kernels.split(",") if k.strip()]
    for k in kernel_ids:
        if k not in KERNELS:
            print(f"[ERRO] Kernel ID {k} inválido. Disponíveis: 1 a 9.")
            sys.exit(1)

    print(f"[*] Gerando configuração para os kernels: {kernel_ids}")

    # Escrever os 3 arquivos
    comp_dir = os.path.abspath(args.component_dir)
    wit_dir = os.path.join(comp_dir, "wit")
    src_dir = os.path.join(comp_dir, "src")
    os.makedirs(wit_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)

    with open(os.path.join(wit_dir, "world.wit"), "w") as f:
        f.write(generate_world_wit(kernel_ids))
    print(" -> wit/world.wit gerado.")

    with open(os.path.join(comp_dir, "wkg.toml"), "w") as f:
        f.write(generate_wkg_toml(kernel_ids))
    print(" -> wkg.toml gerado.")

    with open(os.path.join(src_dir, "lib.rs"), "w") as f:
        f.write(generate_lib_rs(kernel_ids))
    print(" -> src/lib.rs gerado.")

    # Sincronizar dependências WIT via wkg ou cópia
    print("[*] Atualizando dependências WIT com wkg...")
    res_wkg = subprocess.run(["wkg", "wit", "update"], cwd=comp_dir)
    if res_wkg.returncode != 0:
        # Fallback: copiar wit/deps do receiver/all se o wkg falhar offline
        all_deps_dir = os.path.normpath(os.path.join(comp_dir, "../all/wit/deps"))
        target_deps_dir = os.path.join(wit_dir, "deps")
        if os.path.exists(all_deps_dir):
            import shutil
            if os.path.exists(target_deps_dir):
                shutil.rmtree(target_deps_dir)
            shutil.copytree(all_deps_dir, target_deps_dir)
            print(" -> wit/deps sincronizado a partir de receiver/all.")

    # Compilar o componente base Rust
    print("[*] Compilando componente receiver com Cargo...")
    cmd_cargo = ["cargo", "build", "--target", "wasm32-wasip2", "--release"]
    res = subprocess.run(cmd_cargo, cwd=comp_dir)
    if res.returncode != 0:
        print("[ERRO] Falha ao compilar o receiver em Rust.")
        sys.exit(1)

    # Identificar o artefato intermediário compilado
    release_dir = os.path.join(comp_dir, "target/wasm32-wasip2/release")
    crate_name = os.path.basename(comp_dir.rstrip("/"))
    current_wasm = os.path.join(release_dir, f"{crate_name}.wasm")

    if not os.path.exists(current_wasm):
        wasms = [os.path.join(release_dir, f) for f in os.listdir(release_dir) if f.endswith(".wasm") and not f.startswith("custom_") and not f.startswith("temp_")]
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
        k_meta = KERNELS[k]
        k_wasm_path = os.path.normpath(os.path.join(comp_dir, k_meta["c_wasm"]))
        
        if not os.path.exists(k_wasm_path):
            print(f"[ERRO] Binário C do kernel {k_meta['name']} não encontrado em: {k_wasm_path}")
            sys.exit(1)

        if idx == len(kernel_ids) - 1:
            out_step = os.path.abspath(args.output)
        else:
            out_step = os.path.join(comp_dir, f"temp_plug_{idx}.wasm")
            temp_files.append(out_step)

        print(f" -> Plugando Kernel {k} ({k_meta['name']})...")
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
    