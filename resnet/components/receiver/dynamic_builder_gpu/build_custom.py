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
        # 09/11 (4a): a variante GPU recebe os pesos uma vez só (load_weights)
        # e depois só o weight-id; a CPU continua igual.
        "rust_match_variants": {
            "cpu": """                1 => {
                    let out_h = ((p.h_in + 2 * p.padding - p.kh) / p.stride) + 1;
                    let out_w = ((p.w_in + 2 * p.padding - p.kw) / p.stride) + 1;
                    let output_size = (p.cout * out_h * out_w) as usize;
                    let weights = load_weights_file_cached(&p.weight_path);
                    let bias = load_weights_file_cached(&p.bias_path);
                    let conv_groups = if p.groups == 0 { 1 } else { p.groups };
                    let data = bindings::planner::convworld::plan::Data {
                        input: current_input,
                        output: vec![0.0f32; output_size],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                        from_step: origin_step,
                    model_id: input_img.model_id,
                    };
                    let res = bindings::planner::convworld::plan::conv(
                        p.cout, p.cin, p.h_in, p.w_in, p.kh, p.kw, p.stride, p.padding, conv_groups,
                        weights.as_slice(), bias.as_slice(), &data,
                    );
                    computed_output = res.output;
                },""",
            "gpu": """                1 => {
                    // 10/11 (4c): passo GPU local é tratado pela via residente
                    // antes do match; aqui nunca chega.
                    unreachable!("4c: tipo 1 na GPU vai pela via residente")
                },""",
        },
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
        # 09/11 (4a): a variante GPU recebe os pesos uma vez só (load_weights)
        # e depois só o weight-id; a CPU continua igual.
        "rust_match_variants": {
            "cpu": """                2 => {
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
                        from_step: origin_step,
                    model_id: input_img.model_id,
                    };
                    let res = bindings::planner::bnworld::plan::bn(
                        p.channels, p.h_in, p.w_in, p.epsilon,
                        scale.as_slice(), beta.as_slice(), mean.as_slice(), var.as_slice(), &data,
                    );
                    computed_output = res.output;
                },""",
            "gpu": """                2 => {
                    // 10/11 (4c): passo GPU local é tratado pela via residente
                    // antes do match; aqui nunca chega.
                    unreachable!("4c: tipo 2 na GPU vai pela via residente")
                },""",
        },
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
                        from_step: origin_step,
                    model_id: input_img.model_id,
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
        "rust_match_variants": {
            "cpu": """                4 => {
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
                        from_step: origin_step,
                    model_id: input_img.model_id,
                    };
                    let res = bindings::planner::maxpoolworld::plan::maxpool(
                        p.channels, p.h_in, p.w_in, p.kh, p.kw, p.stride, p.padding, &data,
                    );
                    computed_output = res.output;
                },""",
            "gpu": """                4 => {
                    // 10/11 (4c): passo GPU local é tratado pela via residente
                    // antes do match; aqui nunca chega.
                    unreachable!("4c: tipo 4 na GPU vai pela via residente")
                },""",
        },
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
        # Add tem dois casos, igual ao Mul (kernel 11): tensor x constante
        # (params.const_path preenchido, ex: "/model.23/Add_1" do decode do
        # YOLO — x2y2 = anchor_points + rb, dist2bbox) ou tensor x tensor
        # (join-buffer por step, o caso de sempre: resíduo de bloco conv em
        # ResNet/VGG/YOLO). Add é comutativo, então ao contrário de Sub/Div
        # não precisa de const_is_first — a ordem dos operandos não muda o
        # resultado.
        "rust_match": """                5 => {
                    if !p.const_path.is_empty() {
                        let const_vec = load_weights_file_cached(&p.const_path);
                        let data = bindings::planner::addworld::plan::Data {
                            input: current_input.clone(),
                            output: vec![],
                            reply_to: reply_to.clone(),
                            current_kernel: current_step,
                            request: request_id,
                            kernels: vec![],
                            from_step: origin_step,
                        model_id: input_img.model_id,
                        };
                        let res = bindings::planner::addworld::plan::add(&current_input, const_vec.as_slice(), &data);
                        computed_output = res.output;
                    } else if let Some(first_branch) = add_branches.remove(&current_step) {
                        vprintln!("Step {}: 2º ramo recebido! Somando buffers de tamanhos {} e {}...", current_step, first_branch.len(), current_input.len());
                        let data = bindings::planner::addworld::plan::Data {
                            input: current_input.clone(),
                            output: vec![],
                            reply_to: reply_to.clone(),
                            current_kernel: current_step,
                            request: request_id,
                            kernels: vec![],
                            from_step: origin_step,
                        model_id: input_img.model_id,
                        };
                        let res = bindings::planner::addworld::plan::add(&first_branch, &current_input, &data);
                        computed_output = res.output;
                    } else {
                        vprintln!("Step {}: 1º ramo guardado no buffer. Aguardando outro ramo...", current_step);
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
                        from_step: origin_step,
                    model_id: input_img.model_id,
                    };
                    let res = bindings::planner::gapworld::plan::gap(&current_input, &data);
                    computed_output = res.output;
                },""",
            "gpu": """                6 => {
                    // 10/11 (4c): passo GPU local é tratado pela via residente
                    // antes do match; aqui nunca chega.
                    unreachable!("4c: tipo 6 na GPU vai pela via residente")
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
                        from_step: origin_step,
                    model_id: input_img.model_id,
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
                        from_step: origin_step,
                    model_id: input_img.model_id,
                    };
                    let res = bindings::planner::gemmworld::plan::gemm(
                        p.in_features, p.out_features, weights.as_slice(), bias.as_slice(), &data,
                    );
                    computed_output = res.output;
                },""",
            "gpu": """                8 => {
                    // 10/11 (4c): passo GPU local é tratado pela via residente
                    // antes do match; aqui nunca chega.
                    unreachable!("4c: tipo 8 na GPU vai pela via residente")
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
                        from_step: origin_step,
                    model_id: input_img.model_id,
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
    10: {
        "name": "sigmoid",
        "wit_import": "    import planner:sigmoidworld/plan;",
        "rust_match": """                10 => {
                    let data = bindings::planner::sigmoidworld::plan::Data {
                        input: current_input.clone(),
                        output: vec![0.0f32; current_input.len()],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                        from_step: origin_step,
                    model_id: input_img.model_id,
                    };
                    let res = bindings::planner::sigmoidworld::plan::sigmoid(&data);
                    computed_output = res.output;
                },""",
        "variants": {
            "cpu": {
                "wkg_override": '"planner:sigmoidworld" = { path = "../../../kernel/sigmoid/" }',
                "wasm_path": "../../../kernel/sigmoid/sigmoidworld_component.wasm",
            },
            "gpu": {
                "wkg_override": '"planner:sigmoidworld" = { path = "../../../kernel/sigmoid_gpu/wit" }',
                "wasm_path": "../../../kernel/sigmoid_gpu/target/wasm32-wasip2/release/sigmoid_gpu.wasm",
            },
        },
    },
    11: {
        "name": "mul",
        "wit_import": "    import planner:mulworld/plan;",
        # Mul tem dois casos, igual ao receiver isolado (kernel/mul*/src/lib.rs):
        # tensor x constante (params.const_path preenchido, ex: SE-block/scale
        # fixo) ou tensor x tensor (join-buffer por step, igual ao Add).
        "rust_match": """                11 => {
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
                        model_id: input_img.model_id,
                        };
                        let res = bindings::planner::mulworld::plan::mul(&current_input, const_vec.as_slice(), &data);
                        computed_output = res.output;
                    } else if let Some(first_branch) = mul_branches.remove(&current_step) {
                        vprintln!("Step {}: 2º ramo recebido! Multiplicando buffers de tamanhos {} e {}...", current_step, first_branch.len(), current_input.len());
                        let data = bindings::planner::mulworld::plan::Data {
                            input: current_input.clone(),
                            output: vec![],
                            reply_to: reply_to.clone(),
                            current_kernel: current_step,
                            request: request_id,
                            kernels: vec![],
                            from_step: origin_step,
                        model_id: input_img.model_id,
                        };
                        let res = bindings::planner::mulworld::plan::mul(&first_branch, &current_input, &data);
                        computed_output = res.output;
                    } else {
                        vprintln!("Step {}: 1º ramo guardado no buffer. Aguardando outro ramo...", current_step);
                        mul_branches.insert(current_step, current_input);
                        continue;
                    }
                },""",
        "variants": {
            "cpu": {
                "wkg_override": '"planner:mulworld" = { path = "../../../kernel/mul/" }',
                "wasm_path": "../../../kernel/mul/mulworld_component.wasm",
            },
            "gpu": {
                "wkg_override": '"planner:mulworld" = { path = "../../../kernel/mul_gpu/wit" }',
                "wasm_path": "../../../kernel/mul_gpu/target/wasm32-wasip2/release/mul_gpu.wasm",
            },
        },
    },
    12: {
        "name": "silu",
        "wit_import": "    import planner:siluworld/plan;",
        "rust_match": """                12 => {
                    let data = bindings::planner::siluworld::plan::Data {
                        input: current_input.clone(),
                        output: vec![0.0f32; current_input.len()],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                        from_step: origin_step,
                    model_id: input_img.model_id,
                    };
                    let res = bindings::planner::siluworld::plan::silu(&data);
                    computed_output = res.output;
                },""",
        "variants": {
            "cpu": {
                "wkg_override": '"planner:siluworld" = { path = "../../../kernel/silu/" }',
                "wasm_path": "../../../kernel/silu/siluworld_component.wasm",
            },
            "gpu": {
                "wkg_override": '"planner:siluworld" = { path = "../../../kernel/silu_gpu/wit" }',
                "wasm_path": "../../../kernel/silu_gpu/target/wasm32-wasip2/release/silu_gpu.wasm",
            },
        },
    },
    13: {
        "name": "split",
        "wit_import": "    import planner:splitworld/plan;",
        # Split não devolve 1 output — devolve Vec<Vec<f32>> roteado por
        # output_routes (StepNode), não pelo next_steps genérico (destinos
        # diferentes recebem FATIAS diferentes do mesmo step). Por isso este
        # braço nunca passa por computed_output: monta a Data, chama o
        # kernel, e empilha cada fatia direto na queue via `continue`.
        "rust_match": """                13 => {
                    if current_node.output_routes.is_empty() {
                        panic!("Step {} (Split) sem output_routes — grafo desatualizado?", current_step);
                    }
                    let shape: Vec<u32> = p.input_shape.iter().map(|&d| if d > 0 { d as u32 } else { 1 }).collect();
                    let sizes: Vec<u32> = match &p.split {
                        Some(s) => s.clone(),
                        None => {
                            let n = p.num_outputs.max(1);
                            let total = *shape.get(p.axis as usize).unwrap_or(&0);
                            assert_eq!(total % n, 0, "Split step {}: shape[axis]={} não divide igualmente em num_outputs={}", current_step, total, n);
                            vec![total / n; n as usize]
                        }
                    };
                    let data = bindings::planner::splitworld::plan::Data {
                        input: vec![],
                        output: vec![],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                        from_step: origin_step,
                    model_id: input_img.model_id,
                    };
                    let outputs = bindings::planner::splitworld::plan::split(&current_input, &shape, p.axis, &sizes, &data);
                    let out_len: usize = outputs.iter().map(|o| o.len()).sum();
                    {
                        let log_linha = serde_json::json!({
                            "request": request_id,
                            "kernel_id": 13u8,
                            "step": current_step,
                            "layer_name": current_node.name,
                            "dispatch": "local-split",
                            "output_routes": current_node.output_routes.iter().map(|r| (r.to_step, r.output_index)).collect::<Vec<_>>(),
                            "input_len": input_len,
                            "output_len": out_len,
                        });
                        log_push(log_linha.to_string());
                    }
                    for route in &current_node.output_routes {
                        let slice = outputs.get(route.output_index as usize).unwrap_or_else(|| {
                            panic!("Split step {}: output_routes aponta output_index {} mas o kernel só devolveu {} fatias", current_step, route.output_index, outputs.len())
                        });
                        queue.push_back((route.to_step, slice.clone(), current_step));
                    }
                    continue;
                },""",
        "variants": {
            "cpu": {
                "wkg_override": '"planner:splitworld" = { path = "../../../kernel/split/" }',
                "wasm_path": "../../../kernel/split/splitworld_component.wasm",
            },
        },
    },
    14: {
        "name": "concat",
        "wit_import": "    import planner:concatworld/plan;",
        # Join genérico por N ramos (não só 2 como Add/Mul), não-comutativo —
        # ordem vem de input_order, não da ordem de chegada. Guardamos por
        # (step, from_step) -> VecDeque, porque o MESMO from_step pode
        # aparecer 2x em input_order (as duas metades de um Split indo pro
        # mesmo Concat): consumimos em FIFO por from_step, que corresponde à
        # ordem de output_index com que o Split empilhou as rotas.
        "rust_match": """                14 => {
                    if p.input_order.is_empty() {
                        panic!("Step {} (Concat) sem input_order nos params — grafo desatualizado?", current_step);
                    }
                    if p.input_order.iter().any(|e| e.is_const) {
                        panic!("Concat step {}: entrada constante no input_order ainda não é suportada por este dynamic_builder", current_step);
                    }
                    let expected = p.input_order.iter().filter(|e| !e.is_const).count();
                    {
                        let step_slots = concat_branches.entry(current_step).or_insert_with(HashMap::new);
                        step_slots.entry(origin_step).or_insert_with(VecDeque::new).push_back(current_input.clone());
                    }
                    let filled: usize = concat_branches.get(&current_step).map(|m| m.values().map(|v| v.len()).sum()).unwrap_or(0);
                    if filled < expected {
                        vprintln!("Step {}: ramo from_step={} guardado no Concat ({}/{}). Aguardando o resto...", current_step, origin_step, filled, expected);
                        continue;
                    }
                    let mut slots = concat_branches.remove(&current_step).unwrap();
                    let inputs: Vec<Vec<f32>> = p.input_order.iter().map(|e| {
                        let src_step = e.step.unwrap_or_else(|| panic!("Concat step {}: entrada sem step de origem", current_step));
                        slots.get_mut(&src_step).and_then(|dq| dq.pop_front())
                            .unwrap_or_else(|| panic!("Concat step {}: faltou o ramo do step {}", current_step, src_step))
                    }).collect();
                    let shapes: Vec<Vec<u32>> = p.input_order.iter()
                        .map(|e| e.shape.iter().map(|&d| if d > 0 { d as u32 } else { 1 }).collect())
                        .collect();
                    let data = bindings::planner::concatworld::plan::Data {
                        input: vec![],
                        output: vec![],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                        from_step: origin_step,
                    model_id: input_img.model_id,
                    };
                    let res = bindings::planner::concatworld::plan::concat(&inputs, &shapes, p.axis, &data);
                    computed_output = res.output;
                },""",
        "variants": {
            "cpu": {
                "wkg_override": '"planner:concatworld" = { path = "../../../kernel/concat/" }',
                "wasm_path": "../../../kernel/concat/concatworld_component.wasm",
            },
        },
    },
    18: {
        "name": "resize",
        "wit_import": "    import planner:resizeworld/plan;",
        # Upsample nearest 2D (FPN top-down do YOLO) — único input/output,
        # mesma forma dos kernels simples (Conv/MaxPool/GAP). channels/h_in/
        # w_in já existem em LayerParams (compartilhados com Conv/BN/MaxPool);
        # scale_h/scale_w vêm resolvidos pelo onnx_to_dfg.py a partir do
        # initializer "scales" ou "sizes" do nó Resize original.
        "rust_match": """                18 => {
                    let data = bindings::planner::resizeworld::plan::Data {
                        input: current_input.clone(),
                        output: vec![],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                        from_step: origin_step,
                    model_id: input_img.model_id,
                    };
                    let res = bindings::planner::resizeworld::plan::resize(
                        &current_input, p.channels, p.h_in, p.w_in, p.scale_h, p.scale_w, &data,
                    );
                    computed_output = res.output;
                },""",
        "variants": {
            "cpu": {
                "wkg_override": '"planner:resizeworld" = { path = "../../../kernel/resize/" }',
                "wasm_path": "../../../kernel/resize/resizeworld_component.wasm",
            },
        },
    },
    15: {
        "name": "reshape",
        "wit_import": "    import planner:reshapeworld/plan;",
        # Reshape é identidade pura (ONNX Reshape não move memória, só
        # reinterpreta o shape — buffer linear já está no layout certo).
        # Não usa nenhum campo de LayerParams.
        "rust_match": """                15 => {
                    let data = bindings::planner::reshapeworld::plan::Data {
                        input: current_input.clone(),
                        output: vec![],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                        from_step: origin_step,
                    model_id: input_img.model_id,
                    };
                    let res = bindings::planner::reshapeworld::plan::reshape(&current_input, &data);
                    computed_output = res.output;
                },""",
        "variants": {
            "cpu": {
                "wkg_override": '"planner:reshapeworld" = { path = "../../../kernel/reshape/" }',
                "wasm_path": "../../../kernel/reshape/reshapeworld_component.wasm",
            },
        },
    },
    16: {
        "name": "transpose",
        "wit_import": "    import planner:transposeworld/plan;",
        # Transpose é puro gather de memória por permutação de eixos (sem
        # FLOPs, sem ganho em GPU). Reusa input_shape (já existe em
        # LayerParams, compartilhado com Split) + perm (novo).
        "rust_match": """                16 => {
                    let shape: Vec<u32> = p.input_shape.iter().map(|&d| if d > 0 { d as u32 } else { 1 }).collect();
                    let data = bindings::planner::transposeworld::plan::Data {
                        input: current_input.clone(),
                        output: vec![],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                        from_step: origin_step,
                    model_id: input_img.model_id,
                    };
                    let res = bindings::planner::transposeworld::plan::transpose(&current_input, &shape, &p.perm, &data);
                    computed_output = res.output;
                },""",
        "variants": {
            "cpu": {
                "wkg_override": '"planner:transposeworld" = { path = "../../../kernel/transpose/" }',
                "wasm_path": "../../../kernel/transpose/transposeworld_component.wasm",
            },
        },
    },
    17: {
        "name": "matmul",
        "wit_import": "    import planner:matmulworld/plan;",
        # MatMul em lote (batched), usado no bloco de atenção do YOLO — o
        # ÚNICO destes kernels "recentes" (13-18) com FLOPs reais, por isso
        # tem variante GPU (mesma assinatura WIT nas duas, igual Conv/BN/
        # MaxPool/Gemm-CPU). NÃO é comutativo como Add/Mul (kernel 5/11):
        # lhs_step/rhs_step (resolvidos pelo onnx_to_dfg.py a partir do
        # producer real de node.input[0]/node.input[1]) dizem qual ramo é
        # lhs e qual é rhs — join-buffer por (step, origin_step), não por
        # ordem de chegada. Caso lhs/rhs constante (ex: projeção fixa) não
        # espera 2º ramo, igual ao Mul.
        "rust_match": """                17 => {
                    if !p.lhs_const_path.is_empty() {
                        let lhs_const = load_weights_file_cached(&p.lhs_const_path);
                        let lhs_shape: Vec<u32> = p.lhs_shape.iter().map(|&d| if d > 0 { d as u32 } else { 1 }).collect();
                        let rhs_shape: Vec<u32> = p.rhs_shape.iter().map(|&d| if d > 0 { d as u32 } else { 1 }).collect();
                        let data = bindings::planner::matmulworld::plan::Data {
                            input: vec![],
                            output: vec![],
                            reply_to: reply_to.clone(),
                            current_kernel: current_step,
                            request: request_id,
                            kernels: vec![],
                            from_step: origin_step,
                        model_id: input_img.model_id,
                        };
                        let res = bindings::planner::matmulworld::plan::matmul(lhs_const.as_slice(), &current_input, &lhs_shape, &rhs_shape, &data);
                        computed_output = res.output;
                    } else if !p.rhs_const_path.is_empty() {
                        let rhs_const = load_weights_file_cached(&p.rhs_const_path);
                        let lhs_shape: Vec<u32> = p.lhs_shape.iter().map(|&d| if d > 0 { d as u32 } else { 1 }).collect();
                        let rhs_shape: Vec<u32> = p.rhs_shape.iter().map(|&d| if d > 0 { d as u32 } else { 1 }).collect();
                        let data = bindings::planner::matmulworld::plan::Data {
                            input: vec![],
                            output: vec![],
                            reply_to: reply_to.clone(),
                            current_kernel: current_step,
                            request: request_id,
                            kernels: vec![],
                            from_step: origin_step,
                        model_id: input_img.model_id,
                        };
                        let res = bindings::planner::matmulworld::plan::matmul(&current_input, rhs_const.as_slice(), &lhs_shape, &rhs_shape, &data);
                        computed_output = res.output;
                    } else {
                        let lhs_step = p.lhs_step.unwrap_or_else(|| panic!("Step {}: MatMul sem params.lhs_step — grafo desatualizado?", current_step));
                        let rhs_step = p.rhs_step.unwrap_or_else(|| panic!("Step {}: MatMul sem params.rhs_step — grafo desatualizado?", current_step));
                        if lhs_step == rhs_step {
                            panic!("Step {}: MatMul com lhs_step == rhs_step ({}) não é suportado por este dynamic_builder (precisaria de fila por origem, como o Concat) — caso raro, avise se ocorrer de verdade.", current_step, lhs_step);
                        }

                        {
                            let step_slots = matmul_branches.entry(current_step).or_insert_with(HashMap::new);
                            step_slots.insert(origin_step, current_input.clone());
                        }

                        let ready = matmul_branches.get(&current_step)
                            .map(|m| m.contains_key(&lhs_step) && m.contains_key(&rhs_step))
                            .unwrap_or(false);

                        if !ready {
                            vprintln!("Step {}: ramo from_step={} guardado no MatMul (lhs_step={}, rhs_step={}). Aguardando o resto...", current_step, origin_step, lhs_step, rhs_step);
                            continue;
                        }

                        let mut slots = matmul_branches.remove(&current_step).unwrap();
                        let lhs_vec = slots.remove(&lhs_step).unwrap();
                        let rhs_vec = slots.remove(&rhs_step).unwrap();

                        let lhs_shape: Vec<u32> = p.lhs_shape.iter().map(|&d| if d > 0 { d as u32 } else { 1 }).collect();
                        let rhs_shape: Vec<u32> = p.rhs_shape.iter().map(|&d| if d > 0 { d as u32 } else { 1 }).collect();
                        let data = bindings::planner::matmulworld::plan::Data {
                            input: vec![],
                            output: vec![],
                            reply_to: reply_to.clone(),
                            current_kernel: current_step,
                            request: request_id,
                            kernels: vec![],
                            from_step: origin_step,
                        model_id: input_img.model_id,
                        };
                        let res = bindings::planner::matmulworld::plan::matmul(&lhs_vec, &rhs_vec, &lhs_shape, &rhs_shape, &data);
                        computed_output = res.output;
                    }
                },""",
        "variants": {
            "cpu": {
                "wkg_override": '"planner:matmulworld" = { path = "../../../kernel/matmul/" }',
                "wasm_path": "../../../kernel/matmul/matmulworld_component.wasm",
            },
            "gpu": {
                "wkg_override": '"planner:matmulworld" = { path = "../../../kernel/matmul_gpu/wit" }',
                "wasm_path": "../../../kernel/matmul_gpu/target/wasm32-wasip2/release/matmul_gpu.wasm",
            },
        },
    },
    22: {
        "name": "softmax",
        "wit_import": "    import planner:softmaxworld/plan;",
        # Softmax numericamente estável ao longo de UM eixo — ponto-a-ponto
        # com normalização por eixo, sem FLOPs significativos (sem ganho em
        # GPU, sem variante GPU, igual Reshape/Transpose/Resize). Reusa
        # input_shape (já existe em LayerParams, compartilhado com Split/
        # Transpose) + axis (já existe, compartilhado com Split).
        "rust_match": """                22 => {
                    let shape: Vec<u32> = p.input_shape.iter().map(|&d| if d > 0 { d as u32 } else { 1 }).collect();
                    let data = bindings::planner::softmaxworld::plan::Data {
                        input: current_input.clone(),
                        output: vec![],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                        from_step: origin_step,
                    model_id: input_img.model_id,
                    };
                    let res = bindings::planner::softmaxworld::plan::softmax(&current_input, &shape, p.axis, &data);
                    computed_output = res.output;
                },""",
        "variants": {
            "cpu": {
                "wkg_override": '"planner:softmaxworld" = { path = "../../../kernel/softmax/" }',
                "wasm_path": "../../../kernel/softmax/softmaxworld_component.wasm",
            },
        },
    },
    19: {
        "name": "slice",
        "wit_import": "    import planner:sliceworld/plan;",
        # Slice genérico (ONNX Slice), decode do detection head do YOLO —
        # puro gather de memória (sem FLOPs), sem variante GPU (mesmo
        # raciocínio de split/concat/resize/reshape/transpose). starts/ends/
        # steps vêm como i64 no JSON (podem ter sentinelas enormes tipo
        # INT64_MAX pra "até o fim") e são clampados pra i32 aqui, igual ao
        # receiver standalone — o kernel em C já clampa de novo contra o
        # shape real internamente, então o clamp aqui só evita overflow na
        # conversão, não precisa ser exato.
        "rust_match": """                19 => {
                    let shape: Vec<u32> = p.input_shape.iter().map(|&d| if d > 0 { d as u32 } else { 1 }).collect();
                    let starts: Vec<i32> = p.starts.iter().map(|&v| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32).collect();
                    let ends: Vec<i32> = p.ends.iter().map(|&v| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32).collect();
                    let steps: Vec<i32> = p.steps.iter().map(|&v| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32).collect();
                    let data = bindings::planner::sliceworld::plan::Data {
                        input: current_input.clone(),
                        output: vec![],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                        from_step: origin_step,
                    model_id: input_img.model_id,
                    };
                    let res = bindings::planner::sliceworld::plan::slice(&current_input, &shape, &starts, &ends, &p.axes, &steps, &data);
                    computed_output = res.output;
                },""",
        "variants": {
            "cpu": {
                "wkg_override": '"planner:sliceworld" = { path = "../../../kernel/slice/" }',
                "wasm_path": "../../../kernel/slice/sliceworld_component.wasm",
            },
        },
    },
    20: {
        "name": "sub",
        "wit_import": "    import planner:subworld/plan;",
        # Sub tem dois casos, igual ao receiver isolado (kernel/sub/src/lib.rs)
        # e ao padrão já usado pelo Mul: tensor x constante (ex: anchor_points
        # fixo no decode do YOLO) ou tensor x tensor (join-buffer por step).
        # Sub NÃO é comutativo (a - b != b - a) — const_is_first decide a
        # ordem quando um lado é constante; no caso tensor x tensor, usa
        # lhs_step/rhs_step como o MatMul (30/10: o decode do YOLO26n TEM um
        # Sub tensor x tensor — /model.23/Sub_1, wh = x2y2 - x1y1 — e a ordem
        # de chegada invertia o sinal de w/h).
        "rust_match": """                20 => {
                    if !p.const_path.is_empty() {
                        let const_vec = load_weights_file_cached(&p.const_path);
                        let (a, b): (&[f32], &[f32]) = if p.const_is_first {
                            (const_vec.as_slice(), current_input.as_slice())
                        } else {
                            (current_input.as_slice(), const_vec.as_slice())
                        };
                        let data = bindings::planner::subworld::plan::Data {
                            input: current_input.clone(),
                            output: vec![],
                            reply_to: reply_to.clone(),
                            current_kernel: current_step,
                            request: request_id,
                            kernels: vec![],
                            from_step: origin_step,
                        model_id: input_img.model_id,
                        };
                        let res = bindings::planner::subworld::plan::sub(a, b, &data);
                        computed_output = res.output;
                    } else {
                        // tensor x tensor. Sub NÃO é comutativo: lhs_step/rhs_step
                        // (gravados pelo onnx_to_dfg.py a partir do producer real de
                        // node.input[0]/node.input[1]) dizem qual ramo é `a` e qual é `b`
                        // — join por (step, origin_step), igual ao MatMul (17), não por
                        // ordem de chegada. Sem esses campos (DFG antigo) cai no
                        // comportamento anterior (1º a chegar = `a`) com aviso.
                        let pair: Option<(Vec<f32>, Vec<f32>)> = match (p.lhs_step, p.rhs_step) {
                            (Some(lhs_step), Some(rhs_step)) if lhs_step == rhs_step => {
                                Some((current_input.clone(), current_input.clone()))
                            }
                            (Some(lhs_step), Some(rhs_step)) => {
                                let ready = {
                                    let slots = sub_slots.entry(current_step).or_insert_with(HashMap::new);
                                    slots.insert(origin_step, current_input.clone());
                                    slots.contains_key(&lhs_step) && slots.contains_key(&rhs_step)
                                };
                                if ready {
                                    let mut slots = sub_slots.remove(&current_step).unwrap();
                                    let a = slots.remove(&lhs_step).unwrap();
                                    let b = slots.remove(&rhs_step).unwrap();
                                    Some((a, b))
                                } else {
                                    vprintln!("Step {}: ramo from_step={} guardado no Sub (lhs_step={}, rhs_step={}). Aguardando o resto...", current_step, origin_step, lhs_step, rhs_step);
                                    None
                                }
                            }
                            _ => {
                                if let Some(first_branch) = sub_branches.remove(&current_step) {
                                    println!("[AVISO] Step {}: Sub tensor x tensor sem lhs_step/rhs_step no DFG — usando ordem de chegada (pode inverter o sinal/razão). Regere o DFG com o onnx_to_dfg.py atual.", current_step);
                                    Some((first_branch, current_input.clone()))
                                } else {
                                    sub_branches.insert(current_step, current_input.clone());
                                    None
                                }
                            }
                        };
                        let (a, b) = match pair {
                            Some(ab) => ab,
                            None => {
                                vprintln!("Step {}: 1º ramo guardado no buffer. Aguardando outro ramo...", current_step);
                                continue;
                            }
                        };
                        vprintln!("Step {}: Subtraindo buffers de tamanhos {} e {}...", current_step, a.len(), b.len());
                        let data = bindings::planner::subworld::plan::Data {
                            input: current_input.clone(),
                            output: vec![],
                            reply_to: reply_to.clone(),
                            current_kernel: current_step,
                            request: request_id,
                            kernels: vec![],
                            from_step: origin_step,
                        model_id: input_img.model_id,
                        };
                        let res = bindings::planner::subworld::plan::sub(&a, &b, &data);
                        computed_output = res.output;
                    }
                },""",
        "variants": {
            "cpu": {
                "wkg_override": '"planner:subworld" = { path = "../../../kernel/sub/" }',
                "wasm_path": "../../../kernel/sub/subworld_component.wasm",
            },
        },
    },
    21: {
        "name": "div",
        "wit_import": "    import planner:divworld/plan;",
        # Div: mesmo padrão do Sub acima (tensor x constante ou tensor x
        # tensor via join-buffer), também não-comutativo (a / b != b / a).
        "rust_match": """                21 => {
                    if !p.const_path.is_empty() {
                        let const_vec = load_weights_file_cached(&p.const_path);
                        let (a, b): (&[f32], &[f32]) = if p.const_is_first {
                            (const_vec.as_slice(), current_input.as_slice())
                        } else {
                            (current_input.as_slice(), const_vec.as_slice())
                        };
                        let data = bindings::planner::divworld::plan::Data {
                            input: current_input.clone(),
                            output: vec![],
                            reply_to: reply_to.clone(),
                            current_kernel: current_step,
                            request: request_id,
                            kernels: vec![],
                            from_step: origin_step,
                        model_id: input_img.model_id,
                        };
                        let res = bindings::planner::divworld::plan::div(a, b, &data);
                        computed_output = res.output;
                    } else {
                        // tensor x tensor. Div NÃO é comutativo: lhs_step/rhs_step
                        // (gravados pelo onnx_to_dfg.py a partir do producer real de
                        // node.input[0]/node.input[1]) dizem qual ramo é `a` e qual é `b`
                        // — join por (step, origin_step), igual ao MatMul (17), não por
                        // ordem de chegada. Sem esses campos (DFG antigo) cai no
                        // comportamento anterior (1º a chegar = `a`) com aviso.
                        let pair: Option<(Vec<f32>, Vec<f32>)> = match (p.lhs_step, p.rhs_step) {
                            (Some(lhs_step), Some(rhs_step)) if lhs_step == rhs_step => {
                                Some((current_input.clone(), current_input.clone()))
                            }
                            (Some(lhs_step), Some(rhs_step)) => {
                                let ready = {
                                    let slots = div_slots.entry(current_step).or_insert_with(HashMap::new);
                                    slots.insert(origin_step, current_input.clone());
                                    slots.contains_key(&lhs_step) && slots.contains_key(&rhs_step)
                                };
                                if ready {
                                    let mut slots = div_slots.remove(&current_step).unwrap();
                                    let a = slots.remove(&lhs_step).unwrap();
                                    let b = slots.remove(&rhs_step).unwrap();
                                    Some((a, b))
                                } else {
                                    vprintln!("Step {}: ramo from_step={} guardado no Div (lhs_step={}, rhs_step={}). Aguardando o resto...", current_step, origin_step, lhs_step, rhs_step);
                                    None
                                }
                            }
                            _ => {
                                if let Some(first_branch) = div_branches.remove(&current_step) {
                                    println!("[AVISO] Step {}: Div tensor x tensor sem lhs_step/rhs_step no DFG — usando ordem de chegada (pode inverter o sinal/razão). Regere o DFG com o onnx_to_dfg.py atual.", current_step);
                                    Some((first_branch, current_input.clone()))
                                } else {
                                    div_branches.insert(current_step, current_input.clone());
                                    None
                                }
                            }
                        };
                        let (a, b) = match pair {
                            Some(ab) => ab,
                            None => {
                                vprintln!("Step {}: 1º ramo guardado no buffer. Aguardando outro ramo...", current_step);
                                continue;
                            }
                        };
                        vprintln!("Step {}: Dividindo buffers de tamanhos {} e {}...", current_step, a.len(), b.len());
                        let data = bindings::planner::divworld::plan::Data {
                            input: current_input.clone(),
                            output: vec![],
                            reply_to: reply_to.clone(),
                            current_kernel: current_step,
                            request: request_id,
                            kernels: vec![],
                            from_step: origin_step,
                        model_id: input_img.model_id,
                        };
                        let res = bindings::planner::divworld::plan::div(&a, &b, &data);
                        computed_output = res.output;
                    }
                },""",
        "variants": {
            "cpu": {
                "wkg_override": '"planner:divworld" = { path = "../../../kernel/div/" }',
                "wasm_path": "../../../kernel/div/divworld_component.wasm",
            },
        },
    },
    # 09/11: Conv+BN fundidos (23) — mesmo padrão do SiLU (12): o exportador emite este tipo
    # quando acha a sequência no grafo; os kernels individuais continuam
    # existindo. Params = os do Conv + os 4 caminhos e o epsilon do BN.
    23: {
        "name": "convbn",
        "wit_import": "    import planner:convbnworld/plan;",
        # 09/11 (4a): a variante GPU recebe os pesos uma vez só (load_weights)
        # e depois só o weight-id; a CPU continua igual.
        "rust_match_variants": {
            "cpu": """                23 => {
                    let out_h = ((p.h_in + 2 * p.padding - p.kh) / p.stride) + 1;
                    let out_w = ((p.w_in + 2 * p.padding - p.kw) / p.stride) + 1;
                    let output_size = (p.cout * out_h * out_w) as usize;
                    let weights = load_weights_file_cached(&p.weight_path);
                    let bias = load_weights_file_cached(&p.bias_path);
                    let scale = load_weights_file_cached(&p.scale_path);
                    let beta  = load_weights_file_cached(&p.beta_path);
                    let mean  = load_weights_file_cached(&p.mean_path);
                    let var   = load_weights_file_cached(&p.var_path);
                    let conv_groups = if p.groups == 0 { 1 } else { p.groups };
                    let data = bindings::planner::convbnworld::plan::Data {
                        input: current_input,
                        output: vec![0.0f32; output_size],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                        from_step: origin_step,
                    model_id: input_img.model_id,
                    };
                    let res = bindings::planner::convbnworld::plan::convbn(
                        p.cout, p.cin, p.h_in, p.w_in, p.kh, p.kw, p.stride, p.padding, conv_groups,
                        p.epsilon,
                        weights.as_slice(), bias.as_slice(),
                        scale.as_slice(), beta.as_slice(), mean.as_slice(), var.as_slice(), &data,
                    );
                    computed_output = res.output;
                },""",
            "gpu": """                23 => {
                    // 10/11 (4c): passo GPU local é tratado pela via residente
                    // antes do match; aqui nunca chega.
                    unreachable!("4c: tipo 23 na GPU vai pela via residente")
                },""",
        },
        "variants": {
            "cpu": {
                "wkg_override": '"planner:convbnworld" = { path = "../../../kernel/convbn/" }',
                "wasm_path": "../../../kernel/convbn/convbnworld_component.wasm",
            },
            "gpu": {
                "wkg_override": '"planner:convbnworld" = { path = "../../../kernel/convbn_gpu/wit" }',
                "wasm_path": "../../../kernel/convbn_gpu/target/wasm32-wasip2/release/convbn_gpu.wasm",
            },
        },
    },
    # 09/11: Conv+BN+Relu fundidos (24) — mesmo padrão do SiLU (12): o exportador emite este tipo
    # quando acha a sequência no grafo; os kernels individuais continuam
    # existindo. Params = os do Conv + os 4 caminhos e o epsilon do BN.
    24: {
        "name": "convbnrelu",
        "wit_import": "    import planner:convbnreluworld/plan;",
        # 09/11 (4a): a variante GPU recebe os pesos uma vez só (load_weights)
        # e depois só o weight-id; a CPU continua igual.
        "rust_match_variants": {
            "cpu": """                24 => {
                    let out_h = ((p.h_in + 2 * p.padding - p.kh) / p.stride) + 1;
                    let out_w = ((p.w_in + 2 * p.padding - p.kw) / p.stride) + 1;
                    let output_size = (p.cout * out_h * out_w) as usize;
                    let weights = load_weights_file_cached(&p.weight_path);
                    let bias = load_weights_file_cached(&p.bias_path);
                    let scale = load_weights_file_cached(&p.scale_path);
                    let beta  = load_weights_file_cached(&p.beta_path);
                    let mean  = load_weights_file_cached(&p.mean_path);
                    let var   = load_weights_file_cached(&p.var_path);
                    let conv_groups = if p.groups == 0 { 1 } else { p.groups };
                    let data = bindings::planner::convbnreluworld::plan::Data {
                        input: current_input,
                        output: vec![0.0f32; output_size],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: vec![],
                        from_step: origin_step,
                    model_id: input_img.model_id,
                    };
                    let res = bindings::planner::convbnreluworld::plan::convbnrelu(
                        p.cout, p.cin, p.h_in, p.w_in, p.kh, p.kw, p.stride, p.padding, conv_groups,
                        p.epsilon,
                        weights.as_slice(), bias.as_slice(),
                        scale.as_slice(), beta.as_slice(), mean.as_slice(), var.as_slice(), &data,
                    );
                    computed_output = res.output;
                },""",
            "gpu": """                24 => {
                    // 10/11 (4c): passo GPU local é tratado pela via residente
                    // antes do match; aqui nunca chega.
                    unreachable!("4c: tipo 24 na GPU vai pela via residente")
                },""",
        },
        "variants": {
            "cpu": {
                "wkg_override": '"planner:convbnreluworld" = { path = "../../../kernel/convbnrelu/" }',
                "wasm_path": "../../../kernel/convbnrelu/convbnreluworld_component.wasm",
            },
            "gpu": {
                "wkg_override": '"planner:convbnreluworld" = { path = "../../../kernel/convbnrelu_gpu/wit" }',
                "wasm_path": "../../../kernel/convbnrelu_gpu/target/wasm32-wasip2/release/convbnrelu_gpu.wasm",
            },
        },
    },
}

WASI_GFX_OVERRIDE = (
    '"wasi-gfx-runtime:wasi-webgpu" = '
    '{ path = "../../../../../wasi-gfx-runtime/crates/wasi-webgpu-wasmtime/wit/" }'
)


# 10/11 (4c): via residente na GPU.
LANE_CODE = {1: '                1 => {\n                    let wkey = p.weight_path.clone();\n                    let wtag = format!("1:{}", wkey);\n                    if !GPU_W_LOADED.with(|l| l.borrow().contains(&wtag)) {\n                        let weights = load_weights_file_cached(&p.weight_path);\n                        let bias = load_weights_file_cached(&p.bias_path);\n                        bindings::planner::convworld::plan::conv_load_weights(&dev, &wkey, weights.as_slice(), bias.as_slice());\n                        GPU_W_LOADED.with(|l| l.borrow_mut().insert(wtag));\n                    }\n                    let g = if p.groups == 0 { 1 } else { p.groups };\n                    let oh = ((p.h_in + 2 * p.padding - p.kh) / p.stride) + 1;\n                    let ow = ((p.w_in + 2 * p.padding - p.kw) / p.stride) + 1;\n                    let b = bindings::planner::convworld::plan::conv_r(&dev, &wkey, p.cout, p.cin, p.h_in, p.w_in, p.kh, p.kw, p.stride, p.padding, g, &inp.buf);\n                    Some(GpuT { buf: Rc::new(b), len: (p.cout * oh * ow) as usize })\n                }', 2: '                2 => {\n                    let wkey = p.scale_path.clone();\n                    let wtag = format!("2:{}", wkey);\n                    if !GPU_W_LOADED.with(|l| l.borrow().contains(&wtag)) {\n                        let sc = load_weights_file_cached(&p.scale_path);\n                        let be = load_weights_file_cached(&p.beta_path);\n                        let me = load_weights_file_cached(&p.mean_path);\n                        let va = load_weights_file_cached(&p.var_path);\n                        bindings::planner::bnworld::plan::bn_load_weights(&dev, &wkey, sc.as_slice(), be.as_slice(), me.as_slice(), va.as_slice());\n                        GPU_W_LOADED.with(|l| l.borrow_mut().insert(wtag));\n                    }\n                    let b = bindings::planner::bnworld::plan::bn_r(&dev, &wkey, p.channels, p.h_in, p.w_in, p.epsilon, &inp.buf);\n                    Some(GpuT { buf: Rc::new(b), len: inp.len })\n                }', 4: '                4 => {\n                    let oh = ((p.h_in + 2 * p.padding - p.kh) / p.stride) + 1;\n                    let ow = ((p.w_in + 2 * p.padding - p.kw) / p.stride) + 1;\n                    let b = bindings::planner::maxpoolworld::plan::maxpool_r(&dev, p.channels, p.h_in, p.w_in, p.kh, p.kw, p.stride, p.padding, &inp.buf);\n                    Some(GpuT { buf: Rc::new(b), len: (p.channels * oh * ow) as usize })\n                }', 6: '                6 => {\n                    let ch = p.channels.max(1);\n                    let b = bindings::planner::gapworld::plan::gap_r(&dev, ch, (inp.len as u32) / ch, &inp.buf);\n                    Some(GpuT { buf: Rc::new(b), len: ch as usize })\n                }', 8: '                8 => {\n                    let wkey = p.weight_path.clone();\n                    let wtag = format!("8:{}", wkey);\n                    if !GPU_W_LOADED.with(|l| l.borrow().contains(&wtag)) {\n                        let weights = load_weights_file_cached(&p.weight_path);\n                        let bias = load_weights_file_cached(&p.bias_path);\n                        bindings::planner::gemmworld::plan::gemm_load_weights(&dev, &wkey, weights.as_slice(), bias.as_slice());\n                        GPU_W_LOADED.with(|l| l.borrow_mut().insert(wtag));\n                    }\n                    let b = bindings::planner::gemmworld::plan::gemm_r(&dev, &wkey, p.in_features, p.out_features, &inp.buf);\n                    Some(GpuT { buf: Rc::new(b), len: p.out_features as usize })\n                }', 23: '                23 => {\n                    let wkey = p.weight_path.clone();\n                    let wtag = format!("23:{}", wkey);\n                    if !GPU_W_LOADED.with(|l| l.borrow().contains(&wtag)) {\n                        let weights = load_weights_file_cached(&p.weight_path);\n                        let bias = load_weights_file_cached(&p.bias_path);\n                        let sc = load_weights_file_cached(&p.scale_path);\n                        let be = load_weights_file_cached(&p.beta_path);\n                        let me = load_weights_file_cached(&p.mean_path);\n                        let va = load_weights_file_cached(&p.var_path);\n                        bindings::planner::convbnworld::plan::convbn_load_weights(&dev, &wkey, p.epsilon, weights.as_slice(), bias.as_slice(), sc.as_slice(), be.as_slice(), me.as_slice(), va.as_slice());\n                        GPU_W_LOADED.with(|l| l.borrow_mut().insert(wtag));\n                    }\n                    let g = if p.groups == 0 { 1 } else { p.groups };\n                    let oh = ((p.h_in + 2 * p.padding - p.kh) / p.stride) + 1;\n                    let ow = ((p.w_in + 2 * p.padding - p.kw) / p.stride) + 1;\n                    let b = bindings::planner::convbnworld::plan::convbn_r(&dev, &wkey, p.cout, p.cin, p.h_in, p.w_in, p.kh, p.kw, p.stride, p.padding, g, &inp.buf);\n                    Some(GpuT { buf: Rc::new(b), len: (p.cout * oh * ow) as usize })\n                }', 24: '                24 => {\n                    let wkey = p.weight_path.clone();\n                    let wtag = format!("24:{}", wkey);\n                    if !GPU_W_LOADED.with(|l| l.borrow().contains(&wtag)) {\n                        let weights = load_weights_file_cached(&p.weight_path);\n                        let bias = load_weights_file_cached(&p.bias_path);\n                        let sc = load_weights_file_cached(&p.scale_path);\n                        let be = load_weights_file_cached(&p.beta_path);\n                        let me = load_weights_file_cached(&p.mean_path);\n                        let va = load_weights_file_cached(&p.var_path);\n                        bindings::planner::convbnreluworld::plan::convbnrelu_load_weights(&dev, &wkey, p.epsilon, weights.as_slice(), bias.as_slice(), sc.as_slice(), be.as_slice(), me.as_slice(), va.as_slice());\n                        GPU_W_LOADED.with(|l| l.borrow_mut().insert(wtag));\n                    }\n                    let g = if p.groups == 0 { 1 } else { p.groups };\n                    let oh = ((p.h_in + 2 * p.padding - p.kh) / p.stride) + 1;\n                    let ow = ((p.w_in + 2 * p.padding - p.kw) / p.stride) + 1;\n                    let b = bindings::planner::convbnreluworld::plan::convbnrelu_r(&dev, &wkey, p.cout, p.cin, p.h_in, p.w_in, p.kh, p.kw, p.stride, p.padding, g, &inp.buf);\n                    Some(GpuT { buf: Rc::new(b), len: (p.cout * oh * ow) as usize })\n                }'}
LANE_EXTRA_CODE = {3: '                3 => {\n                    let b = bindings::planner::relugpuworld::plan::relu_r(&dev, inp.len as u32, &inp.buf);\n                    Some(GpuT { buf: Rc::new(b), len: inp.len })\n                }', 5: '                5 => {\n                    // Junção do resíduo na GPU. O 1º ramo espera em GPU_ADD_JOIN\n                    // (ou já está no join de CPU, se chegou antes pela rede).\n                    let key = (request_id, current_step);\n                    let other = GPU_ADD_JOIN.with(|j| j.borrow_mut().remove(&key))\n                        .or_else(|| add_branches.remove(&current_step).map(|h| gpu_upload(&dev, &h)));\n                    match other {\n                        Some(o) => {\n                            let b = bindings::planner::addgpuworld::plan::add_r(&dev, inp.len as u32, &o.buf, &inp.buf);\n                            Some(GpuT { buf: Rc::new(b), len: inp.len })\n                        }\n                        None => {\n                            GPU_ADD_JOIN.with(|j| j.borrow_mut().insert(key, inp.clone()));\n                            None\n                        }\n                    }\n                }', 7: '                7 => Some(inp.clone()),', 9: '                9 => Some(inp.clone()),'}
EXTRA_GPU = {3: {'name': 'relugpu', 'wit_import': '    import planner:relugpuworld/plan;', 'wkg_override': '"planner:relugpuworld" = { path = "../../../kernel/relu_gpu/wit" }', 'wasm_path': '../../../kernel/relu_gpu/target/wasm32-wasip2/release/relu_gpu.wasm'}, 5: {'name': 'addgpu', 'wit_import': '    import planner:addgpuworld/plan;', 'wkg_override': '"planner:addgpuworld" = { path = "../../../kernel/add_gpu/wit" }', 'wasm_path': '../../../kernel/add_gpu/target/wasm32-wasip2/release/add_gpu.wasm'}}


def lane_plan(active_kernel_ids, variant_map):
    """Quais tipos vão pela via residente neste grupo e quais componentes
    extra (relu/add na GPU) entram na composição."""
    gpu_lane = [k for k in LANE_CODE if k in active_kernel_ids and variant_map.get(k) == "gpu"]
    if not gpu_lane:
        return [], []
    extras = [k for k in (3, 5) if k in active_kernel_ids]
    ident = [k for k in (7, 9) if k in active_kernel_ids]
    return sorted(gpu_lane + extras + ident), extras

GPU_LANE_HELPERS = '\n// ===== 10/11 (4c): via residente na GPU =====================================\n// Um device por GRUPO (antes: um por kernel). Os kernels GPU recebem o device\n// (borrow) e o tensor de entrada como handle de buffer da GPU, e devolvem a\n// saída como handle. O dado só desce para a CPU quando o próximo passo é de\n// CPU, vai para a rede ou é a saída final. Arquivo `gpu_resident_off` no\n// diretório do sys: baixa depois de TODO passo (comportamento antigo, para\n// medir antes × depois sem recompilar).\nuse bindings::wasi::webgpu::webgpu;\nuse bindings::wasi_gfx_runtime::wasi_webgpu::sync_io;\n\n#[derive(Clone)]\nstruct GpuT {\n    buf: Rc<webgpu::GpuBuffer>,\n    len: usize,\n}\n\nthread_local! {\n    static GROUP_DEV: RefCell<Option<Rc<webgpu::GpuDevice>>> = RefCell::new(None);\n    // (request, passo produtor) -> (tensor, consumidores que ainda faltam)\n    static GPU_REG: RefCell<HashMap<(u32, u32), (GpuT, u32)>> = RefCell::new(HashMap::new());\n    // (request, passo do Add) -> 1º ramo\n    static GPU_ADD_JOIN: RefCell<HashMap<(u32, u32), GpuT>> = RefCell::new(HashMap::new());\n}\n\nfn group_device() -> Rc<webgpu::GpuDevice> {\n    GROUP_DEV.with(|d| {\n        if d.borrow().is_none() {\n            let gpu = webgpu::get_gpu();\n            let opts = webgpu::GpuRequestAdapterOptions {\n                feature_level: None,\n                power_preference: Some(webgpu::GpuPowerPreference::HighPerformance),\n                force_fallback_adapter: None,\n                xr_compatible: None,\n            };\n            let adapter = wit_bindgen::block_on(gpu.request_adapter(Some(opts))).expect("sem adapter de GPU");\n            let info = adapter.info();\n            eprintln!("[GPU4C] device do grupo: vendor=\'{}\' device=\'{}\'", info.vendor(), info.device());\n            // limites altos para as FC grandes (VGG), como o gemm_gpu já fazia\n            let limits = adapter.limits();\n            let rl = webgpu::RecordOptionGpuSize64::new();\n            rl.add("maxStorageBufferBindingSize", Some(limits.max_storage_buffer_binding_size()));\n            rl.add("maxBufferSize", Some(limits.max_buffer_size()));\n            let desc = webgpu::GpuDeviceDescriptor {\n                required_features: None,\n                required_limits: Some(rl),\n                default_queue: None,\n                label: Some("grupo".to_string()),\n            };\n            let dev = wit_bindgen::block_on(adapter.request_device(Some(desc))).expect("request_device falhou");\n            *d.borrow_mut() = Some(Rc::new(dev));\n        }\n        Rc::clone(d.borrow().as_ref().unwrap())\n    })\n}\n\nfn gpu_resident_off() -> bool {\n    std::path::Path::new("gpu_resident_off").exists()\n}\n\nfn gpu_upload(dev: &webgpu::GpuDevice, data: &[f32]) -> GpuT {\n    let n = data.len().max(1);\n    let b = dev.create_buffer(&webgpu::GpuBufferDescriptor {\n        label: None,\n        size: (n * 4) as u64,\n        usage: webgpu::GpuBufferUsage::STORAGE | webgpu::GpuBufferUsage::COPY_SRC | webgpu::GpuBufferUsage::COPY_DST,\n        mapped_at_creation: Some(true),\n    });\n    if !data.is_empty() {\n        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };\n        b.get_mapped_range_set_with_copy(bytes, None, None).unwrap();\n    }\n    b.unmap().unwrap();\n    GpuT { buf: Rc::new(b), len: data.len() }\n}\n\nfn gpu_download(dev: &webgpu::GpuDevice, t: &GpuT) -> Vec<f32> {\n    let n = t.len.max(1);\n    let stg = dev.create_buffer(&webgpu::GpuBufferDescriptor {\n        label: None,\n        size: (n * 4) as u64,\n        usage: webgpu::GpuBufferUsage::MAP_READ | webgpu::GpuBufferUsage::COPY_DST,\n        mapped_at_creation: None,\n    });\n    let enc = dev.create_command_encoder(None);\n    enc.copy_buffer_to_buffer(&t.buf, None, &stg, None, None);\n    dev.queue().submit(&[&enc.finish(None)]);\n    let bytes = sync_io::map_and_read_sync(&stg, webgpu::GpuMapMode::READ, Some(0), None)\n        .expect("map_and_read_sync falhou");\n    let mut v = vec![0.0f32; t.len];\n    let k = (t.len * 4).min(bytes.len());\n    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), v.as_mut_ptr() as *mut u8, k) };\n    v\n}\n\nfn gpu_reg_take(request: u32, producer: u32) -> Option<GpuT> {\n    GPU_REG.with(|r| {\n        let mut r = r.borrow_mut();\n        let key = (request, producer);\n        let (t, left) = r.get_mut(&key)?;\n        let out = t.clone();\n        *left -= 1;\n        if *left == 0 {\n            r.remove(&key);\n        }\n        Some(out)\n    })\n}\n\n/// O passo `step` deste modelo vai pela via residente aqui?\nfn lane_eligible(graph: &ModelGraph, routing: &RoutingTable, model_id: u32, step: u32) -> bool {\n    let Some(node) = graph.steps.iter().find(|s| s.step == step) else { return false };\n    if !LANE_KERNELS.contains(&node.kernel_type) || !LOCAL_KERNELS.contains(&node.kernel_type) {\n        return false;\n    }\n    if node.kernel_type == 5 && !node.params.const_path.is_empty() {\n        return false; // Add com constante: CPU\n    }\n    step_route_dest(routing, model_id, step).is_none()\n}\n'
GPU_LANE_BLOCK = '\n            // ===== 10/11 (4c): via residente na GPU =====\n            let gpu_in = gpu_reg_take(request_id, origin_step);\n            if step_dest.is_none() && lane_eligible(graph, routing, input_img.model_id, current_step) {\n                let dev = group_device();\n                let inp: GpuT = match gpu_in {\n                    Some(t) => t,\n                    None => gpu_upload(&dev, &current_input),\n                };\n                let in_len = inp.len; // [T]\n                let out: Option<GpuT> = match current_node.kernel_type {\n@@LANE_ARMS@@\n                    _ => unreachable!(),\n                };\n                let lane_us = t_step.elapsed().as_micros() as u64; // [T]\n                {\n                    let log_linha = serde_json::json!({\n                        "request": request_id,\n                        "kernel_id": current_node.kernel_type,\n                        "kernel_type": current_node.kernel_type,\n                        "step": current_step,\n                        "layer_name": current_node.name,\n                        "next_steps": current_node.next_steps,\n                        "call_idx": call_idx,\n                        "ts_unix_ms": ts_unix_ms,\n                        "dispatch": "gpu-resident",\n                        "input_len": in_len,\n                        "output_len": out.as_ref().map(|o| o.len).unwrap_or(0),\n                        "timing_us": { "net_in": net_in_us, "deserialize": deserialize_us, "kernel_call": lane_us }\n                    });\n                    log_push(log_linha.to_string());\n                }\n                let Some(out) = out else { continue; }; // Add esperando o outro ramo\n                if current_node.next_steps.is_empty() {\n                    final_output = gpu_download(&dev, &out);\n                    final_from_step = current_step;\n                    reached_final = true;\n                    break;\n                }\n                let resident = !gpu_resident_off();\n                let mut on_gpu: Vec<u32> = Vec::new();\n                let mut on_host: Vec<u32> = Vec::new();\n                for &nxt in current_node.next_steps.iter() {\n                    if resident && lane_eligible(graph, routing, input_img.model_id, nxt) {\n                        on_gpu.push(nxt);\n                    } else {\n                        on_host.push(nxt);\n                    }\n                }\n                if !on_gpu.is_empty() {\n                    GPU_REG.with(|r| r.borrow_mut().insert((request_id, current_step), (out.clone(), on_gpu.len() as u32)));\n                    for nxt in on_gpu {\n                        queue.push_back((nxt, Vec::new(), current_step));\n                    }\n                }\n                if !on_host.is_empty() {\n                    let host = gpu_download(&dev, &out);\n                    for nxt in on_host {\n                        queue.push_back((nxt, host.clone(), current_step));\n                    }\n                }\n                continue;\n            }\n            // Passo que não vai pela via (CPU, rede, Add com constante...) mas\n            // cujo tensor está na GPU: baixa aqui.\n            if let Some(t) = gpu_in {\n                current_input = gpu_download(&group_device(), &t);\n            }\n'

# 10/11 (4c, parte 2): todos os operadores na via residente.
_HELPERS_EXTRA = '\n// ===== 10/11 (4c, parte 2) =====\nuse bindings::planner::opsgpuworld::plan as opsgpu;\n\nthread_local! {\n    // (request, passo do join) -> passo de origem -> ramos (FIFO: o mesmo\n    // produtor pode mandar 2 fatias para o mesmo consumidor, ex. Split->Concat)\n    static GPU_JOIN: RefCell<HashMap<(u32, u32), HashMap<u32, VecDeque<GpuT>>>> = RefCell::new(HashMap::new());\n    // constantes (const_path) já na GPU\n    static GPU_CONST: RefCell<HashMap<String, GpuT>> = RefCell::new(HashMap::new());\n    // buffer de leitura reaproveitado entre downloads (cresce até o maior tensor)\n    static GPU_STG: RefCell<Option<(u64, webgpu::GpuBuffer)>> = RefCell::new(None);\n}\n\nfn gpu_reg_put(request: u32, producer: u32, consumer: u32, t: GpuT) {\n    GPU_REG.with(|r| r.borrow_mut().entry((request, producer, consumer)).or_insert_with(VecDeque::new).push_back(t));\n}\n\nfn gpu_alloc(dev: &webgpu::GpuDevice, n: usize) -> GpuT {\n    let b = dev.create_buffer(&webgpu::GpuBufferDescriptor {\n        label: None,\n        size: (n.max(1) * 4) as u64,\n        usage: webgpu::GpuBufferUsage::STORAGE | webgpu::GpuBufferUsage::COPY_SRC | webgpu::GpuBufferUsage::COPY_DST,\n        mapped_at_creation: None,\n    });\n    GpuT { buf: Rc::new(b), len: n }\n}\n\nfn gpu_const(dev: &webgpu::GpuDevice, path: &str) -> GpuT {\n    if let Some(t) = GPU_CONST.with(|c| c.borrow().get(path).cloned()) {\n        return t;\n    }\n    let v = load_weights_file_cached(path);\n    let t = gpu_upload(dev, v.as_slice());\n    GPU_CONST.with(|c| c.borrow_mut().insert(path.to_string(), t.clone()));\n    t\n}\n\n/// out[i] = a[i] op b[i % len_b] (0 add, 1 sub, 2 mul, 3 div), len = len_a\nfn gpu_binop(dev: &webgpu::GpuDevice, op: u32, a: &GpuT, b: &GpuT) -> GpuT {\n    let r = opsgpu::binop_r(dev, op, a.len as u32, b.len as u32, &a.buf, &b.buf);\n    GpuT { buf: Rc::new(r), len: a.len }\n}\n\nfn gpu_join_put(request: u32, step: u32, origin: u32, t: GpuT) {\n    GPU_JOIN.with(|j| j.borrow_mut().entry((request, step)).or_insert_with(HashMap::new)\n        .entry(origin).or_insert_with(VecDeque::new).push_back(t));\n}\n\nfn gpu_join_has(request: u32, step: u32, origin: u32) -> bool {\n    GPU_JOIN.with(|j| j.borrow().get(&(request, step)).and_then(|m| m.get(&origin)).map(|q| !q.is_empty()).unwrap_or(false))\n}\n\nfn gpu_join_count(request: u32, step: u32) -> usize {\n    GPU_JOIN.with(|j| j.borrow().get(&(request, step)).map(|m| m.values().map(|q| q.len()).sum()).unwrap_or(0))\n}\n\nfn gpu_join_take(request: u32, step: u32, origin: u32) -> Option<GpuT> {\n    GPU_JOIN.with(|j| j.borrow_mut().get_mut(&(request, step)).and_then(|m| m.get_mut(&origin)).and_then(|q| q.pop_front()))\n}\n\n/// Junção por ordem de chegada (Mul tensor x tensor, igual ao braço de CPU).\nfn gpu_join_take_any(request: u32, step: u32) -> Option<GpuT> {\n    GPU_JOIN.with(|j| {\n        let mut j = j.borrow_mut();\n        let m = j.get_mut(&(request, step))?;\n        let t = m.values_mut().find(|q| !q.is_empty()).and_then(|q| q.pop_front());\n        if m.values().all(|q| q.is_empty()) {\n            j.remove(&(request, step));\n        }\n        t\n    })\n}\n\nfn gpu_join_clear(request: u32, step: u32) {\n    GPU_JOIN.with(|j| j.borrow_mut().remove(&(request, step)));\n}\n\n/// Junção por lhs_step/rhs_step (MatMul, Sub, Div): devolve (a, b) quando os\n/// dois ramos chegaram.\nfn gpu_join_pair(request: u32, step: u32, origin: u32, inp: &GpuT, lhs: u32, rhs: u32) -> Option<(GpuT, GpuT)> {\n    gpu_join_put(request, step, origin, inp.clone());\n    if gpu_join_has(request, step, lhs) && gpu_join_has(request, step, rhs) {\n        let a = gpu_join_take(request, step, lhs).unwrap();\n        let b = gpu_join_take(request, step, rhs).unwrap();\n        gpu_join_clear(request, step);\n        Some((a, b))\n    } else {\n        None\n    }\n}\n\nfn shape_u32(v: &[i64]) -> Vec<u32> {\n    v.iter().map(|&d| if d > 0 { d as u32 } else { 1 }).collect()\n}\n'
_GPU_DOWNLOAD_OLD = 'fn gpu_download(dev: &webgpu::GpuDevice, t: &GpuT) -> Vec<f32> {\n    let n = t.len.max(1);\n    let stg = dev.create_buffer(&webgpu::GpuBufferDescriptor {\n        label: None,\n        size: (n * 4) as u64,\n        usage: webgpu::GpuBufferUsage::MAP_READ | webgpu::GpuBufferUsage::COPY_DST,\n        mapped_at_creation: None,\n    });\n    let enc = dev.create_command_encoder(None);\n    enc.copy_buffer_to_buffer(&t.buf, None, &stg, None, None);\n    dev.queue().submit(&[&enc.finish(None)]);\n    let bytes = sync_io::map_and_read_sync(&stg, webgpu::GpuMapMode::READ, Some(0), None)\n        .expect("map_and_read_sync falhou");\n'
_GPU_DOWNLOAD_NEW = 'fn gpu_download(dev: &webgpu::GpuDevice, t: &GpuT) -> Vec<f32> {\n    // 10/11 (4c, parte 2): um staging só, reaproveitado (antes: um novo por\n    // download). O map_and_read_sync já desmapeia antes de voltar.\n    let nbytes = (t.len.max(1) * 4) as u64;\n    let bytes = GPU_STG.with(|s| {\n        let mut s = s.borrow_mut();\n        if s.as_ref().map(|(sz, _)| *sz < nbytes).unwrap_or(true) {\n            let b = dev.create_buffer(&webgpu::GpuBufferDescriptor {\n                label: Some("staging".to_string()),\n                size: nbytes,\n                usage: webgpu::GpuBufferUsage::MAP_READ | webgpu::GpuBufferUsage::COPY_DST,\n                mapped_at_creation: None,\n            });\n            *s = Some((nbytes, b));\n        }\n        let stg = &s.as_ref().unwrap().1;\n        let enc = dev.create_command_encoder(None);\n        enc.copy_buffer_to_buffer(&t.buf, Some(0), stg, Some(0), Some(nbytes));\n        dev.queue().submit(&[&enc.finish(None)]);\n        sync_io::map_and_read_sync(stg, webgpu::GpuMapMode::READ, Some(0), Some(nbytes))\n            .expect("map_and_read_sync falhou")\n    });\n'
_REG_TAKE_OLD = 'fn gpu_reg_take(request: u32, producer: u32) -> Option<GpuT> {\n    GPU_REG.with(|r| {\n        let mut r = r.borrow_mut();\n        let key = (request, producer);\n        let (t, left) = r.get_mut(&key)?;\n        let out = t.clone();\n        *left -= 1;\n        if *left == 0 {\n            r.remove(&key);\n        }\n        Some(out)\n    })\n}\n'
_REG_TAKE_NEW = 'fn gpu_reg_take(request: u32, producer: u32, consumer: u32) -> Option<GpuT> {\n    GPU_REG.with(|r| {\n        let mut r = r.borrow_mut();\n        let key = (request, producer, consumer);\n        let q = r.get_mut(&key)?;\n        let out = q.pop_front();\n        if q.is_empty() {\n            r.remove(&key);\n        }\n        out\n    })\n}\n'
LANE_EXTRA_CODE.update({5: '                5 => {\n                    if !p.const_path.is_empty() {\n                        // Add com constante (decode do YOLO): agora também na GPU\n                        let c = gpu_const(&dev, &p.const_path);\n                        Some(gpu_binop(&dev, 0, &inp, &c))\n                    } else {\n                    // Junção do resíduo na GPU. O 1º ramo espera em GPU_ADD_JOIN\n                    // (ou já está no join de CPU, se chegou antes pela rede).\n                    let key = (request_id, current_step);\n                    let other = GPU_ADD_JOIN.with(|j| j.borrow_mut().remove(&key))\n                        .or_else(|| add_branches.remove(&current_step).map(|h| gpu_upload(&dev, &h)));\n                    match other {\n                        Some(o) => {\n                            let b = bindings::planner::addgpuworld::plan::add_r(&dev, inp.len as u32, &o.buf, &inp.buf);\n                            Some(GpuT { buf: Rc::new(b), len: inp.len })\n                        }\n                        None => {\n                            GPU_ADD_JOIN.with(|j| j.borrow_mut().insert(key, inp.clone()));\n                            None\n                        }\n                    }\n                    }\n                }', 10: '                10 => {\n                    let b = opsgpu::unary_r(&dev, 0, inp.len as u32, &inp.buf);\n                    Some(GpuT { buf: Rc::new(b), len: inp.len })\n                }', 12: '                12 => {\n                    let b = opsgpu::unary_r(&dev, 1, inp.len as u32, &inp.buf);\n                    Some(GpuT { buf: Rc::new(b), len: inp.len })\n                }', 11: '                11 => {\n                    if !p.const_path.is_empty() {\n                        let c = gpu_const(&dev, &p.const_path);\n                        Some(gpu_binop(&dev, 2, &inp, &c))\n                    } else {\n                        match gpu_join_take_any(request_id, current_step) {\n                            Some(first) => Some(gpu_binop(&dev, 2, &first, &inp)),\n                            None => {\n                                gpu_join_put(request_id, current_step, origin_step, inp.clone());\n                                None\n                            }\n                        }\n                    }\n                }', 13: '                13 => {\n                    if current_node.output_routes.is_empty() {\n                        panic!("Step {} (Split) sem output_routes — grafo desatualizado?", current_step);\n                    }\n                    let shape = shape_u32(&p.input_shape);\n                    let sizes: Vec<u32> = match &p.split {\n                        Some(s) => s.clone(),\n                        None => {\n                            let n = p.num_outputs.max(1);\n                            let total = *shape.get(p.axis as usize).unwrap_or(&0);\n                            assert_eq!(total % n, 0, "Split step {}: shape[axis]={} não divide igualmente em num_outputs={}", current_step, total, n);\n                            vec![total / n; n as usize]\n                        }\n                    };\n                    let ax = p.axis as usize;\n                    let outer: u32 = shape[..ax].iter().product();\n                    let inner: u32 = shape[ax + 1..].iter().product();\n                    let total_axis: u32 = sizes.iter().sum();\n                    let mut outs: Vec<GpuT> = Vec::new();\n                    let mut off = 0u32;\n                    for &sz in sizes.iter() {\n                        let chunk = sz * inner;\n                        let o = gpu_alloc(&dev, (outer * chunk) as usize);\n                        opsgpu::copy_strided(&dev, outer, chunk, total_axis * inner, off * inner, chunk, 0, &inp.buf, &o.buf);\n                        outs.push(o);\n                        off += sz;\n                    }\n                    {\n                        let log_linha = serde_json::json!({\n                            "request": request_id,\n                            "kernel_id": 13u8,\n                            "step": current_step,\n                            "layer_name": current_node.name,\n                            "dispatch": "gpu-resident-split",\n                            "output_routes": current_node.output_routes.iter().map(|r| (r.to_step, r.output_index)).collect::<Vec<_>>(),\n                            "input_len": inp.len,\n                            "timing_us": { "net_in": net_in_us, "deserialize": deserialize_us, "kernel_call": t_step.elapsed().as_micros() as u64 }\n                        });\n                        log_push(log_linha.to_string());\n                    }\n                    let resident = !gpu_resident_off();\n                    for route in current_node.output_routes.iter() {\n                        let t = outs.get(route.output_index as usize).unwrap_or_else(|| {\n                            panic!("Split step {}: output_routes aponta output_index {} mas há {} fatias", current_step, route.output_index, outs.len())\n                        }).clone();\n                        if resident && lane_eligible(graph, routing, input_img.model_id, route.to_step) {\n                            gpu_reg_put(request_id, current_step, route.to_step, t);\n                            queue.push_back((route.to_step, Vec::new(), current_step));\n                        } else {\n                            queue.push_back((route.to_step, gpu_download(&dev, &t), current_step));\n                        }\n                    }\n                    continue;\n                }', 14: '                14 => {\n                    if p.input_order.is_empty() {\n                        panic!("Step {} (Concat) sem input_order nos params — grafo desatualizado?", current_step);\n                    }\n                    if p.input_order.iter().any(|e| e.is_const) {\n                        panic!("Concat step {}: entrada constante no input_order ainda não é suportada", current_step);\n                    }\n                    gpu_join_put(request_id, current_step, origin_step, inp.clone());\n                    if gpu_join_count(request_id, current_step) < p.input_order.len() {\n                        None\n                    } else {\n                        let shapes: Vec<Vec<u32>> = p.input_order.iter().map(|e| shape_u32(&e.shape)).collect();\n                        let ax = p.axis as usize;\n                        let outer: u32 = shapes[0][..ax].iter().product();\n                        let inner: u32 = shapes[0][ax + 1..].iter().product();\n                        let total_axis: u32 = shapes.iter().map(|s| s[ax]).sum();\n                        let out = gpu_alloc(&dev, (outer * total_axis * inner) as usize);\n                        let mut run = 0u32;\n                        for (e, sh) in p.input_order.iter().zip(shapes.iter()) {\n                            let src_step = e.step.unwrap_or_else(|| panic!("Concat step {}: entrada sem step de origem", current_step));\n                            let src = gpu_join_take(request_id, current_step, src_step)\n                                .unwrap_or_else(|| panic!("Concat step {}: faltou o ramo do step {}", current_step, src_step));\n                            let chunk = sh[ax] * inner;\n                            opsgpu::copy_strided(&dev, outer, chunk, chunk, 0, total_axis * inner, run * inner, &src.buf, &out.buf);\n                            run += sh[ax];\n                        }\n                        gpu_join_clear(request_id, current_step);\n                        Some(out)\n                    }\n                }', 15: '                15 => Some(inp.clone()),', 16: '                16 => {\n                    let shape = shape_u32(&p.input_shape);\n                    let b = opsgpu::transpose_r(&dev, &shape, &p.perm, &inp.buf);\n                    Some(GpuT { buf: Rc::new(b), len: inp.len })\n                }', 17: '                17 => {\n                    let pair: Option<(GpuT, GpuT)> = if !p.lhs_const_path.is_empty() {\n                        Some((gpu_const(&dev, &p.lhs_const_path), inp.clone()))\n                    } else if !p.rhs_const_path.is_empty() {\n                        Some((inp.clone(), gpu_const(&dev, &p.rhs_const_path)))\n                    } else {\n                        let l = p.lhs_step.unwrap_or_else(|| panic!("Step {}: MatMul sem params.lhs_step — grafo desatualizado?", current_step));\n                        let r = p.rhs_step.unwrap_or_else(|| panic!("Step {}: MatMul sem params.rhs_step — grafo desatualizado?", current_step));\n                        if l == r {\n                            panic!("Step {}: MatMul com lhs_step == rhs_step ({}) não é suportado", current_step, l);\n                        }\n                        gpu_join_pair(request_id, current_step, origin_step, &inp, l, r)\n                    };\n                    match pair {\n                        Some((a, b)) => {\n                            let (buf, n) = opsgpu::matmul_r(&dev, &shape_u32(&p.lhs_shape), &shape_u32(&p.rhs_shape), &a.buf, &b.buf);\n                            Some(GpuT { buf: Rc::new(buf), len: n as usize })\n                        }\n                        None => None,\n                    }\n                }', 18: '                18 => {\n                    let (b, n) = opsgpu::resize_r(&dev, p.channels, p.h_in, p.w_in, p.scale_h, p.scale_w, &inp.buf);\n                    Some(GpuT { buf: Rc::new(b), len: n as usize })\n                }', 19: '                19 => {\n                    let shape = shape_u32(&p.input_shape);\n                    let starts: Vec<i32> = p.starts.iter().map(|&v| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32).collect();\n                    let ends: Vec<i32> = p.ends.iter().map(|&v| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32).collect();\n                    let steps: Vec<i32> = p.steps.iter().map(|&v| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32).collect();\n                    let (b, n) = opsgpu::slice_r(&dev, &shape, &starts, &ends, &p.axes, &steps, &inp.buf);\n                    Some(GpuT { buf: Rc::new(b), len: n as usize })\n                }', 22: '                22 => {\n                    let shape = shape_u32(&p.input_shape);\n                    let b = opsgpu::softmax_r(&dev, &shape, p.axis, &inp.buf);\n                    Some(GpuT { buf: Rc::new(b), len: inp.len })\n                }', 20: '                20 => {\n                    if !p.const_path.is_empty() {\n                        let c = gpu_const(&dev, &p.const_path);\n                        Some(if p.const_is_first { gpu_binop(&dev, 1, &c, &inp) } else { gpu_binop(&dev, 1, &inp, &c) })\n                    } else {\n                        match (p.lhs_step, p.rhs_step) {\n                            (Some(l), Some(r)) if l == r => Some(gpu_binop(&dev, 1, &inp, &inp)),\n                            (Some(l), Some(r)) => gpu_join_pair(request_id, current_step, origin_step, &inp, l, r)\n                                .map(|(a, b)| gpu_binop(&dev, 1, &a, &b)),\n                            _ => match gpu_join_take_any(request_id, current_step) {\n                                Some(first) => Some(gpu_binop(&dev, 1, &first, &inp)),\n                                None => {\n                                    gpu_join_put(request_id, current_step, origin_step, inp.clone());\n                                    None\n                                }\n                            },\n                        }\n                    }\n                }', 21: '                21 => {\n                    if !p.const_path.is_empty() {\n                        let c = gpu_const(&dev, &p.const_path);\n                        Some(if p.const_is_first { gpu_binop(&dev, 3, &c, &inp) } else { gpu_binop(&dev, 3, &inp, &c) })\n                    } else {\n                        match (p.lhs_step, p.rhs_step) {\n                            (Some(l), Some(r)) if l == r => Some(gpu_binop(&dev, 3, &inp, &inp)),\n                            (Some(l), Some(r)) => gpu_join_pair(request_id, current_step, origin_step, &inp, l, r)\n                                .map(|(a, b)| gpu_binop(&dev, 3, &a, &b)),\n                            _ => match gpu_join_take_any(request_id, current_step) {\n                                Some(first) => Some(gpu_binop(&dev, 3, &first, &inp)),\n                                None => {\n                                    gpu_join_put(request_id, current_step, origin_step, inp.clone());\n                                    None\n                                }\n                            },\n                        }\n                    }\n                }'})
EXTRA_GPU[100] = {'name': 'opsgpu', 'wit_import': '    import planner:opsgpuworld/plan;', 'wkg_override': '"planner:opsgpuworld" = { path = "../../../kernel/ops_gpu/wit" }', 'wasm_path': '../../../kernel/ops_gpu/target/wasm32-wasip2/release/ops_gpu.wasm'}

def _lane_sub(src, old, new):
    assert src.count(old) == 1, ("4c-2: trecho não encontrado", old[:80])
    return src.replace(old, new)

GPU_LANE_HELPERS = _lane_sub(GPU_LANE_HELPERS,
    "    static GPU_REG: RefCell<HashMap<(u32, u32), (GpuT, u32)>> = RefCell::new(HashMap::new());",
    "    // (request, produtor, consumidor) -> tensores (FIFO)\n    static GPU_REG: RefCell<HashMap<(u32, u32, u32), VecDeque<GpuT>>> = RefCell::new(HashMap::new());")
GPU_LANE_HELPERS = _lane_sub(GPU_LANE_HELPERS, _GPU_DOWNLOAD_OLD, _GPU_DOWNLOAD_NEW)
GPU_LANE_HELPERS = _lane_sub(GPU_LANE_HELPERS, _REG_TAKE_OLD, _REG_TAKE_NEW)
GPU_LANE_HELPERS = _lane_sub(GPU_LANE_HELPERS,
    "    if node.kernel_type == 5 && !node.params.const_path.is_empty() {\n        return false; // Add com constante: CPU\n    }\n", "")
GPU_LANE_HELPERS = _lane_sub(GPU_LANE_HELPERS,
    "    if !LANE_KERNELS.contains(&node.kernel_type) || !LOCAL_KERNELS.contains(&node.kernel_type) {\n        return false;\n    }\n",
    "    if !LANE_KERNELS.contains(&node.kernel_type) || !LOCAL_KERNELS.contains(&node.kernel_type) {\n        return false;\n    }\n"
    "    if node.kernel_type == 14 && node.params.input_order.iter().any(|e| e.is_const) {\n        return false; // Concat com constante: CPU (que também não suporta)\n    }\n")
GPU_LANE_HELPERS += _HELPERS_EXTRA

GPU_LANE_BLOCK = _lane_sub(GPU_LANE_BLOCK, "let gpu_in = gpu_reg_take(request_id, origin_step);",
    "let gpu_in = gpu_reg_take(request_id, origin_step, current_step);")
GPU_LANE_BLOCK = _lane_sub(GPU_LANE_BLOCK,
    "                    GPU_REG.with(|r| r.borrow_mut().insert((request_id, current_step), (out.clone(), on_gpu.len() as u32)));\n                    for nxt in on_gpu {\n                        queue.push_back((nxt, Vec::new(), current_step));",
    "                    for nxt in on_gpu {\n                        gpu_reg_put(request_id, current_step, nxt, out.clone());\n                        queue.push_back((nxt, Vec::new(), current_step));")
GPU_LANE_BLOCK = _lane_sub(GPU_LANE_BLOCK, "let Some(out) = out else { continue; }; // Add esperando o outro ramo",
    "let Some(out) = out else { continue; }; // junção esperando os outros ramos")

# Tipos leves que vão pela via quando ela existe (componente opsgpu).
OPS_LANE = (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
# Com a via ativa, Sigmoid/Mul/SiLU/MatMul vão pelo opsgpu; os componentes
# GPU antigos deles saem da composição (a variante de CPU fica como reserva).
_LANE_FORCED = [False]


def lane_plan(active_kernel_ids, variant_map):
    """Quais tipos vão pela via residente neste grupo e quais componentes
    extra (relu/add/ops na GPU) entram na composição."""
    gpu_lane = [k for k in LANE_CODE if k in active_kernel_ids and variant_map.get(k) == "gpu"]
    act = bool(gpu_lane) or _LANE_FORCED[0] or any(
        variant_map.get(k) == "gpu" for k in (10, 11, 12, 17) if k in active_kernel_ids)
    if not act:
        return [], []
    light = [k for k in (3, 5) if k in active_kernel_ids]
    ops = [k for k in OPS_LANE if k in active_kernel_ids]
    ident = [k for k in (7, 9) if k in active_kernel_ids]
    extras = light + ([100] if (ops or 5 in active_kernel_ids) else [])
    return sorted(set(gpu_lane + light + ops + ident)), extras

# 12/11: passo recebido pela rede ("pinned") também entra na via residente.
GPU_LANE_BLOCK = _lane_sub(GPU_LANE_BLOCK,
    "if step_dest.is_none() && lane_eligible(graph, routing, input_img.model_id, current_step) {",
    "if step_dest.is_none() && (lane_eligible(graph, routing, input_img.model_id, current_step) || (pinned.contains(&current_step) && lane_kind_ok(graph, current_step))) {")
GPU_LANE_HELPERS += """
/// Tipo do passo vai pela via residente aqui, ignorando a rota (passo recebido pela rede).
fn lane_kind_ok(graph: &ModelGraph, step: u32) -> bool {
    let Some(node) = graph.steps.iter().find(|s| s.step == step) else { return false };
    if !LANE_KERNELS.contains(&node.kernel_type) || !LOCAL_KERNELS.contains(&node.kernel_type) {
        return false;
    }
    !(node.kernel_type == 14 && node.params.input_order.iter().any(|e| e.is_const))
}
"""

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
def generate_world_wit(active_kernel_ids, variant_map=None):
    # O world.wit NUNCA menciona wasi:webgpu, mesmo com kernels GPU
    # selecionados — o wac propaga esse import automaticamente a partir do
    # .wasm plugado, não precisa ser declarado aqui.
    imports = "\n".join([KERNELS[k]["wit_import"] for k in active_kernel_ids])
    # 10/11 (4c): com a via residente o grupo cria o device e lê/escreve
    # buffers, então importa webgpu e sync-io diretamente (os mesmos nomes que
    # os kernels importam; o wac unifica as instâncias na composição).
    lane, extras = lane_plan(active_kernel_ids, variant_map or {})
    if lane:
        imports += "\n    import wasi:webgpu/webgpu@0.3.0-rc.2;\n    import wasi-gfx-runtime:wasi-webgpu/sync-io;"
        imports += "".join("\n" + EXTRA_GPU[k]["wit_import"] for k in extras)
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
    _lane, _extras = lane_plan(active_kernel_ids, variant_map)
    for k in _extras:
        lines.append(EXTRA_GPU[k]["wkg_override"])
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
    # 10/11 (4c): o WIT GPU da via residente não usa o record data nas
    # funções, e o wit-bindgen não gera tipos que nenhuma função usa. O WitData
    # vem então do primeiro kernel que está em CPU neste grupo.
    _cpu_ids = [k for k in active_kernel_ids if variant_map.get(k, "cpu") == "cpu"]
    if first_kernel_mod is not None and variant_map.get(1) == "gpu":
        first_kernel_mod = f"{KERNELS[_cpu_ids[0]]['name']}world" if _cpu_ids else None
        if first_kernel_mod is None:
            first_kernel_mod = "__wire__"
    if first_kernel_mod is None:
        first_kernel_mod = f"{KERNELS[_cpu_ids[0]]['name']}world" if _cpu_ids else "__wire__"
    # 12/11: sem kernel de CPU no grupo não existe Data gerado: usa o WireMeta.
    wit_data_ty = "WireMeta" if first_kernel_mod == "__wire__" else f"bindings::planner::{first_kernel_mod}::plan::Data"

    match_arms = "\n".join(
        get_rust_match(k, variant_map.get(k, "cpu")) for k in active_kernel_ids
    )

    local_kernels_const = generate_local_kernels_const(active_kernel_ids)
    lane_ids, _lane_extras = lane_plan(active_kernel_ids, variant_map)
    lane_const = "const LANE_KERNELS: &[u8] = &[" + ", ".join(str(k) for k in lane_ids) + "];"
    lane_arms = "\n".join((LANE_CODE.get(k) or LANE_EXTRA_CODE[k]) for k in lane_ids)
    lane_helpers = GPU_LANE_HELPERS if lane_ids else ""
    if lane_ids and 100 not in _lane_extras:
        # sem ops_gpu na composição: tira o que depende dele (só usado pelos tipos 5/10–22)
        lane_helpers = lane_helpers.replace("use bindings::planner::opsgpuworld::plan as opsgpu;\n", "")
        _i = lane_helpers.index("/// out[i] = a[i] op b[i % len_b]")
        _j = lane_helpers.index("fn gpu_join_put(")
        lane_helpers = lane_helpers[:_i] + lane_helpers[_j:]
    lane_block = (GPU_LANE_BLOCK.replace("@@LANE_ARMS@@", lane_arms)) if lane_ids else ""
    # Só declara o estado de controle do Gemm GPU se ele de fato estiver
    # presente neste nó — evita código morto/não-usado nos outros casos.
    gemm_gpu_state = ""
    # 09/11 (4a): conjunto de (tipo:weight-id) já enviados aos kernels GPU de
    # Conv/BN/ConvBN/ConvBNReLU neste processo (o gemm tem o seu próprio).
    if any(variant_map.get(k) == "gpu" for k in (1, 2, 8, 23, 24) if k in active_kernel_ids):
        gemm_gpu_state += (
            "thread_local! {\n"
            "    static GPU_W_LOADED: RefCell<std::collections::HashSet<String>> = RefCell::new(std::collections::HashSet::new());\n"
            "}\n"
        )
    if 8 in active_kernel_ids and variant_map.get(8) == "gpu":
        gemm_gpu_state += (
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
use std::rc::Rc;
use serde::Deserialize;

// 04/11 (otimização A3): prints por passo/payload desligados. Para depurar,
// troque VERBOSE para true e recompile o grupo. Erros e [AVISO] continuam
// sempre ligados (println!/eprintln! normais).
const VERBOSE: bool = false;
macro_rules! vprintln {{
    ($($t:tt)*) => {{ if VERBOSE {{ println!($($t)*); }} }};
}}

// 04/11 (otimização A2): log em memória, gravado uma vez por conexão, num
// arquivo por grupo (receiver_logs_<porta>.jsonl). Antes cada passo abria,
// escrevia e fechava o receiver_logs.jsonl, e grupos do mesmo nó misturavam
// linhas no mesmo arquivo. Mesmos campos de antes.
thread_local! {{
    static LOG_PATH: RefCell<String> = RefCell::new(String::from("receiver_logs.jsonl"));
    static LOG_BUF: RefCell<Vec<String>> = RefCell::new(Vec::new());
}}

fn log_push(line: String) {{
    LOG_BUF.with(|b| b.borrow_mut().push(line));
}}

fn flush_logs() {{
    let lines: Vec<String> = LOG_BUF.with(|b| std::mem::take(&mut *b.borrow_mut()));
    if lines.is_empty() {{
        return;
    }}
    let mut out = String::with_capacity(lines.iter().map(|l| l.len() + 1).sum());
    for l in &lines {{
        out.push_str(l);
        out.push(10u8 as char);
    }}
    let path = LOG_PATH.with(|p| p.borrow().clone());
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {{
        let _ = file.write_all(out.as_bytes());
    }}
}}

// 05/11 (otimização B1): formato na rede. Metadados em msgpack + tensor como
// floats crus (little-endian; wasm32 e x86_64 são little-endian). Antes o
// tensor ia dentro do msgpack, 5 bytes por float, codificado um a um.
//   payload = "WOF1" | u32 BE tamanho_meta | meta (msgpack, input vazio) | f32 LE...
// O enquadramento externo (u32 BE com o tamanho do payload) não muda.
// Mensagens sem o "WOF1" (versão antiga) continuam sendo aceitas.
const WIRE_MAGIC: &[u8; 4] = b"WOF1";

// 10/11 (4c): espelho do Data só para o serde (mesmos campos e ordem — os
// bytes na rede não mudam). O grupo agora importa wasi:webgpu, cujos tipos
// têm recursos e não podem derivar Serialize, então o additional_derives saiu.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct WireMeta {{
    input: Vec<f32>,
    output: Vec<f32>,
    reply_to: String,
    kernels: Vec<u8>,
    current_kernel: u32,
    request: u32,
    from_step: u32,
    model_id: u32,
}}

fn to_wire(m: &WitData) -> WireMeta {{
    WireMeta {{ input: m.input.clone(), output: m.output.clone(), reply_to: m.reply_to.clone(), kernels: m.kernels.clone(),
        current_kernel: m.current_kernel, request: m.request, from_step: m.from_step, model_id: m.model_id }}
}}

fn from_wire(w: WireMeta) -> WitData {{
    WitData {{ input: w.input, output: w.output, reply_to: w.reply_to, kernels: w.kernels,
        current_kernel: w.current_kernel, request: w.request, from_step: w.from_step, model_id: w.model_id }}
}}

fn encode_frame(meta: &WitData, tensor: &[f32]) -> Vec<u8> {{
    let meta_bytes = rmp_serde::to_vec(&to_wire(meta)).expect("Erro ao serializar metadados");
    let raw: &[u8] = unsafe {{ std::slice::from_raw_parts(tensor.as_ptr() as *const u8, tensor.len() * 4) }};
    let mut out = Vec::with_capacity(8 + meta_bytes.len() + raw.len());
    out.extend_from_slice(WIRE_MAGIC);
    out.extend_from_slice(&(meta_bytes.len() as u32).to_be_bytes());
    out.extend_from_slice(&meta_bytes);
    out.extend_from_slice(raw);
    out
}}

fn decode_frame(buf: &[u8]) -> WitData {{
    if buf.len() >= 8 && &buf[0..4] == WIRE_MAGIC {{
        let meta_len = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
        let mut meta: WitData = from_wire(rmp_serde::from_slice(&buf[8..8 + meta_len]).expect("Falha ao desserializar metadados"));
        let raw = &buf[8 + meta_len..];
        assert!(raw.len() % 4 == 0, "tensor com {{}} bytes, não é múltiplo de 4", raw.len());
        let n = raw.len() / 4;
        let mut v = vec![0.0f32; n];
        unsafe {{ std::ptr::copy_nonoverlapping(raw.as_ptr(), v.as_mut_ptr() as *mut u8, n * 4) }};
        meta.input = v;
        meta
    }} else {{
        from_wire(rmp_serde::from_slice(buf).expect("Failed to deserialize MessagePack response"))
    }}
}}

/// Grava o log pendente ao sair de handle_client, inclusive em erro (`?`).
struct LogFlushGuard;
impl Drop for LogFlushGuard {{
    fn drop(&mut self) {{
        flush_logs();
    }}
}}

/// 04/11 (otimização A1): lê um JSON inteiro de uma vez. serde_json::from_reader
/// sobre File sem buffer faz uma chamada WASI por byte.
fn read_json_file<T: serde::de::DeserializeOwned>(path: &str) -> Result<T, String> {{
    let text = std::fs::read_to_string(path).map_err(|e| format!("{{:?}}", e))?;
    serde_json::from_str(&text).map_err(|e| format!("{{:?}}", e))
}}

{local_kernels_const}
{gemm_gpu_state}

type WitData = {wit_data_ty};

{lane_const}
{lane_helpers}

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
    }});
    export!(Component);
}}

struct Component;

#[derive(Deserialize)]
struct RoutingTable {{
    table: std::collections::HashMap<u8, String>,
    node_scores: std::collections::HashMap<u8, f32>,
    assignments: HashMap<u8, Vec<u8>>,
    // 09/11: corte por camada. Por modelo (chave = model_id em texto), faixas
    // de passos e o nó que as executa. Tem prioridade sobre `table` (rota por
    // tipo); passo fora de qualquer faixa, ou modelo sem entrada, continua
    // na rota por tipo, como antes.
    #[serde(default)]
    step_routes: HashMap<String, Vec<StepRange>>,
    // 12/11: histórico de cortes por versão (a requisição usa o da versão
    // carimbada no model_id; sem versão ou versão desconhecida = o atual).
    #[serde(default)]
    step_routes_hist: HashMap<String, HashMap<String, Vec<StepRange>>>,
}}

#[derive(Deserialize, Debug, Clone)]
struct StepRange {{
    de: u32,
    ate: u32,
    no: String,
}}

// 09/11: endereço público deste grupo (o mesmo que o sys escreve na tabela:
// ip do nó + porta da caixa de entrada = porta interna − 10000). Usado para
// saber se uma faixa de step_routes é "minha" (executa aqui) ou de outro nó.
thread_local! {{
    static SELF_ADDR: RefCell<String> = RefCell::new(String::new());
}}

/// Destino REMOTO de um passo pela rota por camada: Some(endereço) só quando
/// o passo cai numa faixa de step_routes do modelo e essa faixa é de outro
/// grupo. None = sem faixa (vale a rota por tipo) ou a faixa é deste grupo.
fn step_route_dest(routing: &RoutingTable, model_id: u32, step: u32) -> Option<String> {{
    // model_id = modelo (16 bits baixos) | versão do corte (16 bits altos).
    let m = (model_id & 0xFFFF).to_string();
    let v = model_id >> 16;
    let hist = if v > 0 {{ routing.step_routes_hist.get(&m).and_then(|h| h.get(&v.to_string())) }} else {{ None }};
    let ranges = match hist {{ Some(r) => r, None => routing.step_routes.get(&m)? }};
    let r = ranges.iter().find(|r| r.de <= step && step <= r.ate)?;
    let is_self = SELF_ADDR.with(|a| *a.borrow() == r.no);
    if is_self {{ None }} else {{ Some(r.no.clone()) }}
}}

/// Para onde um passo iria por rede (rota por camada ou, sem ela, por tipo
/// quando o kernel não está neste binário). None = executa aqui.
fn remote_dest_for(routing: &RoutingTable, graph: &ModelGraph, model_id: u32, step: u32) -> Option<String> {{
    if let Some(d) = step_route_dest(routing, model_id, step) {{
        return Some(d);
    }}
    let kt = graph.steps.iter().find(|s| s.step == step)?.kernel_type;
    if LOCAL_KERNELS.contains(&kt) {{ None }} else {{ routing.table.get(&kt).cloned() }}
}}

// Passos extras de um envio agrupado: u32 LE em sequência no campo `kernels`
// (que não era usado). Um tensor que vai para 2+ passos no mesmo nó (ramo
// residual) atravessa a rede uma vez só.
// 09/11 (item 3): CAIXA DE SAÍDA. Conexões de saída persistentes, uma por
// destino, reaproveitadas entre saltos e entre requisições. Antes: conexão
// TCP nova a cada salto (handshake + slow start + espera do fechamento =
// ~6,9 RTTs por salto). Agora: escreve o quadro numa conexão já aberta e
// espera 1 byte de ack da caixa de entrada (~1 RTT + bytes/banda).
// Protocolo compatível: a caixa de entrada nova responde 1 byte e continua
// lendo; um destino antigo (ou um grupo sem caixa de entrada) fecha a
// conexão — aí a leitura do ack devolve 0 e a conexão sai do cache (o envio
// valeu, só não é reaproveitada).
// Desligar (para medir antes × depois, ao vivo, sem recompilar): criar o
// arquivo `outbox_off` no diretório do sys (sys_test/). O runtime não repassa
// variáveis de ambiente ao componente, por isso é arquivo e não variável.
thread_local! {{
    static OUTBOX: RefCell<HashMap<String, TcpStream>> = RefCell::new(HashMap::new());
}}

fn outbox_enabled() -> bool {{
    !std::path::Path::new("outbox_off").exists()
}}

struct SendStats {{
    connect_us: u64,
    write_us: u64,
    ack_us: u64,
    reused: bool,
    retried: bool,
}}

/// Escreve o quadro inteiro numa chamada só (tamanho + conteúdo) e espera o ack.
fn write_and_ack(stream: &mut TcpStream, frame: &[u8]) -> std::io::Result<(u64, u64, usize)> {{
    let t_w = Instant::now();
    stream.write_all(frame)?;
    stream.flush()?;
    let write_us = t_w.elapsed().as_micros() as u64;
    let t_a = Instant::now();
    let mut ack = [0u8; 1];
    let n = loop {{
        match stream.read(&mut ack) {{
            Ok(n) => break n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }}
    }};
    Ok((write_us, t_a.elapsed().as_micros() as u64, n))
}}

/// Envia `frame` (já com o u32 BE do tamanho na frente) para `addr`.
fn send_frame(addr: &str, frame: &[u8]) -> std::io::Result<SendStats> {{
    let use_cache = outbox_enabled();
    // 1ª tentativa: conexão do cache, se houver.
    if use_cache {{
        let cached = OUTBOX.with(|o| o.borrow_mut().remove(addr));
        if let Some(mut s) = cached {{
            match write_and_ack(&mut s, frame) {{
                Ok((write_us, ack_us, 1)) => {{
                    OUTBOX.with(|o| o.borrow_mut().insert(addr.to_string(), s));
                    return Ok(SendStats {{ connect_us: 0, write_us, ack_us, reused: true, retried: false }});
                }}
                // 0 bytes ou erro numa conexão reaproveitada: o outro lado
                // fechou (reinício da caixa de entrada). Reenvia numa nova.
                _ => {{}}
            }}
            return send_new(addr, frame, use_cache, true);
        }}
    }}
    send_new(addr, frame, use_cache, false)
}}

fn send_new(addr: &str, frame: &[u8], use_cache: bool, retried: bool) -> std::io::Result<SendStats> {{
    let t_c = Instant::now();
    let mut s = TcpStream::connect(addr)?;
    let _ = s.set_nodelay(true); // pode não existir no WASI; o quadro vai numa escrita só
    let connect_us = t_c.elapsed().as_micros() as u64;
    let (write_us, ack_us, n) = write_and_ack(&mut s, frame)?;
    if use_cache && n == 1 {{
        OUTBOX.with(|o| o.borrow_mut().insert(addr.to_string(), s));
    }}
    Ok(SendStats {{ connect_us, write_us, ack_us, reused: false, retried }})
}}

fn encode_extra_steps(steps: &[u32]) -> Vec<u8> {{
    let mut v = Vec::with_capacity(steps.len() * 4);
    for s in steps {{ v.extend_from_slice(&s.to_le_bytes()); }}
    v
}}

fn decode_extra_steps(bytes: &[u8]) -> Vec<u32> {{
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}}

#[derive(Deserialize, Debug, Clone, Default)]
struct OutputRoute {{
    to_step: u32,
    output_index: u32,
}}

// Espelha o formato emitido pelo onnx_to_dfg.py pros params de Concat
// (kernel_type 14): {{"const": bool, "step": Option<u32>, "shape": [i64, ...]}}.
#[derive(Deserialize, Debug, Clone, Default)]
struct InputOrderEntry {{
    #[serde(default, rename = "const")]
    is_const: bool,
    #[serde(default)]
    step: Option<u32>,
    #[serde(default)]
    shape: Vec<i64>,
}}

#[derive(Deserialize, Debug, Clone, Default)]
struct LayerParams {{
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
    // Preenchido pelo onnx_to_dfg.py só pra steps com >1 saída roteada
    // individualmente (Split é sempre um deles). Usado em vez de next_steps
    // genérico quando presente.
    #[serde(default)]
    output_routes: Vec<OutputRoute>,
}}

#[derive(Deserialize, Debug)]
struct ModelGraph {{
    model: String,
    steps: Vec<StepNode>,
}}

/// Entrada do registro central de modelos (resnet/models.json, gerado e
/// mantido pelo onnx_to_dfg.py — ver claude/project-state.md). Mesmo schema
/// usado pelo handler/orquestrador.
#[derive(Deserialize, Debug, Clone)]
struct ModelEntry {{
    name: String,
    dfg_path: String,
    #[allow(dead_code)]
    kernels: Vec<u8>,
}}

type ModelsRegistry = HashMap<String, ModelEntry>;

/// Caminho do registro central, FLAT dentro do cwd deste processo composto
/// (herda o cwd de quem o spawna — o sys, em sys_test/). NÃO usar "../" aqui:
/// o wasi-gfx-runtime roda o componente sob um sandbox WASI que proíbe
/// escapar do diretório pré-aberto via ".." mesmo quando o arquivo real
/// existe (erro observado: PermissionDenied / "Operation not permitted" ao
/// tentar abrir "../models.json" — ver claude/project-state.md, 28/10). O
/// sys agora copia resnet/models.json pra dentro de sys_test/models.json
/// antes de cada spawn (ver build_and_launch em sys_test/src/main.rs).
const MODELS_JSON_PATH: &str = "models.json";

/// Carrega TODOS os modelos cadastrados em models.json de uma vez, no boot
/// do processo — substitui o antigo "resnet18DFG.json" fixo, que era
/// compartilhado por qualquer request/modelo (ver claude/project-state.md).
/// Cada request escolhe o grafo certo via input_img.model_id em
/// handle_client, sem reabrir arquivo por request.
fn load_all_model_graphs() -> HashMap<u32, ModelGraph> {{
    let registry: ModelsRegistry = read_json_file(MODELS_JSON_PATH).unwrap_or_else(|e| {{
        panic!("Erro ao ler/parsear {{}} (registro central de modelos): {{}}", MODELS_JSON_PATH, e)
    }});

    let mut graphs: HashMap<u32, ModelGraph> = HashMap::new();
    for (id_str, entry) in registry.iter() {{
        let id: u32 = match id_str.parse() {{
            Ok(v) => v,
            Err(_) => {{
                eprintln!("[AVISO] model_id '{{}}' em {{}} não é um u32 válido, ignorando.", id_str, MODELS_JSON_PATH);
                continue;
            }}
        }};
        // dfg_path no models.json é tipo "sys_test/resnet50.json" (relativo
        // a resnet/), mas o DFG real já está FLAT no cwd deste processo
        // (sys_test/ é o próprio cwd) — então usamos só o nome do arquivo,
        // sem nenhum componente de diretório, pelo mesmo motivo do
        // MODELS_JSON_PATH acima (sandbox WASI não deixa escapar via "..",
        // e aqui nem precisaria: o arquivo já está no mesmo diretório).
        let graph_filename = std::path::Path::new(&entry.dfg_path)
            .file_name()
            .unwrap_or_else(|| panic!("dfg_path '{{}}' do modelo '{{}}' (id {{}}) sem nome de arquivo válido", entry.dfg_path, entry.name, id));
        let graph_path = std::path::Path::new(graph_filename);
        let graph: ModelGraph = read_json_file(&graph_path.to_string_lossy()).unwrap_or_else(|e| {{
            panic!("Erro ao ler/parsear o grafo '{{}}' do modelo '{{}}' (id {{}}): {{}}", graph_path.display(), entry.name, id, e)
        }});
        graphs.insert(id, graph);
    }}
    graphs
}}

// 31/10: estado de junção (join) por request, PERSISTENTE entre conexões.
// Antes os buffers eram locais a cada handle_client (uma conexão TCP): com os
// kernels em grupos diferentes, os dois ramos de um Add/Concat/MatMul/Sub/Div
// chegam por conexões separadas e nunca se encontravam. Agora, no início de
// cada conexão, o estado da request é retirado deste mapa (ou criado vazio) e,
// no fim, devolvido se ainda houver ramo esperando. Os rust_match não mudam.
#[derive(Default)]
struct JoinState {{
    add_branches: HashMap<u32, Vec<f32>>,
    mul_branches: HashMap<u32, Vec<f32>>,
    concat_branches: HashMap<u32, HashMap<u32, VecDeque<Vec<f32>>>>,
    matmul_branches: HashMap<u32, HashMap<u32, Vec<f32>>>,
    sub_branches: HashMap<u32, Vec<f32>>,
    div_branches: HashMap<u32, Vec<f32>>,
    sub_slots: HashMap<u32, HashMap<u32, Vec<f32>>>,
    div_slots: HashMap<u32, HashMap<u32, Vec<f32>>>,
}}

impl JoinState {{
    fn is_empty(&self) -> bool {{
        self.add_branches.is_empty() && self.mul_branches.is_empty()
            && self.concat_branches.is_empty() && self.matmul_branches.is_empty()
            && self.sub_branches.is_empty() && self.div_branches.is_empty()
            && self.sub_slots.is_empty() && self.div_slots.is_empty()
    }}
}}

// 31/10: relê a routing_table.json a cada conexão. Antes ela era lida uma vez
// no boot do grupo, e um grupo que subiu antes de outro nunca sabia que o
// outro existia ("Kernel tipo X não está neste nó nem na routing_table").
// Falha de leitura/parse (ex.: o sys reescrevendo o arquivo) -> usa a do boot.
// 04/11 (otimização A1): só relê quando a data de modificação (ou o tamanho)
// do arquivo muda; e lê inteiro de uma vez (antes: from_reader sem buffer,
// uma chamada WASI por byte, a cada conexão — ~5 ms por salto no E4).
// Editar a tabela continua tendo efeito na conexão seguinte.
thread_local! {{
    static ROUTING_CACHE: RefCell<Option<(std::time::SystemTime, u64, Rc<RoutingTable>)>> = RefCell::new(None);
}}

fn reload_routing() -> Option<Rc<RoutingTable>> {{
    let meta = std::fs::metadata("routing_table.json").ok();
    let key = meta.as_ref().and_then(|m| m.modified().ok().map(|t| (t, m.len())));
    if let Some((mtime, len)) = key {{
        let hit = ROUTING_CACHE.with(|c| {{
            c.borrow().as_ref().and_then(|(t, l, r)| if *t == mtime && *l == len {{ Some(Rc::clone(r)) }} else {{ None }})
        }});
        if hit.is_some() {{
            return hit;
        }}
    }}
    let fresh: RoutingTable = read_json_file("routing_table.json").ok()?;
    let fresh = Rc::new(fresh);
    if let Some((mtime, len)) = key {{
        ROUTING_CACHE.with(|c| *c.borrow_mut() = Some((mtime, len, Rc::clone(&fresh))));
    }}
    Some(fresh)
}}

// Kernels não presentes neste binário caem no braço `_` do match abaixo e
// são despachados via TCP para o endereço indicado na routing_table para
// aquele kernel_type. Kernels presentes rodam direto, sem rede.
fn handle_client(
    mut stream: TcpStream,
    initialized: &mut i32,
    graphs: &HashMap<u32, ModelGraph>,
    routing: &RoutingTable,
    pending_joins: &mut HashMap<u32, JoinState>,
) -> std::io::Result<()> {{
    let _log_guard = LogFlushGuard;
    let fresh_routing = reload_routing();
    let routing: &RoutingTable = fresh_routing.as_deref().unwrap_or(routing);
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
        vprintln!("Recebendo payload de {{}} bytes...", len);

        let mut buffer = vec![0u8; len];
        stream.read_exact(&mut buffer)?;
        let _ = stream.shutdown(Shutdown::Both);
        let net_in_us = start_receive.elapsed().as_micros() as u64; // [T]

        let t_de = Instant::now(); // [T]
        let mut input_img: WitData = decode_frame(&buffer);
        let deserialize_us = t_de.elapsed().as_micros() as u64; // [T]

        // Seleciona o grafo do modelo pedido por esta request (antes: um
        // único grafo fixo pra qualquer model_id — ver claude/project-state.md).
        let graph = graphs.get(&(input_img.model_id & 0xFFFF)).unwrap_or_else(|| {{
            panic!("model_id {{}} não encontrado no registro de grafos (models.json)", input_img.model_id & 0xFFFF)
        }});

        *initialized += 1; // [T] contador de requisições atendidas por este processo
        let call_idx = *initialized; // [T]
        let ts_unix_ms = std::time::SystemTime::now() // [T]
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Terceiro elemento da tupla = step de origem (from_step), propagado
        // igual ao `next_status.from_step = current_step` dos receivers isolados.
        let mut queue: VecDeque<(u32, Vec<f32>, u32)> = VecDeque::new();
        // Buffers de junção desta request: retomados de conexões anteriores
        // (ramos que chegaram antes por outra conexão) — ver JoinState.
        let JoinState {{
            mut add_branches,
            mut mul_branches,
            mut concat_branches,
            mut matmul_branches,
            mut sub_branches,
            mut div_branches,
            mut sub_slots,
            mut div_slots,
        }} = pending_joins.remove(&input_img.request).unwrap_or_default();

        // 09/11: envio agrupado — o mesmo tensor também alimenta os passos
        // extras listados em `kernels` (mesma origem).
        for extra in decode_extra_steps(&input_img.kernels) {{
            queue.push_back((extra, input_img.input.clone(), input_img.from_step));
        }}
        // A4: move o tensor recebido em vez de copiar.
        queue.push_back((input_img.current_kernel, std::mem::take(&mut input_img.input), input_img.from_step));
        // 12/11: passos recebidos pela rede rodam aqui (quem enviou já decidiu
        // a rota); a tabela local só vale para os passos seguintes.
        let pinned: Vec<u32> = queue.iter().map(|q| q.0).collect();

        let mut final_output: Vec<f32> = Vec::new();
        let mut final_from_step: u32 = 0;
        let mut reached_final = false;
        let reply_to = input_img.reply_to.clone();
        let request_id = input_img.request;

        while let Some((current_step, mut current_input, origin_step)) = queue.pop_front() {{
            let current_node = graph.steps.iter().find(|s| s.step == current_step)
                .unwrap_or_else(|| panic!("Step {{}} não encontrado no grafo", current_step));

            vprintln!("==> Executando Step {{}} | Camada: '{{}}' | Tipo: {{}}", current_step, current_node.name, current_node.kernel_type);

            let p = &current_node.params;
            let computed_output: Vec<f32>;
            let input_len = current_input.len(); // [T]

            // [T] Timing no nível de STEP INTEIRO (inclui carregar pesos +
            // executar o kernel), não decomposto como nos receivers isolados
            // (que separam weights_load de kernel_call). Decompor exigiria
            // reescrever os 9 rust_match do catálogo KERNELS individualmente.
            let t_step = Instant::now(); // [T]

            // 09/11: rota por camada primeiro. Se o passo é de outro nó, vai
            // pro braço de rede mesmo que o kernel exista aqui (tipo 0 não
            // existe no catálogo, então cai no `_`).
            let step_dest = if pinned.contains(&current_step) && LOCAL_KERNELS.contains(&current_node.kernel_type) {{
                None
            }} else {{
                step_route_dest(routing, input_img.model_id, current_step)
            }};
            let dispatch_kt: u8 = if step_dest.is_some() {{ 0 }} else {{ current_node.kernel_type }};
{lane_block}

            match dispatch_kt {{
{match_arms}
                _ => {{
                    // Kernel não incluído neste binário (ou passo de outro nó
                    // pela rota por camada): despacha por rede.
                    let t_route = Instant::now(); // [T]
                    let route_kind = if step_dest.is_some() {{ "step" }} else {{ "type" }};
                    let target_address = match &step_dest {{
                        Some(d) => d.clone(),
                        None => routing.table.get(&current_node.kernel_type)
                            .cloned()
                            .unwrap_or_else(|| panic!(
                                "Kernel tipo {{}} não está neste nó nem na routing_table",
                                current_node.kernel_type
                            )),
                    }};

                    // 09/11: outros passos na fila com o MESMO tensor (mesma
                    // origem) e o MESMO destino vão juntos neste envio.
                    let mut merged_steps: Vec<u32> = Vec::new();
                    let mut i_q = 0;
                    while i_q < queue.len() {{
                        let same = {{
                            let (s2, t2, o2) = &queue[i_q];
                            *o2 == origin_step && t2.len() == current_input.len() && *t2 == current_input
                                && remote_dest_for(routing, graph, input_img.model_id, *s2).as_deref() == Some(target_address.as_str())
                        }};
                        if same {{
                            let (s2, _, _) = queue.remove(i_q).unwrap();
                            merged_steps.push(s2);
                        }} else {{
                            i_q += 1;
                        }}
                    }}

                    vprintln!(
                        "Step {{}} (tipo {{}}) não está neste nó — despachando para {{}}...",
                        current_step, current_node.kernel_type, target_address
                    );

                    let payload_data = WitData {{
                        input: vec![],
                        output: vec![],
                        reply_to: reply_to.clone(),
                        current_kernel: current_step,
                        request: request_id,
                        kernels: encode_extra_steps(&merged_steps),
                        from_step: origin_step,
                        model_id: input_img.model_id,
                    }};

                    let t_ser = Instant::now(); // [T]
                    let response_payload = encode_frame(&payload_data, &current_input);
                    // Quadro completo (u32 BE + conteúdo) para uma escrita só.
                    let mut frame = Vec::with_capacity(4 + response_payload.len());
                    frame.extend_from_slice(&(response_payload.len() as u32).to_be_bytes());
                    frame.extend_from_slice(&response_payload);
                    drop(response_payload);
                    let serialize_us = t_ser.elapsed().as_micros() as u64; // [T]
                    let sent_bytes = frame.len() - 4; // [T]

                    let st = send_frame(&target_address, &frame)?;
                    let connect_us = st.connect_us; // [T]
                    let write_us = st.write_us; // [T]
                    let ack_wait_us = st.ack_us; // [T]

                    let step_total_us = t_step.elapsed().as_micros() as u64; // [T]

                    {{ // [T]
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
                            "route": route_kind,
                            "dest": target_address,
                            "merged_steps": merged_steps,
                            "input_len": input_len,
                            "sent_bytes": sent_bytes,
                            "conn_reused": st.reused,
                            "conn_retried": st.retried,
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
                        log_push(log_linha.to_string());
                    }} // [T]

                    continue;
                }},
            }}

            let step_total_us = t_step.elapsed().as_micros() as u64; // [T]
            let output_len = computed_output.len(); // [T]
            let is_local = LOCAL_KERNELS.contains(&current_node.kernel_type); // [T]

            {{
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
                log_push(log_linha.to_string());
            }}

            if current_node.next_steps.is_empty() {{
                vprintln!("Fim da rede alcançado localmente no Step {{}}! Saída com {{}} elementos.", current_step, computed_output.len());
                final_output = computed_output;
                final_from_step = current_step;
                reached_final = true;
                break;
            }} else {{
                // A4: o último sucessor recebe o tensor por movimento; só os
                // anteriores (ramificação, ex.: Add dos ResNets) recebem cópia.
                let n_next = current_node.next_steps.len();
                let mut out_opt = Some(computed_output);
                for (i_next, &nxt) in current_node.next_steps.iter().enumerate() {{
                    let v = if i_next + 1 == n_next {{ out_opt.take().unwrap() }} else {{ out_opt.as_ref().unwrap().clone() }};
                    queue.push_back((nxt, v, current_step));
                }}
            }}
        }}

        // Ramo(s) ainda esperando o par: guarda para a próxima conexão desta request.
        let leftover = JoinState {{
            add_branches,
            mul_branches,
            concat_branches,
            matmul_branches,
            sub_branches,
            div_branches,
            sub_slots,
            div_slots,
        }};
        if !leftover.is_empty() {{
            pending_joins.insert(request_id, leftover);
        }}

        // Só devolve pro cliente se a rede terminou DENTRO deste nó. Se tudo
        // que restava foi despachado por rede, quem responde é o nó remoto.
        if reached_final {{
            let final_data = WitData {{
                input: vec![],
                output: vec![],
                reply_to: reply_to.clone(),
                current_kernel: 0,
                request: request_id,
                kernels: vec![],
                from_step: final_from_step,
                model_id: input_img.model_id,
            }};

            let mut next_stream = TcpStream::connect(&reply_to)?;
            let response_payload = encode_frame(&final_data, &final_output);
            let resp_len = response_payload.len() as u32;
            next_stream.write_all(&resp_len.to_be_bytes())?;
            next_stream.write_all(&response_payload)?;
            next_stream.flush()?;
        }}

        break;
    }}

    vprintln!("Conexão finalizada com o cliente.");
    Ok(())
}}

impl bindings::exports::wasi::cli::run::Guest for Component {{
    fn run() -> Result<(), ()> {{
        let args: Vec<String> = env::args().collect();
        let port = args.get(1).map(|s| s.as_str()).unwrap_or("8080");
        let ip = args.get(2).map(|s| s.as_str()).unwrap_or("0.0.0.0");
        let bind_addr = format!("{{}}:{{}}", ip, port);
        let listener = TcpListener::bind(&bind_addr).expect(&format!("Não conseguiu abrir a porta {{}}", port));
        // 09/11: endereço público = o da caixa de entrada (porta interna − 10000).
        let public_port: u32 = port.parse::<u32>().map(|p| if p >= 20000 {{ p - 10000 }} else {{ p }}).unwrap_or(0);
        SELF_ADDR.with(|a| *a.borrow_mut() = format!("{{}}:{{}}", ip, public_port));

        // Grafos de TODOS os modelos cadastrados em models.json, carregados
        // uma vez no boot do processo (antes: um único "resnet18DFG.json"
        // fixo, compartilhado por qualquer request/modelo — ver
        // claude/project-state.md). Cada request escolhe o grafo certo via
        // input_img.model_id dentro de handle_client.
        let graphs: HashMap<u32, ModelGraph> = load_all_model_graphs();

        let routing: RoutingTable = read_json_file("routing_table.json").unwrap_or_else(|e| panic!("Erro ao ler routing_table.json: {{}}", e));
        // A2: um arquivo de log por grupo (porta).
        LOG_PATH.with(|p| *p.borrow_mut() = format!("receiver_logs_{{}}.jsonl", port));

        println!("Listening on {{}}", bind_addr);

        let mut init = 0;
        let mut pending_joins: HashMap<u32, JoinState> = HashMap::new();
        for stream in listener.incoming() {{
            match stream {{
                Ok(stream) => {{
                    vprintln!("Nova conexão recebida!");
                    if let Err(e) = handle_client(stream, &mut init, &graphs, &routing, &mut pending_joins) {{
                        eprintln!("Error handling client: {{:?}}", e);
                    }}
                }}
                Err(e) => {{
                    eprintln!("Connection failed: {{:?}}", e);
                }}
            }}
            vprintln!("Waiting for a new request...");
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
            print(f"[ERRO] Kernel ID {k} inválido. Disponíveis: {sorted(KERNELS)}.")
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
    # 10/11 (4c, parte 2): com a via residente, os leves vão pelo opsgpu.
    if lane_plan(kernel_ids, variant_map)[0]:
        _LANE_FORCED[0] = True
        for _k in (10, 11, 12, 17):
            if variant_map.get(_k) == "gpu":
                variant_map[_k] = "cpu"

    print(f"[*] Kernels incluídos neste nó: {kernel_ids}")
    print(f"[*] Variantes: {variant_map}")
    print(f"[*] Kernels NÃO incluídos serão despachados por rede via routing_table.json em runtime.")

    comp_dir = os.path.abspath(args.component_dir)
    wit_dir = os.path.join(comp_dir, "wit")
    src_dir = os.path.join(comp_dir, "src")
    os.makedirs(wit_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)

    with open(os.path.join(wit_dir, "world.wit"), "w") as f:
        f.write(generate_world_wit(kernel_ids, variant_map))
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

    # 10/11 (4c): relu/add na GPU entram como componentes extra (a versão de
    # CPU continua no grupo para os casos que não vão pela via residente).
    _lane, _extras = lane_plan(kernel_ids, variant_map)
    plug_list = [(k, KERNELS[k]["variants"][variant_map[k]]["wasm_path"], KERNELS[k]["name"], variant_map[k]) for k in kernel_ids]
    plug_list += [(k, EXTRA_GPU[k]["wasm_path"], EXTRA_GPU[k]["name"], "gpu") for k in _extras]

    for idx, (k, k_rel, k_name, variant) in enumerate(plug_list):
        k_wasm_path = os.path.normpath(os.path.join(comp_dir, k_rel))

        if not os.path.exists(k_wasm_path):
            print(f"[ERRO] Binário do kernel {k_name} ({variant}) não encontrado em: {k_wasm_path}")
            sys.exit(1)

        if idx == len(plug_list) - 1:
            out_step = os.path.abspath(args.output)
        else:
            out_step = os.path.join(comp_dir, f"temp_plug_{idx}.wasm")
            temp_files.append(out_step)

        print(f" -> Plugando Kernel {k} ({k_name}, {variant})...")
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