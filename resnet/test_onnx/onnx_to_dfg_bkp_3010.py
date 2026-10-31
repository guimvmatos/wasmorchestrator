#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

OP_TO_KERNEL = {
    "Conv": 1,
    "BatchNormalization": 2,
    "Relu": 3,
    "MaxPool": 4,
    "Add": 5,
    "GlobalAveragePool": 6,
    "Flatten": 7,
    "Gemm": 8,                 # peso fixo (W, B) — camada densa clássica
    "Dropout": 9,
    "Sigmoid": 10,
    "Mul": 11,
    "Split": 13,
    "Concat": 14,
    "Reshape": 15,
    "Transpose": 16,
    "MatMul": 17,               # tensor x tensor dinâmico — NÃO é Gemm
    "Resize": 18,
    "Slice": 19,
    "Sub": 20,
    "Div": 21,
    "Softmax": 22,
}

SILU_KERNEL_TYPE = 12  # produzido pela fusão Sigmoid+Mul, não é op_type direto

# Registro central de modelos — mora em resnet/models.json (um nível acima
# de test_onnx/), pra handler e orchestrator acharem independente de onde
# rodam. Calculado a partir do caminho deste próprio script, não do cwd,
# pra funcionar não importa de onde o onnx_to_dfg.py é chamado.
MODELS_JSON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models.json"
)


def update_models_registry(model_name, steps, output_json_path):
    """
    Cria/atualiza uma entrada em models.json com base no modelo que acabou
    de ser exportado:
      - "name": o mesmo nome já usado no campo "model" do DFG (derivado
        do nome do .onnx passado em --onnx — não é um parâmetro novo).
      - "kernels": lista ordenada e sem repetição de todos os kernel_type
        usados pelo grafo (já pós-fusão de SiLU) — é a demanda real desse
        modelo, pro orquestrador consumir no lugar do vec![1,...,9] fixo.
      - "dfg_path": caminho do JSON recém-gerado, relativo ao diretório
        onde este models.json mora (resnet/), pra funcionar não importa de
        onde o DFG físico esteja.
    Reexecutar pro mesmo modelo (mesmo "model_name") atualiza a entrada
    existente no lugar — acha pelo nome, nunca duplica nem muda o
    model_id já atribuído antes.
    """
    registry_path = os.path.abspath(MODELS_JSON_PATH)
    registry_dir = os.path.dirname(registry_path)

    if os.path.isfile(registry_path):
        with open(registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {}

    kernels = sorted(set(s["kernel_type"] for s in steps))
    dfg_abs_path = os.path.abspath(output_json_path)
    dfg_rel_path = os.path.relpath(dfg_abs_path, start=registry_dir)

    existing_id = None
    for mid, entry in registry.items():
        if entry.get("name") == model_name:
            existing_id = mid
            break

    if existing_id is None:
        existing_ids = [int(k) for k in registry.keys() if k.isdigit()]
        new_id = str(max(existing_ids, default=0) + 1)
    else:
        new_id = existing_id

    registry[new_id] = {
        "name": model_name,
        "dfg_path": dfg_rel_path,
        "kernels": kernels,
    }

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(
        f"[SUCESSO] models.json atualizado: id {new_id} -> '{model_name}' "
        f"({len(kernels)} kernel_types, dfg_path='{dfg_rel_path}')"
    )

def save_weight_bin(tensor_proto, output_dir):
    if tensor_proto is None:
        return ""
    array = numpy_helper.to_array(tensor_proto).astype(np.float32)
    
    # Usa o nome real do tensor contido no próprio ONNX
    raw_name = tensor_proto.name.replace("/", "_").replace(":", "_")
    filename = f"{raw_name}.bin"
    
    os.makedirs(output_dir, exist_ok=True)
    real_filepath = os.path.join(output_dir, filename)
    array.tofile(real_filepath)
    
    # Adiciona o ../ para a visibilidade do WASM
    dir_name = os.path.basename(os.path.normpath(output_dir))
    return f"../{dir_name}/{filename}"

def try_fuse_silu(valid_nodes):
    """
    Detecta o padrão SiLU (x * sigmoid(x)) e funde o par Sigmoid+Mul em um
    único nó sintético. Critério: o Sigmoid tem exatamente um consumidor,
    esse consumidor é um Mul, e um dos dois inputs do Mul é exatamente o
    mesmo tensor que entrou no Sigmoid (não o output dele). Quando o padrão
    não bate (ex: Sigmoid isolado, ou Mul que não usa o input original),
    cada nó continua sendo emitido separadamente — a fusão é só uma
    otimização quando reconhecida, nunca um requisito.
    """
    output_to_consumers = {}
    for idx, node in enumerate(valid_nodes):
        for inp in node.input:
            output_to_consumers.setdefault(inp, []).append(idx)

    silu_pairs = {}       # sigmoid_idx -> mul_idx
    fused_mul_indices = set()

    for idx, node in enumerate(valid_nodes):
        if node.op_type != "Sigmoid":
            continue
        sig_input = node.input[0]
        sig_output = node.output[0]
        consumers = output_to_consumers.get(sig_output, [])
        if len(consumers) != 1:
            continue
        mul_idx = consumers[0]
        mul_node = valid_nodes[mul_idx]
        if mul_node.op_type != "Mul":
            continue
        if sig_input not in mul_node.input:
            continue
        silu_pairs[idx] = mul_idx
        fused_mul_indices.add(mul_idx)

    fused_indices = set(silu_pairs.keys()) | fused_mul_indices
    result = []
    for idx, node in enumerate(valid_nodes):
        if idx in silu_pairs:
            mul_idx = silu_pairs[idx]
            result.append(("silu", node, valid_nodes[mul_idx]))
        elif idx in fused_indices:
            continue
        else:
            result.append(("raw", node))
    return result


def get_node_attribute(node, attr_name, default=None):
    for attr in node.attribute:
        if attr.name == attr_name:
            if attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            elif attr.type == onnx.AttributeProto.INT:
                return attr.i
            elif attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
            elif attr.type == onnx.AttributeProto.STRING:
                return attr.s.decode("utf-8")
    return default

def normalize_axis(axis, rank):
    """axis negativo (ex: -1 visto no head de detecção do YOLO) vira índice
    positivo quando o rank do tensor é conhecido. Sem rank confiável, devolve
    o axis cru — quem consome o DFG decide."""
    if axis is None:
        return axis
    if axis < 0 and rank:
        return rank + axis
    return axis

def get_split_sizes(node, initializers):
    """
    Tamanhos de cada saída do Split. Dois formatos coexistem no ONNX:
    - opset < 13: atributo INTS "split" no próprio node.
    - opset >= 13: segundo input do node ("split"), um initializer int64.
    Quando nenhum dos dois existe (ex: atributo "num_outputs" do opset 18,
    split igual entre as saídas), devolve None — quem consome o DFG calcula
    o tamanho igual a partir do shape de entrada e do num_outputs.
    """
    attr_split = get_node_attribute(node, "split", None)
    if attr_split is not None:
        return [int(x) for x in attr_split]
    if len(node.input) > 1 and node.input[1] in initializers:
        sizes_tensor = initializers[node.input[1]]
        arr = numpy_helper.to_array(sizes_tensor)
        return [int(x) for x in arr.tolist()]
    return None

def extract_dfg(onnx_path, output_json, weights_dir="bin_weights", input_shape=(1, 3, 224, 224)):
    os.makedirs(weights_dir, exist_ok=True)
    
    # 1. Carrega e infere shapes
    model = onnx.load(onnx_path)
    try:
        inferred_model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[AVISO] Falha no shape_inference automático: {e}. Usando modelo original.")
        inferred_model = model

    graph = inferred_model.graph

    # 2. Mapeia inicializadores (pesos / bias)
    initializers = {init.name: init for init in graph.initializer}

    # 3. Mapeia formatos dos tensores intermediários
    value_info_shapes = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        shape = []
        if vi.type.tensor_type.HasField("shape"):
            for dim in vi.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                else:
                    shape.append(None)
        value_info_shapes[vi.name] = shape

    # 4. Mapeia produtores e consumidores para encontrar next_steps
    tensor_producer = {}   # tensor_name -> step_id
    node_inputs = []       # lista de inputs de cada nó

    # Primeiro passe: identificar nós suportados
    valid_nodes = []
    for idx, node in enumerate(graph.node):
        op_type = node.op_type
        if op_type in OP_TO_KERNEL:
            valid_nodes.append(node)
        else:
            print(f"[AVISO] Ignorando nó '{node.name}' com op_type '{op_type}' (não mapeado)")

    # Funde pares Sigmoid+Mul que formam SiLU antes de numerar os steps —
    # a partir daqui, "fused_nodes" é a lista real usada para gerar o DFG.
    # Cada entrada é ("raw", node) ou ("silu", sigmoid_node, mul_node).
    fused_nodes = try_fuse_silu(valid_nodes)

    # Mapear saídas para os passos. Para um par fundido, só o output do
    # Mul (a saída real do SiLU) conta — o output do Sigmoid interno nunca
    # é consumido por fora do par, então não precisa de entrada própria.
    for step_id, entry in enumerate(fused_nodes, start=1):
        if entry[0] == "silu":
            _, _, mul_node = entry
            for out_name in mul_node.output:
                tensor_producer[out_name] = step_id
        else:
            _, node = entry
            for out_name in node.output:
                tensor_producer[out_name] = step_id

    # 5. Segundo passe: construir os StepNodes e extrair os pesos
    steps = []
    for step_id, entry in enumerate(fused_nodes, start=1):
        if entry[0] == "silu":
            _, sigmoid_node, mul_node = entry
            kernel_type = SILU_KERNEL_TYPE
            layer_name = mul_node.name if mul_node.name else f"SiLU_{step_id}"
            node = mul_node  # usado abaixo só para next_steps (output real do par)
            # O input real do SiLU é o input do Sigmoid, não um input do Mul
            # (que tem dois: o original e a saída do Sigmoid).
            main_input_name = sigmoid_node.input[0]
        else:
            _, node = entry
            kernel_type = OP_TO_KERNEL[node.op_type]
            layer_name = node.name if node.name else f"{node.op_type}_{step_id}"
            main_input_name = node.input[0]
            # Mul com constante: o input[0] pode ser a própria constante
            # (ex: "0.1767..." antes do tensor principal) — nesse caso o
            # shape de referência tem que vir do outro input, não dele.
            if node.op_type == "Mul" and main_input_name in initializers:
                main_input_name = node.input[1]

        params = {}
        in_shape = value_info_shapes.get(main_input_name, [1, 3, 224, 224])
        
        c_in = in_shape[1] if len(in_shape) > 1 and in_shape[1] is not None else 3
        h_in = in_shape[2] if len(in_shape) > 2 and in_shape[2] is not None else 224
        w_in = in_shape[3] if len(in_shape) > 3 and in_shape[3] is not None else 224

        # Preencher atributos específicos de cada operador
        if kernel_type == 1:  # Conv
            strides = get_node_attribute(node, "strides", [1, 1])
            pads = get_node_attribute(node, "pads", [0, 0, 0, 0])
            kernel_shape = get_node_attribute(node, "kernel_shape", [3, 3])
            groups = get_node_attribute(node, "group", 1)  # atributo ONNX chama "group", singular

            w_tensor = initializers.get(node.input[1]) if len(node.input) > 1 else None
            w_path = save_weight_bin(w_tensor, weights_dir) if w_tensor else ""

            # Terceiro input opcional do Conv: bias (separado de BatchNorm,
            # usado quando a camada não é seguida de BN — ex: as 3 convs
            # finais de cada detection head do YOLO, que vão direto pra
            # Reshape sem normalização).
            b_tensor = initializers.get(node.input[2]) if len(node.input) > 2 else None
            b_path = save_weight_bin(b_tensor, weights_dir) if b_tensor else ""

            cout = w_tensor.dims[0] if w_tensor else c_in
            kh = kernel_shape[0] if len(kernel_shape) > 0 else 3
            kw = kernel_shape[1] if len(kernel_shape) > 1 else 3

            # cin REAL do tensor de entrada, não w_tensor.dims[1] — para
            # Conv agrupada/depthwise (groups > 1), o peso ONNX vem no
            # formato [cout, cin/groups, kh, kw], então w_tensor.dims[1] é
            # cin/groups, não cin. Usar sempre o shape do tensor de entrada
            # (já lido acima como c_in) evita essa armadilha silenciosa.
            cin = c_in

            params = {
                "weight_path": w_path,
                "bias_path": b_path,
                "cout": int(cout),
                "cin": int(cin),
                "groups": int(groups),
                "h_in": int(h_in),
                "w_in": int(w_in),
                "kh": int(kh),
                "kw": int(kw),
                "stride": int(strides[0]),
                "padding": int(pads[0])
            }

        elif kernel_type == 2:  # BatchNorm
            epsilon = get_node_attribute(node, "epsilon", 1e-5)
            scale_t = initializers.get(node.input[1]) if len(node.input) > 1 else None
            beta_t  = initializers.get(node.input[2]) if len(node.input) > 2 else None
            mean_t  = initializers.get(node.input[3]) if len(node.input) > 3 else None
            var_t   = initializers.get(node.input[4]) if len(node.input) > 4 else None

            params = {
                "scale_path": save_weight_bin(scale_t, weights_dir),
                "beta_path": save_weight_bin(beta_t, weights_dir),
                "mean_path": save_weight_bin(mean_t, weights_dir),
                "var_path": save_weight_bin(var_t, weights_dir),
                "epsilon": float(epsilon),
                "channels": int(c_in),
                "h_in": int(h_in),
                "w_in": int(w_in)
            }

        elif kernel_type == 4:  # MaxPool
            strides = get_node_attribute(node, "strides", [1, 1])
            pads = get_node_attribute(node, "pads", [0, 0, 0, 0])
            kernel_shape = get_node_attribute(node, "kernel_shape", [2, 2])
            params = {
                "channels": int(c_in),
                "h_in": int(h_in),
                "w_in": int(w_in),
                "kh": int(kernel_shape[0]),
                "kw": int(kernel_shape[1]),
                "stride": int(strides[0]),
                "padding": int(pads[0])
            }

        elif kernel_type == 8:  # Gemm / Linear
            w_tensor = initializers.get(node.input[1]) if len(node.input) > 1 else None
            b_tensor = initializers.get(node.input[2]) if len(node.input) > 2 else None
            
            in_features = w_tensor.dims[1] if (w_tensor and len(w_tensor.dims) > 1) else c_in
            out_features = w_tensor.dims[0] if (w_tensor and len(w_tensor.dims) > 0) else 1000

            params = {
                "weight_path": save_weight_bin(w_tensor, weights_dir),
                "bias_path": save_weight_bin(b_tensor, weights_dir),
                "in_features": int(in_features),
                "out_features": int(out_features)
            }

        elif kernel_type == 6:  # GlobalAveragePool
            # EXCEÇÃO: GlobalAveragePool não tem atributos no ONNX (confirmado
            # via Netron) — a operação em si não precisa de configuração.
            # Mas o kernel que a executa recebe o tensor como buffer linear
            # achatado (list<f32>), sem informação de forma embutida, e
            # precisa saber onde termina um canal e começa o outro
            # (offset = c * spatial_size) para calcular a média por canal
            # corretamente. channels/h_in/w_in não são inventados: é o
            # mesmo shape do tensor de entrada já usado por Conv/BatchNorm/
            # MaxPool acima (c_in/h_in/w_in), apenas também registrado aqui
            # porque GAP não tem outro lugar no schema para carregar esse
            # dado. Sem isso, o kernel teria que assumir um valor fixo de
            # canais (ex: 512), o que quebra silenciosamente em arquiteturas
            # com canal final diferente (ex: ResNet-50 usa 2048).
            params = {
                "channels": int(c_in),
                "h_in": int(h_in),
                "w_in": int(w_in)
            }

        elif kernel_type == 9:  # Dropout (Identity em inferência)
            params = {}

        elif kernel_type == SILU_KERNEL_TYPE:  # SiLU fundido (x * sigmoid(x))
            params = {}

        elif kernel_type == 10:  # Sigmoid isolado (fallback quando não funde com Mul)
            params = {}

        elif kernel_type == 5:  # Add
            # Mesmo padrão do Mul (kernel 11): um dos dois inputs pode ser
            # constante (initializer) — ex: "/model.23/Add_1" no decode do
            # YOLO (x2y2 = anchor_points + rb, dist2bbox), onde um lado é o
            # tensor anchor_points fixo (grid constante pra resolução de
            # entrada fixa). Add é comutativo — diferente de Sub/Div (20/21)
            # — então não precisa de const_is_first, ordem não muda o
            # resultado. Quando os dois lados são dinâmicos (ex: resíduo de
            # bloco conv/residual em ResNet/VGG/YOLO), continua join-buffer
            # de 2 ramos por step, igual a sempre — nada muda pros modelos
            # que só usam Add assim.
            in0, in1 = node.input[0], node.input[1]
            const_tensor = None
            if in0 in initializers:
                const_tensor = initializers[in0]
            elif in1 in initializers:
                const_tensor = initializers[in1]

            if const_tensor is not None:
                params = {
                    "const_path": save_weight_bin(const_tensor, weights_dir),
                    "const_shape": [int(d) for d in const_tensor.dims],
                }
            else:
                params = {}

        elif kernel_type == 11:  # Mul
            in0, in1 = node.input[0], node.input[1]
            const_tensor = None
            if in0 in initializers:
                const_tensor = initializers[in0]
            elif in1 in initializers:
                const_tensor = initializers[in1]

            if const_tensor is not None:
                params = {
                    "const_path": save_weight_bin(const_tensor, weights_dir),
                    "const_shape": [int(d) for d in const_tensor.dims],
                }
            else:
                params = {}

        elif kernel_type == 13:  # Split
            # input_shape é o shape COMPLETO de entrada (não só c_in/h_in/w_in
            # do topo da função) — o receiver precisa do rank inteiro pra
            # calcular outer/inner ao fatiar, igual ao Concat.
            full_in_shape = value_info_shapes.get(main_input_name, in_shape)
            rank = len(full_in_shape) if full_in_shape else None
            axis = normalize_axis(get_node_attribute(node, "axis", 0), rank)
            sizes = get_split_sizes(node, initializers)
            num_outputs = get_node_attribute(node, "num_outputs", None)
            params = {
                "axis": int(axis),
                "input_shape": [int(d) if d is not None else -1 for d in full_in_shape],
                # None quando nem atributo "split" nem input "split" definem
                # tamanhos explícitos — nesse caso é split igual, e quem
                # consome o DFG divide o shape de entrada por num_outputs.
                "split": sizes,
                "num_outputs": int(num_outputs) if num_outputs is not None else (len(sizes) if sizes else len(node.output)),
            }

        elif kernel_type == 14:  # Concat
            # Shape de referência vem do input[0], mas cada ramo pode ter
            # tamanho diferente no próprio axis — por isso cada entrada de
            # input_order carrega seu shape resolvido individualmente
            # (value_info_shapes), não o in_shape genérico do topo da função.
            rank = len(in_shape) if in_shape else None
            axis = normalize_axis(get_node_attribute(node, "axis", 1), rank)
            # Ordem importa: input_order preserva a ordem real do node.input
            # do ONNX, resolvendo cada entrada para o step que a produziu
            # (ou marcando como constante, quando vier de um initializer),
            # junto do shape concreto de cada ramo — necessário pro receiver
            # saber quantos floats copiar de cada um sem ter que inferir em
            # runtime.
            input_order = []
            for inp in node.input:
                if inp in initializers:
                    const_tensor = initializers[inp]
                    input_order.append({
                        "const": True,
                        "step": None,
                        "shape": [int(d) for d in const_tensor.dims],
                    })
                else:
                    branch_shape = value_info_shapes.get(inp, [])
                    input_order.append({
                        "const": False,
                        "step": tensor_producer.get(inp),
                        "shape": [int(d) if d is not None else -1 for d in branch_shape],
                    })
            params = {
                "axis": int(axis),
                "input_order": input_order,
            }

        elif kernel_type == 15:  # Reshape
            # Reshape nunca reordena elementos de um buffer row-major — só
            # reinterpreta as dimensões. Como este pipeline passa tensores
            # como Vec<f32> já achatado (sem metadado de shape viajando
            # junto do dado), o kernel de Reshape é um pass-through puro:
            # os consumidores downstream (Transpose, MatMul, Slice, Softmax)
            # recebem seu próprio shape resolvido estaticamente aqui no
            # grafo, não inferido a partir deste Reshape.
            params = {}

        elif kernel_type == 16:  # Transpose
            full_in_shape = value_info_shapes.get(main_input_name, in_shape)
            rank = len(full_in_shape) if full_in_shape else None
            perm = get_node_attribute(node, "perm", None)
            if perm is None:
                # Sem atributo "perm": default do ONNX é inverter todos os eixos.
                perm = list(reversed(range(rank))) if rank else []
            params = {
                "input_shape": [int(d) if d is not None else -1 for d in full_in_shape],
                "perm": [int(p) for p in perm],
            }

        elif kernel_type == 17:  # MatMul (tensor x tensor dinâmico, não é Gemm)
            # MatMul NÃO é comutativo — diferente de Add/Mul (kernel 5/11),
            # o receiver precisa saber EXATAMENTE qual ramo é lhs e qual é
            # rhs, não só "os dois ramos chegaram". Por isso, igual ao
            # input_order do Concat, resolvemos aqui o producer real de
            # node.input[0] (lhs) e node.input[1] (rhs) via tensor_producer
            # — o receiver usa lhs_step/rhs_step pra casar cada branch que
            # chega (identificado pelo from_step) com o slot certo, em vez
            # de assumir ordem de chegada. Quando um dos lados é constante
            # (initializer), não há join: o outro lado é o único ramo
            # dinâmico esperado.
            in0, in1 = node.input[0], node.input[1]
            in0_shape = value_info_shapes.get(in0, [])
            in1_shape = value_info_shapes.get(in1, [])
            lhs_const = initializers.get(in0)
            rhs_const = initializers.get(in1)
            params = {
                "lhs_shape": [int(d) if d is not None else -1 for d in in0_shape],
                "rhs_shape": [int(d) if d is not None else -1 for d in in1_shape],
                "lhs_step": tensor_producer.get(in0) if lhs_const is None else None,
                "rhs_step": tensor_producer.get(in1) if rhs_const is None else None,
                "lhs_const_path": save_weight_bin(lhs_const, weights_dir) if lhs_const is not None else "",
                "rhs_const_path": save_weight_bin(rhs_const, weights_dir) if rhs_const is not None else "",
            }

        elif kernel_type == 18:  # Resize (upsample nearest, usado no FPN do YOLO)
            mode = get_node_attribute(node, "mode", "nearest")
            # Desde opset 11, scales/sizes chegam como inputs opcionais, não
            # atributos: node.input = [X, roi, scales, sizes]. Resolve o
            # fator de upsample a partir do initializer de "scales" quando
            # presente; cai pra "sizes" (dividido pelo h_in/w_in de entrada)
            # quando o exporter usou sizes explícito em vez de scales.
            scale_h, scale_w = 2.0, 2.0
            if len(node.input) > 2 and node.input[2] and node.input[2] in initializers:
                scales_arr = numpy_helper.to_array(initializers[node.input[2]])
                if len(scales_arr) >= 4:
                    scale_h, scale_w = float(scales_arr[2]), float(scales_arr[3])
            elif len(node.input) > 3 and node.input[3] and node.input[3] in initializers:
                sizes_arr = numpy_helper.to_array(initializers[node.input[3]])
                if len(sizes_arr) >= 4 and h_in and w_in:
                    scale_h = float(sizes_arr[2]) / float(h_in)
                    scale_w = float(sizes_arr[3]) / float(w_in)
            params = {
                "channels": int(c_in),
                "h_in": int(h_in),
                "w_in": int(w_in),
                "scale_h": scale_h,
                "scale_w": scale_w,
                "mode": str(mode),
            }

        elif kernel_type == 19:  # Slice
            full_in_shape = value_info_shapes.get(main_input_name, in_shape)
            rank = len(full_in_shape) if full_in_shape else None

            def _ints_input(idx):
                if len(node.input) > idx and node.input[idx] and node.input[idx] in initializers:
                    return [int(x) for x in numpy_helper.to_array(initializers[node.input[idx]]).tolist()]
                return None

            starts = get_node_attribute(node, "starts", None)
            if starts is None:
                starts = _ints_input(1)
            ends = get_node_attribute(node, "ends", None)
            if ends is None:
                ends = _ints_input(2)
            axes = get_node_attribute(node, "axes", None)
            if axes is None:
                axes = _ints_input(3)
            if axes is None and starts is not None:
                axes = list(range(len(starts)))
            # NUNCA chamar esta variável de "steps" — esse nome já é usado pela
            # lista acumuladora de todo o DFG (steps = [] lá em cima, uma por
            # função). Reatribuir "steps" aqui sobrescreve silenciosamente essa
            # lista inteira a partir deste ponto do loop, descartando todos os
            # StepNodes já acumulados (bug real encontrado: um Slice no meio do
            # grafo truncava o JSON exportado para poucos steps finais).
            slice_strides = get_node_attribute(node, "steps", None)
            if slice_strides is None:
                slice_strides = _ints_input(4)
            if slice_strides is None and starts is not None:
                slice_strides = [1] * len(starts)

            params = {
                "input_shape": [int(d) if d is not None else -1 for d in full_in_shape],
                "starts": [int(s) for s in starts] if starts else [],
                "ends": [int(e) for e in ends] if ends else [],
                "axes": [int(normalize_axis(a, rank)) for a in axes] if axes else [],
                "steps": [int(s) for s in slice_strides] if slice_strides else [],
            }

        elif kernel_type in (20, 21):  # Sub / Div
            # Mesmo padrão do Mul (kernel 11): um dos dois inputs pode ser
            # constante (initializer) — ex: anchor_points fixo no decode do
            # YOLO (grid constante pra resolução de entrada fixa). Sub/Div
            # NÃO são comutativos como Add/Mul, então guardamos qual lado é
            # a constante (const_is_first) pro kernel montar a - b na ordem
            # certa. Quando os dois lados são dinâmicos, o kernel trata como
            # binário elementwise simples (mesmo shape), igual ao Add.
            in0, in1 = node.input[0], node.input[1]
            const_tensor = None
            const_is_first = False
            if in0 in initializers:
                const_tensor = initializers[in0]
                const_is_first = True
            elif in1 in initializers:
                const_tensor = initializers[in1]

            if const_tensor is not None:
                params = {
                    "const_path": save_weight_bin(const_tensor, weights_dir),
                    "const_shape": [int(d) for d in const_tensor.dims],
                    "const_is_first": const_is_first,
                }
            else:
                params = {}

        elif kernel_type == 22:  # Softmax
            full_in_shape = value_info_shapes.get(main_input_name, in_shape)
            rank = len(full_in_shape) if full_in_shape else None
            axis = normalize_axis(get_node_attribute(node, "axis", -1), rank)
            params = {
                "input_shape": [int(d) if d is not None else -1 for d in full_in_shape],
                "axis": int(axis),
            }

        # Encontrar next_steps observando quem consome a saída deste nó.
        # Para um SiLU fundido, "node" já foi redirecionado para o mul_node
        # acima, então node.output continua correto (é a saída real do par).
        next_steps = []
        output_routes = []  # só populado quando este step tem >1 saída nomeada (ex: Split)
        for out_idx, out_name in enumerate(node.output):
            for consumer_step, consumer_entry in enumerate(fused_nodes, start=1):
                if consumer_step == step_id:
                    continue
                consumer_node = consumer_entry[-1]  # mul_node para "silu", node para "raw"
                if out_name in consumer_node.input:
                    if consumer_step not in next_steps:
                        next_steps.append(consumer_step)
                    output_routes.append({"to_step": consumer_step, "output_index": out_idx})

        step_entry = {
            "step": step_id,
            "name": layer_name,
            "kernel_type": kernel_type,
            "params": params,
            "next_steps": next_steps
        }
        if len(node.output) > 1:
            step_entry["output_routes"] = output_routes
        steps.append(step_entry)

    model_dfg = {
        "model": os.path.basename(onnx_path).replace(".onnx", ""),
        "steps": steps
    }

    with open(output_json, "w") as f:
        json.dump(model_dfg, f, indent=2)

    print(f"[SUCESSO] DFG exportado para '{output_json}' com {len(steps)} etapas.")
    print(f"[SUCESSO] Pesos binários gravados na pasta '{weights_dir}/'.")

    update_models_registry(model_dfg["model"], steps, output_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversor de Modelo ONNX para JSON DFG")
    parser.add_argument("--onnx", type=str, required=True, help="Caminho para o arquivo .onnx")
    parser.add_argument("--output", type=str, default="modelDFG.json", help="Arquivo JSON de saída")
    parser.add_argument("--weights-dir", type=str, default="bin_weights", help="Diretório de saída dos pesos binários")
    args = parser.parse_args()

    extract_dfg(args.onnx, args.output, args.weights_dir)
