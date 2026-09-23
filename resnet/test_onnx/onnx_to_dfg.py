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
    "Gemm": 8,
    "MatMul": 8,
    "Dropout": 9,
}

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

    # Mapear saídas para os passos
    for step_id, node in enumerate(valid_nodes, start=1):
        for out_name in node.output:
            tensor_producer[out_name] = step_id

    # 5. Segundo passe: construir os StepNodes e extrair os pesos
    steps = []
    for step_id, node in enumerate(valid_nodes, start=1):
        kernel_type = OP_TO_KERNEL[node.op_type]
        layer_name = node.name if node.name else f"{node.op_type}_{step_id}"
        params = {}

        # Determinar dimensões de entrada a partir do tensor de entrada principal
        main_input_name = node.input[0]
        in_shape = value_info_shapes.get(main_input_name, [1, 3, 224, 224])
        
        c_in = in_shape[1] if len(in_shape) > 1 and in_shape[1] is not None else 3
        h_in = in_shape[2] if len(in_shape) > 2 and in_shape[2] is not None else 224
        w_in = in_shape[3] if len(in_shape) > 3 and in_shape[3] is not None else 224

        # Preencher atributos específicos de cada operador
        if kernel_type == 1:  # Conv
            strides = get_node_attribute(node, "strides", [1, 1])
            pads = get_node_attribute(node, "pads", [0, 0, 0, 0])
            kernel_shape = get_node_attribute(node, "kernel_shape", [3, 3])

            w_tensor = initializers.get(node.input[1]) if len(node.input) > 1 else None
            w_path = save_weight_bin(w_tensor, weights_dir) if w_tensor else ""

            cout = w_tensor.dims[0] if w_tensor else c_in
            cin = w_tensor.dims[1] if w_tensor else c_in
            kh = kernel_shape[0] if len(kernel_shape) > 0 else 3
            kw = kernel_shape[1] if len(kernel_shape) > 1 else 3

            params = {
                "weight_path": w_path,
                "cout": int(cout),
                "cin": int(cin),
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

        elif kernel_type == 9:  # Dropout (Identity em inferência)
            params = {}

        # Encontrar next_steps observando quem consome a saída deste nó
        next_steps = []
        for out_name in node.output:
            for consumer_step, other_node in enumerate(valid_nodes, start=1):
                if consumer_step != step_id and out_name in other_node.input:
                    if consumer_step not in next_steps:
                        next_steps.append(consumer_step)

        steps.append({
            "step": step_id,
            "name": layer_name,
            "kernel_type": kernel_type,
            "params": params,
            "next_steps": next_steps
        })

    model_dfg = {
        "model": os.path.basename(onnx_path).replace(".onnx", ""),
        "steps": steps
    }

    with open(output_json, "w") as f:
        json.dump(model_dfg, f, indent=2)

    print(f"[SUCESSO] DFG exportado para '{output_json}' com {len(steps)} etapas.")
    print(f"[SUCESSO] Pesos binários gravados na pasta '{weights_dir}/'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversor de Modelo ONNX para JSON DFG")
    parser.add_argument("--onnx", type=str, required=True, help="Caminho para o arquivo .onnx")
    parser.add_argument("--output", type=str, default="modelDFG.json", help="Arquivo JSON de saída")
    parser.add_argument("--weights-dir", type=str, default="bin_weights", help="Diretório de saída dos pesos binários")
    args = parser.parse_args()

    extract_dfg(args.onnx, args.output, args.weights_dir)