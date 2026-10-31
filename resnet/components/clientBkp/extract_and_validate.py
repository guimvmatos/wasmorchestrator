import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

ONNX_MODEL = "resnet18-v1-7.onnx"   # Coloque o caminho do seu modelo .onnx
INPUT_PB   = "test_data_set_0/input_0.pb"
OUTPUT_PB  = "test_data_set_0/output_0.pb"

# 1. Carregar o input_0.pb oficial
input_tensor = onnx.TensorProto()
with open(INPUT_PB, "rb") as f:
    input_tensor.ParseFromString(f.read())
input_array = numpy_helper.to_array(input_tensor).astype(np.float32)

print(f"Formato do Tensor de Entrada: {input_array.shape}, Dtype: {input_array.dtype}")

# 2. Garantir que está contíguo em memória e salvar em binário
# Formato NCHW puro (1x3x224x224 = 150.528 floats = 602.112 bytes)
input_array_c = np.ascontiguousarray(input_array)
input_array_c.tofile("input_image.bin")
print(f"Salvo 'input_image.bin' ({input_array_c.nbytes} bytes, {input_array_c.size} floats)")

# 3. Carregar o output_0.pb oficial e salvar para comparação
output_tensor = onnx.TensorProto()
with open(OUTPUT_PB, "rb") as f:
    output_tensor.ParseFromString(f.read())
expected_output = numpy_helper.to_array(output_tensor).flatten().astype(np.float32)
expected_output.tofile("expected_output.bin")

# 4. Validar o modelo de referência com o ONNX Runtime
session = ort.InferenceSession(ONNX_MODEL)
input_name = session.get_inputs()[0].name
ort_outs = session.run(None, {input_name: input_array})
ort_logits = ort_outs[0].flatten()

top1_pb  = np.argmax(expected_output)
top1_ort = np.argmax(ort_logits)

print("\n--- Validação do Ground Truth ---")
print(f"Top-1 no output_0.pb:      Classe {top1_pb} (Logit: {expected_output[top1_pb]:.4f})")
print(f"Top-1 no ONNX Runtime:     Classe {top1_ort} (Logit: {ort_logits[top1_ort]:.4f})")
print(f"Diferença Máx (PB vs ORT): {np.max(np.abs(expected_output - ort_logits)):.6e}")