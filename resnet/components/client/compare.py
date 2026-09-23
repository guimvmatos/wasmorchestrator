import numpy as np

expected = np.fromfile("expected_output.bin", dtype=np.float32)
actual   = np.fromfile("output_received.bin", dtype=np.float32)

if expected.shape != actual.shape:
    print(f"Erro: Tamanhos incompatíveis! Esperado: {expected.shape}, Obtido: {actual.shape}")
    exit(1)

max_diff = np.max(np.abs(expected - actual))
mean_diff = np.mean(np.abs(expected - actual))
expected_top1 = np.argmax(expected)
actual_top1 = np.argmax(actual)

print(f"Top-1 Esperado (ONNX): {expected_top1}")
print(f"Top-1 Obtido (WASM):   {actual_top1}")
print(f"Diferença Máxima:      {max_diff:.6e}")
print(f"Diferença Média:       {mean_diff:.6e}")
print(f"Match Exato no Top-1:  {'SIM' if expected_top1 == actual_top1 else 'NÃO'}")