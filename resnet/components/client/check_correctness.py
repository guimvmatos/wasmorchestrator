#!/usr/bin/env python3
"""check_correctness.py — Fase 0: compara a saída do pipeline WASM com o ONNX Runtime.

Uso:
    python3 check_correctness.py --onnx resnet50-v1-7.onnx \
        --input input_image2.bin --output output_received.bin [--tol 1e-3] [--save-ref ref.npy]

- --input  : o MESMO .bin float32 (NCHW) que foi passado ao client.
- --output : o output_received.bin que o client grava (floats crus, antes do softmax/top-k).

O script:
  1. lê o shape de entrada do .onnx e CONFERE se o .bin tem o número certo de floats
     (pega o caso de mandar uma imagem 640x640 para um modelo 224x224);
  2. roda o ONNX Runtime (CPU) com essa entrada;
  3. compara com a saída do pipeline: erro absoluto máx/médio, erro relativo,
     top-1/top-5 (quando a saída parece classificação);
  4. imprime PASS/FAIL segundo a tolerância.
"""
import argparse
import sys

import numpy as np
import onnx
import onnxruntime as ort


def input_shape(model_path):
    m = onnx.load(model_path, load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    real = [i for i in m.graph.input if i.name not in inits]
    if len(real) != 1:
        sys.exit(f"Esperava 1 entrada, o modelo tem {len(real)}: {[i.name for i in real]}")
    dims = []
    for d in real[0].type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(d.dim_value)
        else:
            dims.append(1 if not dims else None)  # batch dinâmico -> 1
    if None in dims:
        sys.exit(f"Shape de entrada com dimensão dinâmica além do batch: {dims}")
    return real[0].name, dims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tol", type=float, default=1e-3, help="tolerância de erro absoluto máximo")
    ap.add_argument("--save-ref", help="salva a saída do ORT (.npy)")
    a = ap.parse_args()

    name, shape = input_shape(a.onnx)
    x = np.fromfile(a.input, dtype=np.float32)
    need = int(np.prod(shape))
    print(f"modelo : {a.onnx}")
    print(f"entrada: {name} shape={shape} ({need} floats)  | arquivo: {x.size} floats")
    if x.size != need:
        sys.exit(f"FAIL: o .bin tem {x.size} floats, o modelo espera {need}. "
                 f"Entrada errada para este modelo.")
    x = x.reshape(shape)

    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    ref = sess.run(None, {name: x})[0].astype(np.float32).ravel()
    got = np.fromfile(a.output, dtype=np.float32)
    print(f"saída  : ORT {ref.size} floats | pipeline {got.size} floats")
    if a.save_ref:
        np.save(a.save_ref, ref)
    if ref.size != got.size:
        sys.exit("FAIL: tamanhos de saída diferentes.")

    diff = np.abs(ref - got)
    rel = diff / (np.abs(ref) + 1e-6)
    print(f"erro abs máx  : {diff.max():.6e}  (índice {int(diff.argmax())})")
    print(f"erro abs médio: {diff.mean():.6e}")
    print(f"erro rel médio: {rel.mean():.6e}")
    print(f"NaN/Inf no pipeline: {int((~np.isfinite(got)).sum())}")

    if ref.size <= 1000:  # parece classificação
        t5r, t5g = np.argsort(-ref)[:5], np.argsort(-got)[:5]
        print(f"top-1 ORT {t5r[0]} | pipeline {t5g[0]} -> {'OK' if t5r[0] == t5g[0] else 'DIFERENTE'}")
        print(f"top-5 ORT {t5r.tolist()} | pipeline {t5g.tolist()} -> "
              f"{'OK' if set(t5r) == set(t5g) else 'DIFERENTE'}")

    ok = diff.max() < a.tol and np.isfinite(got).all()
    print(f"\n{'PASS' if ok else 'FAIL'} (tol {a.tol:g})")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
