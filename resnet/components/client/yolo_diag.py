#!/usr/bin/env python3
"""yolo_diag.py — diagnostica onde a saída do YOLO no pipeline diverge do ONNX Runtime.

Uso:
    python3 yolo_diag.py --onnx ../../test_onnx/yolo26n.onnx \
        --input input_image.bin --output output_received.bin

Mostra:
  1. shape real da saída do ORT;
  2. erro por canal (ex.: [1,84,8400] -> canais 0-3 = caixa, 4-83 = classes);
  3. erro por âncora, agrupado pelas 3 escalas do head (80x80, 40x40, 20x20);
  4. testes de hipótese de layout (saída transposta? escalas em outra ordem?);
  5. com --dump, salva ref/pipeline em .npy para inspeção.
"""
import argparse

import numpy as np
import onnx
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()

    m = onnx.load(a.onnx, load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    inp = [i for i in m.graph.input if i.name not in inits][0]
    shape = [d.dim_value or 1 for d in inp.type.tensor_type.shape.dim]
    x = np.fromfile(a.input, dtype=np.float32).reshape(shape)

    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    ref = sess.run(None, {inp.name: x})[0].astype(np.float32)
    got = np.fromfile(a.output, dtype=np.float32)
    print(f"saída ORT: shape {list(ref.shape)} ({ref.size} floats) | pipeline {got.size} floats")
    if a.dump:
        np.save("yolo_ref.npy", ref)
        np.save("yolo_got.npy", got)

    r = ref.reshape(ref.shape[-2], ref.shape[-1]) if ref.ndim == 3 else ref.reshape(ref.shape[0], -1)
    C, N = r.shape
    g = got.reshape(C, N)
    print(f"interpretando como [{C} canais x {N} âncoras]\n")

    # 1. Erro por canal
    d = np.abs(r - g)
    print("--- erro por canal (só os 12 piores) ---")
    print(f"{'canal':>6} {'erro máx':>11} {'erro médio':>11} {'|ref| médio':>11} {'corr':>7}")
    worst = np.argsort(-d.max(axis=1))[:12]
    for c in sorted(worst):
        corr = np.corrcoef(r[c], g[c])[0, 1] if r[c].std() > 0 and g[c].std() > 0 else float("nan")
        print(f"{c:6d} {d[c].max():11.4e} {d[c].mean():11.4e} {np.abs(r[c]).mean():11.4e} {corr:7.4f}")
    if C >= 5:
        print(f"\ncanais 0-3 (caixa): erro médio {d[:4].mean():.4e} | máx {d[:4].max():.4e}")
        print(f"canais 4-{C-1} (classes): erro médio {d[4:].mean():.4e} | máx {d[4:].max():.4e}")

    # 2. Erro por escala do head (para 640: 80x80=6400, 40x40=1600, 20x20=400)
    if N == 8400:
        print("\n--- erro por escala ---")
        for name, s, e in [("P3 80x80", 0, 6400), ("P4 40x40", 6400, 8000), ("P5 20x20", 8000, 8400)]:
            print(f"{name}: erro médio {d[:, s:e].mean():.4e} | máx {d[:, s:e].max():.4e}")

    # 3. Hipóteses de layout
    print("\n--- hipóteses de layout ---")
    gt = got.reshape(N, C).T
    print(f"pipeline transposto ([{N},{C}] lido como [{C},{N}]): erro máx {np.abs(r - gt).max():.4e}")
    frac = (d < 1e-2 * (np.abs(r) + 1)).mean()
    print(f"fração de valores que batem (erro < 1% + 0,01): {frac*100:.2f}%")
    bad = np.argwhere(d > 1e-2 * (np.abs(r) + 1))
    if len(bad):
        ch, an = bad[:, 0], bad[:, 1]
        print(f"valores divergentes: {len(bad)} | canais afetados: {len(np.unique(ch))} | "
              f"âncoras afetadas: {len(np.unique(an))}")
        print(f"primeiras âncoras divergentes: {np.unique(an)[:15].tolist()}")
        print(f"canais divergentes: {np.unique(ch)[:30].tolist()}")


if __name__ == "__main__":
    main()
