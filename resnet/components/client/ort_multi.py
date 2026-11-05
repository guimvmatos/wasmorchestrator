#!/usr/bin/env python3
"""ort_multi.py — E10: vários modelos (ou várias cópias do mesmo) num ÚNICO processo ONNX Runtime.

Mede a memória do processo (VmRSS/VmHWM do /proc) depois de carregar as sessões e depois de
rodar, e a latência mediana por modelo, em rodízio (uma inferência de cada sessão por rodada).

Exemplos (em components/client/):
  # 4 modelos diferentes num processo
  python3 ort_multi.py --onnx ../../test_onnx/resnet18-v1-7.onnx ../../test_onnx/resnet34-v1-7.onnx \
      ../../test_onnx/resnet50-v1-7.onnx ../../test_onnx/vgg16-bn-7.onnx --input input_image2.bin --rounds 20
  # 4 sessões do mesmo modelo num processo
  python3 ort_multi.py --onnx ../../test_onnx/resnet18-v1-7.onnx --copies 4 --input input_image2.bin --rounds 50
"""
import argparse
import time

import numpy as np
import onnxruntime as ort


def mem_mb():
    out = {}
    for line in open("/proc/self/status"):
        if line.startswith(("VmRSS:", "VmHWM:")):
            k, v = line.split(":")
            out[k] = int(v.split()[0]) / 1024
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", nargs="+", required=True)
    ap.add_argument("--copies", type=int, default=1, help="sessões por modelo")
    ap.add_argument("--input", required=True)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()

    m0 = mem_mb()
    x = np.fromfile(a.input, dtype=np.float32).reshape(1, 3, 224, 224)
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    sessions = []
    for path in a.onnx:
        for c in range(a.copies):
            s = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
            sessions.append((f"{path.split('/')[-1]}#{c}", s, s.get_inputs()[0].name))
    m1 = mem_mb()
    lat = {n: [] for n, _, _ in sessions}
    for _ in range(a.rounds):
        for n, s, inp in sessions:
            t = time.perf_counter()
            s.run(None, {inp: x})
            lat[n].append((time.perf_counter() - t) * 1000)
    m2 = mem_mb()
    print(f"sessões: {len(sessions)} | threads por sessão: {a.threads}")
    print(f"memória: início {m0['VmRSS:']:.0f} MB | depois de carregar {m1['VmRSS:']:.0f} MB | "
          f"depois de rodar {m2['VmRSS:']:.0f} MB | pico {m2['VmHWM:']:.0f} MB")
    for n, v in lat.items():
        print(f"  {n:28s} mediana {np.median(v):8.2f} ms")


if __name__ == "__main__":
    main()
