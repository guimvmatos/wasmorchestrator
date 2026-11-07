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
  # 07/11: os 5 modelos (YOLO tem entrada 640x640): passe as duas imagens; cada sessão usa a
  # imagem cujo tamanho bate com a entrada do modelo
  python3 ort_multi.py --onnx ... ../../test_onnx/yolo26n.onnx --input input_image2.bin input_image.bin
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
            out[k + ":"] = int(v.split()[0]) / 1024  # 07/11: a chave ficava sem os ':'

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", nargs="+", required=True)
    ap.add_argument("--copies", type=int, default=1, help="sessões por modelo")
    ap.add_argument("--input", nargs="+", required=True, help="uma ou mais imagens .bin (float32)")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()

    m0 = mem_mb()
    # 07/11: a forma vem do próprio modelo (o YOLO é 1x3x640x640), não mais fixa em 224x224
    imgs = [np.fromfile(p, dtype=np.float32) for p in a.input]
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    sessions = []
    for path in a.onnx:
        for c in range(a.copies):
            s = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
            inp = s.get_inputs()[0]
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            n = int(np.prod(shape))
            match = [im for im in imgs if im.size == n]
            if not match:
                raise SystemExit(f"{path}: nenhuma imagem de --input tem {n} floats (forma {inp.shape})")
            sessions.append((f"{path.split('/')[-1]}#{c}", s, inp.name, match[0].reshape(shape)))
    m1 = mem_mb()
    lat = {n: [] for n, _, _, _ in sessions}
    for _ in range(a.rounds):
        for n, s, inp, x in sessions:
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
