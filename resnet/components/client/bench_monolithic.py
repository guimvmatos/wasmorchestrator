#!/usr/bin/env python3
"""bench_monolithic.py — Fase 1: baselines monolíticos (ONNX Runtime / PyTorch), mesmo protocolo
e mesmo formato de CSV do client do wasmorchestrator, para as linhas caírem no mesmo arquivo.

Exemplos (no diretório components/client/):
  # ONNX Runtime, CPU
  python3 bench_monolithic.py --framework ort --device cpu --model-id 1 \
      --onnx ../../test_onnx/resnet50-v1-7.onnx --input input_image2.bin \
      --warmup 10 --reps 200 --csv fase1.csv --label resnet50-notebook-ort-cpu

  # PyTorch (o MESMO .onnx convertido com onnx2torch: mesmos pesos, mesmo grafo), GPU
  python3 bench_monolithic.py --framework torch --device cuda --model-id 1 \
      --onnx ../../test_onnx/resnet50-v1-7.onnx --input input_image2.bin \
      --warmup 10 --reps 200 --csv fase1.csv --label resnet50-notebook-torch-gpu

Colunas do CSV (iguais às do client): serialize/send/deserialize = 0 (não há rede nem
serialização num processo monolítico) e wait_ms = e2e_ms = tempo da inferência.
Uma linha extra com rep = -1 registra o cold start: do início do processo (antes de importar
o framework) até o fim da 1ª inferência.

Dependências: numpy, onnx; onnxruntime (CPU) ou onnxruntime-gpu (CUDA); torch + onnx2torch
para --framework torch.
"""
import time

T_PROCESS_START = time.perf_counter()  # antes de qualquer import pesado (cold start)

import argparse
import os
import sys


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", choices=["ort", "torch"], required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--input", required=True, help="o mesmo .bin float32 NCHW usado no client")
    ap.add_argument("--model-id", type=int, required=True, help="só para a coluna model_id do CSV")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--threads", type=int, default=0, help="0 = padrão do framework")
    return ap.parse_args()


def input_shape(onnx_path):
    import onnx
    m = onnx.load(onnx_path, load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    real = [i for i in m.graph.input if i.name not in inits]
    dims = [d.dim_value if d.HasField("dim_value") else 1 for d in real[0].type.tensor_type.shape.dim]
    return real[0].name, dims


def build_ort(a, in_name):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if a.threads:
        so.intra_op_num_threads = a.threads
    if a.device == "cuda":
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            sys.exit("CUDAExecutionProvider indisponível: instale onnxruntime-gpu "
                     f"(providers atuais: {ort.get_available_providers()})")
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        prov = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(a.onnx, sess_options=so, providers=prov)
    print(f"ORT {ort.__version__} | providers ativos: {sess.get_providers()}")

    def run(x):
        return sess.run(None, {in_name: x})[0]
    return run


def load_for_onnx2torch(onnx_path, target_opset=13):
    """Os ResNet/VGG do ONNX Model Zoo (*-v1-7, vgg16-bn-7) são opset 7, e o onnx2torch não tem
    conversor para ops tão antigos (ex.: BatchNormalization v7). Sobe o opset com o
    version_converter oficial do ONNX — muda só a versão dos ops, não os pesos nem o grafo."""
    import onnx
    from onnx import version_converter
    m = onnx.load(onnx_path)
    cur = max((o.version for o in m.opset_import if o.domain in ("", "ai.onnx")), default=0)
    if cur < target_opset:
        # Modelos IR < 4 exigem que todo initializer também apareça em graph.input; o
        # version_converter falha ("Input W is undefined") quando falta algum.
        # BatchNormalization v7 tem o atributo `spatial`; o version_converter só aceita
        # spatial=1. O vgg16-bn-7 do Model Zoo grava spatial=0, mas com scale/bias/mean/var
        # de forma [C] (um valor por canal), que é exatamente a semântica de spatial=1 —
        # é assim que o ONNX Runtime executa esse modelo. Só corrige quando a forma é [C].
        inits_dims = {t.name: list(t.dims) for t in m.graph.initializer}
        for n in m.graph.node:
            if n.op_type == "BatchNormalization":
                for at in n.attribute:
                    if at.name == "spatial" and at.i != 1 and len(inits_dims.get(n.input[1], [0, 0])) == 1:
                        at.i = 1
        if m.ir_version < 4:
            have = {i.name for i in m.graph.input}
            for t in m.graph.initializer:
                if t.name not in have:
                    m.graph.input.append(onnx.helper.make_tensor_value_info(t.name, t.data_type, list(t.dims)))
        m = version_converter.convert_version(m, target_opset)
        print(f"opset {cur} -> {target_opset} (onnx.version_converter) para o onnx2torch")
    return m


def build_torch(a):
    import torch
    from onnx2torch import convert
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("torch.cuda.is_available() == False")
    dev = torch.device(a.device)
    model = convert(load_for_onnx2torch(a.onnx)).eval().to(dev)
    print(f"torch {torch.__version__} | device {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if a.device == "cuda" else "")
          + f" | threads {torch.get_num_threads()}")

    def run(x):
        with torch.inference_mode():
            t = torch.from_numpy(x).to(dev)
            y = model(t)
            if isinstance(y, (list, tuple)):
                y = y[0]
            out = y.cpu().numpy()  # .cpu() sincroniza a GPU: o tempo inclui a cópia de volta
        return out
    return run


def main():
    a = parse()
    import numpy as np

    in_name, shape = input_shape(a.onnx)
    x = np.fromfile(a.input, dtype=np.float32)
    need = int(np.prod(shape))
    if x.size != need:
        sys.exit(f"Entrada com {x.size} floats; o modelo espera {need} {shape}.")
    x = x.reshape(shape)
    input_bytes = x.nbytes

    run = build_ort(a, in_name) if a.framework == "ort" else build_torch(a)

    new = not os.path.exists(a.csv) or os.path.getsize(a.csv) == 0
    f = open(a.csv, "a")
    if new:
        f.write("timestamp_ms,label,model_id,request,rep,input_bytes,output_floats,top1,"
                "serialize_ms,send_ms,wait_ms,deserialize_ms,e2e_ms\n")
    label = a.label.replace(",", ";")

    def row(rep, y, ms):
        top1 = str(int(np.argmax(y))) if y.size <= 1000 else ""
        f.write(f"{int(time.time()*1000)},{label},{a.model_id},0,{rep},{input_bytes},{y.size},{top1},"
                f"0,0,{ms:.4f},0,{ms:.4f}\n")

    # 1ª inferência = cold start (processo -> 1º resultado)
    t = time.perf_counter()
    y = run(x)
    first_ms = (time.perf_counter() - t) * 1000
    cold_ms = (time.perf_counter() - T_PROCESS_START) * 1000
    row(-1, y, cold_ms)
    print(f"cold start (processo -> 1ª saída): {cold_ms:.1f} ms | 1ª inferência: {first_ms:.1f} ms")

    for _ in range(max(0, a.warmup - 1)):
        run(x)

    lat = []
    for r in range(a.reps):
        t = time.perf_counter()
        y = run(x)
        ms = (time.perf_counter() - t) * 1000
        lat.append(ms)
        row(r, y, ms)
    f.close()

    s = np.sort(np.array(lat))
    print(f"{a.label}: mediana {np.median(s):.3f} ms | p95 {np.percentile(s,95):.3f} | "
          f"p99 {np.percentile(s,99):.3f} | média {s.mean():.3f} | desvio {s.std():.3f} "
          f"({a.reps} medidas, {a.warmup} warmup) -> {a.csv}")


if __name__ == "__main__":
    main()
