#!/usr/bin/env python3
"""prio_app.py — "aplicação de maior prioridade" que toma a GPU (T-MIG, 15/11).

Um processo independente do nosso sistema: multiplica matrizes num laço, por WebGPU
(wgpu-py, Vulkan), na GPU escolhida pelo VK_ICD_FILENAMES. Enquanto roda, deixa o PID em
--pidfile (é o "sinal" de que a GPU foi tomada, como o escalonador/orquestrador saberia).
Registra a própria vazão (multiplicações por segundo) a cada segundo em --csv: é a métrica
da aplicação prioritária (ela recupera a GPU quando a inferência sai?).

  python3 prio_app.py --seconds 60 --n 1024 --csv prio.csv --label S2 --pidfile /tmp/prio.pid
"""
import argparse
import csv
import os
import time

T_PROC_START_MS = int(time.time() * 1000)   # antes de abrir a GPU: base do atraso de detecção

import numpy as np
import wgpu

WGSL = """
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> N: u32;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let n = N;
  if (g.x >= n || g.y >= n) { return; }
  var acc = 0.0;
  for (var k = 0u; k < n; k = k + 1u) { acc = acc + A[g.y * n + k] * B[k * n + g.x]; }
  C[g.y * n + g.x] = acc;
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4, help="multiplicações por submissão")
    ap.add_argument("--csv", default="prio.csv")
    ap.add_argument("--label", default="prio")
    ap.add_argument("--pidfile", default="/tmp/prio.pid")
    ap.add_argument("--window", type=float, default=0.25, help="janela de medição da vazão (s)")
    a = ap.parse_args()
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    print(f"[prio] GPU: {adapter.info.get('device')} ({adapter.info.get('vendor')}, {adapter.info.get('adapter_type')})",
          flush=True)
    dev = adapter.request_device_sync()
    t_device_ms = int(time.time() * 1000)   # contexto da GPU criado: a partir daqui o driver vê o processo
    n = a.n
    rng = np.random.default_rng(0)
    A = rng.random((n, n), dtype=np.float32)
    B = rng.random((n, n), dtype=np.float32)
    ba = dev.create_buffer_with_data(data=A, usage=wgpu.BufferUsage.STORAGE)
    bb = dev.create_buffer_with_data(data=B, usage=wgpu.BufferUsage.STORAGE)
    bc = dev.create_buffer(size=A.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    bn = dev.create_buffer_with_data(data=np.array([n], dtype=np.uint32), usage=wgpu.BufferUsage.UNIFORM)
    pipe = dev.create_compute_pipeline(layout="auto",
                                       compute={"module": dev.create_shader_module(code=WGSL), "entry_point": "main"})
    bg = dev.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
        {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate((ba, bb, bc, bn))])
    wg = (n + 15) // 16

    def step():
        enc = dev.create_command_encoder()
        for _ in range(a.batch):
            p = enc.begin_compute_pass()
            p.set_pipeline(pipe)
            p.set_bind_group(0, bg)
            p.dispatch_workgroups(wg, wg)
            p.end()
        dev.queue.submit([enc.finish()])
        dev.queue.read_buffer(bc, 0, 4)     # espera a GPU terminar

    step()  # aquece
    new = not os.path.exists(a.csv)
    f = open(a.csv, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["timestamp_ms", "label", "matmul_per_s", "gflops", "t_proc_start_ms", "t_gpu_ready_ms", "t_device_ms"])
    t_gpu_ready_ms = int(time.time() * 1000)
    with open(a.pidfile, "w") as pf:
        pf.write(str(os.getpid()))
    print(f"[prio] GPU tomada (pid {os.getpid()} em {a.pidfile}) por {a.seconds:.0f} s; n={n}", flush=True)
    t_end = time.time() + a.seconds
    try:
        while time.time() < t_end:
            t0 = time.time()
            done = 0
            while time.time() - t0 < a.window:
                step()
                done += a.batch
            dt = time.time() - t0
            rate = done / dt
            w.writerow([int(time.time() * 1000), a.label, round(rate, 2), round(rate * 2 * n ** 3 / 1e9, 1),
                        T_PROC_START_MS, t_gpu_ready_ms, t_device_ms])
            f.flush()
            if a.window >= 1 or int(time.time() * 4) % 4 == 0:
                print(f"[prio] {time.strftime('%H:%M:%S')} {rate:.1f} matmul/s ({rate * 2 * n ** 3 / 1e9:.1f} GFLOP/s)", flush=True)
    finally:
        try:
            os.remove(a.pidfile)
        except FileNotFoundError:
            pass
        f.close()
        print("[prio] GPU liberada", flush=True)


if __name__ == "__main__":
    main()
