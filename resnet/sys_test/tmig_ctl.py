#!/usr/bin/env python3
"""tmig_ctl.py — reage quando a GPU é tomada por uma aplicação de maior prioridade (T-MIG, 15/11).

Vigia o sinal "GPU tomada" (o arquivo --pidfile da aplicação prioritária existe e o processo
está vivo; é o que o escalonador/orquestrador da plataforma saberia) e, ao mudar o estado,
move a inferência pelo nosso mecanismo (prepara → grava, versão por requisição, sys:9999):

  --mode off      não move nada (referência: inferência e aplicação dividem a GPU)
  --mode full     GPU tomada → TODA a inferência vai para o grupo B (outro dispositivo);
                  GPU livre → volta para o grupo A (rota por tipo = grupo do sys)
  --mode partial  GPU tomada → passos 1..K vão para B, o resto fica em A (alivia a GPU);
                  GPU livre → volta tudo para A
  --mode cold     como full, mas o grupo B NÃO está no ar: o controlador sobe o runtime B
                  (outra GPU via VK_ICD_FILENAMES) na hora e só então move

Cada evento vira uma linha em --log (horários em ms Unix): sinal visto, preparado, gravado.
Rode de dentro de resnet/sys_test/ (usa cut_ctl.py e fase3_table.py).

  python3 tmig_ctl.py --mode full --model 6 --a 200.18.102.33:18080 --b 200.18.102.33:18090 \
      --sys 200.18.102.33:9999 --log tmig_S2.csv --label S2 --stop-after 1
"""
import argparse
import csv
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cut_ctl as cc  # noqa: E402
import fase3_table as ft  # noqa: E402


def prio_running(pidfile):
    try:
        pid = int(open(pidfile).read().strip())
    except (FileNotFoundError, ValueError):
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


IGNORE = ("runtime", "Xorg", "Xwayland", "gnome-shell")


def gpu_procs():
    """Processos na GPU NVIDIA segundo o driver: tabela 'Processes' do nvidia-smi (inclui Vulkan/gráfico,
    que o --query-compute-apps pode não listar)."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=2).stdout
    except Exception:  # noqa: BLE001
        return []
    procs, inside = [], False
    for line in out.splitlines():
        if "Processes:" in line:
            inside = True
            continue
        if not inside or not line.startswith("|"):
            continue
        tok = line.strip("| ").split()
        if len(tok) >= 6 and tok[3].isdigit():
            procs.append((int(tok[3]), tok[5]))
    return procs


def gpu_taken_nvml():
    """GPU tomada = há na GPU um processo que não é o nosso runtime (nem o servidor gráfico)."""
    return any(not any(name.endswith(i) or i in name for i in IGNORE) for _, name in gpu_procs())


def pidfile_ms(pidfile):
    try:
        return int(os.path.getmtime(pidfile) * 1000)
    except OSError:
        return ""


def port_open(addr):
    host, port = addr.rsplit(":", 1)
    try:
        with socket.create_connection((host, int(port)), timeout=0.2):
            return True
    except OSError:
        return False


def main():
    if os.path.basename(os.getcwd()) != "sys_test":
        sys.exit("rode de dentro de resnet/sys_test/")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["off", "full", "partial", "cold"], required=True)
    ap.add_argument("--model", type=int, default=6)
    ap.add_argument("--a", required=True, help="grupo A (GPU que será tomada), ip:porta")
    ap.add_argument("--b", required=True, help="grupo B (outro dispositivo), ip:porta")
    ap.add_argument("--sys", required=True)
    ap.add_argument("--k", type=int, default=19, help="partial: passos 1..K vão para B")
    ap.add_argument("--pidfile", default="/tmp/prio.pid")
    ap.add_argument("--trigger", choices=["pidfile", "gpu"], default="pidfile",
                    help="pidfile = sinal de admissão da aplicação; gpu = detecta pelo driver (nvidia-smi) outro processo na GPU")
    ap.add_argument("--poll-ms", type=float, default=20)
    ap.add_argument("--log", default="tmig.csv")
    ap.add_argument("--label", default="tmig")
    ap.add_argument("--stop-after", type=int, default=1, help="encerra depois de N ciclos tomada→liberada")
    ap.add_argument("--b-icd", default="/usr/share/vulkan/icd.d/intel_icd.json", help="cold: ICD do dispositivo B")
    ap.add_argument("--b-wasm", default="kernel_0_gpu.wasm", help="cold: grupo a subir em B")
    a = ap.parse_args()
    sysa = [x for x in a.sys.split(",") if x]
    _, steps = ft.load_dfg(a.model)
    last = max(st["step"] for st in steps)
    valid = {k for k, *_ in ft.valid_cuts(steps)}
    if a.mode == "partial" and a.k not in valid:
        sys.exit(f"K={a.k} não é corte válido do modelo {a.model}: {sorted(valid)}")
    new = not os.path.exists(a.log)
    f = open(a.log, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["t_signal_ms", "t_ready_ms", "t_prepared_ms", "t_committed_ms", "event", "mode", "ranges", "ok",
                    "label", "trigger", "t_pidfile_ms"])
    procB = None
    sense = (lambda: gpu_taken_nvml()) if a.trigger == "gpu" else (lambda: prio_running(a.pidfile))
    taken = sense()
    cycles = 0
    print(f"[tmig] gatilho {a.trigger}; modo {a.mode}; modelo {a.model} ({last} passos); A={a.a} B={a.b}; GPU tomada agora: {taken}",
          flush=True)

    def move(ranges, event, t_sig, t_ready):
        ver = cc.next_version()
        if ranges is None:
            t1 = time.time()
            rep = cc.to_all(sysa, {"op": "uncut", "model": str(a.model), "version": ver})
            ok = all(r.get("ok") for r in rep.values())
        else:
            prep = cc.to_all(sysa, {"op": "cut", "model": str(a.model), "ranges": ranges, "commit": False})
            t1 = time.time()
            ok = all(r.get("ok") for r in prep.values())
            if ok:
                rep = cc.to_all(sysa, {"op": "cut", "model": str(a.model), "ranges": ranges, "commit": True,
                                       "version": ver})
                ok = all(r.get("ok") for r in rep.values())
        t2 = time.time()
        desc = "volta para A" if ranges is None else " | ".join(f"{r['de']}..{r['ate']}->{r['no']}" for r in ranges)
        print(f"[tmig] {time.strftime('%H:%M:%S')} {event}: {desc} [versão {ver}; pronto {1000 * (t_ready - t_sig):.0f} ms, "
              f"prepara {1000 * (t1 - t_ready):.0f} ms, grava {1000 * (t2 - t1):.0f} ms; sinal→gravado "
              f"{1000 * (t2 - t_sig):.0f} ms] {'ok' if ok else 'FALHOU'}", flush=True)
        w.writerow([int(t_sig * 1000), int(t_ready * 1000), int(t1 * 1000), int(t2 * 1000), event, a.mode,
                    desc, int(ok), a.label, a.trigger, pidfile_ms(a.pidfile)])
        f.flush()

    try:
        while True:
            now = sense()
            if now != taken:
                t_sig = time.time()
                taken = now
                if a.mode == "off":
                    w.writerow([int(t_sig * 1000), "", "", "", "tomada" if now else "liberada", a.mode, "", 1, a.label,
                                a.trigger, pidfile_ms(a.pidfile)])
                    f.flush()
                    print(f"[tmig] {time.strftime('%H:%M:%S')} GPU {'tomada' if now else 'liberada'} (modo off: nada muda)",
                          flush=True)
                elif now:
                    t_ready = t_sig
                    if a.mode == "cold":
                        env = dict(os.environ, VK_ICD_FILENAMES=a.b_icd)
                        host, port = a.b.rsplit(":", 1)
                        procB = subprocess.Popen(["../../wasi-gfx-runtime/target/release/runtime", "--wasm", a.b_wasm,
                                                  "--port", port, "--ip", host], env=env,
                                                 stdout=open(f"tmig_runtimeB_{a.label}.log", "w"),
                                                 stderr=subprocess.STDOUT)
                        while not port_open(a.b):
                            time.sleep(0.01)
                        t_ready = time.time()
                    if a.mode == "partial":
                        ranges = [{"de": 1, "ate": a.k, "no": a.b}, {"de": a.k + 1, "ate": last, "no": a.a}]
                    else:
                        ranges = [{"de": 1, "ate": last, "no": a.b}]
                    move(ranges, "tomada", t_sig, t_ready)
                else:
                    move(None, "liberada", t_sig, t_sig)
                    if procB is not None:      # cold: desliga o B depois que ele esvaziar
                        time.sleep(5)
                        procB.terminate()
                        procB = None
                if not now:
                    cycles += 1
                    if cycles >= a.stop_after:
                        break
            time.sleep(a.poll_ms / 1000)
    finally:
        if procB is not None:
            procB.terminate()
        f.close()
    print("[tmig] fim", flush=True)


if __name__ == "__main__":
    main()
