#!/usr/bin/env python3
"""cut_ctl.py — controlador do corte por camada, ao vivo, em vários nós (12/11).

Fala com o sys de cada nó pela porta 9999 (plano de controle novo do sys).
Rode de dentro de resnet/sys_test/ (usa o fase3_table.py para achar os pontos de corte válidos).

  python3 cut_ctl.py show  --sys 200.18.102.33:9999,200.18.102.27:9999
  python3 cut_ctl.py apply 6 19 200.18.102.33:18080 200.18.102.27:18080 --sys 200.18.102.33:9999,200.18.102.27:9999
        # passos 1..19 no 1º grupo, o resto no 2º. Vários pontos: "10,28" e um grupo a mais por ponto.
  python3 cut_ctl.py apply 6 none --sys ...       # tira o corte (volta para a rota por tipo)
  python3 cut_ctl.py run roteiro.txt --sys ...    # aplica uma sequência de cortes (ver abaixo)

Cada troca tem duas fases, em todos os nós ao mesmo tempo (uma thread por nó):
  1. prepara: cada sys confere se tem os kernels da faixa que vai rodar nele; se faltar,
     sobe um grupo com o que falta. Nenhuma rota muda nesta fase.
  2. grava: cada sys grava a rota nova na routing_table.json dele.
Toda troca vira uma linha em cut_ctl_log.csv (horários em ms Unix, do relógio desta máquina).

Roteiro (um passo por linha; # comenta):
  <espera_s> <modelo> <pontos|none> [<grupo_1> <grupo_2> ...] [# rótulo]
Exemplo:
  10 6 none                                                   # tudo no nó de entrada
  10 6 19 200.18.102.33:18080 192.168.1.182:18080             # corte em 2
  10 6 10,28 200.18.102.33:18080 192.168.1.182:18080 200.18.102.27:18080
"""
import json
import os
import socket
import struct
import sys
import threading
import time

LOG = "cut_ctl_log.csv"
VER_FILE = "cut_ctl_version.txt"


def next_version():
    """Versão do corte, igual em todos os nós (vai nos 16 bits altos do model_id)."""
    try:
        v = int(open(VER_FILE).read().strip())
    except (OSError, ValueError):
        v = int(time.time()) % 60000
    v = v + 1 if v < 65535 else 1
    with open(VER_FILE, "w") as f:
        f.write(str(v))
    return v


def send(addr, msg, timeout=300.0):
    host, port = addr.rsplit(":", 1)
    data = json.dumps(msg).encode()
    with socket.create_connection((host, int(port)), timeout=timeout) as s:
        s.sendall(struct.pack(">I", len(data)) + data)
        hdr = b""
        while len(hdr) < 4:
            c = s.recv(4 - len(hdr))
            if not c:
                raise ConnectionError(f"{addr}: conexão fechada sem resposta (sys antigo, sem plano de controle?)")
            hdr += c
        n = struct.unpack(">I", hdr)[0]
        body = b""
        while len(body) < n:
            c = s.recv(n - len(body))
            if not c:
                raise ConnectionError(f"{addr}: resposta incompleta")
            body += c
    return json.loads(body)


def to_all(addrs, msg):
    """Manda a mesma mensagem para todos os sys em paralelo; devolve {addr: resposta}."""
    out = {}

    def one(a):
        try:
            out[a] = send(a, msg)
        except Exception as e:  # noqa: BLE001
            out[a] = {"ok": False, "err": str(e)}

    ths = [threading.Thread(target=one, args=(a,)) for a in addrs]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return out


def ranges_for(model, points_tok, groups):
    import fase3_table as ft  # mesmo diretório (sys_test)
    _, steps = ft.load_dfg(int(model))
    by_name = {st["name"]: st["step"] for st in steps}
    valid = {k for k, *_ in ft.valid_cuts(steps)}
    points = []
    for tok in points_tok.split(","):
        k = int(tok) if tok.isdigit() else by_name.get(tok)
        if k is None:
            sys.exit(f"camada {tok!r} não existe no modelo {model}")
        if k not in valid:
            sys.exit(f"passo {k} não é ponto de corte válido (veja: python3 fase3_table.py cuts {model})")
        points.append(k)
    points = sorted(set(points))
    if len(groups) != len(points) + 1:
        sys.exit(f"{len(points)} ponto(s) de corte pedem {len(points) + 1} grupos; vieram {len(groups)}")
    last = max(st["step"] for st in steps)
    b = [0] + points + [last]
    return [{"de": b[i] + 1, "ate": b[i + 1], "no": groups[i]} for i in range(len(groups))]


def apply(model, points_tok, groups, sys_addrs, label=""):
    t0 = time.time()
    ver = next_version()
    if points_tok == "none":
        rep = to_all(sys_addrs, {"op": "uncut", "model": str(model), "version": ver})
        t1 = t2 = time.time()
        ok = all(r.get("ok") for r in rep.values())
        built = []
        desc = "sem corte"
    else:
        ranges = ranges_for(model, points_tok, groups)
        prep = to_all(sys_addrs, {"op": "cut", "model": str(model), "ranges": ranges, "commit": False})
        t1 = time.time()
        bad = {a: r for a, r in prep.items() if not r.get("ok")}
        if bad:
            print(f"[ERRO] preparação falhou, nenhuma rota foi trocada: {bad}")
            log_line(t0, t1, t1, model, points_tok, groups, False, [], label)
            return False
        built = sorted({k for r in prep.values() for k in r.get("built", [])})
        rep = to_all(sys_addrs, {"op": "cut", "model": str(model), "ranges": ranges, "commit": True, "version": ver})
        t2 = time.time()
        ok = all(r.get("ok") for r in rep.values())
        desc = " | ".join(f"{r['de']}..{r['ate']} -> {r['no']}" for r in ranges)
    status = "ok" if ok else f"FALHOU {rep}"
    print(f"{time.strftime('%H:%M:%S')}.{int((t2 % 1) * 1000):03d} modelo {model}: {desc} "
          f"[versão {ver}; prepara {1000 * (t1 - t0):.0f} ms, grava {1000 * (t2 - t1):.0f} ms, kernels montados: {built or 'nenhum'}] {status}")
    log_line(t0, t1, t2, model, points_tok, groups, ok, built, label)
    return ok


def log_line(t0, t1, t2, model, points_tok, groups, ok, built, label):
    import csv
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["t_start_ms", "t_prepared_ms", "t_committed_ms", "model", "points", "groups", "ok", "built", "label"])
        w.writerow([int(t0 * 1000), int(t1 * 1000), int(t2 * 1000), model, points_tok,
                    "|".join(groups), int(ok), "|".join(map(str, built)), label])


def parse_sys(args):
    if "--sys" not in args:
        sys.exit("faltou --sys ip:9999[,ip:9999...] (todos os nós com grupo ou handler)")
    i = args.index("--sys")
    addrs = [a for a in args[i + 1].split(",") if a]
    return args[:i] + args[i + 2:], addrs


def main():
    if os.path.basename(os.getcwd()) != "sys_test":
        sys.exit("rode de dentro de resnet/sys_test/")
    args, addrs = parse_sys(sys.argv[1:])
    if not args:
        sys.exit(__doc__)
    if args[0] == "show":
        for a, r in to_all(addrs, {"op": "show"}).items():
            print(a, json.dumps(r, ensure_ascii=False))
    elif args[0] == "apply":
        if len(args) < 3:
            sys.exit("uso: apply <modelo> <pontos|none> [grupos...] --sys ...")
        ok = apply(args[1], args[2], args[3:], addrs, label="apply")
        sys.exit(0 if ok else 1)
    elif args[0] == "run":
        steps = []
        for raw in open(args[1]):
            line, _, label = raw.partition("#")
            tok = line.split()
            if not tok:
                continue
            steps.append((float(tok[0]), tok[1], tok[2], tok[3:], label.strip()))
        print(f"roteiro com {len(steps)} trocas; log em {LOG}")
        for wait, model, pts, groups, label in steps:
            time.sleep(wait)
            apply(model, pts, groups, addrs, label=label)
        print("fim do roteiro")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
