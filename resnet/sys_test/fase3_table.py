#!/usr/bin/env python3
"""fase3_table.py — monta a routing_table.json de um nó para os testes multi-nó (Fase 3).

Rode SEMPRE de dentro de resnet/sys_test/ (o sys e os grupos leem ./routing_table.json).

  python3 fase3_table.py init 1 1,2,3,4          # tabela nova: este nó (id 1) deve rodar os kernels 1,2,3,4
  python3 fase3_table.py init 1 1,2,3,4,5,6,7,8,9   # G1: tudo neste nó
  python3 fase3_table.py peer 200.18.102.27:8080 5,6,7,8,9   # kernels que rodam no OUTRO nó
  python3 fase3_table.py assign 1 1,2,3,4,5,6,7,8,9   # muda só o que este nó deve rodar (mantém a tabela)
  python3 fase3_table.py assign 2 none                # este nó (id 2) não roda mais nada (o sys drena e mata os grupos)
  python3 fase3_table.py show

Corte por camada (09/11) — tem prioridade sobre a rota por tipo, só para o modelo indicado:
  python3 fase3_table.py cuts 6                       # lista os pontos de corte válidos do modelo 6
  python3 fase3_table.py cut 6 resnetv15_stage2_activation1 200.18.102.33:18080 200.18.102.27:18080
        # passos 1..(camada) no 1º nó, o resto no 2º. O ponto pode ser o nome da camada ou o nº
        # do passo; vários pontos separados por vírgula = vários cortes (um nó a mais por ponto).
  python3 fase3_table.py uncut 6                      # volta o modelo 6 para a rota por tipo
  Os grupos releem a tabela a cada conexão: o corte vale a partir da próxima requisição/salto,
  sem montar nem derrubar grupo. Os dois nós precisam ter os kernels dos passos da sua faixa.

Por que existe: o sys de cada máquina só escreve na tabela os grupos que ELE subiu. Sem a
entrada "peer", o grupo deste nó não sabe para onde mandar o tensor quando o próximo kernel
roda em outra máquina. Use "peer" só DEPOIS que o sys local já subiu os grupos dele (senão a
primeira gravação do sys pode acontecer depois e, como ele parte do arquivo atual, não apaga a
entrada — mas confira com "show").

Toda gravação é atômica (arquivo .tmp + rename), igual ao sys, para nenhum grupo ler a
tabela pela metade.
"""
import json
import os
import sys

PATH = "routing_table.json"


def load():
    with open(PATH) as f:
        return json.load(f)


def save(d):
    tmp = PATH + ".tmp.py"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, PATH)


def kernels(csv, allow_empty=False):
    if allow_empty and csv == "none":
        return []
    ks = [int(x) for x in csv.split(",") if x.strip()]
    if not ks or any(k < 1 or k > 255 for k in ks):
        sys.exit(f"lista de kernels inválida: {csv!r}")
    return ks


def load_dfg(model_id):
    reg_path = os.path.join("..", "models.json")
    with open(reg_path) as f:
        reg = json.load(f)
    if str(model_id) not in reg:
        sys.exit(f"model_id {model_id} não está em {reg_path}")
    entry = reg[str(model_id)]
    with open(os.path.join("..", entry["dfg_path"])) as f:
        return entry["name"], json.load(f)["steps"]


def step_floats(steps_by_id, step_id, depth=0):
    """Tamanho (em floats) do tensor que ENTRA no passo, pelos params dele;
    para ops que não mudam o tamanho (Relu, Add...), olha o próximo passo."""
    p = steps_by_id[step_id]["params"]
    if p.get("cin") and p.get("h_in"):
        return p["cin"] * p["h_in"] * p["w_in"]
    if p.get("channels") and p.get("h_in"):
        return p["channels"] * p["h_in"] * p["w_in"]
    if p.get("in_features"):
        return p["in_features"]
    if p.get("input_shape"):
        n = 1
        for d in p["input_shape"]:
            n *= max(1, d)
        return n
    if depth < 8:
        for nxt in steps_by_id[step_id]["next_steps"]:
            v = step_floats(steps_by_id, nxt, depth + 1)
            if v:
                return v
    return None


def step_macs(st):
    p = st["params"]; kt = st["kernel_type"]
    if kt in (1, 23, 24):
        oh = (p["h_in"] + 2 * p["padding"] - p["kh"]) // p["stride"] + 1
        ow = (p["w_in"] + 2 * p["padding"] - p["kw"]) // p["stride"] + 1
        return p["cout"] * (p["cin"] // max(1, p["groups"])) * p["kh"] * p["kw"] * oh * ow
    if kt == 8:
        return p.get("in_features", 0) * p.get("out_features", 0)
    return 0


def valid_cuts(steps):
    """Pontos k em que só UM tensor atravessa de {1..k} para {k+1..N}."""
    by_id = {st["step"]: st for st in steps}
    ids = sorted(by_id)
    total = sum(step_macs(st) for st in steps) or 1
    cuts, acc = [], 0
    for k in ids[:-1]:
        acc += step_macs(by_id[k])
        srcs = {st["step"] for st in steps if st["step"] <= k and any(n > k for n in st["next_steps"])}
        if len(srcs) == 1:
            src = srcs.pop()
            # Split (várias saídas): cada saída é um tensor diferente.
            outs = {r["output_index"] for r in by_id[src].get("output_routes", []) if r["to_step"] > k}
            if len(outs) > 1:
                continue
            dst = min(n for n in by_id[src]["next_steps"] if n > k)
            fl = step_floats(by_id, dst)
            cuts.append((k, by_id[k]["name"], fl, acc / total))
    return cuts


def main():
    if not os.path.basename(os.getcwd()) == "sys_test":
        sys.exit("rode de dentro de resnet/sys_test/")
    a = sys.argv[1:]
    if not a or a[0] not in ("init", "peer", "assign", "show", "cuts", "cut", "uncut"):
        sys.exit(__doc__)
    if a[0] == "init":
        if len(a) != 3:
            sys.exit("uso: init <node_id> <kernels>")
        if os.path.exists(PATH):
            os.replace(PATH, PATH + ".anterior")
        save({"table": {}, "node_scores": {}, "assignments": {str(int(a[1])): kernels(a[2])}})
        print(f"tabela nova: nó {a[1]} roda {a[2]} (anterior salva em {PATH}.anterior)")
    elif a[0] == "peer":
        if len(a) != 3 or ":" not in a[1]:
            sys.exit("uso: peer <ip:porta> <kernels>")
        d = load()
        for k in kernels(a[2]):
            d["table"][str(k)] = a[1]
        save(d)
        print(f"kernels {a[2]} -> {a[1]}")
    elif a[0] == "assign":
        if len(a) != 3:
            sys.exit("uso: assign <node_id> <kernels|none>")
        d = load()
        d["assignments"] = {str(int(a[1])): kernels(a[2], allow_empty=True)}
        save(d)
        print(f"nó {a[1]} agora deve rodar: {a[2]}")
    elif a[0] == "cuts":
        if len(a) != 2:
            sys.exit("uso: cuts <model_id>")
        name, steps = load_dfg(int(a[1]))
        print(f"modelo {a[1]} ({name}), {len(steps)} passos. Pontos de corte válidos (1 tensor atravessa):")
        print(f"  {'passo':>5}  {'MB':>7}  {'% MACs antes':>12}  camada")
        for k, lname, fl, frac in valid_cuts(steps):
            mb = f"{fl * 4 / 1e6:7.3f}" if fl else "      ?"
            print(f"  {k:5d}  {mb}  {100 * frac:11.1f}%  {lname}")
    elif a[0] == "cut":
        if len(a) < 5:
            sys.exit("uso: cut <model_id> <ponto[,ponto...]> <nó_1> <nó_2> [<nó_3> ...]")
        mid = int(a[1])
        name, steps = load_dfg(mid)
        by_name = {st["name"]: st["step"] for st in steps}
        valid = {k for k, *_ in valid_cuts(steps)}
        points = []
        for tok in a[2].split(","):
            k = int(tok) if tok.isdigit() else by_name.get(tok)
            if k is None:
                sys.exit(f"camada {tok!r} não existe no modelo {mid}")
            if k not in valid:
                sys.exit(f"passo {k} ({tok}) não é ponto de corte válido (mais de 1 tensor atravessa). "
                         f"Veja: python3 fase3_table.py cuts {mid}")
            points.append(k)
        points = sorted(set(points))
        nodes = a[3:]
        if len(nodes) != len(points) + 1:
            sys.exit(f"{len(points)} ponto(s) de corte pedem {len(points) + 1} nós; vieram {len(nodes)}")
        for n in nodes:
            if ":" not in n:
                sys.exit(f"nó {n!r} tem de ser ip:porta (a porta pública do grupo, ex: 18080)")
        last = max(st["step"] for st in steps)
        bounds = [0] + points + [last]
        ranges = [{"de": bounds[i] + 1, "ate": bounds[i + 1], "no": nodes[i]} for i in range(len(nodes))]
        d = load()
        d.setdefault("step_routes", {})[str(mid)] = ranges
        # 12/11: corte manual vale para todas as requisições novas: apaga a
        # versão/histórico do cut_ctl.py para este modelo (senão o handler
        # carimbaria uma versão antiga).
        d.get("route_version", {}).pop(str(mid), None)
        d.get("step_routes_hist", {}).pop(str(mid), None)
        save(d)
        print(f"modelo {mid} ({name}) cortado:")
        for r in ranges:
            print(f"  passos {r['de']:3d}..{r['ate']:3d} -> {r['no']}")
    elif a[0] == "uncut":
        if len(a) != 2:
            sys.exit("uso: uncut <model_id>")
        d = load()
        removed = d.get("step_routes", {}).pop(str(int(a[1])), None)
        d.get("route_version", {}).pop(str(int(a[1])), None)
        d.get("step_routes_hist", {}).pop(str(int(a[1])), None)
        if not d.get("step_routes"):
            d.pop("step_routes", None)
        save(d)
        print(f"modelo {a[1]}: " + ("volta para a rota por tipo" if removed else "não tinha corte"))
    if a[0] in ("show", "peer", "assign", "cut", "uncut"):
        d = load()
        print("assignments:", d.get("assignments"))
        for k in sorted(d.get("table", {}), key=int):
            print(f"  kernel {int(k):2d} -> {d['table'][k]}")
        for mid, rs in sorted(d.get("step_routes", {}).items(), key=lambda x: int(x[0])):
            print(f"  modelo {mid}: " + " | ".join(f"{r['de']}..{r['ate']} -> {r['no']}" for r in rs))


if __name__ == "__main__":
    main()
