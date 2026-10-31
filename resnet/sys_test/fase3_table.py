#!/usr/bin/env python3
"""fase3_table.py — monta a routing_table.json de um nó para os testes multi-nó (Fase 3).

Rode SEMPRE de dentro de resnet/sys_test/ (o sys e os grupos leem ./routing_table.json).

  python3 fase3_table.py init 1 1,2,3,4          # tabela nova: este nó (id 1) deve rodar os kernels 1,2,3,4
  python3 fase3_table.py init 1 1,2,3,4,5,6,7,8,9   # G1: tudo neste nó
  python3 fase3_table.py peer 200.18.102.27:8080 5,6,7,8,9   # kernels que rodam no OUTRO nó
  python3 fase3_table.py assign 1 1,2,3,4,5,6,7,8,9   # muda só o que este nó deve rodar (mantém a tabela)
  python3 fase3_table.py assign 2 none                # este nó (id 2) não roda mais nada (o sys drena e mata os grupos)
  python3 fase3_table.py show

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


def main():
    if not os.path.basename(os.getcwd()) == "sys_test":
        sys.exit("rode de dentro de resnet/sys_test/")
    a = sys.argv[1:]
    if not a or a[0] not in ("init", "peer", "assign", "show"):
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
    if a[0] in ("show", "peer", "assign"):
        d = load()
        print("assignments:", d.get("assignments"))
        for k in sorted(d.get("table", {}), key=int):
            print(f"  kernel {int(k):2d} -> {d['table'][k]}")


if __name__ == "__main__":
    main()
