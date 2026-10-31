#!/usr/bin/env bash
# fase3_netem.sh — atraso/banda SÓ no tráfego para UM IP (o outro nó). SSH, VPN e o client
# não são afetados, porque o filtro casa apenas o IP de destino do par.
#
#   sudo bash fase3_netem.sh set 200.18.102.27 5          # +5 ms de ida para .27
#   sudo bash fase3_netem.sh set 200.18.102.27 5 100mbit  # +5 ms e banda de 100 Mbit/s
#   sudo bash fase3_netem.sh show
#   sudo bash fase3_netem.sh clear                        # volta ao normal
#
# Para somar D ms ao RTT entre dois nós, rode "set <ip do outro> D/2" NOS DOIS nós.
set -euo pipefail
cmd="${1:-show}"
iface_for() { ip -o route get "$1" | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}'; }

case "$cmd" in
  set)
    peer="$2"; delay="$3"; rate="${4:-}"
    IF=$(iface_for "$peer")
    tc qdisc del dev "$IF" root 2>/dev/null || true
    tc qdisc add dev "$IF" root handle 1: prio bands 4
    if [ -n "$rate" ]; then
      tc qdisc add dev "$IF" parent 1:4 handle 40: netem delay "${delay}ms" rate "$rate" limit 100000
    else
      tc qdisc add dev "$IF" parent 1:4 handle 40: netem delay "${delay}ms" limit 100000
    fi
    tc filter add dev "$IF" parent 1: protocol ip prio 1 u32 match ip dst "$peer"/32 flowid 1:4
    echo "[netem] $IF: +${delay} ms ${rate:+e $rate }só para $peer"
    ;;
  clear)
    for IF in $(ls /sys/class/net); do
      tc qdisc show dev "$IF" | grep -q 'prio 1:' && tc qdisc del dev "$IF" root && echo "[netem] $IF limpo"
    done
    true
    ;;
  show)
    for IF in $(ls /sys/class/net); do echo "== $IF"; tc qdisc show dev "$IF"; done
    ;;
  *) echo "uso: set <ip_par> <ms> [rate] | clear | show"; exit 1;;
esac
