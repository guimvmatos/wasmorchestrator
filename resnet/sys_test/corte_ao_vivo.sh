#!/usr/bin/env bash
# 09/11: troca o corte do modelo 6 (R18 fundido) a cada 2 s, ao vivo, entre
# os grupos A (127.0.0.1:18080) e B (127.0.0.1:18081), enquanto o client roda.
# Rode de dentro de resnet/sys_test/. Registra cada troca em /tmp/corte_ao_vivo.log.
cd /home/guimvmatos/Documents/wasmorchestrator/resnet/sys_test || exit 1
A=127.0.0.1:18080
B=127.0.0.1:18081
LOG=/tmp/corte_ao_vivo.log
: > "$LOG"
for i in $(seq 1 6); do
  for spec in "10 $A $B" "19 $A $B" "28 $A $B" "10,28 $A $B $A" "uncut"; do
    if [ "$spec" = "uncut" ]; then
      python3 fase3_table.py uncut 6 > /dev/null
    else
      python3 fase3_table.py cut 6 $spec > /dev/null
    fi
    echo "$(date +%T.%3N) rodada $i: $spec" | tee -a "$LOG"
    sleep 2
  done
done
python3 fase3_table.py uncut 6 > /dev/null
echo "$(date +%T.%3N) fim (sem corte)" | tee -a "$LOG"
