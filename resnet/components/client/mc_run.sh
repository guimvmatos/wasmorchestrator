#!/usr/bin/env bash
# mc_run.sh — E10: vários clients ao mesmo tempo, cada um com a sua porta de resposta.
# Rode de dentro de resnet/components/client/.
#
#   bash mc_run.sh same N BASE MODEL_ID SERVIDOR REPS LABEL
#       N clients do mesmo modelo. Ex.: bash mc_run.sh same 4 130000 4 200.18.102.33:8050 30 r18-F10-peach-G1
#   bash mc_run.sh mix  BASE SERVIDOR REPS LABEL
#       4 clients, um de cada modelo (R18, R34, R50, VGG16). Ex.: bash mc_run.sh mix 139000 127.0.0.1:8050 20 mix-N10-G1
#
# Cada client i usa a porta 9000+i, request_ids a partir de BASE + i*1000, e grava
# mc/<LABEL>-c<i>.csv e mc/<LABEL>-c<i>.log. Cada client tem limite de 30 min (timeout):
# se algum travar, o script segue e o .log mostra onde parou.
set -u
mode=$1; shift
mkdir -p mc
start=$(date +%s.%N)
if [ "$mode" = same ]; then
  N=$1; BASE=$2; MID=$3; SRV=$4; REPS=$5; LAB=$6
  for i in $(seq 1 "$N"); do
    timeout 1800 ./target/release/client $((BASE + i*1000)) "$MID" input_image2.bin "$SRV" --warmup 3 --reps "$REPS" \
      --listen 0.0.0.0:$((9000 + i)) --csv "mc/${LAB}-c${i}.csv" --label "${LAB}-c${i}" > "mc/${LAB}-c${i}.log" 2>&1 &
  done
elif [ "$mode" = mix ]; then
  BASE=$1; SRV=$2; REPS=$3; LAB=$4
  i=0
  for MID in 4 5 1 2; do
    i=$((i + 1))
    timeout 1800 ./target/release/client $((BASE + i*1000)) "$MID" input_image2.bin "$SRV" --warmup 3 --reps "$REPS" \
      --listen 0.0.0.0:$((9000 + i)) --csv "mc/${LAB}-c${i}.csv" --label "${LAB}-m${MID}-c${i}" > "mc/${LAB}-c${i}.log" 2>&1 &
  done
else
  echo "modo: same | mix"; exit 1
fi
wait
end=$(date +%s.%N)
echo "[mc_run] ${mode} terminou em $(awk "BEGIN{printf \"%.1f\", $end - $start}") s"
for f in mc/"${LAB}"-c*.log; do grep -q 'LATÊNCIA' "$f" || echo "[mc_run] ATENÇÃO: $f não terminou (ver o log)"; done
