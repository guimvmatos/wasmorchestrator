#!/usr/bin/env bash
# 10/11 (4c, parte 2): compila o ops_gpu (Sigmoid, SiLU, Mul/Sub/Div/Add-const,
# Concat, Split, Transpose, Slice, Resize, Softmax, MatMul na via residente).
# Rodar como root, a partir de resnet/components/kernel/.
set -euo pipefail
[ -f /root/.cargo/env ] && source /root/.cargo/env
export PATH="/root/.cargo/bin:$PATH"
cd ops_gpu
if [ ! -d wit/deps ]; then wkg wit fetch; fi
rm -f target/wasm32-wasip2/release/ops_gpu.wasm
CC="" cargo build --target=wasm32-wasip2 --release 2>&1 | tail -1
ls -l target/wasm32-wasip2/release/ops_gpu.wasm
echo "=== ops_gpu (4c parte 2) compilado ==="
