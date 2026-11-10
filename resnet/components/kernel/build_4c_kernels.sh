#!/usr/bin/env bash
# 10/11 (4c): compila os 9 kernels GPU da via residente (7 alterados + relu_gpu
# e add_gpu novos). Os kernels de CPU e os outros GPU (sigmoid/mul/silu/matmul)
# não mudam. Rodar como root, a partir de resnet/components/kernel/.
set -uo pipefail
[ -f /root/.cargo/env ] && source /root/.cargo/env
export PATH="/root/.cargo/bin:$PATH"
FAILED=()
for name in conv_gpu bn_gpu convbn_gpu convbnrelu_gpu maxpool_gpu gap_gpu gemm_gpu relu_gpu add_gpu; do
  echo "--- $name ---"
  (
    set -e
    cd "$name"
    if [ ! -d wit/deps ]; then wkg wit fetch; fi
    rm -f "target/wasm32-wasip2/release/${name}.wasm"
    CC="" cargo build --target=wasm32-wasip2 --release 2>&1 | tail -1
    ls -l "target/wasm32-wasip2/release/${name}.wasm"
  )
  if [ $? -ne 0 ]; then FAILED+=("$name"); fi
done
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "=== 9 kernels GPU (4c) compilados com sucesso ==="
else
  echo "=== Falharam: ${FAILED[*]} ==="; exit 1
fi
