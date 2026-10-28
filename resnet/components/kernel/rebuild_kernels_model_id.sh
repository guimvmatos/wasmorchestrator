#!/usr/bin/env bash
# Regera bindings + recompila os kernels WASM depois da mudança de WIT que
# adicionou o campo `model-id` ao record `data` (Problema 1 / Problema 2 —
# ver claude/project-state.md).
#
# Rode este script a partir de resnet/components/kernel/:
#   cd ~/Documents/wasmorchestrator/resnet/components/kernel
#   bash rebuild_kernels_model_id.sh
#
# Pré-requisito: WASI_SDK apontando pro wasi-sdk instalado (ajuste se não
# for /opt/wasi-sdk na sua máquina).

set -uo pipefail
# (sem -e no nível do script principal — cada subshell de kernel ativa o
# próprio -e internamente; -e no escopo externo, combinado com `( ... ) ||`,
# é desativado pelo bash dentro do subshell e deixava o script seguir pro
# clang mesmo com o wit-bindgen tendo falhado antes — bug corrigido em 28/10.)

# Garante que ~/.cargo/bin (wit-bindgen, cargo) está no PATH mesmo rodando
# via `bash script.sh` (shell não-interativo não lê .bashrc/.profile, onde
# o rustup normalmente adiciona isso ao PATH).
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
export PATH="$HOME/.cargo/bin:$PATH"

export WASI_SDK="${WASI_SDK:-/opt/wasi-sdk}"

if [ ! -x "$WASI_SDK/bin/clang" ]; then
  echo "[ERRO] wasi-sdk clang não encontrado em $WASI_SDK/bin/clang" >&2
  echo "       Ajuste a variável WASI_SDK antes de rodar, ex:" >&2
  echo "       WASI_SDK=/caminho/certo bash rebuild_kernels_model_id.sh" >&2
  exit 1
fi

if ! command -v wit-bindgen >/dev/null 2>&1; then
  echo "[ERRO] wit-bindgen não encontrado no PATH (nem em ~/.cargo/bin)." >&2
  echo "       Rode 'which wit-bindgen' no seu terminal normal e me diga o caminho," >&2
  echo "       ou exporte o PATH certo antes de rodar este script." >&2
  exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "[ERRO] cargo não encontrado no PATH (nem em ~/.cargo/bin)." >&2
  exit 1
fi

# 22 kernels CPU (C) — wit-bindgen c + wasi-sdk clang, mesma receita de
# components/Compilation_resnet.md, pra cada um.
CPU_KERNELS=(
  add
  bn
  concat
  conv
  div
  dropout
  flatten
  gap
  gemm
  matmul
  maxpool
  mul
  relu
  reshape
  resize
  sigmoid
  silu
  slice
  softmax
  split
  sub
  transpose
)

# 9 kernels GPU (Rust + wit_bindgen::generate! macro) — só precisa recompilar,
# os bindings são regerados automaticamente pela macro no build.
GPU_KERNELS=(
  bn_gpu
  conv_gpu
  gap_gpu
  gemm_gpu
  matmul_gpu
  maxpool_gpu
  mul_gpu
  sigmoid_gpu
  silu_gpu
)

FAILED=()

echo "=== Kernels CPU (C) ==="
for name in "${CPU_KERNELS[@]}"; do
  echo "--- $name ---"
  if [ ! -d "$name" ]; then
    echo "[AVISO] diretório $name não encontrado, pulando."
    continue
  fi
  (
    set -e
    cd "$name"
    wit-bindgen c --world "${name}world" "${name}.wit"
    rm -f "${name}world_component.wasm"
    "$WASI_SDK/bin/clang" --target=wasm32-wasip2 \
      --sysroot="$WASI_SDK/share/wasi-sysroot" \
      -I"$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2" \
      -O3 -msimd128 \
      "${name}.c" "${name}world.c" "${name}world_component_type.o" \
      -o "${name}world_component.wasm" -mexec-model=reactor
  )
  # NÃO usar "(...) || FAILED+=(...)" — o bash desativa o -e de dentro do
  # subshell quando ele é operando de ||/&&/if, mesmo com `set -e` explícito
  # lá dentro, e o script seguiria pro clang mesmo com o wit-bindgen tendo
  # falhado antes (foi exatamente isso que quebrou na 1ª tentativa, 28/10).
  if [ $? -ne 0 ]; then
    FAILED+=("$name")
  fi
done

echo ""
echo "=== Kernels GPU (Rust) ==="
for name in "${GPU_KERNELS[@]}"; do
  echo "--- $name ---"
  if [ ! -d "$name" ]; then
    echo "[AVISO] diretório $name não encontrado, pulando."
    continue
  fi
  (
    set -e
    cd "$name"
    CC="" cargo build --target=wasm32-wasip2 --release
  )
  if [ $? -ne 0 ]; then
    FAILED+=("$name")
  fi
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "=== Tudo rebuildado com sucesso (22 CPU + 9 GPU) ==="
else
  echo "=== Falharam: ${FAILED[*]} ==="
  echo "Corrija e rode de novo só pra esses (o script é idempotente por kernel)."
  exit 1
fi
