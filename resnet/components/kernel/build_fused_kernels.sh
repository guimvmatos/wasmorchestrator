#!/usr/bin/env bash
# 09/11: compila SÓ os 4 kernels fundidos novos (convbn, convbnrelu,
# convbn_gpu, convbnrelu_gpu), sem tocar nos outros. Mesma receita do
# rebuild_kernels_model_id.sh. Rode a partir de resnet/components/kernel/.
set -uo pipefail
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
export PATH="$HOME/.cargo/bin:$PATH"
export WASI_SDK="${WASI_SDK:-/opt/wasi-sdk}"
FAILED=()
for name in convbn convbnrelu; do
  echo "--- $name (CPU) ---"
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
    ls -l "${name}world_component.wasm"
  )
  if [ $? -ne 0 ]; then FAILED+=("$name"); fi
done
for name in convbn_gpu convbnrelu_gpu; do
  echo "--- $name (GPU) ---"
  (
    set -e
    cd "$name"
    if [ ! -d wit/deps ]; then wkg wit fetch; fi
    CC="" cargo build --target=wasm32-wasip2 --release
    ls -l "target/wasm32-wasip2/release/${name}.wasm"
  )
  if [ $? -ne 0 ]; then FAILED+=("$name"); fi
done
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "=== 4 kernels fundidos compilados com sucesso ==="
else
  echo "=== Falharam: ${FAILED[*]} ==="; exit 1
fi
