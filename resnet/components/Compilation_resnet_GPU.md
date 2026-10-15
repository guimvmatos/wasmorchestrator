git clone wasmorchestrator.... 
export WOR="$(pwd)"

# Runtime
cd "$WOR/wasi-gfx-runtime"
cargo build -p runtime --release

# Kernels/Receivers
## Conv
cd "$WOR/resnet/components/kernel/conv_gpu"
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release

cd "$WOR/resnet/components/receiver/conv_gpu"
wkg wit fetch
cargo build --target=wasm32-wasip2 --release

cd ..
wac plug conv_gpu/target/wasm32-wasip2/release/convserver.wasm \
  --plug ../kernel/conv_gpu/target/wasm32-wasip2/release/conv_gpu.wasm \
  -o ../../sys_test/kernel_1_gpu.wasm

## BatchNorm
cd "$WOR/resnet/components/kernel/bn_gpu"
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release

cd "$WOR/resnet/components/receiver/bn_gpu"
wkg wit fetch
cargo build --target=wasm32-wasip2 --release

cd ..
wac plug bn_gpu/target/wasm32-wasip2/release/bnserver.wasm \
  --plug ../kernel/bn_gpu/target/wasm32-wasip2/release/bn_gpu.wasm \
  -o ../../sys_test/kernel_2_gpu.wasm

## MaxPool
cd "$WOR/resnet/components/kernel/maxpool_gpu"
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release

cd "$WOR/resnet/components/receiver/maxpool_gpu"
wkg wit fetch
cargo build --target=wasm32-wasip2 --release

cd ..
wac plug maxpool_gpu/target/wasm32-wasip2/release/maxpoolserver.wasm \
  --plug ../kernel/maxpool_gpu/target/wasm32-wasip2/release/maxpool_gpu.wasm \
  -o ../../sys_test/kernel_4_gpu.wasm

## GAP
cd "$WOR/resnet/components/kernel/gap_gpu"
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release

cd "$WOR/resnet/components/receiver/gap_gpu"
wkg wit fetch
cargo build --target=wasm32-wasip2 --release

cd ..
wac plug gap_gpu/target/wasm32-wasip2/release/gapserver.wasm \
  --plug ../kernel/gap_gpu/target/wasm32-wasip2/release/gap_gpu.wasm \
  -o ../../sys_test/kernel_6_gpu.wasm

## GEMM
cd "$WOR/resnet/components/kernel/gemm_gpu"
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release

cd "$WOR/resnet/components/receiver/gemm_gpu"
wkg wit fetch
cargo build --target=wasm32-wasip2 --release

cd ..
wac plug gemm_gpu/target/wasm32-wasip2/release/gemmserver.wasm \
  --plug ../kernel/gemm_gpu/target/wasm32-wasip2/release/gemm_gpu.wasm \
  -o ../../sys_test/kernel_8_gpu.wasm

## Sigmoid
cd "$WOR/resnet/components/kernel/sigmoid_gpu"
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release

cd "$WOR/resnet/components/receiver/sigmoid_gpu"
wkg wit fetch
cargo build --target=wasm32-wasip2 --release

cd ..
wac plug sigmoid_gpu/target/wasm32-wasip2/release/sigmoidserver.wasm \
  --plug ../kernel/sigmoid_gpu/target/wasm32-wasip2/release/sigmoid_gpu.wasm \
  -o ../../sys_test/kernel_10_gpu.wasm

## mul
cd "$WOR/resnet/components/kernel/mul_gpu"
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release

cd "$WOR/resnet/components/receiver/mul_gpu"
wkg wit fetch
cargo build --target=wasm32-wasip2 --release

cd ..
wac plug mul_gpu/target/wasm32-wasip2/release/mulserver.wasm \
  --plug ../kernel/mul_gpu/target/wasm32-wasip2/release/mul_gpu.wasm \
  -o ../../sys_test/kernel_11_gpu.wasm

## SILU
cd "$WOR/resnet/components/kernel/silu_gpu"
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release

cd "$WOR/resnet/components/receiver/silu_gpu"
wkg wit fetch
cargo build --target=wasm32-wasip2 --release

cd ..
wac plug silu_gpu/target/wasm32-wasip2/release/siluserver.wasm \
  --plug ../kernel/silu_gpu/target/wasm32-wasip2/release/silu_gpu.wasm \
  -o ../../sys_test/kernel_12_gpu.wasm

## MatMul
# MatMul em lote (batched), bloco de atenção do YOLO — o ÚNICO kernel entre
# 13-18 com FLOPs reais, por isso tem variante GPU (os demais — split,
# concat, reshape, transpose, resize — são puro rearranjo/gather de
# memória, sem ganho em shader). Mesma assinatura WIT nas duas variantes
# (lhs, rhs, lhs-shape, rhs-shape, data), igual Conv/BN/MaxPool/Gemm-CPU.
cd "$WOR/resnet/components/kernel/matmul_gpu"
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release

cd "$WOR/resnet/components/receiver/matmul_gpu"
wkg wit fetch
cargo build --target=wasm32-wasip2 --release

cd ..
wac plug matmul_gpu/target/wasm32-wasip2/release/matmulserver.wasm \
  --plug ../kernel/matmul_gpu/target/wasm32-wasip2/release/matmul_gpu.wasm \
  -o ../../sys_test/kernel_17_gpu.wasm

# to run

cd "$WOR/resnet/sys_test"
"$WOR/wasi-gfx-runtime/target/release/runtime" \
  --wasm ./kernel_1_gpu.wasm \
  --port 8081 \
  --ip 127.0.0.1

cd "$WOR/resnet/sys_test"
"$WOR/wasi-gfx-runtime/target/release/runtime" \
  --wasm ./kernel_2_gpu.wasm \
  --port 8082 \
  --ip 127.0.1

cd "$WOR/resnet/sys_test"
"$WOR/wasi-gfx-runtime/target/release/runtime" \
  --wasm ./kernel_4_gpu.wasm \
  --port 8084 \
  --ip 127.0.0.1

cd "$WOR/resnet/sys_test"
"$WOR/wasi-gfx-runtime/target/release/runtime" \
  --wasm ./kernel_6_gpu.wasm \
  --port 8086 \
  --ip 127.0.0.1

cd "$WOR/resnet/sys_test"
"$WOR/wasi-gfx-runtime/target/release/runtime" \
  --wasm ./kernel_8_gpu.wasm \
  --port 8088 \
  --ip 127.0.0.1

cd "$WOR/resnet/sys_test"
"$WOR/wasi-gfx-runtime/target/release/runtime" \
  --wasm ./kernel_10_gpu.wasm \
  --port 8090 \
  --ip 127.0.0.1

cd "$WOR/resnet/sys_test"
"$WOR/wasi-gfx-runtime/target/release/runtime" \
  --wasm ./kernel_11_gpu.wasm \
  --port 8091 \
  --ip 127.0.0.1  

cd "$WOR/resnet/sys_test"
"$WOR/wasi-gfx-runtime/target/release/runtime" \
  --wasm ./kernel_12_gpu.wasm \
  --port 8092 \
  --ip 127.0.0.1

cd "$WOR/resnet/sys_test"
"$WOR/wasi-gfx-runtime/target/release/runtime" \
  --wasm ./kernel_17_gpu.wasm \
  --port 8097 \
  --ip 127.0.0.1

# onnx to dfg.py
cd "$WOR/resnet/test_onnx"
python3 onnx_to_dfg.py --onnx resnet50-v1-7.onnx --output ../sys_test/resnet18DFG.json --weights-dir ../bin_weights


# AutoGenerate
cd "$WOR/resnet/components/receiver/dynamic_builder_gpu"
python3 build_custom.py \
  --kernels 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,22 \
  --gpu 1,2,4,6,8,10,11,12,17 \
  --component-dir ./target_component \
  --output ../../../sys_test/kernel_0_gpu.wasm
# relu(3), add(5), flatten(7), dropout(9), split(13), concat(14), reshape(15),
# transpose(16), resize(18) e softmax(22) não têm variante GPU (puro
# rearranjo/gather de memória, identidade/permutação, ou normalização
# ponto-a-ponto por eixo — sem FLOPs significativos — sem ganho em shader) —
# ficam de fora de --gpu e são plugados na variante CPU automaticamente
# (fallback do catálogo). matmul(17) É o único entre 13-18/22 com FLOPs
# reais e por isso TEM variante GPU — incluído em --gpu acima; sem --gpu 17,
# cai na variante CPU automaticamente (mesmo fallback).
#
# split(13), concat(14), resize(18), reshape(15), transpose(16) e
# softmax(22) são kernels CPU em C (wit-bindgen c + wasi-sdk clang, ver
# Compilation_resnet.md), não crates Rust — exigem que o
# <nome>world_component.wasm já tenha sido compilado antes, já que o
# build_custom.py só faz `wac plug` sobre o .wasm pronto, não compila os
# kernels em si:
cd "$WOR/resnet/components/kernel/split" && wit-bindgen c --world splitworld split.wit && \
  $WASI_SDK/bin/clang --target=wasm32-wasip2 --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 -O3 -msimd128 \
  split.c splitworld.c splitworld_component_type.o -o splitworld_component.wasm -mexec-model=reactor

cd "$WOR/resnet/components/kernel/concat" && wit-bindgen c --world concatworld concat.wit && \
  $WASI_SDK/bin/clang --target=wasm32-wasip2 --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 -O3 -msimd128 \
  concat.c concatworld.c concatworld_component_type.o -o concatworld_component.wasm -mexec-model=reactor

cd "$WOR/resnet/components/kernel/resize" && wit-bindgen c --world resizeworld resize.wit && \
  $WASI_SDK/bin/clang --target=wasm32-wasip2 --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 -O3 -msimd128 \
  resize.c resizeworld.c resizeworld_component_type.o -o resizeworld_component.wasm -mexec-model=reactor

cd "$WOR/resnet/components/kernel/reshape" && wit-bindgen c --world reshapeworld reshape.wit && \
  $WASI_SDK/bin/clang --target=wasm32-wasip2 --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 -O3 -msimd128 \
  reshape.c reshapeworld.c reshapeworld_component_type.o -o reshapeworld_component.wasm -mexec-model=reactor

cd "$WOR/resnet/components/kernel/transpose" && wit-bindgen c --world transposeworld transpose.wit && \
  $WASI_SDK/bin/clang --target=wasm32-wasip2 --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 -O3 -msimd128 \
  transpose.c transposeworld.c transposeworld_component_type.o -o transposeworld_component.wasm -mexec-model=reactor

cd "$WOR/resnet/components/kernel/softmax" && wit-bindgen c --world softmaxworld softmax.wit && \
  $WASI_SDK/bin/clang --target=wasm32-wasip2 --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 -O3 -msimd128 \
  softmax.c softmaxworld.c softmaxworld_component_type.o -o softmaxworld_component.wasm -mexec-model=reactor

# matmul(17) só precisa deste build se você NÃO passar 17 em --gpu (cai na
# variante CPU em C); se passar 17 em --gpu, o build prévio é o do
# kernel/matmul_gpu (crate Rust, ver seção "## MatMul" acima), não este:
cd "$WOR/resnet/components/kernel/matmul" && wit-bindgen c --world matmulworld matmul.wit && \
  $WASI_SDK/bin/clang --target=wasm32-wasip2 --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 -O3 -msimd128 \
  matmul.c matmulworld.c matmulworld_component_type.o -o matmulworld_component.wasm -mexec-model=reactor

cd "$WOR/resnet/sys_test"
"$WOR/wasi-gfx-runtime/target/release/runtime" \
  --wasm ./kernel_0_gpu.wasm \
  --port 8080 \
  --ip 127.0.0.1