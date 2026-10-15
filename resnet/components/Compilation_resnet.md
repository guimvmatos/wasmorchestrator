#### Conv (kernel_1)
##### Kernel
cd "$WOR/resnet/components"

cd kernel/conv/
wit-bindgen c --world convworld conv.wit
export WASI_SDK=/opt/wasi-sdk
$WASI_SDK/bin/clang --target=wasm32-wasip2 \
  --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 \
  -O3 -msimd128 \
  conv.c convworld.c convworld_component_type.o \
  -o convworld_component.wasm -mexec-model=reactor
  
##### Receiver
cd ../../receiver/conv
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release
cd ..
wac plug conv/target/wasm32-wasip2/release/convserver.wasm --plug ../kernel/conv/convworld_component.wasm -o convFinal.wasm

#### batchnorm (kernel_2)
##### Kernel
cd ../kernel/bn/
wit-bindgen c --world bnworld bn.wit
export WASI_SDK=/opt/wasi-sdk
$WASI_SDK/bin/clang --target=wasm32-wasip2 \
  --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 \
  -O3 -msimd128 \
  bn.c bnworld.c bnworld_component_type.o \
  -o bnworld_component.wasm -mexec-model=reactor
  
##### Receiver
cd ../../receiver/bn
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release
cd ..
wac plug bn/target/wasm32-wasip2/release/bnserver.wasm --plug ../kernel/bn/bnworld_component.wasm -o bnFinal.wasm

#### relu (kernel_3)
##### Kernel
cd ../kernel/relu/
wit-bindgen c --world reluworld relu.wit
export WASI_SDK=/opt/wasi-sdk
$WASI_SDK/bin/clang --target=wasm32-wasip2 \
  --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 \
  -O3 -msimd128 \
  relu.c reluworld.c reluworld_component_type.o \
  -o reluworld_component.wasm -mexec-model=reactor
  
##### Receiver
cd ../../receiver/relu
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release
cd ..
wac plug relu/target/wasm32-wasip2/release/reluserver.wasm --plug ../kernel/relu/reluworld_component.wasm -o reluFinal.wasm

#### maxpool (kernel_4)
##### Kernel
cd ../kernel/maxpool/
wit-bindgen c --world maxpoolworld maxpool.wit
export WASI_SDK=/opt/wasi-sdk
$WASI_SDK/bin/clang --target=wasm32-wasip2 \
  --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 \
  -O3 -msimd128 \
  maxpool.c maxpoolworld.c maxpoolworld_component_type.o \
  -o maxpoolworld_component.wasm -mexec-model=reactor
  
##### Receiver
cd ../../receiver/maxpool
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release
cd ..
wac plug maxpool/target/wasm32-wasip2/release/maxpoolserver.wasm --plug ../kernel/maxpool/maxpoolworld_component.wasm -o maxpoolFinal.wasm

#### add (kernel_5)
##### Kernel
cd ../kernel/add/
wit-bindgen c --world addworld add.wit
export WASI_SDK=/opt/wasi-sdk
$WASI_SDK/bin/clang --target=wasm32-wasip2 \
  --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 \
  -O3 -msimd128 \
  add.c addworld.c addworld_component_type.o \
  -o addworld_component.wasm -mexec-model=reactor
  
##### Receiver
cd ../../receiver/add
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release
cd ..
wac plug add/target/wasm32-wasip2/release/addserver.wasm --plug ../kernel/add/addworld_component.wasm -o addFinal.wasm

#### GAP (global average pool) (kernel_6)
##### Kernel
cd ../kernel/gap/
wit-bindgen c --world gapworld gap.wit
export WASI_SDK=/opt/wasi-sdk
$WASI_SDK/bin/clang --target=wasm32-wasip2 \
  --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 \
  -O3 -msimd128 \
  gap.c gapworld.c gapworld_component_type.o \
  -o gapworld_component.wasm -mexec-model=reactor
  
##### Receiver
cd ../../receiver/gap
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release
cd ..
wac plug gap/target/wasm32-wasip2/release/gapserver.wasm --plug ../kernel/gap/gapworld_component.wasm -o gapFinal.wasm

#### flatten (kernel_7)
##### Kernel
cd ../kernel/flatten/
wit-bindgen c --world flattenworld flatten.wit
export WASI_SDK=/opt/wasi-sdk
$WASI_SDK/bin/clang --target=wasm32-wasip2 \
  --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 \
  -O3 -msimd128 \
  flatten.c flattenworld.c flattenworld_component_type.o \
  -o flattenworld_component.wasm -mexec-model=reactor
  
##### Receiver
cd ../../receiver/flatten
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release
cd ..
wac plug flatten/target/wasm32-wasip2/release/flattenserver.wasm --plug ../kernel/flatten/flattenworld_component.wasm -o flattenFinal.wasm

#### gemm (kernel_8)
##### Kernel
cd ../kernel/gemm/
wit-bindgen c --world gemmworld gemm.wit
export WASI_SDK=/opt/wasi-sdk
$WASI_SDK/bin/clang --target=wasm32-wasip2 \
  --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 \
  -O3 -msimd128 \
  gemm.c gemmworld.c gemmworld_component_type.o \
  -o gemmworld_component.wasm -mexec-model=reactor
  
##### Receiver
cd ../../receiver/gemm
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release
cd ..
wac plug gemm/target/wasm32-wasip2/release/gemmserver.wasm --plug ../kernel/gemm/gemmworld_component.wasm -o gemmFinal.wasm

#### dropout (kernel_9)
##### Kernel
cd ../kernel/dropout/
wit-bindgen c --world dropoutworld dropout.wit
export WASI_SDK=/opt/wasi-sdk
$WASI_SDK/bin/clang --target=wasm32-wasip2 \
  --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 \
  -O3 -msimd128 \
  dropout.c dropoutworld.c dropoutworld_component_type.o \
  -o dropoutworld_component.wasm -mexec-model=reactor
  
##### Receiver
cd ../../receiver/dropout
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release
cd ..
wac plug dropout/target/wasm32-wasip2/release/dropoutserver.wasm --plug ../kernel/dropout/dropoutworld_component.wasm -o dropoutFinal.wasm

#### sigmoid (kernel_10)
##### Kernel
cd ../kernel/sigmoid/
wit-bindgen c --world sigmoidworld sigmoid.wit
export WASI_SDK=/opt/wasi-sdk
$WASI_SDK/bin/clang --target=wasm32-wasip2 \
  --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 \
  -O3 -msimd128 \
  sigmoid.c sigmoidworld.c sigmoidworld_component_type.o \
  -o sigmoidworld_component.wasm -mexec-model=reactor
  
##### Receiver
cd ../../receiver/sigmoid
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release
cd ..
wac plug sigmoid/target/wasm32-wasip2/release/sigmoidserver.wasm --plug ../kernel/sigmoid/sigmoidworld_component.wasm -o sigmoidFinal.wasm

#### mul (kernel_11)
##### Kernel
cd ../kernel/mul/
wit-bindgen c --world mulworld mul.wit
export WASI_SDK=/opt/wasi-sdk
$WASI_SDK/bin/clang --target=wasm32-wasip2 \
  --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 \
  -O3 -msimd128 \
  mul.c mulworld.c mulworld_component_type.o \
  -o mulworld_component.wasm -mexec-model=reactor
  
##### Receiver
cd ../../receiver/mul
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release
cd ..
wac plug mul/target/wasm32-wasip2/release/mulserver.wasm --plug ../kernel/mul/mulworld_component.wasm -o mulFinal.wasm

#### silu (kernel_12)
##### Kernel
cd ../kernel/silu/
wit-bindgen c --world siluworld silu.wit
export WASI_SDK=/opt/wasi-sdk
$WASI_SDK/bin/clang --target=wasm32-wasip2 \
  --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 \
  -O3 -msimd128 \
  silu.c siluworld.c siluworld_component_type.o \
  -o siluworld_component.wasm -mexec-model=reactor
  
##### Receiver
cd ../../receiver/silu
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release
cd ..
wac plug silu/target/wasm32-wasip2/release/siluserver.wasm --plug ../kernel/silu/siluworld_component.wasm -o siluFinal.wasm

#### split (kernel_13)
cd ../kernel/split/
wit-bindgen c --world splitworld split.wit
export WASI_SDK=/opt/wasi-sdk
$WASI_SDK/bin/clang --target=wasm32-wasip2 \
  --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 \
  -O3 -msimd128 \
  split.c splitworld.c splitworld_component_type.o \
  -o splitworld_component.wasm -mexec-model=reactor

##### Receiver
cd ../../receiver/split
wkg wit fetch
cargo build --target wasm32-wasip2 --release
cd ..
wac plug split/target/wasm32-wasip2/release/splitserver.wasm --plug ../kernel/split/splitworld_component.wasm -o splitFinal.wasm


#### concat (kernel_14)
##### Kernel
cd ../kernel/concat/
wit-bindgen c --world concatworld concat.wit
export WASI_SDK=/opt/wasi-sdk
$WASI_SDK/bin/clang --target=wasm32-wasip2 \
  --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 \
  -O3 -msimd128 \
  concat.c concatworld.c concatworld_component_type.o \
  -o concatworld_component.wasm -mexec-model=reactor

##### Receiver
cd ../../receiver/concat
wkg wit fetch
cargo build --target wasm32-wasip2 --release
cd ..
wac plug concat/target/wasm32-wasip2/release/concatserver.wasm --plug ../kernel/concat/concatworld_component.wasm -o concatFinal.wasm

#### resize (kernel_18)
# upsample nearest 2D (FPN top-down do YOLO) — puro CPU, sem variante GPU.
##### Kernel
cd ../kernel/resize/
wit-bindgen c --world resizeworld resize.wit
export WASI_SDK=/opt/wasi-sdk
$WASI_SDK/bin/clang --target=wasm32-wasip2 \
  --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 \
  -O3 -msimd128 \
  resize.c resizeworld.c resizeworld_component_type.o \
  -o resizeworld_component.wasm -mexec-model=reactor

##### Receiver
cd ../../receiver/resize
wkg wit fetch
cargo build --target wasm32-wasip2 --release
cd ..
wac plug resize/target/wasm32-wasip2/release/resizeserver.wasm --plug ../kernel/resize/resizeworld_component.wasm -o resizeFinal.wasm

#### all (kernel_0) — DEPRECADO, ver "Dynamic Builder" abaixo
# O crate receiver/all é estático (código Rust escrito à mão, não gerado) e
# ficou para trás em duas frentes:
#   1. Nunca ganhou os match arms de sigmoid/mul/silu (daí os comentários
#      abaixo) — exigiria editar receiver/all/src/lib.rs igual ao que foi
#      feito no catálogo do dynamic_builder.
#   2. O record `data` dos .wit ganhou o campo `from-step: u32` (patch
#      aplicado em todo o repo); qualquer `Data { .. }` literal escrito à mão
#      em receiver/all/src/lib.rs vai quebrar a compilação até adicionar
#      `from_step` nos literais, igual ao que foi feito nos dois
#      build_custom.py do dynamic_builder.
# allFinal.wasm no disco está desatualizado (mtime anterior ao patch from-step
# e aos kernels 10/11/12). Prefira a seção "Dynamic Builder" — ela já cobre
# os 12 kernels e compila sem exigir mudanças manuais no receiver/all.
##### Receiver
cd "$WOR/resnet/components/receiver/all/"
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release
cd ..
wac plug all/target/wasm32-wasip2/release/allserver.wasm \
--plug ../kernel/add/addworld_component.wasm \
--plug ../kernel/bn/bnworld_component.wasm \
--plug ../kernel/conv/convworld_component.wasm \
--plug ../kernel/dropout/dropoutworld_component.wasm \
--plug ../kernel/flatten/flattenworld_component.wasm \
--plug ../kernel/gap/gapworld_component.wasm \
--plug ../kernel/gemm/gemmworld_component.wasm \
--plug ../kernel/maxpool/maxpoolworld_component.wasm \
--plug ../kernel/relu/reluworld_component.wasm \
--plug ../kernel/sigmoid/sigmoidworld_component.wasm \ # receiver/all ainda não trata kernel_type 10 — ver nota acima
--plug ../kernel/mul/mulworld_component.wasm \ # receiver/all ainda não trata kernel_type 11 — ver nota acima
--plug ../kernel/silu/siluworld_component.wasm \ # receiver/all ainda não trata kernel_type 12 — ver nota acima
-o allFinal.wasm


cp *.wasm ../../sys_test/
cd ../../sys_test/
mv convFinal.wasm kernel_1.wasm
mv bnFinal.wasm kernel_2.wasm
mv reluFinal.wasm kernel_3.wasm
mv maxpoolFinal.wasm kernel_4.wasm
mv addFinal.wasm kernel_5.wasm
mv gapFinal.wasm kernel_6.wasm
mv flattenFinal.wasm kernel_7.wasm
mv gemmFinal.wasm kernel_8.wasm
mv dropoutFinal.wasm kernel_9.wasm
mv sigmoidFinal.wasm kernel_10.wasm
mv mulFinal.wasm kernel_11.wasm
mv siluFinal.wasm kernel_12.wasm
mv splitFinal.wasm kernel_13.wasm
mv concatFinal.wasm kernel_14.wasm
mv resizeFinal.wasm kernel_18.wasm
mv allFinal.wasm kernel_0.wasm

#### Dynamic Builder (kernel_0) 
cd "$WOR/resnet/components/receiver/dynamic_builder"
python3 build_custom.py --kernels 1,2,3,4,5,6,7,8,9,10,11,12,13,14,18 --component-dir ./target_component --output ./node_all.wasm
Change the name of the .wasm and put it on the sys_test folder
# split(13), concat(14) e resize(18) são kernels CPU puros (puro rearranjo/
# gather de memória, sem ganho em GPU) — ver Compilation_resnet_GPU.md pro
# --gpu equivalente e pro build prévio das crates kernel/split, kernel/concat
# e kernel/resize.

#### Onnx to Json
cd "$WOR/resnet/test_onnx/"
python3 onnx_to_dfg.py --onnx resnet50-v1-7.onnx --output ../sys_test/resnet18DFG.json --weights-dir ../bin_weights


### Running

####
Orquestrator
/home/guimvmatos/Documents/wasmorchestrator/resnet/orchestrator

Sys
/home/guimvmatos/Documents/wasmorchestrator/resnet/sys_test

Handler
/home/guimvmatos/Documents/wasmorchestrator/resnet/components/handler

Client
/home/guimvmatos/Documents/wasmorchestrator/resnet/components/client

portas:
orquestrador: escuta em 0.0.0.0:9998. envia para 9999 dos sys
handler escuta em 0.0.0.0:8050 (era 8090 — porta tomada pelo kernel_10/sigmoid_gpu)
client envia para server address: 127.0.0.1:8050 e escute em 0.0.0.0:9000
sys: escuta em 0.0.0.0:9999 envia para 0.0.0.0:9998 do orquestrador
receiver escuta em 0.0.0.0:8080

kernels GPU (ver Compilation_resnet_GPU.md): 8081(1) 8082(2) 8084(4) 8086(6) 8088(8) 8090(10) 8091(11) 8092(12)
