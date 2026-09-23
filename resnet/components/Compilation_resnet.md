#### Conv (kernel_1)
##### Kernel
cd "$WOR/resnet/components"

cd kernel/conv/
wit-bindgen c --world convworld conv.wit
export WASI_SDK=/opt/wasi-sdk
$WASI_SDK/bin/clang --target=wasm32-wasip2 \
  --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 \
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
  dropout.c dropoutworld.c dropoutworld_component_type.o \
  -o dropoutworld_component.wasm -mexec-model=reactor
  
##### Receiver
cd ../../receiver/dropout
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release
cd ..
wac plug dropout/target/wasm32-wasip2/release/dropoutserver.wasm --plug ../kernel/dropout/dropoutworld_component.wasm -o dropoutFinal.wasm

#### all (kernel_0)  
##### Receiver
cd ../../receiver/all
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
mv allFinal.wasm kernel_0.wasm

#### Dynamic Builder (kernel_0) 
Go to components/receiver/dynamic_builder folder
python3 build_custom.py --kernels 1,2,3,4,5,6,7,8,9 --component-dir ./target_component --output ./node_all.wasm
Change the name of the .wasm and put it on the sys_test folder

#### Onnx to Json
Go to resnet/text_onnx folder
python3 onnx_to_dfg.py --onnx resnet18.onnx --output resnet18DFG.json --weights-dir ./bin_weights
Put the .json file on sys_test\ folder and bin_weights on resnet\



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
handler escuta em 0.0.0.0:8090
client envia para server address: 127..1:8090 e escute em 0.0.0.0:9000
sys: escuta em 0.0.0.0:9999 envia para 0.0.0.0:9998 do orquestrador
receiver escuta em 0.0.0.0:8080
