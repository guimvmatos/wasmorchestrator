# Runtime
cd /home/guimvmatos/Documents/wasi-gfx-runtime
cargo build -p runtime --release

# Kernels/Receivers
## Conv
cd /home/guimvmatos/Documents/wasmorchestrator/resnet/components/kernel/conv_gpu
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release

cd /home/guimvmatos/Documents/wasmorchestrator/resnet/components/receiver/conv_gpu
wkg wit fetch
cargo build --target=wasm32-wasip2 --release

cd ..
wac plug conv_gpu/target/wasm32-wasip2/release/convserver.wasm \
  --plug ../kernel/conv_gpu/target/wasm32-wasip2/release/conv_gpu.wasm \
  -o ../../sys_test/kernel_1.wasm


cd /home/guimvmatos/Documents/wasmorchestrator/resnet/sys_test
/home/guimvmatos/Documents/wasi-gfx-runtime/target/release/runtime \
  --wasm ./kernel_1.wasm \
  --port 8081 \
  --ip 127.0.0.1

## BatchNorm
cd /home/guimvmatos/Documents/wasmorchestrator/resnet/components/kernel/bn_gpu
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release

cd /home/guimvmatos/Documents/wasmorchestrator/resnet/components/receiver/bn_gpu
wkg wit fetch
cargo build --target=wasm32-wasip2 --release

cd ..
wac plug bn_gpu/target/wasm32-wasip2/release/bnserver.wasm \
  --plug ../kernel/bn_gpu/target/wasm32-wasip2/release/bn_gpu.wasm \
  -o ../../sys_test/kernel_2.wasm


cd /home/guimvmatos/Documents/wasmorchestrator/resnet/sys_test
/home/guimvmatos/Documents/wasi-gfx-runtime/target/release/runtime \
  --wasm ./kernel_2.wasm \
  --port 8082 \
  --ip 127.0.0.1