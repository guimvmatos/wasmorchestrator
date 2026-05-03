cd Tests/kernel/
export WASI_SDK=/opt/wasi-sdk
wit-bindgen c --world kernelworld kernel.wit
$WASI_SDK/bin/clang --target=wasm32-wasip2 \
  --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 \
  kernel.c kernelworld.c kernelworld_component_type.o \
  -o kernelworld_component.wasm -mexec-model=reactor

cd ../receiver/
