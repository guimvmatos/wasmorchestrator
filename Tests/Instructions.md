cd Tests/kernel/
export WASI_SDK=/opt/wasi-sdk
wit-bindgen c --world kernelworld kernel.wit
$WASI_SDK/bin/clang --target=wasm32-wasip2 \
  --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 \
  kernel.c kernelworld.c kernelworld_component_type.o \
  -o kernelworld_component.wasm -mexec-model=reactor

cd ../receiver/
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release
cd ..
wac plug receiver/target/wasm32-wasip2/release/kernelserver.wasm --plug kernel/kernelworld_component.wasm -o final.wasm

mv final.wasm sys/

wasmtime run --wasi inherit-network --dir . final.wasm 8081 10.68.119.168






git reset --hard origin/main
git pull