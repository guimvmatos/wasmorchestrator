## Deployment

### Initial dependencies:
``` bash
curl https://sh.rustup.rs -sSf | sh
#apt install cargo rustup
#apt install rustup
rustup target add wasm32-wasip1
rustup target add wasm32-wasip2
cargo install wit-bindgen-cli wkg wac-cli
export PATH="/root/.cargo/bin:$PATH"

wget https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-25/wasi-sdk-25.0-x86_64-linux.tar.gz
tar -xvf wasi-sdk-25.0-x86_64-linux.tar.gz
sudo mv wasi-sdk-25.0-x86_64-linux /opt/wasi-sdk
export WASI_SDK="/opt/wasi-sdk"
rm -rf wasi-sdk-25.0-x86_64-linux*

curl -L -O https://github.com/bytecodealliance/wasmtime/releases/download/v29.0.0/wasmtime-v29.0.0-x86_64-linux.tar.xz
tar -xf wasmtime-v29.0.0-x86_64-linux.tar.xz
mv wasmtime-v29.0.0-x86_64-linux/wasmtime /usr/local/bin/
rm -rf wasmtime-v29.0.0-x86_64-linux*
```



### Components
#### Grayscale (kernel_1)
##### Kernel
cd Distributed/kernel/grayscale/
wit-bindgen c --world grayscaleworld grayscale.wit
export WASI_SDK=/opt/wasi-sdk
$WASI_SDK/bin/clang --target=wasm32-wasip2 \
  --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 \
  grayscale.c grayscaleworld.c grayscaleworld_component_type.o \
  -o grayscaleworld_component.wasm -mexec-model=reactor

##### Receiver
cd ../../receivers/grayscale
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release
cd ..
wac plug grayscale/target/wasm32-wasip2/release/grayscaleserver.wasm --plug ../kernel/grayscale/grayscaleworld_component.wasm -o grayscaleFinal.wasm
cd ../..


#### Sobel (kernel_2=)
##### Kernel
cd Distributed/kernel/sobel
wit-bindgen c --world sobelworld sobel.wit
export WASI_SDK=/opt/wasi-sdk
$WASI_SDK/bin/clang --target=wasm32-wasip2 \
  --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 \
  sobel.c sobelworld.c sobelworld_component_type.o \
  -o sobelworld_component.wasm -mexec-model=reactor

##### Receiver
cd ../../receivers/sobel
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release
cd ..
wac plug sobel/target/wasm32-wasip2/release/sobelserver.wasm --plug ../kernel/sobel/sobelworld_component.wasm -o sobelFinal.wasm
cd ../..


#### Negative
##### Kernel
cd Distributed/kernel/negative
wit-bindgen c --world negativeworld negative.wit
export WASI_SDK=/opt/wasi-sdk
$WASI_SDK/bin/clang --target=wasm32-wasip2 \
  --sysroot=$WASI_SDK/share/wasi-sysroot \
  -I$WASI_SDK/share/wasi-sysroot/include/wasm32-wasip2 \
  negative.c negativeworld.c negativeworld_component_type.o \
  -o negativeworld_component.wasm -mexec-model=reactor

##### Receiver
cd ../../receivers/negative
wkg wit fetch
CC="" cargo build --target=wasm32-wasip2 --release
cd ..
wac plug negative/target/wasm32-wasip2/release/negativeserver.wasm --plug ../kernel/negative/negativeworld_component.wasm -o negativeFinal.wasm



cp *.wasm ../sys/
cd ../sys/
mv grayscaleFinal.wasm kernel_1.wasm
mv sobelFinal.wasm kernel_2.wasm
mv negativeFinal.wasm kernel_3.wasm
cp *.wasm ../sys_test/


## Client
cd Distributed/client
CC="" cargo run
convert resultado.ppm resultado.jpg

## orchestrator
cd Distributed/orchestrator
CC="" cargo run

## sys or sys_test (preferred)
cd Distributed/sys
CC="" cargo run


## Tests env Ok
For distributed, use:
Tests/orchestrator Ok
Distributed/sys_test Ok
Distributed/client Ok
Distributed/kernel Ok
Distributed/receivers Ok



For monolithic, use:
Tests/orchestrator confirmed?
Tests/receiver confirmed?
Tests/kernel confirmed?
Distributed/client (with only 1 as request) Ok
Distributed/sys_test Ok





## to run manually:
/home/eiumgat/wace2026/code/distributed/receivers
wasmtime run --wasi inherit-network --dir . grayscaleFinal.wasm 8081
wasmtime run --wasi inherit-network --dir . sobelFinal.wasm 8082
wasmtime run --wasi inherit-network --dir . negativeFinal.wasm 8083




git reset --hard origin/main