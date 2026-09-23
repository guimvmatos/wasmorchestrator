use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use wasmtime::component::*;
use wasmtime::{Config, Engine, Store};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiView};


use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use cudarc::driver::DeviceRepr;

bindgen!({
    world: "convworld",
    path: "../kernel/conv/conv.wit",
});

// Kernel CUDA compilado em runtime com NVRTC
const CONV2D_KERNEL: &str = r#"
struct ConvParams {
    int c_out;
    int c_in;
    int h_in;
    int w_in;
    int kh;
    int kw;
    int stride;
    int padding;
    int out_h;
    int out_w;
};

extern "C" __global__ void conv2d_forward(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__ output,
    ConvParams p
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = p.c_out * p.out_h * p.out_w;
    if (idx >= total_elements) return;

    // Decodifica o índice linear de saída: [co, i, j]
    int j = idx % p.out_w;
    int tmp = idx / p.out_w;
    int i = tmp % p.out_h;
    int co = tmp / p.out_h;

    float sum = 0.0f;
    int in_base_h = i * p.stride - p.padding;
    int in_base_w = j * p.stride - p.padding;

    for (int ci = 0; ci < p.c_in; ci++) {
        for (int ki = 0; ki < p.kh; ki++) {
            for (int kj = 0; kj < p.kw; kj++) {
                int in_h = in_base_h + ki;
                int in_w = in_base_w + kj;

                if (in_h >= 0 && in_h < p.h_in && in_w >= 0 && in_w < p.w_in) {
                    int input_idx = ci * (p.h_in * p.w_in) + in_h * p.w_in + in_w;
                    int kernel_idx = co * (p.c_in * p.kh * p.kw) + ci * (p.kh * p.kw) + ki * p.kw + kj;
                    sum += input[input_idx] * weights[kernel_idx];
                }
            }
        }
    }
    output[idx] = sum;
}
"#;

struct HostServerState {
    wasi: WasiCtx,
    resource_table: wasmtime_wasi::ResourceTable,
    cuda_dev: Arc<CudaDevice>,
}



#[repr(C)]
#[derive(Clone, Copy)]
struct CudaConvParams {
    c_out: i32,
    c_in: i32,
    h_in: i32,
    w_in: i32,
    kh: i32,
    kw: i32,
    stride: i32,
    padding: i32,
    out_h: i32,
    out_w: i32,
}

unsafe impl DeviceRepr for CudaConvParams {}

impl WasiView for HostServerState {
    fn ctx(&mut self) -> &mut WasiCtx {
        &mut self.wasi
    }
    fn table(&mut self) -> &mut wasmtime_wasi::ResourceTable {
        &mut self.resource_table
    }
}

impl planner::convworld::accelerator::Host for HostServerState {
    fn exec_conv(
        &mut self,
        c_out: u32,
        c_in: u32,
        h_in: u32,
        w_in: u32,
        kh: u32,
        kw: u32,
        stride: u32,
        padding: u32,
        weights: Vec<f32>,
        input: Vec<f32>,
    ) -> Vec<f32> {
        let out_h = ((h_in + 2 * padding - kh) / stride) + 1;
        let out_w = ((w_in + 2 * padding - kw) / stride) + 1;
        let out_size = (c_out * out_h * out_w) as usize;

        // 1. Aloca e copia os dados para a VRAM da GPU
        let dev = &self.cuda_dev;
        let d_input = dev.htod_copy(input).expect("Falha ao copiar input para GPU");
        let d_weights = dev.htod_copy(weights).expect("Falha ao copiar pesos para GPU");
        let mut d_output = dev.alloc_zeros::<f32>(out_size).expect("Falha ao alocar saída na GPU");

        // 2. Configura os parâmetros e a grid de execução
        let params = CudaConvParams {
            c_out: c_out as i32,
            c_in: c_in as i32,
            h_in: h_in as i32,
            w_in: w_in as i32,
            kh: kh as i32,
            kw: kw as i32,
            stride: stride as i32,
            padding: padding as i32,
            out_h: out_h as i32,
            out_w: out_w as i32,
        };

        let threads = 256;
        let blocks = ((out_size + threads - 1) / threads) as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // 3. Executa o kernel via tupla de 4 elementos nativa do cudarc
        let func = dev
            .get_func("conv2d", "conv2d_forward")
            .expect("Kernel CUDA conv2d_forward não encontrado");

        unsafe {
            func.launch(cfg, (&d_input, &d_weights, &mut d_output, params))
                .expect("Falha ao disparar kernel na GPU");
        }

        // 4. Copia o resultado da VRAM de volta para a memória RAM
        dev.dtoh_sync_copy(&d_output)
            .expect("Falha ao copiar saída de volta da GPU")
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Uso: wasm_gpu_runner <caminho_para_convFinal.wasm> [porta] [ip]");
        std::process::exit(1);
    }

    let wasm_path = PathBuf::from(&args[1]);

    // 1. Inicializa o dispositivo CUDA (RTX 4050)
    println!("Inicializando CUDA Device 0...");
    let cuda_dev = CudaDevice::new(0).context("Falha ao inicializar dispositivo CUDA")?;

    // 2. Compila o kernel CUDA em runtime
    println!("Compilando kernel de convolução via NVRTC...");
    let ptx = compile_ptx(CONV2D_KERNEL).context("Falha ao compilar PTX")?;
    cuda_dev.load_ptx(ptx, "conv2d", &["conv2d_forward"])
        .context("Falha ao carregar PTX no dispositivo")?;

    // 3. Configura o Wasmtime
    let mut config = Config::new();
    config.wasm_component_model(true);
    config.async_support(true);

    let engine = Engine::new(&config)?;
    let component = Component::from_file(&engine, &wasm_path)
        .context("Falha ao carregar o arquivo .wasm")?;

    let mut linker = Linker::new(&engine);
    wasmtime_wasi::add_to_linker_async(&mut linker)?;
    planner::convworld::accelerator::add_to_linker(&mut linker, |state: &mut HostServerState| state)?;

    let guest_args = &args[1..];
    let wasi_ctx = WasiCtxBuilder::new()
        .inherit_stdio()
        .inherit_network()
        .args(guest_args)
        .build();

    let state = HostServerState {
        wasi: wasi_ctx,
        resource_table: wasmtime_wasi::ResourceTable::new(),
        cuda_dev,
    };

    let mut store = Store::new(&engine, state);
    let instance = linker.instantiate_async(&mut store, &component).await?;

    let run_func = instance
        .get_typed_func::<(), (Result<(), ()>,)>(&mut store, "wasi:cli/run@0.2.7#run")
        .or_else(|_| instance.get_typed_func::<(), (Result<(), ()>,)>(&mut store, "wasi:cli/run@0.2.0#run"))?;

    println!("Host Runner pronto. Despachando execução do Wasm...");
    let _ = run_func.call_async(&mut store, ()).await?;

    Ok(())
}