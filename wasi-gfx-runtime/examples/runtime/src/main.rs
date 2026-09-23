use std::sync::Arc;

use clap::Parser;
use frame_buffer_wasmtime::{FrameBufferCtx, FrameBufferCtxView};
use surface_wasmtime::{
    winit::WasiWinitEventLoopProxy, SurfaceCtxView, SurfaceFrameBufferCtx,
    SurfaceFrameBufferCtxView, SurfaceWebgpuCtx, SurfaceWebgpuCtxView,
};
use wasi_webgpu_wasmtime::{WasiWebGpuCtx, WasiWebGpuCtxView, WasiWebGpuOptions};
use wasmtime::{
    component::{Component, Linker},
    error::Context,
    Config, Engine, Store,
};

use wasmtime_wasi::{
    FsPerms, ResourceTable, WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView,
};

#[derive(clap::Parser, Debug)]
struct RuntimeArgs {
    /// Caminho do componente .wasm (ex: ../../receiver_conv.wasm)
    #[arg(long)]
    wasm: String,

    /// Porta TCP para escuta (ex: 8081)
    #[arg(long, default_value = "8081")]
    port: String,

    /// IP para bind (ex: 127.0.0.1 ou 0.0.0.0)
    #[arg(long, default_value = "127.0.0.1")]
    ip: String,
}



struct HostState {
    instance: Arc<wgpu_core::global::Global>,
    webgpu_options: WasiWebGpuOptions,
    main_thread_proxy: Arc<surface_wasmtime::winit::WasiWinitEventLoopProxy>,
}

impl HostState {
    fn new(main_thread_proxy: surface_wasmtime::winit::WasiWinitEventLoopProxy) -> Self {
        Self {
            instance: Arc::new(wgpu_core::global::Global::new(
                "webgpu",
                wgpu_types::InstanceDescriptor {
                    backends: wgpu_types::Backends::all(),
                    flags: wgpu_types::InstanceFlags::from_build_config(),
                    backend_options: Default::default(),
                    memory_budget_thresholds: Default::default(),
                    display: None,
                },
                None,
            )),
            webgpu_options: WasiWebGpuOptions::default(),
            main_thread_proxy: Arc::new(main_thread_proxy),
        }
    }
    pub fn add_workload(&self, args: &[String]) -> anyhow::Result<WorkloadState> {
        let mut wasi_builder = WasiCtxBuilder::new();
        
        // 1. Repassa os argumentos de linha de comando (ex: porta e IP)
        wasi_builder.args(args);

        // 2. Libera a rede para abrir o TcpListener (equivalente a --wasi inherit-network)
        // 2. Libera a rede completa no WASI P2 (incluindo bind e listen para servidores TCP)
        wasi_builder.inherit_network();
        wasi_builder.allow_ip_name_lookup(true);
        wasi_builder.allow_tcp(true);
        wasi_builder.allow_udp(true);

        // 3. Libera as pastas do disco com permissões totais de leitura e escrita
        wasi_builder.preopened_dir(".", ".", FsPerms::ReadWrite)?;

        let real_weights_path = "/home/guimvmatos/Documents/wasmorchestrator/resnet/bin_weights";

        if std::path::Path::new(real_weights_path).exists() {
            wasi_builder.preopened_dir(real_weights_path, "../bin_weights", FsPerms::ReadWrite)?;
            wasi_builder.preopened_dir(real_weights_path, "bin_weights", FsPerms::ReadWrite)?;
        } else {
            eprintln!("[AVISO] Pasta de pesos não encontrada no host em: {}", real_weights_path);
        }

        // Conecta saída padrão para você ver os prints no terminal
        wasi_builder.inherit_stdout().inherit_stderr();

        Ok(WorkloadState {
            table: ResourceTable::new(),
            wasi_ctx: wasi_builder.build(),
            instance: Arc::clone(&self.instance),
            webgpu_options: self.webgpu_options.clone(),
            main_thread_proxy: Arc::clone(&self.main_thread_proxy),
        })
    }
}

struct WorkloadState {
    table: ResourceTable,
    wasi_ctx: WasiCtx,
    instance: Arc<wgpu_core::global::Global>,
    webgpu_options: WasiWebGpuOptions,
    main_thread_proxy: Arc<WasiWinitEventLoopProxy>,
}
impl WasiView for WorkloadState {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.wasi_ctx,
            table: &mut self.table,
        }
    }
}

impl wasmtime::component::HasData for WorkloadState {
    type Data<'a> = &'a mut WorkloadState;
}

impl WasiWebGpuCtxView for WorkloadState {
    fn webgpu_ctx(&mut self) -> WasiWebGpuCtx<'_> {
        WasiWebGpuCtx {
            instance: &self.instance,
            table: &mut self.table,
            options: &self.webgpu_options,
        }
    }
}

impl FrameBufferCtxView for WorkloadState {
    fn frame_buffer_ctx<'a>(&'a mut self) -> FrameBufferCtx<'a> {
        FrameBufferCtx {
            table: &mut self.table,
        }
    }
}

impl SurfaceCtxView for WorkloadState {
    type Spawner = WasiWinitEventLoopProxy;
    fn surface_ctx(&mut self) -> surface_wasmtime::SurfaceCtx<'_, WasiWinitEventLoopProxy> {
        surface_wasmtime::SurfaceCtx {
            table: &mut self.table,
            main_thread_spawner: &self.main_thread_proxy,
        }
    }
}
impl SurfaceWebgpuCtxView for WorkloadState {
    type Spawner = WasiWinitEventLoopProxy;
    fn surface_webgpu_ctx(&mut self) -> SurfaceWebgpuCtx<'_, WasiWinitEventLoopProxy> {
        SurfaceWebgpuCtx {
            table: &mut self.table,
            instance: &self.instance,
            main_thread_spawner: &self.main_thread_proxy,
        }
    }
}
impl SurfaceFrameBufferCtxView for WorkloadState {
    type Spawner = WasiWinitEventLoopProxy;
    fn surface_frame_buffer_ctx(&mut self) -> SurfaceFrameBufferCtx<'_, WasiWinitEventLoopProxy> {
        SurfaceFrameBufferCtx {
            table: &mut self.table,
            instance: &self.instance,
            main_thread_spawner: &self.main_thread_proxy,
        }
    }
}



#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();

    let args = RuntimeArgs::parse();

    let (main_thread_loop, main_thread_proxy) =
        surface_wasmtime::winit::create_wasi_winit_event_loop();
    let host_state = HostState::new(main_thread_proxy);

    let mut config = Config::default();
    config.wasm_component_model(true);
    config.wasm_component_model_async(true);
    config.async_support(true);
    let engine = Engine::new(&config)?;
    let mut linker: Linker<WorkloadState> = Linker::new(&engine);

    // 1. Conecta WASI P2 (rede, fs, args) ao linker
    wasmtime_wasi::p2::add_to_linker_async(&mut linker)?;

    // 2. Conecta as interfaces WebGPU ao linker
    wasi_webgpu_wasmtime::add_to_linker(&mut linker)?;
    frame_buffer_wasmtime::add_to_linker(&mut linker)?;
    surface_wasmtime::add_all_to_linker(&mut linker)?;

    // 3. Monta os argumentos reais passados na CLI (nome_app, porta, ip)
    let kernel_args = vec![
        args.wasm.clone(),
        args.port.clone(),
        args.ip.clone(),
    ];

    // Inicia um executor de background para processar a fila de eventos do wgpu.
    // Usamos std::thread::spawn (não tokio::task::spawn_blocking) de propósito:
    // uma thread nativa é abandonada quando o processo termina, enquanto uma
    // task do blocking pool do Tokio trava o shutdown do Runtime esperando
    // esse loop terminar — o que nunca acontece.
    let gpu_global = Arc::clone(&host_state.instance);
    std::thread::spawn(move || {
        loop {
            let _ = gpu_global.poll_all_devices(true);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    });

    let workload_state = host_state.add_workload(&kernel_args)?;
    let mut store = Store::new(&engine, workload_state);

    println!("[HOST] Carregando componente WASM de: {}", args.wasm);
    let component = Component::from_file(&engine, &args.wasm)
        .context(format!("Componente não encontrado em: {}", args.wasm))?;

    println!("[HOST] Instanciando o componente...");
    let instance = linker.instantiate_async(&mut store, &component).await
        .context("Falha ao instanciar o componente no Linker")?;

    println!("[HOST] Buscando exportação wasi:cli/run...");

    // Busca a função 'run' exportada pelo componente desestruturando a tupla (_, idx)
    let (_, run_idx) = instance
        .get_export(&mut store, None, "wasi:cli/run@0.2.7")
        .and_then(|(_, iface_idx)| instance.get_export(&mut store, Some(&iface_idx), "run"))
        .or_else(|| {
            instance
                .get_export(&mut store, None, "wasi:cli/run@0.2.0")
                .and_then(|(_, iface_idx)| instance.get_export(&mut store, Some(&iface_idx), "run"))
        })
        .or_else(|| instance.get_export(&mut store, None, "run"))
        .context("Não foi possível localizar o export 'run' no componente")?;

    let run_func = instance
        .get_typed_func::<(), (Result<(), ()>,)>(&mut store, &run_idx)
        .context("Falha ao vincular tipos da função 'run'")?;

    println!("[HOST] Executando run() no componente...");
    let (run_res,) = run_func.call_async(&mut store, ()).await?;
    match run_res {
        Ok(()) => println!("[HOST] Componente finalizou com sucesso."),
        Err(()) => eprintln!("[HOST] Componente retornou erro em run()."),
    }

    Ok(())
}
