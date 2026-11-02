use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use clap::Parser;
use frame_buffer_wasmtime::{FrameBufferCtx, FrameBufferCtxView};
use surface_wasmtime::{
    winit::WasiWinitEventLoopProxy, MainThreadSpawner, Surface, SurfaceCtxView, SurfaceDesc,
    SurfaceFrameBufferCtx, SurfaceFrameBufferCtxView, SurfaceWebgpuCtx, SurfaceWebgpuCtxView,
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

    /// Força o modo sem tela (headless). Sem esta flag, o modo é escolhido sozinho:
    /// headless quando não há DISPLAY / WAYLAND_DISPLAY / WAYLAND_SOCKET no ambiente.
    #[arg(long, default_value_t = false)]
    headless: bool,
}

// 02/11: modo sem tela (headless), para servidores acessados por SSH.
//
// O winit exige uma tela (X11/Wayland) só para CRIAR o event loop; sem tela ele
// aborta com "neither WAYLAND_DISPLAY nor WAYLAND_SOCKET nor DISPLAY is set".
// Nossos componentes não usam janela (wasi:surface): só WebGPU de computação e
// rede/arquivos. Então, sem tela, não criamos o event loop e usamos este spawner,
// que roda o closure direto (Spawn) e recusa criar janela (create_surface).
// Com tela, nada muda: continua o proxy do winit, exatamente como antes.
#[derive(Clone)]
enum MainSpawner {
    Winit(WasiWinitEventLoopProxy),
    Headless,
}

impl MainThreadSpawner for MainSpawner {
    fn spawn<F, T>(&self, f: F) -> impl Future<Output = T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let this = self.clone();
        async move {
            match this {
                MainSpawner::Winit(proxy) => proxy.spawn(f).await,
                MainSpawner::Headless => f(),
            }
        }
    }

    fn create_surface(&self, desc: SurfaceDesc) -> Pin<Box<dyn Future<Output = Surface> + Send>> {
        match self {
            MainSpawner::Winit(proxy) => proxy.create_surface(desc),
            MainSpawner::Headless => panic!(
                "[HOST] Modo sem tela (headless): o componente pediu uma janela (wasi:surface), \
                 que não existe neste modo. Rode com DISPLAY/WAYLAND_DISPLAY definido."
            ),
        }
    }
}

/// true quando o ambiente tem uma tela que o winit consegue usar
/// (as mesmas variáveis que o winit confere antes de abortar).
fn has_display() -> bool {
    ["DISPLAY", "WAYLAND_DISPLAY", "WAYLAND_SOCKET"]
        .iter()
        .any(|v| std::env::var_os(v).map_or(false, |s| !s.is_empty()))
}



struct HostState {
    instance: Arc<wgpu_core::global::Global>,
    webgpu_options: WasiWebGpuOptions,
    main_thread_proxy: Arc<MainSpawner>,
}

impl HostState {
    fn new(main_thread_proxy: MainSpawner) -> Self {
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
    main_thread_proxy: Arc<MainSpawner>,
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
    type Spawner = MainSpawner;
    fn surface_ctx(&mut self) -> surface_wasmtime::SurfaceCtx<'_, MainSpawner> {
        surface_wasmtime::SurfaceCtx {
            table: &mut self.table,
            main_thread_spawner: &self.main_thread_proxy,
        }
    }
}
impl SurfaceWebgpuCtxView for WorkloadState {
    type Spawner = MainSpawner;
    fn surface_webgpu_ctx(&mut self) -> SurfaceWebgpuCtx<'_, MainSpawner> {
        SurfaceWebgpuCtx {
            table: &mut self.table,
            instance: &self.instance,
            main_thread_spawner: &self.main_thread_proxy,
        }
    }
}
impl SurfaceFrameBufferCtxView for WorkloadState {
    type Spawner = MainSpawner;
    fn surface_frame_buffer_ctx(&mut self) -> SurfaceFrameBufferCtx<'_, MainSpawner> {
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

    // Com tela: cria o event loop do winit como antes (a variável segura o loop
    // vivo até o fim do main, igual ao comportamento anterior). Sem tela: headless.
    let headless = args.headless || !has_display();
    let (_main_thread_loop, main_thread_proxy) = if headless {
        println!("[HOST] Sem tela (DISPLAY/WAYLAND ausentes ou --headless): modo headless. WebGPU de computação continua disponível; janelas (wasi:surface) não.");
        (None, MainSpawner::Headless)
    } else {
        let (event_loop, proxy) = surface_wasmtime::winit::create_wasi_winit_event_loop();
        (Some(event_loop), MainSpawner::Winit(proxy))
    };
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

    // CVE-2026-27204: Wasmtime limita por padrão quantos bytes um guest pode
    // copiar pro host numa única hostcall (128MiB default), para prevenir
    // exhaustão de recursos por guest malicioso. Nossos próprios tensores de
    // pesos (ex: camada FC densa do Gemm) legitimamente excedem isso, então
    // ampliamos o teto — este runtime só executa componentes nossos, não
    // recebe wasm de terceiros não confiável.
    store.set_hostcall_fuel(1024 * 1024 * 1024); // 1 GiB

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
