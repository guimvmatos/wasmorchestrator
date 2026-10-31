use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use chrono::Local;
use sysinfo::{System, Cpu};
use std::process::{Child, Command};
use std::time::Instant;


#[derive(Serialize, Deserialize, Debug, Clone)]
struct RoutingTable {
    table: HashMap<u8, String>,
    node_scores: HashMap<u8, f32>,
    assignments: HashMap<u8, Vec<u8>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct FunctionInfo {
    id: u8,
    endpoint: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct SystemMetrics {
    node_id: u8,
    cpu_free_pct: f32,
    mem_free_pct: f32,
    health_score: f32,
    net_incoming_bytes: u64,
    functions: Vec<FunctionInfo>,
}

// ============================================================================
// COMPOSIÇÃO DINÂMICA POR GRUPO (ver claude/project-state.md, seção "DESIGN
// CONFIRMADO (26/10)"): o sys builda e roda sozinho, sem intervenção manual.
//
// Regra 1 (adição é sempre isolada): kernel novo que nenhum grupo cobre ainda
// -> builda só o delta, sobe grupo novo numa porta nova, nunca mexe nos
// grupos que já estão rodando.
// Regra 2 (remoção é o único gatilho de consolidação): kernel que saiu de
// `assignments` -> reconstrói a união de tudo que sobrou num `.wasm` novo,
// sobe, atualiza a tabela (via FunctionInfo/telemetria, mecanismo que já
// existe), e só DEPOIS mata os grupos antigos que ficaram redundantes.
// ============================================================================

/// Kernels que têm variante GPU e devem usá-la quando presentes num grupo —
/// mesmo conjunto já validado manualmente (`--gpu 1,2,4,6,8,10,11,12,17`).
/// #### TODO: se um novo kernel ganhar variante GPU, incluir aqui.
const GPU_CAPABLE_KERNELS: &[u8] = &[1, 2, 4, 6, 8, 10, 11, 12, 17];

/// Caminho do crate do Dynamic Builder relativo ao cwd do sys (sys_test/).
/// #### TODO: ajustar se a árvore de pastas mudar.
const BUILDER_DIR: &str = "../components/receiver/dynamic_builder_gpu";
const BUILDER_SCRIPT: &str = "build_custom.py";
const COMPONENT_DIR: &str = "./target_component";

/// Caminho do runtime wasi-gfx relativo ao cwd do sys (sys_test/).
/// #### TODO: ajustar se a árvore de pastas mudar.
const RUNTIME_BIN: &str = "../../wasi-gfx-runtime/target/release/runtime";

/// Porta inicial pra alocação de grupos. Nunca reutiliza uma porta já usada
/// nesta execução do processo (mesmo depois de um grupo morrer), pra evitar
/// correr risco de socket ainda em TIME_WAIT.
const BASE_PORT: u16 = 8080;

/// Intervalo entre checagens de conexão ativa antes de matar um grupo
/// redundante (ver `drain_and_kill_group`).
const DRAIN_POLL_INTERVAL_MS: u64 = 200;

/// Tempo máximo de espera por grupo antes de matar forçado mesmo com
/// conexão ainda ativa — fallback de segurança pra não deixar processo
/// preso indefinidamente esperando uma conexão que nunca fecha.
const DRAIN_MAX_WAIT: std::time::Duration = std::time::Duration::from_secs(15);

struct Group {
    kernels: HashSet<u8>,
    port: u16,
    child: Child,
}

struct SysRuntime {
    groups: HashMap<u32, Group>,
    next_group_id: u32,
    next_port: u16,
}

impl SysRuntime {
    fn new() -> Self {
        SysRuntime {
            groups: HashMap::new(),
            next_group_id: 0,
            next_port: BASE_PORT,
        }
    }

    fn covered_kernels(&self) -> HashSet<u8> {
        self.groups
            .values()
            .flat_map(|g| g.kernels.iter().cloned())
            .collect()
    }

    fn alloc_group_id(&mut self) -> u32 {
        let id = self.next_group_id;
        self.next_group_id += 1;
        id
    }

    fn alloc_port(&mut self) -> u16 {
        let p = self.next_port;
        self.next_port += 1;
        p
    }
}

/// Builda (bloqueante) o grupo com exatamente `kernels` e sobe o runtime.
/// Só retorna Ok depois que a porta do novo processo já aceita conexão.
fn build_and_launch(
    state: &mut SysRuntime,
    kernels: &[u8],
    node_ip: &str,
) -> io::Result<(u32, Group)> {
    let group_id = state.alloc_group_id();
    let port = state.alloc_port();

    let mut sorted_kernels = kernels.to_vec();
    sorted_kernels.sort_unstable();

    let kernels_csv = sorted_kernels
        .iter()
        .map(|k| k.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let gpu_csv = sorted_kernels
        .iter()
        .filter(|k| GPU_CAPABLE_KERNELS.contains(k))
        .map(|k| k.to_string())
        .collect::<Vec<_>>()
        .join(",");

    let cwd = std::env::current_dir()?;
    let output_path = cwd.join(format!("kernel_{}_gpu.wasm", group_id));

    println!(
        "[BUILD] Grupo {} — kernels [{}] (gpu: [{}]) -> {}",
        group_id,
        kernels_csv,
        gpu_csv,
        output_path.display()
    );

    let mut cmd = Command::new("python3");
    cmd.current_dir(BUILDER_DIR)
        .arg(BUILDER_SCRIPT)
        .args(["--kernels", &kernels_csv])
        .args(["--component-dir", COMPONENT_DIR])
        .arg("--output")
        .arg(&output_path);
    if !gpu_csv.is_empty() {
        cmd.args(["--gpu", &gpu_csv]);
    }

    let status = cmd.status()?;
    if !status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "build_custom.py falhou (grupo {}, kernels {:?}): exit {:?}",
                group_id, sorted_kernels, status.code()
            ),
        ));
    }

    println!("[BUILD] Grupo {} compilado com sucesso.", group_id);

    // Copia o registro central de modelos (resnet/models.json) pra dentro
    // do cwd deste processo (sys_test/), FLAT, sem "..". O componente WASM
    // roda sob o sandbox WASI do wasi-gfx-runtime, que proíbe o guest de
    // escapar do diretório pré-aberto via ".." mesmo quando o arquivo real
    // existe no disco (achado em 28/10, 1ª execução real pós-rebuild: panic
    // "Erro ao abrir ../models.json ... PermissionDenied / Operation not
    // permitted" — ver claude/project-state.md). Refeito a cada grupo pra
    // sempre refletir o registro atual.
    let models_registry_src = cwd.join("../models.json");
    let models_registry_dst = cwd.join("models.json");
    std::fs::copy(&models_registry_src, &models_registry_dst).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!(
                "Erro copiando {:?} -> {:?} (registro de modelos pro cwd do grupo): {:?}",
                models_registry_src, models_registry_dst, e
            ),
        )
    })?;

    let child = Command::new(RUNTIME_BIN)
        .arg("--wasm")
        .arg(&output_path)
        .arg("--port")
        .arg(port.to_string())
        .arg("--ip")
        .arg(node_ip)
        .spawn()?;

    // Aguarda o runtime abrir a porta (mesmo padrão de polling já usado
    // antes em spawn_wasm_worker).
    let mut up = false;
    for _ in 0..200 {
        if TcpStream::connect(format!("{}:{}", node_ip, port)).is_ok() {
            up = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    if !up {
        eprintln!(
            "[AVISO] Grupo {} não respondeu em {}:{} depois de 2s — seguindo mesmo assim.",
            group_id, node_ip, port
        );
    }

    println!(
        "[GROUPS] Grupo {} no ar em {}:{} (kernels: {:?})",
        group_id, node_ip, port, sorted_kernels
    );

    Ok((
        group_id,
        Group {
            kernels: sorted_kernels.into_iter().collect(),
            port,
            child,
        },
    ))
}

fn kill_group(group_id: u32, mut g: Group) {
    let _ = g.child.kill();
    let _ = g.child.wait();
    println!(
        "[GROUPS] Grupo {} encerrado (kernels: {:?}, porta {}).",
        group_id, g.kernels, g.port
    );
}

/// Checa `/proc/net/tcp` e `/proc/net/tcp6` por alguma conexão em estado
/// ESTABLISHED (hex "01") com porta local igual a `port`. Não conta o
/// próprio socket de escuta do grupo (estado LISTEN, hex "0A", que
/// sempre aparece enquanto o processo está vivo) — só conexão de verdade
/// aceita, que é sinal de que tem (ou acabou de ter) um request em
/// andamento.
fn port_has_established_connection(port: u16) -> bool {
    const ESTABLISHED: &str = "01";
    for path in ["/proc/net/tcp", "/proc/net/tcp6"] {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        for line in content.lines().skip(1) {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 4 {
                continue;
            }
            let local_addr = fields[1];
            let state = fields[3];
            let Some(port_hex) = local_addr.rsplit(':').next() else {
                continue;
            };
            if let Ok(local_port) = u16::from_str_radix(port_hex, 16) {
                if local_port == port && state.eq_ignore_ascii_case(ESTABLISHED) {
                    return true;
                }
            }
        }
    }
    false
}

/// Espera o grupo "esvaziar" (nenhuma conexão ESTABLISHED na sua porta)
/// antes de matar o processo — proteção pra não derrubar um grupo que
/// ainda está no meio de uma inferência. Poll a cada
/// `DRAIN_POLL_INTERVAL_MS`; se estourar `DRAIN_MAX_WAIT` com conexão
/// ainda ativa, mata forçado mesmo assim (fallback de segurança, pra não
/// deixar processo zumbi preso pra sempre).
///
/// #### LIMITAÇÃO CONHECIDA: essa espera é bloqueante dentro do loop
/// principal do sys (mesmo padrão síncrono já usado em `build_and_launch`)
/// — telemetria e novas atualizações de `routing_table.json` não são
/// processadas enquanto um grupo está sendo drenado. Pode levar até
/// `DRAIN_MAX_WAIT` no pior caso.
fn drain_and_kill_group(group_id: u32, g: Group) {
    let port = g.port;
    let start = Instant::now();
    loop {
        if !port_has_established_connection(port) {
            println!(
                "[GROUPS] Grupo {} (porta {}) drenado — sem conexões ativas, encerrando.",
                group_id, port
            );
            break;
        }
        if start.elapsed() >= DRAIN_MAX_WAIT {
            eprintln!(
                "[AVISO] Grupo {} (porta {}) ainda com conexão ativa após {:?} — matando forçado (timeout de segurança).",
                group_id, port, DRAIN_MAX_WAIT
            );
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(DRAIN_POLL_INTERVAL_MS));
    }
    kill_group(group_id, g);
}

/// Aplica a Regra 1 / Regra 2, devolve a lista de funções ativas (pra
/// telemetria) e um booleano dizendo se algo de fato mudou nesta chamada
/// (kernel novo ou removido) — o chamador só precisa persistir
/// `routing_table.json` de volta quando esse booleano for `true`.
///
/// Recebe o `RoutingTable` inteiro (não só o mapa `table`) porque agora
/// esta função pode precisar persistir o arquivo ELA MESMA, antes de
/// terminar: quando um grupo vai ser substituído, o `table`/`routing_table.json`
/// é atualizado e gravado em disco com o endereço do grupo NOVO assim que
/// ele sobe — ANTES de drenar/matar o(s) grupo(s) antigo(s). Isso importa:
/// sem essa ordem, qualquer componente que leia `routing_table.json` pra
/// rotear um step continuaria mandando tráfego NOVO pro grupo antigo
/// durante toda a janela de dreno (até `DRAIN_MAX_WAIT`), e o dreno nunca
/// terminaria de esvaziar de verdade. Com a tabela já apontando pro grupo
/// novo, só sobra nos grupos antigos o tráfego que já estava em trânsito
/// antes da troca — que é exatamente o que `drain_and_kill_group` espera
/// escoar antes de matar.
fn reconcile_groups(
    state: &mut SysRuntime,
    wanted: &HashSet<u8>,
    node_ip: &str,
    routing_data: &mut RoutingTable,
) -> (Vec<FunctionInfo>, bool) {
    let covered = state.covered_kernels();
    let to_add: Vec<u8> = wanted.difference(&covered).cloned().collect();
    let to_remove: Vec<u8> = covered.difference(wanted).cloned().collect();
    let changed = !to_add.is_empty() || !to_remove.is_empty();

    // Grupos que vão ser mortos, SÓ depois de a tabela nova já estar
    // persistida (ver comentário acima) — preenchido pelas Regras 1/2
    // abaixo, dreno/morte acontece no fim desta função.
    let mut groups_to_drain: Vec<(u32, Group)> = Vec::new();

    if !to_remove.is_empty() {
        println!(
            "[GROUPS] Kernels removidos de assignments: {:?} — consolidando.",
            to_remove
        );

        if wanted.is_empty() {
            // Nada mais é necessário neste nó: todos viram candidatos a dreno.
            groups_to_drain.extend(state.groups.drain());
        } else {
            let consolidated: Vec<u8> = wanted.iter().cloned().collect();
            match build_and_launch(state, &consolidated, node_ip) {
                Ok((new_gid, new_group)) => {
                    groups_to_drain.extend(state.groups.drain());
                    state.groups.insert(new_gid, new_group);
                }
                Err(e) => {
                    eprintln!(
                        "[ERRO] Falha ao consolidar grupo: {}. Mantendo grupos antigos no ar.",
                        e
                    );
                }
            }
        }
    } else if !to_add.is_empty() {
        println!(
            "[GROUPS] Kernels novos em assignments: {:?} — build isolado.",
            to_add
        );
        match build_and_launch(state, &to_add, node_ip) {
            Ok((gid, new_group)) => {
                state.groups.insert(gid, new_group);
            }
            Err(e) => {
                eprintln!(
                    "[ERRO] Falha ao buildar grupo novo: {}. Kernels {:?} ficam sem cobertura por ora.",
                    e, to_add
                );
            }
        }
    }

    // `state.groups` já reflete o estado final (grupo novo incluso, grupos
    // substituídos já fora da struct, mesmo que o processo deles ainda
    // esteja de pé aguardando dreno) — reconstrói `table`/`functions` a
    // partir dele.
    for k in &to_remove {
        routing_data.table.remove(k);
    }
    let mut functions = Vec::new();
    for g in state.groups.values() {
        for &k in &g.kernels {
            let endpoint = format!("{}:{}", node_ip, g.port);
            routing_data.table.insert(k, endpoint.clone());
            functions.push(FunctionInfo { id: k, endpoint });
        }
    }

    // Persiste a tabela JÁ ATUALIZADA antes de drenar/matar os grupos
    // substituídos (ver docstring acima pra entender por que a ordem
    // importa). Se não há nada a drenar (só Regra 1, adição isolada), o
    // chamador ainda persiste normalmente depois do retorno — essa
    // gravação aqui é só pro caso de Regra 2.
    if changed && !groups_to_drain.is_empty() {
        if let Ok(json_data) = serde_json::to_string_pretty(&routing_data) {
            let _ = std::fs::write("routing_table.json", json_data);
            println!(
                "[GROUPS] Tabela persistida antes do dreno — tráfego novo já roteado pro grupo novo."
            );
        }
    }

    // Só agora, com a tabela já refletindo o estado novo, drena e mata os
    // grupos redundantes (ver drain_and_kill_group).
    for (gid, g) in groups_to_drain {
        drain_and_kill_group(gid, g);
    }

    (functions, changed)
}

fn collect_free_metrics(node_id: u8, sys: &mut System, networks: &mut sysinfo::Networks,
    functions: Vec<FunctionInfo>) -> SystemMetrics {
    sys.refresh_cpu_all();
    sys.refresh_memory();
    networks.refresh(true);

    let cpu_usage: f32 = sys.cpus().iter()
        .map(|cpu: &Cpu| cpu.cpu_usage())
        .sum::<f32>() / sys.cpus().len() as f32;

    let cpu_free = 100.0 - cpu_usage;
    let total_mem = sys.total_memory() as f32;
    let avail_mem = sys.available_memory() as f32;
    let mem_free = (avail_mem / total_mem) * 100.0;

    let health_score = cpu_free + mem_free;

    let total_rx = networks.into_iter()
        .map(|(_, data)| data.received())
        .sum();

    println!("Node ID: {} | Memory Free: {:.2}% | CPU Free: {:.2}% | Functions: {} | Health: {}", node_id, mem_free, cpu_free, functions.len(), health_score);

    SystemMetrics {
        node_id,
        cpu_free_pct: cpu_free,
        mem_free_pct: mem_free,
        health_score,
        net_incoming_bytes: total_rx,
        functions,
    }
}

fn send_to_orchestrator(metrics: SystemMetrics) {
    let addr = "0.0.0.0:9998"; // #### TODO: Porta onde o Orchestrador estará ouvindo telemetria. Colocar o Ip do nó onde esta o orquestrador

    match TcpStream::connect(addr) {
        Ok(mut stream) => {
            let payload = rmp_serde::to_vec(&metrics)
                .expect("Falha ao serializar telemetria");

            let len = (payload.len() as u32).to_be_bytes();

            if stream.write_all(&len).is_ok() && stream.write_all(&payload).is_ok() {
                let _ = stream.flush();
                println!("[TELEMETRIA] Dados enviados com sucesso para {}", addr);
            }
        }
        Err(_) => {
            eprintln!("[TELEMETRIA] Orquestrador em {} não respondeu (Offline).", addr);
        }
    }
}

fn handle_orchestrator(mut stream: TcpStream, state: &mut SysRuntime, node_id: u8, node_ip: &str) -> Result<Vec<FunctionInfo>, io::Error> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_be_bytes(len_buf) as usize;
    let mut buffer = vec![0u8; len];
    stream.read_exact(&mut buffer)?;

    let mut routing_data: RoutingTable = rmp_serde::from_slice(&buffer).expect("Falha ao deserializar");

    let wanted: HashSet<u8> = routing_data.assignments.get(&node_id).cloned().unwrap_or_default().into_iter().collect();
    let (functions, changed) = reconcile_groups(state, &wanted, node_ip, &mut routing_data);

    // Só reescreve o arquivo quando algo de fato mudou nos grupos deste
    // nó (kernel novo ou removido) — uma vez só, logo depois de subir/
    // consolidar, nunca em toda chamada só porque chegou uma RoutingTable
    // (que pode trazer os mesmos assignments de antes).
    if changed {
        let json_data = serde_json::to_string_pretty(&routing_data).unwrap();
        let mut file = File::create("routing_table.json")?;
        file.write_all(json_data.as_bytes())?;
        println!("[{}] Tabela recebida via rede, atualizada e salva.", Local::now().format("%H:%M:%S"));
    } else {
        println!("[{}] Tabela recebida via rede — sem mudança nos grupos deste nó, nada a escrever.", Local::now().format("%H:%M:%S"));
    }

    Ok(functions)
}

fn main() -> std::io::Result<()> {
    let listener = TcpListener::bind("0.0.0.0:9999")?; //#### TODO: colocar o ip da maquina local ou 0.0.0.0. é por onde o seu sys vai ouvir
    let my_node_id = 1; //#### TODO: Trocar o numero do nó
    let my_ip = "127.0.0.1"; //#### TODO: Colocar o numero do ip da maquina onde esta nó esta.

    let mut my_functions: Vec<FunctionInfo> = Vec::new();
    let mut state = SysRuntime::new();

    listener.set_nonblocking(true)?;
    let mut sys = System::new_all();
    let mut networks = sysinfo::Networks::new_with_refreshed_list();
    let mut last_telemetry = std::time::Instant::now();

    println!("SYS iniciado (composição dinâmica por grupo). Aguardando atualizações na porta 9999...");

    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                if let Ok(updated) = handle_orchestrator(stream, &mut state, my_node_id, my_ip) {
                    my_functions = updated;
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // Se não recebeu nada da rede, tenta sincronizar com o arquivo local
                if let Ok(mut file) = File::open("routing_table.json") {
                    let mut content = String::new();
                    if file.read_to_string(&mut content).is_ok() {
                        if let Ok(mut routing_data) = serde_json::from_str::<RoutingTable>(&content) {
                            let wanted: HashSet<u8> = routing_data.assignments.get(&my_node_id).cloned().unwrap_or_default().into_iter().collect();
                            let (functions, changed) = reconcile_groups(&mut state, &wanted, my_ip, &mut routing_data);
                            my_functions = functions;
                            // Só persiste de volta quando algo mudou nos
                            // grupos deste nó — senão este branch roda a cada
                            // ~50ms (todo tick do loop em que não chega nada
                            // pela rede) e ficaria reescrevendo o arquivo sem
                            // parar mesmo parado/estável.
                            if changed {
                                if let Ok(json_data) = serde_json::to_string_pretty(&routing_data) {
                                    let _ = std::fs::write("routing_table.json", json_data);
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => eprintln!("Erro na conexão: {:?}", e),
        }
        if last_telemetry.elapsed() >= std::time::Duration::from_secs(15) {
            println!("--- Coletando Telemetria ---");
            let metrics = collect_free_metrics(my_node_id, &mut sys, &mut networks, my_functions.clone());

            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("sys_hardware_metrics.jsonl")
            {
                let log_linha = serde_json::json!({
                    "node_id": my_node_id,
                    "cpu_free": format!("{:.2}", metrics.cpu_free_pct),
                    "mem_free": format!("{:.2}", metrics.mem_free_pct),
                    "health_score": format!("{:.2}", metrics.health_score),
                });

                if let Ok(texto) = serde_json::to_string(&log_linha) {
                    use std::io::Write;
                    let _ = writeln!(file, "{}", texto);
                }
            }

            send_to_orchestrator(metrics);

            last_telemetry = Instant::now();
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}
