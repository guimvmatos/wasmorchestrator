use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
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

//fn spawn_wasm_worker(func_id: u8, port: u16) -> Child {
fn spawn_wasm_worker(func_id: u8, port: u16, node_ip: &str) -> Child {
    let wasm_file = format!("kernel_{}.wasm", func_id);

    Command::new("wasmtime")
        .arg("run")
        .args(["--wasi", "inherit-network", "--dir", "."])
        .arg(&wasm_file)
        .arg(port.to_string())
        .arg(node_ip)
        .spawn()
        .expect("Falha ao iniciar o worker WASM")
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
    //let mem_free = sys.available_memory() / 1024 / 1024;
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
    let addr = "10.68.119.168:9998"; // #### TODO: Porta onde o Orchestrador estará ouvindo telemetria. Colocar o Ip do nó onde esta o orquestrador

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

fn handle_orchestrator(mut stream: TcpStream, active_workers: &mut HashMap<u8, Child>, node_id: u8, node_ip: &str) -> Result<Vec<FunctionInfo>, std::io::Error> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_be_bytes(len_buf) as usize;
    let mut buffer = vec![0u8; len];
    stream.read_exact(&mut buffer)?;

    let routing_data: RoutingTable = rmp_serde::from_slice(&buffer).expect("Falha ao deserializar");

    // Salva no JSON para persistência/configuração manual
    let json_data = serde_json::to_string_pretty(&routing_data).unwrap();
    let mut file = File::create("routing_table.json")?;
    file.write_all(json_data.as_bytes())?;

    println!("[{}] Tabela recebida via rede e salva.", Local::now().format("%H:%M:%S"));
    Ok(sync_workers(&routing_data, active_workers, node_id, node_ip))
}

fn sync_workers(routing_data: &RoutingTable, active_workers: &mut HashMap<u8, Child>, node_id: u8, node_ip: &str) -> Vec<FunctionInfo> {
    let mut current_functions = Vec::new();

    // 1. Iniciar o que está assinado para mim e não está rodando
    if let Some(my_tasks) = routing_data.assignments.get(&node_id) {
        for &func_id in my_tasks {
            if !active_workers.contains_key(&func_id) {
                let port = match func_id { 1 => 8081, 2 => 8082, 3 => 8083, _ => 8080 };
                let start_exec = Instant::now();
                let child = spawn_wasm_worker(func_id, port, node_ip);

                // Aguarda o worker subir
                for _ in 0..100 {
                    if TcpStream::connect(format!("{}:{}", node_ip, port)).is_ok() { break; }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }

                let duration_micros = start_exec.elapsed().as_micros();
                active_workers.insert(func_id, child);
                println!("[WATCHDOG] Kernel {} iniciado (Porta {}) Tempo: {}µs", func_id, port, duration_micros);
            }
        }
    }

    // 2. Matar o que está rodando mas não está mais nos assignments
    let to_kill: Vec<u8> = active_workers.keys()
        .filter(|id| !routing_data.assignments.get(&node_id).map_or(true, |tasks| tasks.contains(id)))
        .cloned().collect();

    for id in to_kill {
        if let Some(mut child) = active_workers.remove(&id) {
            let _ = child.kill();
            let _ = child.wait();
            println!("[WATCHDOG] Kernel {} encerrado por falta de demanda.", id);
        }
    }

    // 3. Atualizar lista de funções ativas para telemetria
    for &id in active_workers.keys() {
        let port = match id { 1 => 8081, 2 => 8082, 3 => 8083, _ => 8080 };
        current_functions.push(FunctionInfo { id, endpoint: format!("{}:{}", node_ip, port) });
    }
    current_functions
}

fn main() -> std::io::Result<()> {
    let listener = TcpListener::bind("10.68.119.168:9999")?; //#### TODO: colocar o ip da maquina local ou 0.0.0.0. é por onde o seu sys vai ouvir
    let my_node_id = 1; //#### TODO: Trocar o numero do nó
    let my_ip = "10.68.119.168"; //#### TODO: Colocar o numero do ip da maquina onde esta nó esta.

    let mut my_functions: Vec<FunctionInfo> = Vec::new();

    listener.set_nonblocking(true)?;
    let mut sys = System::new_all();
    let mut networks = sysinfo::Networks::new_with_refreshed_list(); // Adicione esta linha
    let mut last_telemetry = std::time::Instant::now();
    let mut active_workers: HashMap<u8, Child> = HashMap::new(); //(port>processo)

    println!("SYS iniciado. Aguardando atualizações na porta 9999...");

    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                if let Ok(updated) = handle_orchestrator(stream, &mut active_workers, my_node_id, my_ip) {
                    my_functions = updated;
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // Se não recebeu nada da rede, tenta sincronizar com o arquivo local
                if let Ok(mut file) = File::open("routing_table.json") {
                    let mut content = String::new();
                    if file.read_to_string(&mut content).is_ok() {
                        if let Ok(routing_data) = serde_json::from_str::<RoutingTable>(&content) {
                            my_functions = sync_workers(&routing_data, &mut active_workers, my_node_id, my_ip);
                        }
                    }
                }
            }
            Err(e) => eprintln!("Erro na conexão: {:?}", e),
        }
        if last_telemetry.elapsed() >= std::time::Duration::from_secs(15) {
            println!("--- Coletando Telemetria ---");
            let metrics = collect_free_metrics(my_node_id, &mut sys, &mut networks, my_functions.clone());

            // ====================================================================
            // NOVO: SALVA AS MÉTRICAS DE HARDWARE EM JSON LINES DINÂMICO
            // ====================================================================
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("sys_hardware_metrics.jsonl") 
            {
                let log_linha = serde_json::json!({
                    "node_id": my_node_id,
                    //"cpu_free": metrics.cpu_free_pct,       // TODO: Confirme se o campo na sua struct chama 'cpu_free'
                    //"mem_free": metrics.mem_free_pct,       // TODO: Confirme se o campo na sua struct chama 'mem_free'
                    //"health_score": metrics.health_score // TODO: Confirme se o campo na sua struct chama 'health_score'
                    "cpu_free": (metrics.cpu_free_pct * 100.0).round() / 100.0,       
                    "mem_free": (metrics.mem_free_pct * 100.0).round() / 100.0,       
                    "health_score": (metrics.health_score * 100.0).round() / 100.0,
                });

                if let Ok(texto) = serde_json::to_string(&log_linha) {
                    use std::io::Write; // Garante o trait Write no escopo local
                    let _ = writeln!(file, "{}", texto);
                }
            }
            // ====================================================================

            send_to_orchestrator(metrics);

            last_telemetry = std::time::Instant::now();
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    Ok(())
}
