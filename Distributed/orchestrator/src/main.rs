use std::collections::HashMap;
use std::net::TcpStream;
use std::io::{Write, Read};
use std::thread;
use std::time::Duration;
use serde::{Serialize, Deserialize};
use std::net::TcpListener;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct RoutingTable {
    // ID do Kernel (1, 2, 3) -> Endereço (IP:Porta)
    table: HashMap<u8, String>,
    node_scores: HashMap<u8, f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct FunctionInfo {
    id: u8,
    endpoint: String, // Agora aceita "127.0.0.1:8081"
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct SystemMetrics {
    node_id: u8,
    cpu_free_pct: f32,
    mem_free_pct: f32,
    health_score: f32,
    net_incoming_bytes: u64,
    functions: Vec<FunctionInfo>,
}

struct NodeStatus {
    metrics: SystemMetrics,
    last_seen: std::time::Instant,
}

fn decide_routing(
    nodes_health: &HashMap<String, NodeStatus>, 
    demand: &Vec<u8> 
) -> (HashMap<u8, String>, HashMap<u8, f32>) {
    let mut new_table = HashMap::new();
    let mut current_scores = HashMap::new();

    // 1. Coleta e Ordena
    let mut sorted_nodes: Vec<SystemMetrics> = nodes_health.values()
        .map(|n| {
            // Aproveitamos para salvar o score de cada nó conhecido
            current_scores.insert(n.metrics.node_id, n.metrics.health_score);
            n.metrics.clone()
        })
        .collect();
    
    sorted_nodes.sort_by(|a, b| b.health_score.partial_cmp(&a.health_score).unwrap_or(std::cmp::Ordering::Equal));

    // 2. Aplica a Regra 190/180    
    for &func_id in demand {
        for node in &sorted_nodes {
            if let Some(f) = node.functions.iter().find(|f| f.id == func_id) {
                let score = node.health_score;
                
                if score >= 190.0 {
                    new_table.insert(func_id, f.endpoint.clone());
                    break;
                } else if score >= 180.0 {
                    let max_id = node.functions.iter().map(|f| f.id).max().unwrap_or(0);
                    if func_id == max_id { continue; } 
                    
                    new_table.insert(func_id, f.endpoint.clone());
                    break;
                }
            }
        }
    }
    (new_table, current_scores)
}



fn main() -> std::io::Result<()> {
    // 1. Inicializa a tabela com os valores padrão
    let mut routing_table = RoutingTable {
        table: HashMap::from([
            (1, "127.0.0.1:8081".to_string()), // Grayscale
            (2, "127.0.0.1:8082".to_string()), // Sobel
            (3, "127.0.0.1:8083".to_string()), // Negative
        ]),
        // Adicionando o campo que estava faltando para o compilador aceitar
        node_scores: HashMap::new(), 
    };

    // 1. O "Banco de Dados" em memória do AGORA
    let mut nodes_health: HashMap<String, NodeStatus> = HashMap::new();

    // 2. Listener de Telemetria (Non-blocking)
    let listener = TcpListener::bind("10.68.119.168:9998")?; //#### Ip do orquestrador
    listener.set_nonblocking(true)?;

    let mut last_table_update = std::time::Instant::now();

    println!("Orquestrador iniciado...");
    println!(" -> Porta 9998: Recebendo Telemetria");
    println!(" -> Porta 9999: Enviando Tabelas (a cada 15s)");


    loop {
        // --- PARTE A: RECEBER TELEMETRIA (Ouvinte) ---
        match listener.accept() {
            Ok((mut stream, addr)) => {
                let ip = addr.ip().to_string();
                let mut len_buf = [0u8; 4];
                if stream.read_exact(&mut len_buf).is_ok() {
                    let len = u32::from_be_bytes(len_buf) as usize;
                    let mut buffer = vec![0u8; len];
                    if stream.read_exact(&mut buffer).is_ok() {
                        if let Ok(metrics) = rmp_serde::from_slice::<SystemMetrics>(&buffer) {
                            // ATUALIZA O AGORA:
                            nodes_health.insert(ip.clone(), NodeStatus {
                                metrics: metrics.clone(),
                                last_seen: std::time::Instant::now(),
                            });
                            println!("[TELEMETRIA] Nó {} (ID: {}) Score: {:.2}", ip, metrics.node_id, metrics.health_score);
                        }
                    }
                }
            }
            _ => {}
        }

        // --- PARTE B: LÓGICA DE DECISÃO E ENVIO ---
        // --- PARTE B: LÓGICA DE DECISÃO E ENVIO ---
        if last_table_update.elapsed() >= Duration::from_secs(15) {
            println!("--- Ciclo de Atualização de Tabela ---");

            // 1. Limpeza de "Zumbis": Se o nó sumiu há mais de 30s, tiramos do inventário
            nodes_health.retain(|ip, status| {
                if status.last_seen.elapsed() > Duration::from_secs(30) {
                    println!("[AVISO] Nó {} offline. Removendo do inventário.", ip);
                    false
                } else {
                    true
                }
            });

            // 2. Definimos a demanda (O que queremos que aconteça no cluster)
            let current_demand = vec![1,2,3]; 

            // 3. Calculamos QUEM faz O QUE baseado no 190/180
            let (new_table, scores) = decide_routing(&nodes_health, &current_demand);

            routing_table.table = new_table;
            routing_table.node_scores = scores;

            println!("Demanda {:?} -> Configuração: {:?}", current_demand, routing_table.table);

            // 4. ENVIO PARA TODOS: Percorre o inventário de IPs conhecidos
            for ip in nodes_health.keys() {
                let target = format!("{}:9999", ip);
                match TcpStream::connect(&target) {
                    Ok(mut stream) => {
                        let payload = rmp_serde::to_vec(&routing_table).expect("Erro ao serializar");
                        let _ = stream.write_all(&(payload.len() as u32).to_be_bytes());
                        let _ = stream.write_all(&payload);
                        println!("Tabela sincronizada com o SYS em {}", target);
                    }
                    Err(_) => {
                        eprintln!("[ERRO] Não foi possível sincronizar com o nó {}", target);
                    }
                }
            }

            last_table_update = std::time::Instant::now();
        }
        thread::sleep(Duration::from_millis(10));
    }
}