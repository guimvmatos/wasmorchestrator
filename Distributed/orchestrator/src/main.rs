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
    assignments: HashMap<u8, Vec<u8>>,
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

fn deploy_demand(
    nodes_health: &HashMap<String, NodeStatus>, 
    demand: &Vec<u8> 
) -> (HashMap<u8, String>, HashMap<u8, f32>, HashMap<u8, Vec<u8>>) {
    let mut new_table = HashMap::new();
    let mut current_scores = HashMap::new();
    let mut assignments: HashMap<u8, Vec<u8>> = HashMap::new();

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
        if let Some(best_node) = sorted_nodes.first() {
            //let node_id = best_node.metrics.node_id;
            let node_id = best_node.node_id;

            let tasks = assignments.entry(node_id).or_insert(vec![]);
                if !tasks.contains(&func_id) {
                    tasks.push(func_id);
                }
        }
    }
    (new_table, current_scores, assignments)
}

fn decide_routing(
    nodes_health: &HashMap<String, NodeStatus>, 
    demand: &Vec<u8> 
) -> (HashMap<u8, String>, HashMap<u8, f32>, HashMap<u8, Vec<u8>>) {
    let mut new_table = HashMap::new();
    let mut current_scores = HashMap::new();
    let mut assignments: HashMap<u8, Vec<u8>> = HashMap::new();

    let mut sorted_nodes: Vec<SystemMetrics> = nodes_health.values()
        .map(|n| {
            current_scores.insert(n.metrics.node_id, n.metrics.health_score);
            n.metrics.clone()
        })
        .collect();
    
    sorted_nodes.sort_by(|a, b| b.health_score.partial_cmp(&a.health_score).unwrap_or(std::cmp::Ordering::Equal));

    for &func_id in demand {
        let mut found = false;

        for node in &sorted_nodes {
            if let Some(f) = node.functions.iter().find(|f| f.id == func_id) {
                let score = node.health_score;
                let node_id = node.node_id;

                if score >= 190.0 {
                    new_table.insert(func_id, f.endpoint.clone());
                    assignments.entry(node_id).or_insert(vec![]).push(func_id);
                    found = true;
                    break;
                } else if score >= 180.0 {
                    //let max_id = node.functions.iter().map(|f| f.id).max();
                    let max_id = node.functions.iter().map(|f| f.id).max().unwrap_or(0);
                    
                    //if let Some(m_id) = max_id {
                        if func_id == max_id && sorted_nodes.len() > 1 { 
                            continue; 
                        }
                    //}
                    
                    new_table.insert(func_id, f.endpoint.clone());
                    assignments.entry(node_id).or_insert(vec![]).push(func_id);
                    found = true;
                    break;
                }
            }
        }

        if !found {
            if let Some(best_node) = sorted_nodes.first() {
                let node_id = best_node.node_id;
                // Se cair aqui, o Orquestrador manda o melhor nó iniciar a função
                let tasks = assignments.entry(node_id).or_insert(vec![]);
                if !tasks.contains(&func_id) {
                    tasks.push(func_id);
                }
            }
        }
    }

    (new_table, current_scores, assignments)
}


fn main() -> std::io::Result<()> {
    let mut routing_table = RoutingTable {
        table: HashMap::new(),
        node_scores: HashMap::new(),
        assignments: HashMap::new(),
    };

    let mut nodes_health: HashMap<String, NodeStatus> = HashMap::new();

    let listener = TcpListener::bind("10.68.119.168:9998")?; //#### TODO Ip do orquestrador
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
        if last_table_update.elapsed() >= Duration::from_secs(15) {
            println!("--- Ciclo de Atualização de Tabela ---");

            nodes_health.retain(|ip, status| {
                if status.last_seen.elapsed() > Duration::from_secs(30) {
                    println!("[AVISO] Nó {} offline. Removendo do inventário.", ip);
                    false
                } else {
                    true
                }
            });

            let current_demand = vec![1,2,3]; 

            let (new_table, scores, new_assignments) = decide_routing(&nodes_health, &current_demand);

            routing_table.table = new_table;
            routing_table.node_scores = scores;
            routing_table.assignments = new_assignments;

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