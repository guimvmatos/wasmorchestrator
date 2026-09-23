/*
TODO LIST
1 - Definição de como ele receberá a informação de qual é o primeiro kernel. na pratica, queroq ue ele receba demanda, comunique o orquestrador, o orquestrador vai processar e enfim, fazer o trabalho dele e vai enviar só qual é o ip do primeiro kernel. Por hora, estou usando manual: manual_kernel_address
2 - 


*/


use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write, Read};
use std::net::{TcpStream, TcpListener};
use std::collections::HashMap;
use rmp_serde::{Deserializer, Serializer};
use std::fs::File;
use std::time::Instant;
use std::env;


#[derive(Serialize, Deserialize, Debug)]
struct Data {
    input: Vec<f32>,
    output: Vec<f32>,
    reply_to: String,
    kernels: Vec<u8>,
    current_kernel: u32,
    request: u32
}

#[derive(Serialize, Deserialize, Debug)]
struct ClientRequest {
    request_id: u32,
    model_id: u32,
    image_bytes: Vec<u8>,
}

#[derive(Deserialize, Debug)]
struct RoutingTable {
    table: HashMap<u8, String>,
    node_scores: HashMap<u8, f32>,
    assignments: HashMap<u8, Vec<u8>>,
}

#[derive(Deserialize, Debug)]
struct StepNode {
    step: u32,
    kernel_type: u8,
    #[serde(default)]
    next_steps: Vec<u32>,
}

#[derive(Deserialize, Debug)]
struct ModelGraph {
    model: String,
    steps: Vec<StepNode>,
}

//fn main() -> std::io::Result<()> {
fn handle_client(mut stream: TcpStream) -> std::io::Result<()> {
    let start_total = Instant::now(); // Time counter 1 measures 

    // 1. RECEBER O CABEÇALHO COM O TAMANHO DO PACOTE DO CLIENTE
    let mut len_buf = [0u8; 4];
    if let Err(e) = stream.read_exact(&mut len_buf) {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            return Ok(()); // Cliente desconectou normalmente
        }
        return Err(e);
    }

    let payload_len = u32::from_be_bytes(len_buf) as usize;
    println!("Recebendo requisição do cliente ({} bytes)...", payload_len);


    // 2. LER E DESSERIALIZAR O CLIENTREQUEST (MessagePack)
    let mut buffer = vec![0u8; payload_len];
    stream.read_exact(&mut buffer)?;

    let client_req: ClientRequest = rmp_serde::from_slice(&buffer)
        .expect("Falha ao desserializar o ClientRequest do MessagePack");

    println!(
        "Demanda recebida! Request ID: {}, Modelo Solicitado: {}",
        client_req.request_id, client_req.model_id
    );

    // 3. CONVERTER OS BYTES DA IMAGEM EM ELEMENTOS F32 (150.528 floats)
    let start_read = Instant::now();

    if client_req.image_bytes.len() % 4 != 0 {
        eprintln!("Erro: Tamanho de bytes da imagem incompatível com float32.");
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Corrupted binary payload",
        ));
    }

    let data_raw: Vec<f32> = unsafe {
        std::slice::from_raw_parts(
            client_req.image_bytes.as_ptr() as *const f32,
            client_req.image_bytes.len() / 4,
        )
        .to_vec()
    };

    let readduration_millis = start_read.elapsed().as_micros() as f64 / 1000.0;
    println!(
        "Payload convertido com sucesso em {} elementos f32 em {:.3}ms",
        data_raw.len(),
        readduration_millis
    );

    // 4. CONSULTAR O MODEL_GRAPH.JSON PARA OBTER O GRAFO DO MODELO
    let graph_file = std::fs::File::open("resnet18DFG.json").expect("Erro ao abrir resnet18DFG.json");
let graph: ModelGraph = serde_json::from_reader(graph_file).expect("Erro no parser do resnet18DFG.json");

    let first_step_node = graph.steps.first()
        .expect("Erro: Grafo do modelo está vazio sem steps");

    let first_step = first_step_node.step;
    let first_kernel_type = first_step_node.kernel_type;

    // 5. MONTAR A STRUCT DATA UNIVERSAL
    //let client_ip = stream.peer_addr()?.to_string();
    let client_ip_only = stream.peer_addr()?.ip();
    let reply_to_address = format!("{}:9000", client_ip_only);

    let data = Data {
        input: data_raw,
        output: vec![],
        reply_to: reply_to_address,
        kernels: vec![], // A rota agora é guiada pelos steps no model_graph.json
        current_kernel: first_step, // Carrega o step inicial (ex: 1)
        request: client_req.request_id,
    };

    println!(
        "Struct Data criada com sucesso para a requisição {}! Step inicial: {}, Tipo de Kernel: {}",
        data.request, data.current_kernel, first_kernel_type
    );

    // TODO: Consultar tabela de roteamento e despachar 'data' via MessagePack para o primeiro Kernel
    let total_time_ms = start_total.elapsed().as_micros() as f64 / 1000.0;
    println!("Processamento da demanda concluído localmente em {:.3}ms\n", total_time_ms);

    let route_file = std::fs::File::open("routing_table.json").expect("Erro ao abrir routing_table.json");
    let routing: RoutingTable = serde_json::from_reader(route_file).expect("Erro no parser de routing_table.json");

    // Se o Kernel 0 (All) estiver ativo, ele assume a execução completa; senão, despacha para o kernel individual
    let (target_kernel_id, target_address) = if let Some(addr) = routing.table.get(&0) {
        (0u8, addr.clone())
    } else {
        let addr = routing.table.get(&first_kernel_type)
            .cloned()
            .unwrap_or_else(|| panic!("Tipo de Kernel {} não encontrado na routing_table.json", first_kernel_type));
        (first_kernel_type, addr)
    };

    println!("Despachando demanda para o Kernel Tipo {} em {}...", target_kernel_id, target_address);

    let start_serialize = Instant::now();
    let serialized_data = rmp_serde::to_vec(&data)
        .expect("Falha ao serializar a struct Data via MessagePack");
    let serialize_ms = start_serialize.elapsed().as_micros() as f64 / 1000.0;

    let start_send = Instant::now();
    let mut target_stream = TcpStream::connect(target_address)?;

    let payload_len = serialized_data.len() as u32;
    target_stream.write_all(&payload_len.to_be_bytes())?;
    target_stream.write_all(&serialized_data)?;
    target_stream.flush()?;

    // 🤝 AGUARDA O ACK DO RECEIVER (Mesmo mecanismo de segurança)
    let mut ack_buf = [0u8; 1];
    let _ = target_stream.read(&mut ack_buf);

    let send_ms = start_send.elapsed().as_micros() as f64 / 1000.0;
    let total_time_ms = start_total.elapsed().as_micros() as f64 / 1000.0;

    println!(
        "Data despachado com sucesso! ({} bytes | Serialização: {:.3}ms | Envio: {:.3}ms | Total Handler: {:.3}ms)\n",
        payload_len, serialize_ms, send_ms, total_time_ms
    );



    Ok(())
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let port = args.get(1).map(|s| s.as_str()).unwrap_or("8090");
    let ip = args.get(2).map(|s| s.as_str()).unwrap_or("0.0.0.0");

    let bind_addr = format!("{}:{}", ip, port);
    let listener = TcpListener::bind(&bind_addr)?;

    println!("Servidor Receptor iniciado e ouvindo em {}", bind_addr);

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                println!("New connection received!");
                if let Err(e) = handle_client(stream) {
                    eprintln!("Error handling client: {:?}", e);
                }
            }
            Err(e) => {
                eprintln!("Connection failed: {:?}", e);
            }
        }
        println!("Waiting for new incoming requests...");
    }

    Ok(())
}