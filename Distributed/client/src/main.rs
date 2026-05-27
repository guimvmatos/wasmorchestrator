use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write, Read};
use std::net::{TcpStream, TcpListener};
use std::collections::HashMap;
use rmp_serde::{Deserializer, Serializer};
use std::fs::File;
use std::time::Instant;
use std::env;


#[derive(Serialize, Deserialize, Debug)]
struct ImageData {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    reply_to: String,
    kernels: Vec<u8>,
    current_kernel: u32,
    request: u32
}

fn read_ppm(path: &str) -> std::io::Result<ImageData> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();

    reader.read_line(&mut line)?; // P6

    line.clear();
    while reader.read_line(&mut line)? > 0 {
        if !line.trim().starts_with('#') && !line.trim().is_empty() {
            break;
        }
        line.clear();
    }

    let dims: Vec<u32> = line.split_whitespace().map(|s| s.parse().unwrap()).collect();
    let (width, height) = (dims[0], dims[1]);

    line.clear();
    reader.read_line(&mut line)?; // MaxVal (255)

    let mut pixels = vec![0u8; (width * height * 3) as usize];
    reader.read_exact(&mut pixels)?;

    Ok(ImageData { width, height, pixels, reply_to: String::new(), kernels: Vec::new(), current_kernel: 1, request: 0})
}

fn save_ppm(path: &str, img: &ImageData) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    
    write!(file, "P6\n{} {}\n255\n", img.width, img.height)?;
    
    file.write_all(&img.pixels)?;
    
    println!("Imagem salva com sucesso em: {}", path);
    Ok(())
}

fn main() -> std::io::Result<()> {
    // 1. CAPTURA DOS PARÂMETROS DE INICIALIZAÇÃO
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 3 {
        eprintln!("Erro: Parâmetros insuficientes.");
        eprintln!("Uso correto: {} <id_do_request> <caminho_da_imagem.ppm>", args[0]);
        std::process::exit(1);
    }

    // Faz o parse do argumento string para u32
    let request_id: u32 = args[1].parse().unwrap_or_else(|_| {
        eprintln!("Erro: O ID do request precisa ser um número inteiro válido (u32).");
        std::process::exit(1);
    });

    let path = &args[2];


    //LOADING IMAGE
    //let path = "image.ppm";
    println!("Carregando {}...", path);
    //let data = read_ppm(path).expect("Erro ao ler o arquivo PPM");
    let start_read = Instant::now();
    let data_raw = read_ppm(path).expect("Erro ao ler PPM");
    let readduration_micros = start_read.elapsed().as_micros(); 
    let readduration_millis = readduration_micros as f64 / 1000.0;
    

    let data = ImageData {
        reply_to: "10.68.119.168:9000".to_string(), //TODO CLIENT'S IP
        kernels: vec![1], //TODO KERNELS TO PROCESS
        current_kernel: 1,
        request: request_id,
        ..data_raw
    };
    
    //IMAGE SERIALIZATION
    let start_serialize = Instant::now();
    let serialized_msgpack = rmp_serde::to_vec(&data).expect("MSGPACK Serialization failed");
    let serializeduration_micros = start_serialize.elapsed().as_micros(); 
    let serializeduration_millis = serializeduration_micros as f64 / 1000.0;

    //IMAGE SEND
    let start_send = Instant::now(); //time to send begin
    let mut stream = TcpStream::connect("10.68.119.168:8081")?;  //TODO FIRST KERNEL IP

    let len = serialized_msgpack.len() as u32;
    stream.write_all(&len.to_be_bytes())?; 

    stream.write_all(&serialized_msgpack)?;
    stream.flush()?;

    let sendduration_micros = start_send.elapsed().as_micros(); 
    let sendduration_millis = sendduration_micros as f64 / 1000.0;
    

    println!("Enviado: {} bytes de payload.", len);
    
    //PIPELINE... (TIME TO PROCESS KERNELS AND RECEIVE AN OUTPUT)
    let start_exec = Instant::now(); //time after send until receives all data back
    drop(stream);
    let listener = TcpListener::bind("10.68.119.168:9000")?; //TODO CLIENT'S IP

    let (mut stream_resposta, addr) = listener.accept()?; 
    println!("Conexão de resposta vinda de: {}", addr);

    
    let mut len_buf = [0u8; 4];
    stream_resposta.read_exact(&mut len_buf).expect("Failed to read size header");
    let response_len = u32::from_be_bytes(len_buf) as usize;

    let mut response_payload = vec![0u8; response_len];
    stream_resposta.read_exact(&mut response_payload).expect("Failed to read payload");
    drop(stream_resposta);
    let duration_micros = start_exec.elapsed().as_micros(); 
    let duration_millis = duration_micros as f64 / 1000.0;


    //IMAGEM DESERIALIZATION
    let start_deserialize = Instant::now(); //time after send until receives all data back
    let result: ImageData = rmp_serde::from_slice(&response_payload).expect("Failed to deserialize MessagePack response");
    let deserializeduration_micros = start_deserialize.elapsed().as_micros(); 
    let deserializeduration_millis = deserializeduration_micros as f64 / 1000.0;
    
    
    println!("Sucess! Image received: {}x{} | Time to process: {}µs ({:.3}ms) | Time to send: {}µs ({:.3}ms)", result.width, result.height, duration_micros, duration_millis, sendduration_micros, sendduration_millis);

    save_ppm("resultado.ppm", &result).expect("Erro ao salvar o arquivo de saída");
    if let Ok(mut file) = std::fs::OpenOptions::new()
    .create(true)
    .append(true)
    .open("client_logs.jsonl") 
    {
        let log_linha = serde_json::json!({
            "request": request_id,
            "sla_ms": 15,
            "img_width": result.width,
            "img_height": result.height,
            "read_time_ms": readduration_millis,
            "serialize_time_ms": serializeduration_millis,
            "send_time_ms": sendduration_millis,
            "exec_time_ms": duration_millis,
            "deserialize_time_ms": deserializeduration_millis,
            "pipeline_demanda": result.kernels 
        });

        if let Ok(texto) = serde_json::to_string(&log_linha) {
            let _ = writeln!(file, "{}", texto);
            println!("Log da execução {} salvo com sucesso em client_logs.jsonl!", result.request);
        }
    }
    
    Ok(())
}