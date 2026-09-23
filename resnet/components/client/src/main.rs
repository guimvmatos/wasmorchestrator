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

fn process_top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exps: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| (e / sum_exps) * 100.0).collect();

    let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).collect()
}

fn main() -> std::io::Result<()> {
    let start_total = Instant::now(); // Time counter 1 measures 
    // 1. Initiation parameters capture
    
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 4 {
        eprintln!("Correct usage: {} <id_request> <model> <input_image.bin>", args[0]);
        std::process::exit(1);
    }

    let request_id: u32 = args[1].parse().unwrap_or_else(|_| {
        eprintln!("Error: request's ID must be a valid integer (u32).");
        std::process::exit(1);
    });
    let model_id: u32 = args[2].parse().unwrap_or_else(|_| {
        eprintln!("Error: Model's number must be a valid integer (u32).");
        std::process::exit(1);
    });

    let path = &args[3];

    let server_address = args.get(4).map(|s| s.as_str()).unwrap_or("127.0.0.1:8090");
    let listen_address = "0.0.0.0:9000"; // Porta onde o cliente aguarda o retorno final
    
    let start_read = Instant::now(); // Time counter 2 measures time to reads the input
    
    //LOADING IMAGE
    let path = "input_image.bin"; // needs to be removed in the future
    println!("Loading {}...", path);

    let mut file = std::fs::File::open(path)?;
    let mut image_bytes = Vec::new();
    file.read_to_end(&mut image_bytes)?;
    
    let readduration_millis = start_read.elapsed().as_micros() as f64 / 1000.0;

    if image_bytes.len() % 4 != 0 {
        eprintln!("Error: Binary file corrupted or size incompatible with float32.");
        std::process::exit(1);
    }
    
    let client_req = ClientRequest {
        request_id,
        model_id,
        image_bytes,
    };

    let start_serialize = Instant::now();
    let serialized_payload =
        rmp_serde::to_vec(&client_req).expect("Failed to serialize ClientRequest via MessagePack");
    let serialize_millis = start_serialize.elapsed().as_micros() as f64 / 1000.0;

    // 4. ENVIO PARA O SERVIDOR RECEPTOR
    let start_send = Instant::now();
    println!("Connecting to server at {}...", server_address);
    let mut stream = TcpStream::connect(server_address)?;

    let payload_len = serialized_payload.len() as u32;
    stream.write_all(&payload_len.to_be_bytes())?;
    stream.write_all(&serialized_payload)?;
    stream.flush()?;

    let send_millis = start_send.elapsed().as_micros() as f64 / 1000.0;
    println!(
        "Payload sent successfully! ({} bytes | Send time: {:.3}ms)",
        payload_len, send_millis
    );

    // Fecha a conexão de envio
    drop(stream);

    // 5. ESCUTA A RESPOSTA FINAL DA PIPELINE (RETORNO DO ÚLTIMO KERNEL)
    println!("Waiting for response pipeline on {}...", listen_address);
    let listener = TcpListener::bind(listen_address)?;

    let start_exec = Instant::now();
    let (mut response_stream, peer_addr) = listener.accept()?;
    println!("Incoming response from: {}", peer_addr);

    let mut len_buf = [0u8; 4];
    response_stream.read_exact(&mut len_buf)?;
    let response_len = u32::from_be_bytes(len_buf) as usize;

    let mut response_payload = vec![0u8; response_len];
    response_stream.read_exact(&mut response_payload)?;

    let exec_millis = start_exec.elapsed().as_micros() as f64 / 1000.0;

    // 6. DESSERIALIZAÇÃO DO OBJETO DATA PROCESSADO
    let start_deserialize = Instant::now();
    let result: Data = rmp_serde::from_slice(&response_payload)
        .expect("Failed to deserialize Data response from pipeline");
    let deserialize_millis = start_deserialize.elapsed().as_micros() as f64 / 1000.0;

    let total_client_time = start_total.elapsed().as_secs_f64() * 1000.0;

    println!("\n=== INFERENCE RESULT ===");
    println!("Request ID: {}", result.request);
    println!("Output elements count: {}", result.input.len());
    println!("Total client latency: {:.3}ms", total_client_time);

    let top5 = process_top_k(&result.input, 5);
    println!("\n--- TOP-5 PREDICTIONS ---");
    for (rank, (class_idx, prob)) in top5.iter().enumerate() {
        println!("  #{}: Class {} -> {:.2}%", rank + 1, class_idx, prob);
    }
    println!("-------------------------\n");

    // Salva os floats puros em formato binário para bater direto com o output_0.pb
    let raw_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            result.input.as_ptr() as *const u8,
            result.input.len() * std::mem::size_of::<f32>(),
        )
    };
    let _ = std::fs::write("output_received.bin", raw_bytes);

    // 7. REGISTRO DE LOGS EM client_logs.jsonl
    if let Ok(mut log_file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("client_logs.jsonl")
    {
        use std::io::Write as IoWrite;
        let log_entry = serde_json::json!({
            "request": result.request,
            "top1_class": top5.first().map(|(c, _)| *c),
            "top1_prob": top5.first().map(|(_, p)| *p),
            "output_logits": result.input,
            "read_time_ms": readduration_millis,
            "serialize_time_ms": serialize_millis,
            "send_time_ms": send_millis,
            "exec_time_ms": exec_millis,
            "deserialize_time_ms": deserialize_millis,
            "total_client_time_ms": total_client_time,
            "pipeline_demanda": result.kernels
        });

        if let Ok(text) = serde_json::to_string(&log_entry) {
            let _ = writeln!(log_file, "{}", text);
            println!("Log and raw logits saved into client_logs.jsonl");
        }
    }

    Ok(())
}