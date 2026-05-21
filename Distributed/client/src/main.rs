use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write, Read};
use std::net::{TcpStream, TcpListener};
use std::collections::HashMap;
use rmp_serde::{Deserializer, Serializer};
use std::fs::File;
use std::time::Instant;


#[derive(Serialize, Deserialize, Debug)]
struct ImageData {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    reply_to: String,
    kernels: Vec<u8>,
    current_kernel: u32
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

    Ok(ImageData { width, height, pixels, reply_to: String::new(), kernels: Vec::new(), current_kernel: 1})
}

fn save_ppm(path: &str, img: &ImageData) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    
    write!(file, "P6\n{} {}\n255\n", img.width, img.height)?;
    
    file.write_all(&img.pixels)?;
    
    println!("Imagem salva com sucesso em: {}", path);
    Ok(())
}

fn main() -> std::io::Result<()> {
    let path = "image.ppm";
    println!("Carregando {}...", path);
    //let data = read_ppm(path).expect("Erro ao ler o arquivo PPM");
    let data_raw = read_ppm(path).expect("Erro ao ler PPM");

    
    let data = ImageData {
        reply_to: "10.68.119.168:9000".to_string(),
        kernels: vec![1],
        current_kernel: 1,
        ..data_raw
    };
    
    let mut stream = TcpStream::connect("10.68.119.168:8081")?;  

    let serialized_msgpack = rmp_serde::to_vec(&data).expect("MSGPACK Serialization failed");
    let len = serialized_msgpack.len() as u32;
    stream.write_all(&len.to_be_bytes())?; 

    stream.write_all(&serialized_msgpack)?;
    stream.flush()?;
    let start_exec = Instant::now();

    println!("Enviado: {} bytes de payload.", len);

    drop(stream);
    //===============
    let listener = TcpListener::bind("10.68.119.168:9000")?;
    let (mut stream_resposta, addr) = listener.accept()?; 
    println!("Conexão de resposta vinda de: {}", addr);

    let mut len_buf = [0u8; 4];
    stream_resposta.read_exact(&mut len_buf).expect("Failed to read size header");

    let response_len = u32::from_be_bytes(len_buf) as usize;

    let mut response_payload = vec![0u8; response_len];
    
    //stream.read_exact(&mut response_payload).expect("Failed to read payload");
    stream_resposta.read_exact(&mut response_payload).expect("Failed to read payload");

    let result: ImageData = rmp_serde::from_slice(&response_payload).expect("Failed to deserialize MessagePack response");
    
    let duration_micros = start_exec.elapsed().as_micros(); 
    let duration_millis = duration_micros as f64 / 1000.0;
    
    println!("Sucesso! Recebida imagem de {}x{} Tempo: {}µs ({:.3}ms", result.width, result.height, duration_micros, duration_millis);

    save_ppm("resultado.ppm", &result).expect("Erro ao salvar o arquivo de saída");
    
    Ok(())
}