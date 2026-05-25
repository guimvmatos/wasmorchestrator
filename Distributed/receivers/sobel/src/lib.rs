use clap::Parser;
use std::io::{BufRead, BufReader, Write, Read};
use std::net::{TcpListener, TcpStream, Shutdown};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use std::env;
use std::collections::HashMap;

type WitImagedata = bindings::planner::sobelworld::plan::Imagedata;

mod bindings {
    use super::Component;
    wit_bindgen::generate!({
        generate_all,
        additional_derives: [
            serde::Deserialize,
            serde::Serialize,
        ],
    });
    export!(Component);
}

struct Component;

#[derive(Deserialize)]
struct RoutingTable {
    table: std::collections::HashMap<u8, String>,
    node_scores: std::collections::HashMap<u8, f32>,
    assignments: HashMap<u8, Vec<u8>>,
}

fn handle_client(mut stream: TcpStream, initialized: &mut i32) -> std::io::Result<()> {

    let start_total = Instant::now(); // TEMP

    let MY_ID: u8 = 2; //#### TODO colocar o numero da funcao aqui...

    let mut len_buf = [0u8; 4];

    loop {
        let start_receive = Instant::now(); // TEMP
        if let Err(e) = stream.read_exact(&mut len_buf) {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                break; // Cliente desconectou normalmente
            }
            return Err(e); // Erro real de rede
        }

        
        // 2. Converter os bytes para o tamanho real da imagem
        let len = u32::from_be_bytes(len_buf) as usize;
        println!("Recebendo payload de {} bytes...", len);

        let mut buffer = vec![0u8; len];
        stream.read_exact(&mut buffer)?;
        let _ = stream.shutdown(Shutdown::Both); //em teste

        let input_img: WitImagedata = rmp_serde::from_slice(&buffer).expect("Failed to deserialize MessagePack response");

        let receiveduration = start_receive.elapsed().as_millis();

        let start_exec = Instant::now();

        let mut status = bindings::planner::sobelworld::plan::sobel(&input_img);

        let execduration = start_exec.elapsed().as_millis();

        let file = std::fs::File::open("routing_table.json").expect("Erro ao abrir JSON");
        let routing: RoutingTable = serde_json::from_reader(file).expect("Erro no JSON");

        // Captura o score deste nó (estou assumindo que este nó físico é o ID 1)
        let current_score = routing.node_scores.get(&1).cloned().unwrap_or(0.0); //#### TODO colocar ip numero do NÓ aqui

        let itinerary = &status.kernels; 
        let my_pos = itinerary.iter().position(|&id| id == MY_ID);

        let next_address = if let Some(current_idx) = my_pos {
    
            // 3. VERIFICA SE EXISTE ALGUÉM DEPOIS
            if current_idx + 1 < itinerary.len() {
                // Pega o VALOR do ID que está na próxima posição
                let next_kernel_id = itinerary[current_idx + 1]; // Aqui pegamos o "7"
                
                println!("Eu sou o ID {}. Localizei o próximo ID: {}", MY_ID, next_kernel_id);

                status.current_kernel = next_kernel_id as u32;

                // 4. CONSULTA KEY/VALUE NO JSON
                let file = std::fs::File::open("routing_table.json").expect("Erro ao abrir JSON");
                let routing: RoutingTable = serde_json::from_reader(file).expect("Erro no JSON");

                // Buscamos pela CHAVE "7"
                routing.table.get(&next_kernel_id)
                    .cloned()
                    .expect("ID não encontrado na tabela de roteamento")
            } else {
                // 5. SE NÃO HOUVER PRÓXIMO (Ex: você era o ID 3)
                println!("Sou o último da lista. Retornando para o cliente.");
                status.reply_to.clone()
            }
        } else {
            // Caso bizarro: seu ID não está na lista que o cliente mandou
            status.reply_to.clone()
        };

        let start_send = Instant::now();
        println!("Conectando a {}...", next_address);
        let mut next_stream = TcpStream::connect(&next_address)?;

        let response_payload = rmp_serde::to_vec(&status).expect("MSGPACK Serialization failed");
        
        let resp_len = response_payload.len() as u32;
        next_stream.write_all(&resp_len.to_be_bytes())?;
        next_stream.write_all(&response_payload)?;
        next_stream.flush()?;

        println!("Payload enviado para o Sobel com sucesso!");
        
        let mut wait = [0u8; 1];
        //let _ = next_stream.shutdown(Shutdown::Write); //em teste
        //let _ = stream.shutdown(Shutdown::Both); //em teste
        let _ = next_stream.read(&mut wait);
        
        let sendduration = start_send.elapsed().as_millis();

        let totalduration = start_total.elapsed().as_millis();

        println!(
            "METRIC_DATA: request={}, id={}, w={}, h={}, total_ms={}, score={:.2}, receive_ms{}, exec_ms{}, send_ms{}",
            input_img.request, MY_ID, input_img.width, input_img.height, totalduration, current_score, receiveduration, execduration, sendduration
        );

        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("receiver_logs.jsonl") 
        {
            let log_linha = serde_json::json!({
                "request": input_img.request,
                "kernel_id": MY_ID,
                "total_receiver_ms": totalduration,
                "receive_ms": receiveduration,
                "exec_ms": execduration,
                "send_ms": sendduration
            });

            if let Ok(texto) = serde_json::to_string(&log_linha) {
                let _ = writeln!(file, "{}", texto);
            }
        }
        
        break;
    }

    println!("Conexão finalizada com o cliente.");
    Ok(())
}

impl bindings::exports::wasi::cli::run::Guest for Component {
    fn run() -> Result<(), ()> {

        let args: Vec<String> = env::args().collect();
        let port = args.get(1).map(|s| s.as_str()).unwrap_or("8080"); //fallback.. se nao passar por argumento ele vai coloar essa
        let ip = args.get(2).map(|s| s.as_str()).unwrap_or("0.0.0.0");
        let bind_addr = format!("{}:{}", ip, port);
        let listener = TcpListener::bind(&bind_addr).expect(&format!("Não conseguiu abrir a porta {}", port));

        println!("Listening on {}", bind_addr);

        let mut init = 0;
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    //call function to handle client
                    println!("entrei!!");

                    if let Err(e) = handle_client(stream, &mut init) {
                        eprintln!("Error handling client: {:?}", e);
                    }
                }
                Err(e) => {
                    eprintln!("Connection failed: {:?}", e);
                }
            }
            println!("Waiting for a new request...");
        }
        Ok(())
    }
}