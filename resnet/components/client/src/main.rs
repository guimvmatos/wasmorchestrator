// client — envia uma imagem (float32 NCHW cru) para o pipeline de kernels WASM
// e espera o resultado final na porta 9000.
//
// Uso:
//   client <id_request> <model_id> <input.bin> [server_addr]
//          [--reps N] [--warmup W] [--csv arquivo.csv] [--label texto]
//          [--log-logits] [--listen 0.0.0.0:9000]
//
// Sem flags o comportamento é o de antes: 1 inferência, imprime top-5,
// grava output_received.bin e uma linha em client_logs.jsonl.
//
// Mudanças em 30/10 (backup do original em ../clientBkp):
//   1. O listener da porta 9000 é aberto ANTES do envio. Antes ele só abria
//      depois de mandar o pedido; com pipeline rápido (ex.: grupo único em
//      loopback) o último kernel podia tentar conectar antes do bind e levar
//      "Connection refused".
//   2. Modo benchmark: --warmup W descarta W inferências, --reps N mede N,
//      tudo no mesmo processo (sem custo de subir processo em cada medida).
//      Cada repetição usa request_id = id_request + i.
//   3. --csv grava uma linha por repetição medida; resumo (mediana/p95/p99)
//      no fim.
//   4. Os logits só vão para client_logs.jsonl com --log-logits (o arquivo
//      tinha chegado a 205 MB). output_received.bin continua sendo gravado
//      com a saída crua da última repetição (usado pelo check_correctness.py).
//
// O formato das mensagens na rede (ClientRequest / Data via MessagePack,
// prefixo de 4 bytes big-endian) NÃO mudou: handler e kernels não precisam
// ser recompilados.

use serde::{Deserialize, Serialize};
use std::env;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Serialize, Deserialize, Debug)]
struct Data {
    input: Vec<f32>,
    output: Vec<f32>,
    reply_to: String,
    kernels: Vec<u8>,
    current_kernel: u32,
    request: u32,
    from_step: u32, // precisa bater com o record `data` dos .wit (campo from-step)
    model_id: u32,  // precisa bater com o record `data` dos .wit (campo model-id)
}

#[derive(Serialize, Deserialize, Debug)]
struct ClientRequest {
    request_id: u32,
    model_id: u32,
    image_bytes: Vec<u8>,
}

struct Args {
    request_id: u32,
    model_id: u32,
    path: String,
    server: String,
    listen: String,
    reps: usize,
    warmup: usize,
    csv: Option<String>,
    label: String,
    log_logits: bool,
}

/// Tempos de uma inferência (ms).
struct RunTimes {
    serialize_ms: f64,
    send_ms: f64,
    wait_ms: f64,
    deserialize_ms: f64,
    e2e_ms: f64,
}

fn usage(prog: &str) -> ! {
    eprintln!(
        "Uso: {} <id_request> <model_id> <input.bin> [server_addr] \
         [--reps N] [--warmup W] [--csv arquivo.csv] [--label texto] \
         [--log-logits] [--listen 0.0.0.0:9000]",
        prog
    );
    std::process::exit(1);
}

fn parse_args() -> Args {
    let raw: Vec<String> = env::args().collect();
    let prog = raw[0].clone();
    let mut pos: Vec<String> = Vec::new();
    let mut reps = 1usize;
    let mut warmup = 0usize;
    let mut csv = None;
    let mut label = String::new();
    let mut log_logits = false;
    let mut listen = "0.0.0.0:9000".to_string();

    let mut i = 1;
    while i < raw.len() {
        let a = raw[i].as_str();
        let mut value = || -> String {
            i += 1;
            raw.get(i).cloned().unwrap_or_else(|| {
                eprintln!("Erro: {} precisa de um valor.", a);
                usage(&prog)
            })
        };
        match a {
            "--reps" => reps = value().parse().unwrap_or_else(|_| usage(&prog)),
            "--warmup" => warmup = value().parse().unwrap_or_else(|_| usage(&prog)),
            "--csv" => csv = Some(value()),
            "--label" => label = value(),
            "--listen" => listen = value(),
            "--log-logits" => log_logits = true,
            "-h" | "--help" => usage(&prog),
            s if s.starts_with("--") => {
                eprintln!("Erro: flag desconhecida {}", s);
                usage(&prog)
            }
            _ => pos.push(raw[i].clone()),
        }
        i += 1;
    }

    if pos.len() < 3 || reps == 0 {
        usage(&prog);
    }
    let request_id: u32 = pos[0].parse().unwrap_or_else(|_| {
        eprintln!("Erro: id_request precisa ser um inteiro (u32).");
        std::process::exit(1);
    });
    let model_id: u32 = pos[1].parse().unwrap_or_else(|_| {
        eprintln!("Erro: model_id precisa ser um inteiro (u32).");
        std::process::exit(1);
    });

    Args {
        request_id,
        model_id,
        path: pos[2].clone(),
        server: pos.get(3).cloned().unwrap_or_else(|| "127.0.0.1:8050".to_string()),
        listen,
        reps,
        warmup,
        csv,
        label,
        log_logits,
    }
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn process_top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exps: f32 = exps.iter().sum();
    let mut indexed: Vec<(usize, f32)> = exps
        .iter()
        .map(|&e| (e / sum_exps) * 100.0)
        .enumerate()
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).collect()
}

fn argmax(v: &[f32]) -> Option<usize> {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
}

/// Uma inferência completa: serializa, envia, espera a resposta no listener
/// (já aberto) e desserializa.
fn run_once(
    listener: &TcpListener,
    server: &str,
    request_id: u32,
    model_id: u32,
    image_bytes: &[u8],
) -> std::io::Result<(Data, RunTimes)> {
    let t0 = Instant::now();

    // 1. Serialização
    let t = Instant::now();
    let req = ClientRequest {
        request_id,
        model_id,
        image_bytes: image_bytes.to_vec(),
    };
    let payload = rmp_serde::to_vec(&req).expect("Falha ao serializar ClientRequest (MessagePack)");
    let serialize_ms = ms(t);

    // 2. Envio
    let t = Instant::now();
    {
        let mut stream = TcpStream::connect(server)?;
        stream.set_nodelay(true)?;
        stream.write_all(&(payload.len() as u32).to_be_bytes())?;
        stream.write_all(&payload)?;
        stream.flush()?;
    } // conexão de envio fechada aqui, como antes
    let send_ms = ms(t);

    // 3. Espera pela resposta do último kernel (listener já estava aberto)
    let t = Instant::now();
    let (mut resp, _peer) = listener.accept()?;
    let mut len_buf = [0u8; 4];
    resp.read_exact(&mut len_buf)?;
    let len = u32::from_be_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    resp.read_exact(&mut buf)?;
    let wait_ms = ms(t);

    // 4. Desserialização
    let t = Instant::now();
    let data: Data = rmp_serde::from_slice(&buf).expect("Falha ao desserializar Data da pipeline");
    let deserialize_ms = ms(t);

    let e2e_ms = ms(t0);
    Ok((
        data,
        RunTimes {
            serialize_ms,
            send_ms,
            wait_ms,
            deserialize_ms,
            e2e_ms,
        },
    ))
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo as f64)
}

fn main() -> std::io::Result<()> {
    let args = parse_args();

    // Leitura da entrada (uma vez só)
    let t = Instant::now();
    let mut image_bytes = Vec::new();
    std::fs::File::open(&args.path)?.read_to_end(&mut image_bytes)?;
    let read_ms = ms(t);
    if image_bytes.len() % 4 != 0 {
        eprintln!("Erro: arquivo de entrada não é múltiplo de 4 bytes (float32).");
        std::process::exit(1);
    }
    println!(
        "Entrada: {} ({} bytes, {} floats) | modelo {} | servidor {}",
        args.path,
        image_bytes.len(),
        image_bytes.len() / 4,
        args.model_id,
        args.server
    );

    // Listener aberto ANTES de qualquer envio (correção da condição de corrida)
    let listener = TcpListener::bind(&args.listen)?;
    println!("Escutando a resposta em {}", args.listen);

    // CSV (append; cabeçalho só se o arquivo for novo/vazio)
    let mut csv = match &args.csv {
        Some(p) => {
            let new = std::fs::metadata(p).map(|m| m.len() == 0).unwrap_or(true);
            let mut f = OpenOptions::new().create(true).append(true).open(p)?;
            if new {
                writeln!(
                    f,
                    "timestamp_ms,label,model_id,request,rep,input_bytes,output_floats,top1,\
                     serialize_ms,send_ms,wait_ms,deserialize_ms,e2e_ms"
                )?;
            }
            Some(f)
        }
        None => None,
    };

    let total = args.warmup + args.reps;
    let mut e2e: Vec<f64> = Vec::with_capacity(args.reps);
    let mut last: Option<Data> = None;

    for i in 0..total {
        let rid = args.request_id.wrapping_add(i as u32);
        let warm = i < args.warmup;
        let (data, rt) = run_once(&listener, &args.server, rid, args.model_id, &image_bytes)?;

        if data.request != rid {
            eprintln!(
                "AVISO: resposta com request {} != enviado {} (resposta atrasada de outra execução?)",
                data.request, rid
            );
        }

        if !warm {
            let rep = i - args.warmup;
            let top1 = if data.input.len() <= 1000 { argmax(&data.input) } else { None };
            e2e.push(rt.e2e_ms);

            if let Some(f) = csv.as_mut() {
                writeln!(
                    f,
                    "{},{},{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4}",
                    now_unix_ms(),
                    args.label.replace(',', ";"),
                    args.model_id,
                    rid,
                    rep,
                    image_bytes.len(),
                    data.input.len(),
                    top1.map(|c| c.to_string()).unwrap_or_default(),
                    rt.serialize_ms,
                    rt.send_ms,
                    rt.wait_ms,
                    rt.deserialize_ms,
                    rt.e2e_ms
                )?;
            }

            if let Ok(mut log) = OpenOptions::new().create(true).append(true).open("client_logs.jsonl") {
                let mut entry = serde_json::json!({
                    "request": data.request,
                    "model_id": args.model_id,
                    "label": args.label,
                    "rep": rep,
                    "top1_class": top1,
                    "output_floats": data.input.len(),
                    "read_time_ms": read_ms,
                    "serialize_time_ms": rt.serialize_ms,
                    "send_time_ms": rt.send_ms,
                    "wait_time_ms": rt.wait_ms,
                    "deserialize_time_ms": rt.deserialize_ms,
                    "e2e_time_ms": rt.e2e_ms,
                    "pipeline_demanda": data.kernels,
                });
                if args.log_logits {
                    entry["output_logits"] = serde_json::json!(data.input);
                }
                if let Ok(text) = serde_json::to_string(&entry) {
                    let _ = writeln!(log, "{}", text);
                }
            }
        }

        if total > 1 {
            println!(
                "[{}/{}]{} request {} -> e2e {:.3} ms (espera {:.3} ms)",
                i + 1,
                total,
                if warm { " warmup" } else { "" },
                rid,
                rt.e2e_ms,
                rt.wait_ms
            );
        }
        last = Some(data);
    }

    // Saída crua da última repetição (para check_correctness.py)
    if let Some(data) = &last {
        let raw: Vec<u8> = data.input.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write("output_received.bin", &raw)?;

        println!("\n=== RESULTADO (última repetição) ===");
        println!("Request: {} | floats na saída: {}", data.request, data.input.len());
        if data.input.len() <= 1000 {
            println!("--- TOP-5 ---");
            for (rank, (c, p)) in process_top_k(&data.input, 5).iter().enumerate() {
                println!("  #{}: classe {} -> {:.2}%", rank + 1, c, p);
            }
        } else {
            println!("(saída não é de classificação; top-5 omitido)");
        }
        println!("Saída crua gravada em output_received.bin");
    }

    // Resumo
    let mut s = e2e.clone();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len() as f64;
    let mean = s.iter().sum::<f64>() / n;
    let std = (s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
    println!(
        "\n=== LATÊNCIA E2E ({} medidas, {} warmup) ===\n\
         mediana {:.3} ms | p95 {:.3} ms | p99 {:.3} ms | média {:.3} ms | desvio {:.3} ms | mín {:.3} | máx {:.3}",
        s.len(),
        args.warmup,
        percentile(&s, 50.0),
        percentile(&s, 95.0),
        percentile(&s, 99.0),
        mean,
        std,
        s.first().copied().unwrap_or(f64::NAN),
        s.last().copied().unwrap_or(f64::NAN)
    );
    if let Some(p) = &args.csv {
        println!("CSV: {}", p);
    }
    Ok(())
}
