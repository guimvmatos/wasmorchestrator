// 05/11 — CAIXA DE ENTRADA (relay) por grupo, para acabar com o travamento
// entre grupos.
//
// O problema: cada grupo (componente WASM) tem uma thread só. Ele aceita uma
// conexão, lê a mensagem, processa e, quando o próximo kernel está em outro
// grupo, CONECTA e ENVIA para ele e espera o "ack" (o outro lado fechar a
// conexão depois de ler tudo). Enquanto envia, ele não aceita nada. Se o
// grupo A está enviando para B e, ao mesmo tempo, B está enviando para A,
// os dois esperam um pelo outro para sempre. Acontece com 2+ requisições em
// voo (vários clients) e com modelos que ramificam entre grupos (YOLO).
//
// A correção: o endereço público do grupo (o que vai na routing_table) passa
// a ser esta caixa de entrada, que roda no sys (processo nativo, com
// threads). Ela SEMPRE aceita na hora, lê a mensagem inteira, devolve o ack e
// guarda numa fila. Uma thread de entrega passa as mensagens, uma por vez e
// na ordem de chegada, para o grupo, na porta interna dele. Assim quem envia
// nunca fica esperando um grupo ocupado, e o ciclo de espera não se forma.
// Nada muda no componente WASM, no handler nem no client.

use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

pub struct Relay {
    stop: Arc<AtomicBool>,
    pending: Arc<AtomicUsize>,
    public_addr: String,
}

/// Lê um quadro inteiro: u32 BE com o tamanho + o conteúdo. Devolve os bytes
/// já com o prefixo, prontos para reenviar sem mexer.
fn read_frame(s: &mut TcpStream) -> io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    s.read_exact(&mut len_buf)?;
    let len = u32::from_be_bytes(len_buf) as usize;
    let mut msg = vec![0u8; 4 + len];
    msg[..4].copy_from_slice(&len_buf);
    s.read_exact(&mut msg[4..])?;
    Ok(msg)
}

/// Entrega um quadro ao grupo e espera o ack dele (o grupo fecha a conexão
/// depois de ler tudo), igual a como os grupos fazem entre si.
fn deliver(inner_addr: &str, msg: &[u8]) -> io::Result<()> {
    let mut s = TcpStream::connect(inner_addr)?;
    s.write_all(msg)?;
    s.flush()?;
    let mut ack = [0u8; 64];
    loop {
        match s.read(&mut ack) {
            Ok(0) => break,
            Ok(_) => continue,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => break, // reset depois do shutdown do grupo também conta como entregue
        }
    }
    Ok(())
}

impl Relay {
    /// Abre `bind_ip:public_port` e repassa tudo para `inner_addr`.
    pub fn start(bind_ip: &str, public_port: u16, inner_addr: String, label: String) -> io::Result<Relay> {
        let listener = TcpListener::bind((bind_ip, public_port))?;
        let stop = Arc::new(AtomicBool::new(false));
        let pending = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = mpsc::channel::<Vec<u8>>();

        // Entrega: uma thread, ordem de chegada.
        {
            let stop = stop.clone();
            let pending = pending.clone();
            let label = label.clone();
            std::thread::spawn(move || {
                for msg in rx {
                    let mut tries = 0u32;
                    loop {
                        match deliver(&inner_addr, &msg) {
                            Ok(()) => break,
                            Err(e) => {
                                if stop.load(Ordering::SeqCst) {
                                    break;
                                }
                                tries += 1;
                                if tries >= 250 {
                                    eprintln!("[RELAY {}] desisti de entregar {} bytes para {}: {}", label, msg.len(), inner_addr, e);
                                    break;
                                }
                                std::thread::sleep(Duration::from_millis(20));
                            }
                        }
                    }
                    pending.fetch_sub(1, Ordering::SeqCst);
                }
            });
        }

        // Aceitação: uma thread; cada conexão é lida numa thread própria, para
        // um remetente lento não segurar os outros.
        {
            let stop = stop.clone();
            let pending = pending.clone();
            std::thread::spawn(move || {
                for conn in listener.incoming() {
                    if stop.load(Ordering::SeqCst) {
                        break;
                    }
                    let mut s = match conn {
                        Ok(s) => s,
                        Err(_) => continue,
                    };
                    pending.fetch_add(1, Ordering::SeqCst);
                    let tx = tx.clone();
                    let pending = pending.clone();
                    std::thread::spawn(move || match read_frame(&mut s) {
                        Ok(msg) => {
                            // ack para quem enviou: a mensagem já está guardada aqui.
                            let _ = s.shutdown(Shutdown::Both);
                            if tx.send(msg).is_err() {
                                pending.fetch_sub(1, Ordering::SeqCst);
                            }
                        }
                        Err(_) => {
                            pending.fetch_sub(1, Ordering::SeqCst);
                        }
                    });
                }
                // listener e tx saem de escopo aqui: a porta pública fecha e,
                // quando as leituras em curso terminam, a thread de entrega sai.
            });
        }

        Ok(Relay {
            stop,
            pending,
            public_addr: format!("{}:{}", bind_ip, public_port),
        })
    }

    /// Mensagens recebidas e ainda não entregues ao grupo.
    pub fn pending(&self) -> usize {
        self.pending.load(Ordering::SeqCst)
    }

    /// Fecha a porta pública (acorda a thread de aceitação com uma conexão vazia).
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(&self.public_addr);
    }
}
