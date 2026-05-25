import json
import os
import pandas as pd


def carregar_jsonl(caminho_arquivo):
    """Lê um arquivo .jsonl e retorna uma lista de dicionários."""
    dados = []
    if not os.path.exists(caminho_arquivo):
        print(f"Aviso: Arquivo {caminho_arquivo} não encontrado.")
        return dados
    with open(caminho_arquivo, "r") as f:
        for linha in f:
            if linha.strip():
                dados.append(json.loads(linha))
    return dados


def consolidar_dataset():
    time.sleep(0.5)
    print("Iniciando a consolidação dos dados para a IA...")

    # 1. CARREGA OS LOGS DO CLIENTE (Caminho Atualizado)
    logs_cliente = carregar_jsonl("../client/client_logs.jsonl")
    if not logs_cliente:
        print("Erro: Nenhum log de cliente encontrado para processar.")
        return
    df_cliente = pd.DataFrame(logs_cliente)

    # 2. CARREGA OS LOGS DOS RECEIVERS (Caminho Atualizado e Unificado)
    logs_receivers = carregar_jsonl("../sys_test/receiver_logs.jsonl")
    if logs_receivers:
        df_receivers = pd.DataFrame(logs_receivers)
    else:
        print("Aviso: Nenhum log de receiver encontrado.")
        df_receivers = pd.DataFrame(
            columns=[
                "request",
                "node_id",
                "total_receiver_ms",
                "receive_ms",
                "exec_ms",
                "send_ms",
            ]
        )

    # 3. CARREGA A TELEMETRIA DE HARDWARE DO SYS (Caminho Atualizado)
    logs_hardware = carregar_jsonl("../sys_test/sys_hardware_metrics.jsonl")
    df_hardware = pd.DataFrame(logs_hardware)

    # 4. O ENGENHO DE CRUZAMENTO (Pivoteamento por Request)
    dataset_final = []

    # Vamos iterar sobre cada requisição que o cliente finalizou
    for _, req_client in df_cliente.iterrows():
        req_id = req_client["request"]

        # Inicializa a linha do dataset com os dados macros do cliente
        linha_dataset = {
            "request_id": int(req_id),
            "img_width": int(req_client["img_width"]),
            "img_height": int(req_client["img_height"]),
            "send_time_ms": float(req_client["send_time_ms"]),
            "total_client_time_ms": float(req_client["total_time_ms"]),
            "sla_ms": int(req_client["sla_ms"]),
            "pipeline_demanda": req_client["pipeline_demanda"],
        }

        # Busca tudo o que foi gerado pelos receivers para ESSA request específica
        req_receivers = df_receivers[df_receivers["request"] == req_id]

        for _, row_rec in req_receivers.iterrows():
            n_id = int(row_rec["kernel_id"])

            # Injeta os tempos específicos desse nó de forma dinâmica na linha do CSV
            linha_dataset[f"node_{n_id}_total_ms"] = float(
                row_rec["total_receiver_ms"]
            )
            linha_dataset[f"node_{n_id}_wasm_ms"] = float(row_rec["exec_ms"])
            linha_dataset[f"node_{n_id}_network_receive_ms"] = float(
                row_rec["receive_ms"]
            )
            linha_dataset[f"node_{n_id}_network_send_ms"] = float(
                row_rec["send_ms"]
            )

            # 5. CRUZAMENTO SEMÂNTICO DO HARDWARE
            # Como combinamos, pegamos o último estado de hardware conhecido desse nó específico
            if not df_hardware.empty:
                hw_no = df_hardware[df_hardware["node_id"] == n_id]
                if not hw_no.empty:
                    # Pega a última linha de telemetria registrada antes/durante o teste
                    ultimo_hw = hw_no.iloc[-1]
                    linha_dataset[f"node_{n_id}_cpu_free_pct"] = float(
                        ultimo_hw["cpu_free"]
                    )
                    linha_dataset[f"node_{n_id}_mem_free_pct"] = float(
                        ultimo_hw["mem_free"]
                    )
                    linha_dataset[f"node_{n_id}_health_score"] = float(
                        ultimo_hw["health_score"]
                    )

        dataset_final.append(linha_dataset)

    # 6. SALVA O DATASET DE TREINO PRONTINHO
    df_final = pd.DataFrame(dataset_final)
    df_final.to_csv("dataset_treino_ia.csv", index=False)
    print(
        f"Sucesso! Dataset consolidado com {len(df_final)} exemplos em 'dataset_treino_ia.csv'"
    )


if __name__ == "__main__":
    consolidar_dataset()