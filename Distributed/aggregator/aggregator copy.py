import json
import os
import sys
import time
import pandas as pd


def carregar_jsonl(caminho_arquivo):
    dados = []
    if not os.path.exists(caminho_arquivo):
        return dados
    with open(caminho_arquivo, "r") as f:
        for linha in f:
            if linha.strip():
                dados.append(json.loads(linha))
    return dados


def consolidar_dataset():
    # 1. VALIDAÇÃO DO ARGUMENTO DO RUST
    if len(sys.argv) < 2:
        print(
            "Erro: O agregador precisa receber o request_id atual como argumento."
        )
        return

    target_request_id = int(sys.argv[1])
    print(
        f"Consolidando exclusivamente os dados temporais e de hardware para o Request ID: {target_request_id}"
    )

    # 2. TRAVA DE SEGURANÇA: Garante que o Receiver já escreveu a linha atual no disco
    tentativas = 0
    req_receivers = []
    while tentativas < 10:
        logs_receivers = carregar_jsonl("../sys_test/receiver_logs.jsonl")
        req_receivers = [
            x for x in logs_receivers if x["request"] == target_request_id
        ]

        if req_receivers:
            break  # Achou pelo menos um kernel desse request, pode prosseguir
        time.sleep(0.5)
        tentativas += 1

    if not req_receivers:
        print(
            f"Erro: Nenhum log do receiver encontrado para o request {target_request_id} após 5s."
        )
        return

    # 3. FILTRA APENAS A LINHA CORRETA DO CLIENTE
    logs_cliente = carregar_jsonl("../client/client_logs.jsonl")
    req_client = next(
        (x for x in reversed(logs_cliente) if x["request"] == target_request_id),
        None,
    )

    if not req_client:
        print(
            f"Erro: Request {target_request_id} não encontrado no arquivo do cliente."
        )
        return

    # 4. CAPTURA A TELEMETRIA DO HARDWARE ATUAL (A última linha gravada no sys_test)
    logs_hardware = carregar_jsonl("../sys_test/sys_hardware_metrics.jsonl")

    # 5. MONTA A LINHA ÚNICA DA EXECUÇÃO ATUAL
    lista_kernels = req_client["pipeline_demanda"]
    if isinstance(lista_kernels, list):
        demanda_formatada = "|".join(str(k) for k in lista_kernels)
    else:
        demanda_formatada = str(lista_kernels).replace("[", "").replace("]", "").replace(", ", "|")

    # Montagem da estrutura base da linha do dataset
    linha_dataset = {
        "request_id": target_request_id,
        "img_width": int(req_client["img_width"]),
        "img_height": int(req_client["img_height"]),
        "client_read_time_ms": float(req_client.get("read_time_ms", 0.0)),
        "client_serialize_time_ms": float(req_client.get("serialize_time_ms", 0.0)),
        "send_time_ms": float(req_client["send_time_ms"]),
        "total_client_time_ms": float(req_client["exec_time_ms"]),  # CORRIGIDO: De total_time_ms para exec_time_ms
        "client_deserialize_time_ms": float(req_client.get("deserialize_time_ms", 0.0)),
        "sla_ms": int(req_client["sla_ms"]),
        "pipeline_demanda": demanda_formatada,
    }

    # Varre todos os kernels que processaram essa request (caso haja mais de um)
    for row_rec in req_receivers:
        k_id = int(row_rec["kernel_id"])

        # Mapeamento dinâmico baseado no kernel_id executado
        linha_dataset[f"node_{k_id}_total_ms"] = float(
            row_rec["total_receiver_ms"]
        )
        linha_dataset[f"node_{k_id}_wasm_ms"] = float(row_rec["exec_ms"])
        linha_dataset[f"node_{k_id}_network_receive_ms"] = float(
            row_rec["receive_ms"]
        )
        linha_dataset[f"node_{k_id}_network_send_ms"] = float(
            row_rec["send_ms"]
        )

        # Associa com a foto do hardware desse exato momento do teste
        if logs_hardware:
            # Filtra o histórico de hardware desse nó específico
            hw_no = [x for x in logs_hardware if x["node_id"] == k_id]
            if hw_no:
                # Pega o último estado de hardware registrado ATÉ AGORA
                ultimo_hw = hw_no[-1]
                linha_dataset[f"node_{k_id}_cpu_free_pct"] = float(
                    ultimo_hw["cpu_free"]
                )
                linha_dataset[f"node_{k_id}_mem_free_pct"] = float(
                    ultimo_hw["mem_free"]
                )
                linha_dataset[f"node_{k_id}_health_score"] = float(
                    ultimo_hw["health_score"]
                )

    # 6. SALVA VIA APPEND REAL (Preserva o histórico imutável)
    df_nova_linha = pd.DataFrame([linha_dataset])
    csv_path = "dataset_treino_ia.csv"

    if not os.path.exists(csv_path):
        # Se for a primeira execução da história, cria o arquivo com o cabeçalho
        df_nova_linha.to_csv(csv_path, index=False)
    else:
        # Se o arquivo já existir, apenas cola a linha nova embaixo, mantendo o passado intocado
        df_nova_linha.to_csv(csv_path, mode="a", header=False, index=False)

    print(
        f"Sucesso! Dados do request {target_request_id} anexados permanentemente ao dataset."
    )


if __name__ == "__main__":
    consolidar_dataset()