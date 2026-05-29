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
    # 1. VALIDAÇÃO DO ARGUMENTO
    if len(sys.argv) < 2:
        print("Erro: O agregador precisa receber o request_id atual como argumento.")
        return

    target_request_id = int(sys.argv[1])
    print(f"Consolidando exclusivamente os dados temporais e de hardware para o Request ID: {target_request_id}")

    # 2. TRAVA DE SEGURANÇA
    tentativas = 0
    req_receivers = []
    while tentativas < 10:
        logs_receivers = carregar_jsonl("../sys_test/receiver_logs.jsonl")
        req_receivers = [x for x in logs_receivers if x["request"] == target_request_id]

        if req_receivers:
            break
        time.sleep(0.5)
        tentativas += 1

    if not req_receivers:
        print(f"Erro: Nenhum log do receiver encontrado para o request {target_request_id} após 5s.")
        return

    # 3. FILTRA APENAS A LINHA CORRETA DO CLIENTE
    logs_cliente = carregar_jsonl("../client/client_logs.jsonl")
    req_client = next(
        (x for x in reversed(logs_cliente) if x["request"] == target_request_id),
        None,
    )

    if not req_client:
        print(f"Erro: Request {target_request_id} não encontrado no arquivo do cliente.")
        return

    # 4. CAPTURA A TELEMETRIA DO HARDWARE ATUAL
    logs_hardware = carregar_jsonl("../sys_test/sys_hardware_metrics.jsonl")

    # 5. PARSE DA DEMANDA REAL
    lista_kernels = req_client["pipeline_demanda"]
    if isinstance(lista_kernels, list):
        kernels_solicitados = [int(k) for k in lista_kernels]
    else:
        clean_str = str(lista_kernels).replace("[", "").replace("]", "").replace(" ", "")
        kernels_solicitados = [int(k) for k in clean_str.split(",") if k]

    demanda_formatada = "|".join(str(k) for k in kernels_solicitados)

    # =========================================================================
    # CONFIGURAÇÃO DE LARGURA FIXA E SEPARADA: METADADOS -> NÓ -> KERNELS (1,2,3)
    # =========================================================================
    linha_dataset = {
        # Bloco A: Metadados do Request e Cliente
        "request_id": target_request_id,
        "img_width": int(req_client["img_width"]),
        "img_height": int(req_client["img_height"]),
        "total_client_time_ms": float(req_client.get("total_client_time_ms", 0.0)), # Tempo mestre fim a fim real
        "client_read_time_ms": float(req_client.get("read_time_ms", 0.0)),
        "client_serialize_time_ms": float(req_client.get("serialize_time_ms", 0.0)),
        "send_time_ms": float(req_client["send_time_ms"]),
        "client_exec_time_ms": float(req_client["exec_time_ms"]), # Tempo da pipeline (rede + wasm)
        "client_deserialize_time_ms": float(req_client.get("deserialize_time_ms", 0.0)),
        "sla_ms": int(req_client["sla_ms"]),
        "pipeline_demanda": demanda_formatada,
        
        # Bloco B: Dados Gerais de Telemetria do Nó Físico (Sempre presentes no teste atual)
        "node_cpu_free_pct": "", 
        "node_mem_free_pct": "", 
        "node_health_score": "",
        
        # Bloco C: Métricas de Tempo do Kernel 1 (Grayscale)
        "k1_total_ms": "", "k1_wasm_ms": "", "k1_network_receive_ms": "", "k1_network_send_ms": "",
        
        # Bloco D: Métricas de Tempo do Kernel 2 (Sobel)
        "k2_total_ms": "", "k2_wasm_ms": "", "k2_network_receive_ms": "", "k2_network_send_ms": "",
        
        # Bloco E: Métricas de Tempo do Kernel 3 (Negative)
        "k3_total_ms": "", "k3_wasm_ms": "", "k3_network_receive_ms": "", "k3_network_send_ms": "",
    }

    colunas_fidelidade = list(linha_dataset.keys())

    # PREENCHE O BLOCO DE HARDWARE DO NÓ (Independente de qual kernel rodou)
    if logs_hardware:
        # Como o teste atual está centralizado no Node 1, buscamos a última métrica dele
        hw_no = [x for x in logs_hardware if x["node_id"] == 1]
        if hw_no:
            ultimo_hw = hw_no[-1]
            linha_dataset["node_cpu_free_pct"] = float(ultimo_hw["cpu_free"])
            linha_dataset["node_mem_free_pct"] = float(ultimo_hw["mem_free"])
            linha_dataset["node_health_score"] = float(ultimo_hw["health_score"])

    # PREENCHE OS BLOCOS DE TEMPO DOS KERNELS BASEADO NA SEQUÊNCIA DA DEMANDA
    for idx, row_rec in enumerate(req_receivers):
        if idx >= len(kernels_solicitados):
            break
            
        k_id = kernels_solicitados[idx]

        # Injeta estritamente no bloco correspondente ao k_id (k1, k2 ou k3)
        linha_dataset[f"k{k_id}_total_ms"] = float(row_rec["total_receiver_ms"])
        linha_dataset[f"k{k_id}_wasm_ms"] = float(row_rec["exec_ms"])
        linha_dataset[f"k{k_id}_network_receive_ms"] = float(row_rec["receive_ms"])
        linha_dataset[f"k{k_id}_network_send_ms"] = float(row_rec["send_ms"])

    # 6. GRAVAÇÃO COMPACTA COM GARANTIA DE CABEÇALHO SINCRO
    df_nova_linha = pd.DataFrame([linha_dataset])
    df_nova_linha = df_nova_linha[colunas_fidelidade]
    
    csv_path = "dataset_treino_ia.csv"

    # Se o arquivo não existir ou estiver zerado, força a criação do cabeçalho de colunas
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        df_nova_linha.to_csv(csv_path, index=False)
    else:
        df_nova_linha.to_csv(csv_path, mode="a", header=False, index=False)

    print(f"Sucesso! Dados do request {target_request_id} anexados permanentemente ao dataset.")


if __name__ == "__main__":
    consolidar_dataset()