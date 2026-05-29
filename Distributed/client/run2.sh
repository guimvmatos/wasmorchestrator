#!/bin/bash

# Array com as suas imagens de teste
IMAGENS=("image_4k.ppm" "image_cnn.ppm" "image_fhd.ppm" "image_hd.ppm")

# ID inicial do request (você pode mudar se quiser continuar de onde parou)
REQUEST_ID=1

# Quantas vezes você quer rodar a bateria completa de testes?
REPETICOES=1

echo "========================================================="
echo "   Iniciando Bateria de Testes Automatizados - IA Continuum"
echo "========================================================="

for ((i=1; i<=REPETICOES; i++))
do
    echo "---------------------------------------------------------"
    echo " RODADA DE TESTES NÚMERO: $i"
    echo "---------------------------------------------------------"

    for IMG in "${IMAGENS[@]}"
    do
        # Valida se o arquivo de fato existe antes de quebrar o Rust
        if [ ! -f "$IMG" ]; then
            echo "Aviso: Arquivo $IMG não encontrado na pasta. Pulando..."
            continue
        fi

        echo "[Request #$REQUEST_ID] Disparando pipeline para a imagem: $IMG"

        # 1. Executa o cliente Rust passando o ID e o caminho da imagem por parâmetro
        # Nota: Usando --release para a CPU não chorar na serialização de 9 segundos!
        cargo run --release -- $REQUEST_ID "$IMG"

        # 2. Pequena folga de 1 segundo para garantir que os logs dos Kernels
        # foram transmitidos e gravados no sys_test pelo receptor assíncrono
        sleep 1

        # 3. Executa o agregador Python para amarrar os logs e atualizar o CSV de treino
        echo "[Request #$REQUEST_ID] Agregando telemetria e atualizando o dataset..."
        python3 ../aggregator/aggregator.py $REQUEST_ID

        # Incrementa o ID para o próximo request
        REQUEST_ID=$((REQUEST_ID + 1))
        
        echo "Aguardando próximo disparo..."
        sleep 2
    done
done

echo "========================================================="
echo " Bateria concluída! Dataset atualizado com sucesso."
echo "========================================================="