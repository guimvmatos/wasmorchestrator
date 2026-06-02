#!/bin/bash

IMAGENS=("image_hd.ppm" "image_fhd.ppm" "image_4k.ppm" "image_cnn.ppm")
DEMANDAS=("1" "2" "3" "1,2" "1,3" "2,3" "1,2,3")

REQUEST_ID=1
REPETICOES=1

echo "=== INICIANDO BATERIA DE 280 TESTES (10 RODADAS) ==="

for ((r=1; r<=REPETICOES; r++))
do
    echo "================================================"
    echo "INICIANDO RODADA DE REPETIÇÃO NÚMERO: $r / $REPETICOES"
    echo "================================================"

    for IMG in "${IMAGENS[@]}"
    do
        if [ ! -f "$IMG" ]; then
            echo "Aviso: $IMG não encontrada. Pulando..."
            continue
        fi

        for DEMANDA in "${DEMANDAS[@]}"
        do
            echo "------------------------------------------------"
            echo "[Req #$REQUEST_ID] [Rodada $r] Imagem: $IMG | Pipeline: [$DEMANDA]"
            
            # Executa o cliente passando: ID, Caminho da Imagem e a String da Demanda
            cargo run --release -- $REQUEST_ID "$IMG" "$DEMANDA"
            
            sleep 1

            # pegar arquivos em outro pc.
            #sshpass -p "Kr4pn1kn1l" rsync -avzP --include="*.jsonl" --exclude="*" ssh guimvmatos@10.147.172.163:/home/guimvmatos/wasmorchestrator/Distributed/sys_test/. ../sys_test/
            
            # O seu agregador Python original faz o papel dele usando o ID
            python3 ../aggregator/aggregator.py $REQUEST_ID
            
            REQUEST_ID=$((REQUEST_ID + 1))
            sleep 2
        done
    done
done

echo "=== BATERIA DE TESTES CONCLUÍDA! ==="