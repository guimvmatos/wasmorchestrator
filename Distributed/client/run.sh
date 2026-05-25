#!/bin/bash

# CONFIGURAÇÕES DA SUA BATERIA DE TESTES
ID_INICIAL=1       # Mude para o ID que você quer começar
QUANTIDADE=10       # Quantos requests quer rodar em sequência?
ID_FINAL=$((ID_INICIAL + QUANTIDADE - 1))

echo "==============================================="
echo " Iniciando Esteira Automatizada Segura (v2) "
echo " Executando requests do ID $ID_INICIAL até $ID_FINAL"
echo "==============================================="

for (( id=$ID_INICIAL; id<=$ID_FINAL; id++ ))
do
    echo ""
    echo "-----------------------------------------------"
    echo " [PASSO 1/2] Executando Client com Request ID: $id"
    echo "-----------------------------------------------"
    
    # 1. Roda o Client (Isso bloqueia o Bash até o ciclo de rede fechar 100%)
    CC="" cargo run $id

    echo ""
    echo "-----------------------------------------------"
    echo " [PASSO 2/2] Sincronizando e Agregando dados para o ID: $id"
    echo "-----------------------------------------------"
    
    # 2. Chama o Python de forma cirúrgica para o ID atual
    # Como o Client e o Receiver já fecharam os arquivos, o dado está garantido no disco!
    python3 ../aggregator/aggregator.py $id
    
    # Mini delay de 1 segundo para estabilização de hardware antes do próximo teste
    sleep 1
done

echo ""
echo "==============================================="
echo " Bateria concluída! Dataset atualizado sem falhas. "
echo "==============================================="