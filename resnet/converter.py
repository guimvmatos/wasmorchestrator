import numpy as np
import os
import re

def converter_todos():
    pasta_entrada = 'npy'
    pasta_saida = 'headers'
    
    # Cria a pasta de saída se ela não existir
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)
    
    # Lista todos os arquivos .npy na pasta
    arquivos = [f for f in os.listdir(pasta_entrada) if f.endswith('.npy')]
    
    if not arquivos:
        print(f"Nenhum arquivo .npy encontrado na pasta '{pasta_entrada}'.")
        return

    for nome_arquivo in arquivos:
        caminho_entrada = os.path.join(pasta_entrada, nome_arquivo)
        
        # 1. Limpa o nome para ser uma variável C válida (remove pontos e traços)
        # Ex: "resnet_v1.weight.npy" vira "resnet_v1_weight"
        nome_base = os.path.splitext(nome_arquivo)[0]
        nome_variavel = re.sub(r'[^a-zA-Z0-9_]', '_', nome_base)
        
        caminho_saida = os.path.join(pasta_saida, nome_base + '.h')
        
        # 2. Carrega e processa
        try:
            dados = np.load(caminho_entrada)
            dados_flat = dados.flatten()
            
            # 3. Escreve o arquivo .h
            with open(caminho_saida, 'w') as f:
                f.write(f"// Pesos extraídos de {nome_arquivo}\n")
                f.write(f"// Shape original: {dados.shape}\n\n")
                
                # Usamos um Header Guard para evitar múltiplas inclusões
                f.write(f"#ifndef {nome_variavel.upper()}_H\n")
                f.write(f"#define {nome_variavel.upper()}_H\n\n")
                
                f.write(f"const float {nome_variavel}[] = {{\n")
                
                for i, valor in enumerate(dados_flat):
                    f.write(f"{valor:f}f, ")
                    if (i + 1) % 10 == 0:
                        f.write("\n")
                
                f.write("\n};\n\n")
                f.write(f"const int {nome_variavel}_len = {len(dados_flat)};\n\n")
                f.write(f"#endif // {nome_variavel.upper()}_H\n")
            
            print(f"Convertido: {nome_arquivo} -> {caminho_saida}")
            
        except Exception as e:
            print(f"Erro ao converter {nome_arquivo}: {e}")

def gerar_weights_h():
    pasta_headers = 'headers'
    arquivo_mestre = 'weights.h'
    
    arquivos_h = [f for f in os.listdir(pasta_headers) if f.endswith('.h')]
    
    with open(arquivo_mestre, 'w') as f:
        f.write("// Arquivo mestre de pesos - Gerado automaticamente\n")
        f.write("#ifndef WEIGHTS_H\n")
        f.write("#define WEIGHTS_H\n\n")
        
        for h in arquivos_h:
            # Note que o path deve ser relativo ao local onde o test.c está
            f.write(f'#include "headers/{h}"\n')
            
        f.write("\n#endif // WEIGHTS_H\n")
    print(f"\nArquivo mestre '{arquivo_mestre}' gerado com {len(arquivos_h)} inclusões.")

# Chame essa função no seu if __name__ == "__main__":
if __name__ == "__main__":
    converter_todos()
    gerar_weights_h()            

