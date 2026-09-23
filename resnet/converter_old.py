import numpy as np

def converter():
    arquivo_entrada = 'resnetv15_conv0_weight.npy'
    arquivo_saida = 'pesos_conv0.h'
    
    # 1. Carrega o arquivo NPY
    dados = np.load(arquivo_entrada)
    
    # 2. Transforma em uma lista única de números (flat)
    dados_flat = dados.flatten()
    
    # 3. Escreve o arquivo .h
    with open(arquivo_saida, 'w') as f:
        f.write(f"// Pesos extraidos de {arquivo_entrada}\n")
        f.write(f"// Shape original: {dados.shape}\n\n")
        f.write(f"const float pesos_conv0[] = {{\n")
        
        # Escreve os números separados por vírgula
        for i, valor in enumerate(dados_flat):
            f.write(f"{valor:f}f, ")
            if (i + 1) % 8 == 0: # Quebra linha a cada 8 números para organizar
                f.write("\n")
                
        f.write("\n};\n\n")
        f.write(f"const int pesos_conv0_len = {len(dados_flat)};\n")

    print(f"Pronto! O arquivo {arquivo_saida} foi criado.")

if __name__ == "__main__":
    converter()