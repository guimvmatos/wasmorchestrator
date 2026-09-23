import numpy as np
import os

def npy_para_bin():
    pasta_entrada = 'npy'
    pasta_saida = 'bin_weights'
    
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)

    for f in os.listdir(pasta_entrada):
        if f.endswith('.npy'):
            caminho_in = os.path.join(pasta_entrada, f)
            nome_base = os.path.splitext(f)[0]
            caminho_out = os.path.join(pasta_saida, f"{nome_base}.bin")

            dados = np.load(caminho_in).astype(np.float32)
            # Salva exatamente a memória contígua dos floats de 32 bits
            dados.tofile(caminho_out)
            print(f"Gerado: {caminho_out} ({dados.size} floats)")

if __name__ == "__main__":
    npy_para_bin()