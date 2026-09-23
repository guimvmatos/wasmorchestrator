import numpy as np
import onnx
from onnx import numpy_helper

def converter_pb_para_bin():
    # Se o script está dentro de c_code, buscamos o .pb na pasta pai
    arquivo_pb = 'input_0.pb'
    arquivo_bin = 'input_image.bin'
    
    try:
        # Carrega o tensor usando o método robusto de leitura do ONNX
        with open(arquivo_pb, 'rb') as f:
            tensor = onnx.load_tensor(f)
        
        # Converte o objeto do ONNX para uma matriz do NumPy
        dados_np = numpy_helper.to_array(tensor)
        
        # Garante float32
        dados_flat = dados_np.astype(np.float32)
        
        # Salva em disco como bytes brutos
        dados_flat.tofile(arquivo_bin)
        
        print(f"✅ Sucesso! Tensor extraído com o novo parser.")
        print(f"   Shape original: {dados_np.shape}")
        print(f"   Arquivo gerado: {arquivo_bin}")
        print(f"   Primeiro valor do input: {dados_flat.flatten()[0]:.6f}")
        
    except FileNotFoundError:
        print(f"❌ Erro: O arquivo '{arquivo_pb}' não foi encontrado.")
    except Exception as e:
        print(f"❌ Erro ao converter o arquivo: {e}")

if __name__ == "__main__":
    converter_pb_para_bin()