#define _POSIX_C_SOURCE 199309L
#define _GNU_SOURCE

// Compilar: gcc -O3 relu.c -o test_relu
// Executar (1a camada 224x224 -> 1*64*112*112 = 802816): ./test_relu 802816
// Executar (1a camada 512x512 -> 1*64*256*256 = 4194304): ./test_relu 4194304

#include <stdio.h>
#include <stdlib.h>

void relu_onnx(int total_elementos, const float *input, float *output) {
    for (int i = 0; i < total_elementos; i++) {
        // Se o valor for maior que 0, mantém. Se for menor, vira 0.0f
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Uso correto: %s <total_de_elementos>\n", argv[0]);
        return 1;
    }

    int total_elementos = atoi(argv[1]);

    // Alocação dinâmica para os tensores de entrada e saída
    float *input_tensor  = malloc(total_elementos * sizeof(float));
    float *output_tensor = malloc(total_elementos * sizeof(float));

    if (!input_tensor || !output_tensor) {
        fprintf(stderr, "Erro de alocação de memória.\n");
        return 1;
    }

    // --- LEITURA DO OUTPUT DO BATCHNORM ---
    // Ajuste o nome do arquivo para bater com o que o seu test_bn salvou em disco
    FILE *file_in = fopen("bn_output.bin", "rb"); 
    if (!file_in) {
        fprintf(stderr, "Erro: Arquivo 'bn_output.bin' nao encontrado!\n");
        free(input_tensor); free(output_tensor);
        return 1;
    }
    fread(input_tensor, sizeof(float), total_elementos, file_in);
    fclose(file_in);

    // Executa a ativação ReLU
    relu_onnx(total_elementos, input_tensor, output_tensor);

    // --- SALVA O RESULTADO PARA O PRÓXIMO KERNEL (MAXPOOL) ---
    FILE *file_out = fopen("relu_output.bin", "wb");
    if (!file_out) {
        fprintf(stderr, "Erro ao criar o arquivo 'relu_output.bin'.\n");
        free(input_tensor); free(output_tensor);
        return 1;
    }
    fwrite(output_tensor, sizeof(float), total_elementos, file_out);
    fclose(file_out);

    printf("ReLU finalizada com sucesso! Dados salvos em 'relu_output.bin'\n");
    printf("Primeiro pixel retificado: %f\n", output_tensor[0]);

    free(input_tensor);
    free(output_tensor);
    return 0;
}