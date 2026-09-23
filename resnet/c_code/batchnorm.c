#define _POSIX_C_SOURCE 199309L
#define _GNU_SOURCE

//gcc -O3 -I.. batchnorm.c -o test_bn
//./test_bn resnetv15_batchnorm0 64 112 112
//./test_bn resnetv15_batchnorm0 64 256 256

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Inclui a árvore de pesos e definições que estão na pasta pai
#include "weights.h"

#define B 1

// --- Função de mapeamento X-Macro adaptada para buscar os pesos ---
const float* get_weight_pointer(const char* name, int *out_len) {
    const float* ptr = NULL;
    
    #define X(var_name) \
    if (strcmp(name, #var_name) == 0) { \
        ptr = var_name; \
        *out_len = var_name ## _len; \
    }

    #include "weights.def"
    #undef X

    return ptr;
}

// --- Algoritmo de Batch Normalization de Inferência do ONNX ---
void batchnorm2d_onnx(
    int channels, int h, int w, float epsilon,
    const float *input,
    const float *gamma,
    const float *beta,
    const float *running_mean,
    const float *running_var,
    float *output)
{
    for (int c = 0; c < channels; c++) {
        // Coeficientes estáticos para o canal atual
        float g = gamma[c];
        float b = beta[c];
        float mean = running_mean[c];
        float var  = running_var[c];

        // Pré-calcula o denominador para poupar CPU
        float denom = sqrtf(var + epsilon);

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                int idx = c * (h * w) + i * w + j;

                // Fórmula matemática oficial de inferência do ONNX
                float normalized = (input[idx] - mean) / denom;
                output[idx] = g * normalized + b;
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Uso correto: %s <nome_base_batchnorm> <canais> <h_img> <w_img>\n", argv[0]);
        fprintf(stderr, "Exemplo: %s resnetv15_batchnorm0 64 112 112\n", argv[0]);
        return 1;
    }

    char *base_name = argv[1];
    int channels    = atoi(argv[2]);
    int h           = atoi(argv[3]);
    int w           = atoi(argv[4]);
    float epsilon   = 1e-5f; // Valor extraído do atributo epsilon do Netron

    // Strings temporárias para montar o nome exato das variáveis guardadas no weights.h
    char gamma_name[256], beta_name[256], mean_name[256], var_name[256];
    sprintf(gamma_name, "%s_gamma", base_name);
    sprintf(beta_name,  "%s_beta", base_name);
    sprintf(mean_name,  "%s_running_mean", base_name);
    sprintf(var_name,   "%s_running_var", base_name);

    int len_g = 0, len_b = 0, len_m = 0, len_v = 0;
    
    // Busca os 4 ponteiros de memória usando as strings geradas
    const float *gamma_ptr = get_weight_pointer(gamma_name, &len_g);
    const float *beta_ptr  = get_weight_pointer(beta_name, &len_b);
    const float *mean_ptr  = get_weight_pointer(mean_name, &len_m);
    const float *var_ptr   = get_weight_pointer(var_name, &len_v);

    // Validação de segurança: confere se todos os 4 headers existem e se o tamanho bate com os canais
    if (!gamma_ptr || !beta_ptr || !mean_ptr || !var_ptr) {
        fprintf(stderr, "Erro: Não foi possível localizar um ou mais componentes para a camada %s\n", base_name);
        return 1;
    }
    if (len_g != channels || len_b != channels || len_m != channels || len_v != channels) {
        fprintf(stderr, "Erro: O número de canais informado (%d) diverge do tamanho das matrizes no cabeçalho (%d).\n", channels, len_g);
        return 1;
    }

    printf("Carregado BatchNorm: %s | Canais: %d | Formato Espacial: %dx%d\n", base_name, channels, h, w);

    // Alocação dinâmica da entrada simulada (vinda do kernel anterior) e da saída
    float *input_tensor  = malloc(B * channels * h * w * sizeof(float));
    float *output_tensor = malloc(B * channels * h * w * sizeof(float));

    // --- CARREGA A ENTRADA VINDA DO ARQUIVO INTERMEDIÁRIO ---
    FILE *file_in = fopen("conv_output.bin", "rb");
    if (!file_in) {
        fprintf(stderr, "Erro: Arquivo 'conv_output.bin' nao encontrado!\n");
        return 1;
    }
    fread(input_tensor, sizeof(float), B * channels * h * w, file_in);
    fclose(file_in);

    // Executa a normalização
    batchnorm2d_onnx(channels, h, w, epsilon, input_tensor, gamma_ptr, beta_ptr, mean_ptr, var_ptr, output_tensor);

    printf("Batch Normalization finalizado com sucesso!\n");
    printf("Primeiro pixel normalizado: %f\n", output_tensor[0]);

    FILE *file_out = fopen("bn_output.bin", "wb");
    if (!file_out) {
        fprintf(stderr, "Erro ao criar o arquivo de saída 'bn_output.bin'.\n");
        free(input_tensor);
        free(output_tensor);
        return 1;
    }
    fwrite(output_tensor, sizeof(float), B * channels * h * w, file_out);
    fclose(file_out);
    printf("Resultado do BatchNorm salvo com sucesso em 'bn_output.bin'\n");

    free(input_tensor);
    free(output_tensor);
    return 0;
}