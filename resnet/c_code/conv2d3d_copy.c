#define _POSIX_C_SOURCE 199309L
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Inclui a árvore de pesos e definições geradas pelo seu Python
#include "weights.h"

// Mantemos apenas o lote fixado em 1 por padrão
#define B 1

// --- Função de mapeamento X-Macro para capturar o ponteiro do peso ---
const float* get_kernel_pointer(const char* name, int *out_len) {
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

// --- Convolução Totalmente Dinâmica (Aceita qualquer camada e dimensão) ---
void conv2d3d(
    int c_out, int c_in, int h_in, int w_in, int kh, int kw, int stride, int padding,
    float input[B][c_in][h_in][w_in], 
    const float *kernel, 
    float *output) 
{
    // Fórmula universal para cálculo de saída espacial com padding duplo
    int out_h = ((h_in + 2 * padding - kh) / stride) + 1;
    int out_w = ((w_in + 2 * padding - kw) / stride) + 1;

    for (int b = 0; b < B; b++) {
        for (int co = 0; co < c_out; co++) {
            for (int i = 0; i < out_h; i++) {
                for (int j = 0; j < out_w; j++) {
                    float sum = 0.0f;
                    
                    int in_base_h = i * stride - padding;
                    int in_base_w = j * stride - padding;

                    for (int ci = 0; ci < c_in; ci++) {
                        for (int ki = 0; ki < kh; ki++) {
                            for (int kj = 0; kj < kw; kj++) {
                                int in_h = in_base_h + ki;
                                int in_w = in_base_w + kj;

                                // Tratamento de Zero-Padding genérico para qualquer borda
                                if (in_h >= 0 && in_h < h_in && in_w >= 0 && in_w < w_in) {
                                    int kernel_idx = co * (c_in * kh * kw) 
                                                   + ci * (kh * kw) 
                                                   + ki * kw 
                                                   + kj;
                                    
                                    sum += input[b][ci][in_h][in_w] * kernel[kernel_idx];
                                }
                            }
                        }
                    }
                    
                    // Mapeamento linear seguro da saída estruturada dinamicamente [b][co][i][j]
                    int out_idx = b * (c_out * out_h * out_w)
                                + co * (out_h * out_w)
                                + i * out_w
                                + j;
                                
                    output[out_idx] = sum;
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 10) {
        fprintf(stderr, "Uso correto: %s <nome_do_peso> <c_out> <c_in> <h_in> <w_in> <kh> <kw> <stride> <padding>\n", argv[0]);
        fprintf(stderr, "Exemplo (1a camada): %s resnetv15_conv0_weight 64 3 224 224 7 7 2 3\n", argv[0]);
        return 1;
    }

    // Captura dos 9 parâmetros necessários
    char *kernel_name = argv[1];
    int c_out   = atoi(argv[2]);
    int c_in    = atoi(argv[3]);
    int h_in    = atoi(argv[4]);
    int w_in    = atoi(argv[5]);
    int kh      = atoi(argv[6]);
    int kw      = atoi(argv[7]);
    int stride  = atoi(argv[8]);
    int padding = atoi(argv[9]);

    // 1. Localiza os pesos corretos na árvore binária compilada
    int kernel_len = 0;
    const float *weights_ptr = get_kernel_pointer(kernel_name, &kernel_len);

    if (!weights_ptr) {
        fprintf(stderr, "Erro: O peso '%s' não foi localizado nos cabeçalhos.\n", kernel_name);
        return 1;
    }

    // Validação estrita do tensor tridimensional de pesos
    if (kernel_len != (c_out * c_in * kh * kw)) {
        fprintf(stderr, "Erro: O shape do kernel fornecido (%dx%dx%dx%d = %d) não condiz com o tamanho do array carregado (%d).\n",
                c_out, c_in, kh, kw, (c_out * c_in * kh * kw), kernel_len);
        return 1;
    }

    // Calcula os tamanhos espaciais da saída baseado na fórmula da convolução
    int out_h = ((h_in + 2 * padding - kh) / stride) + 1;
    int out_w = ((w_in + 2 * padding - kw) / stride) + 1;

    printf("Executando camada: %s\n", kernel_name);
    printf("Input Shape:  (%d, %d, %d, %d)\n", B, c_in, h_in, w_in);
    printf("Kernel Shape: (%d, %d, %d, %d)\n", c_out, c_in, kh, kw);
    printf("Stride: %d | Padding: %d\n", stride, padding);

    // Alocação dinâmica utilizando VLAs (Variable Length Arrays) para mapear os saltos de ponteiros automaticamente
    float (*input4d)[c_in][h_in][w_in] = malloc(B * sizeof *input4d);
    float *output_flat = malloc(B * c_out * out_h * out_w * sizeof(float));

    if (!input4d || !output_flat) {
        fprintf(stderr, "Erro de alocação de memória.\n");
        return 1;
    }

    // --- CARREGA A IMAGEM DE ENTRADA DO ARQUIVO BINÁRIO ---
    FILE *file_in = fopen("input_image.bin", "rb");
    if (!file_in) {
        fprintf(stderr, "Erro: Arquivo 'input_image.bin' nao encontrado!\n");
        free(input4d); free(output_flat);
        return 1;
    }
    fread(input4d, sizeof(float), B * c_in * h_in * w_in, file_in);
    fclose(file_in);

    // Executa a Convolução
    conv2d3d(c_out, c_in, h_in, w_in, kh, kw, stride, padding, input4d, weights_ptr, output_flat);

    // --- SALVA A SAÍDA PARA O PRÓXIMO KERNEL ---
    FILE *file_out = fopen("conv_output.bin", "wb");
    if (!file_out) {
        fprintf(stderr, "Erro ao criar o arquivo de saída 'conv_output.bin'.\n");
        free(input4d); free(output_flat);
        return 1;
    }
    fwrite(output_flat, sizeof(float), B * c_out * out_h * out_w, file_out);
    fclose(file_out);
    
    printf("Convolução finalizada! Output shape esperado pelo ONNX: (%d, %d, %d, %d)\n", B, c_out, out_h, out_w);
    printf("Primeiro pixel de saída calculado: %f\n", output_flat[0]);

    // Desalocação limpa
    free(input4d);
    free(output_flat);

    return 0;
}