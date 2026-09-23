#define _POSIX_C_SOURCE 199309L
#define _GNU_SOURCE

// Compilar: gcc -O3 maxpool.c -o test_maxpool -lm
// Executar: ./test_maxpool 64 112 112 3 3 2 1 

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define B 1

void maxpool2d_onnx(
    int channels, int h_in, int w_in, int kh, int kw, int stride, int padding,
    const float *input, float *output)
{
    int out_h = ((h_in + 2 * padding - kh) / stride) + 1;
    int out_w = ((w_in + 2 * padding - kw) / stride) + 1;

    for (int b = 0; b < B; b++) {
        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < out_h; i++) {
                for (int j = 0; j < out_w; j++) {
                    
                    float max_val = -FLT_MAX; // Inicializa com o menor float possível
                    int in_base_h = i * stride - padding;
                    int in_base_w = j * stride - padding;

                    for (int ki = 0; ki < kh; ki++) {
                        for (int kj = 0; kj < kw; kj++) {
                            int in_h = in_base_h + ki;
                            int in_w = in_base_w + kj;

                            // Se estiver dentro da imagem, avalia o máximo. 
                            // Se cair no padding, o ONNX adota o comportamento de ignorar ou tratar como borda.
                            if (in_h >= 0 && in_h < h_in && in_w >= 0 && in_w < w_in) {
                                int in_idx = b * (channels * h_in * w_in)
                                           + c * (h_in * w_in)
                                           + in_h * w_in
                                           + in_w;
                                
                                if (input[in_idx] > max_val) {
                                    max_val = input[in_idx];
                                }
                            }
                        }
                    }

                    int out_idx = b * (channels * out_h * out_w)
                                + c * (out_h * out_w)
                                + i * out_w
                                + j;
                    
                    output[out_idx] = max_val;
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 8) {
        fprintf(stderr, "Uso: %s <canais> <h_in> <w_in> <kh> <kw> <stride> <padding>\n", argv[0]);
        fprintf(stderr, "Exemplo: %s 64 112 112 3 3 2 1\n", argv[0]);
        return 1;
    }

    int channels = atoi(argv[1]);
    int h_in     = atoi(argv[2]);
    int w_in     = atoi(argv[3]);
    int kh       = atoi(argv[4]);
    int kw       = atoi(argv[5]);
    int stride   = atoi(argv[6]);
    int padding  = atoi(argv[7]);

    int out_h = ((h_in + 2 * padding - kh) / stride) + 1;
    int out_w = ((w_in + 2 * padding - kw) / stride) + 1;

    float *input_tensor  = malloc(B * channels * h_in * w_in * sizeof(float));
    float *output_tensor = malloc(B * channels * out_h * out_w * sizeof(float));

    if (!input_tensor || !output_tensor) {
        fprintf(stderr, "Erro de alocação de memória.\n");
        return 1;
    }

    // --- LEITURA DO OUTPUT DA RELU ---
    FILE *file_in = fopen("relu_output.bin", "rb");
    if (!file_in) {
        fprintf(stderr, "Erro: Arquivo 'relu_output.bin' nao encontrado!\n");
        free(input_tensor); free(output_tensor);
        return 1;
    }
    fread(input_tensor, sizeof(float), B * channels * h_in * w_in, file_in);
    fclose(file_in);

    // Executa MaxPool
    maxpool2d_onnx(channels, h_in, w_in, kh, kw, stride, padding, input_tensor, output_tensor);

    // --- SALVA O RESULTADO ---
    FILE *file_out = fopen("maxpool_output.bin", "wb");
    if (!file_out) {
        fprintf(stderr, "Erro ao criar 'maxpool_output.bin'.\n");
        free(input_tensor); free(output_tensor);
        return 1;
    }
    fwrite(output_tensor, sizeof(float), B * channels * out_h * out_w, file_out);
    fclose(file_out);

    printf("MaxPool finalizado com sucesso! Output shape: (%d, %d, %d, %d)\n", B, channels, out_h, out_w);
    printf("Primeiro pixel pós-pooling: %f\n", output_tensor[0]);

    free(input_tensor);
    free(output_tensor);
    return 0;
}