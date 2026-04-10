#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

// --- Estruturas e IO ---

typedef struct {
    int width, height;
    uint8_t *data;
} Image;

Image read_ppm(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    Image img = {0, 0, NULL};
    if (!fp) return img;

    char format[3];
    int max_val;
    // Lê cabeçalho: Tipo, Largura, Altura e Valor Máximo
    if (fscanf(fp, "%s\n%d %d\n%d\n", format, &img.width, &img.height, &max_val) != 4) {
        fclose(fp);
        return img;
    }

    if (format[0] != 'P' || format[1] != '6') {
        fclose(fp);
        return img;
    }

    img.data = (uint8_t*)malloc(img.width * img.height * 3);
    fread(img.data, 3, img.width * img.height, fp);
    fclose(fp);
    return img;
}

void write_ppm(const char *filename, Image img) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) return;
    fprintf(fp, "P6\n%d %d\n255\n", img.width, img.height);
    fwrite(img.data, 3, img.width * img.height, fp);
    fclose(fp);
}

// --- Kernels de Processamento ---

void grayscale_kernel(uint8_t* data, int width, int height) {
    for (int i = 0; i < width * height * 3; i += 3) {
        uint8_t gray = (uint8_t)(0.299f * data[i] + 0.587f * data[i+1] + 0.114f * data[i+2]);
        data[i] = data[i+1] = data[i+2] = gray;
    }
}

uint8_t get_pixel(uint8_t* data, int w, int h, int x, int y) {
    if (x < 0 || x >= w || y < 0 || y >= h) return 0;
    return data[(y * w + x) * 3]; 
}

void sobel_filter(uint8_t* input, uint8_t* output, int w, int h) {
    int gx_mask[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int gy_mask[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int sumX = 0;
            int sumY = 0;

            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    uint8_t pixel = get_pixel(input, w, h, x + j, y + i);
                    sumX += pixel * gx_mask[i + 1][j + 1];
                    sumY += pixel * gy_mask[i + 1][j + 1];
                }
            }

            int magnitude = abs(sumX) + abs(sumY);
            if (magnitude > 255) magnitude = 255;

            int idx = (y * w + x) * 3;
            output[idx] = output[idx + 1] = output[idx + 2] = (uint8_t)magnitude;
        }
    }
}

void threshold_kernel(uint8_t* data, int width, int height, uint8_t limit) {
    for (int i = 0; i < width * height * 3; i += 3) {
        // Se o valor (qualquer canal, já que é cinza) for maior que o limite
        uint8_t value = (data[i] > limit) ? 255 : 0;
        
        data[i]     = value;
        data[i + 1] = value;
        data[i + 2] = value;
    }
}

// --- Main ---

int main() {
    // 1. Carregar
    Image img = read_ppm("entrada.ppm");
    if (!img.data) {
        printf("Erro: Certifique-se de que 'gl.ppm' existe.\n");
        return 1;
    }

    // 2. Grayscale (O Sobel precisa de uma base monocromática)
    grayscale_kernel(img.data, img.width, img.height);

    // 3. Preparar buffer de saída para o Sobel
    uint8_t* sobel_data = (uint8_t*)malloc(img.width * img.height * 3);
    
    printf("Processando Sobel em imagem %dx%d...\n", img.width, img.height);
    sobel_filter(img.data, sobel_data, img.width, img.height);

    printf("Aplicando Threshold para binarizar as bordas...\n");
    // 100 é um bom valor inicial, mas você pode testar entre 50 e 150
    threshold_kernel(sobel_data, img.width, img.height, 100);

    // 4. Salvar
    Image final_img = {img.width, img.height, sobel_data};
    write_ppm("pipeline_final.ppm", final_img);

    printf("Sucesso! Verifique 'pipeline_final.ppm'.\n");

    // 5. Limpar
    free(img.data);
    free(sobel_data);
    return 0;
}