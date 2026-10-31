#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "bnworld.h"

#define B 1

void exports_planner_bnworld_plan_bn(
    uint32_t channels, 
    uint32_t h_in, 
    uint32_t w_in, 
    float epsilon, 
    bnworld_list_f32_t *scale, 
    bnworld_list_f32_t *beta, 
    bnworld_list_f32_t *mean, 
    bnworld_list_f32_t *var, 
    exports_planner_bnworld_plan_data_t *img, 
    exports_planner_bnworld_plan_data_t *ret) {

    // 1. Extração dos ponteiros lineares flat gerados pelo wit-bindgen
    float *input_flat  = img->input.ptr;
    float *output_flat = img->output.ptr;

    float *scale_ptr = scale->ptr;
    float *beta_ptr  = beta->ptr;
    float *mean_ptr  = mean->ptr;
    float *var_ptr   = var->ptr;

    int spatial_size = h_in * w_in;
for (int b = 0; b < B; b++) {
        for (int c = 0; c < channels; c++) {
            float c_mean  = mean_ptr[c];
            float c_var   = var_ptr[c];
            float c_scale = scale_ptr[c];
            float c_beta  = beta_ptr[c];

            // Pré-calcula a transformação por canal: y = (x * inv_std * scale) + (beta - mean * inv_std * scale)
            float inv_std = 1.0f / sqrtf(c_var + epsilon);
            float a = c_scale * inv_std;
            float b_val = c_beta - (c_mean * a);

            int channel_offset = b * (channels * spatial_size) + c * spatial_size;

            for (int i = 0; i < spatial_size; i++) {
                int idx = channel_offset + i;
                output_flat[idx] = (input_flat[idx] * a) + b_val;
            }
        }
    }

    *ret = *img;

    // --- 30/10: libera os argumentos recebidos (fix de vazamento de memória) ---
    // No ABI de componentes, a função exportada é DONA dos argumentos: o
    // adapter gerado pelo wit-bindgen não os libera, e o post-return só libera
    // o valor de RETORNO. Sem isso, cada chamada vazava os buffers de entrada
    // na memória linear do kernel até estourar os 4 GiB do wasm32
    // (cabi_realloc -> abort). Não liberado aqui: *ret = *img: o record inteiro vai no retorno.
    bnworld_list_f32_free(scale);
    bnworld_list_f32_free(beta);
    bnworld_list_f32_free(mean);
    bnworld_list_f32_free(var);
}
