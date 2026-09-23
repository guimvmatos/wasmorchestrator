#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "gapworld.h"

void exports_planner_gapworld_plan_gap(
    gapworld_list_f32_t *a, 
    exports_planner_gapworld_plan_data_t *result_container, 
    exports_planner_gapworld_plan_data_t *ret) {

    // Dimensões do tensor de entrada (512 canais de 7x7)
    const uint32_t channels = 512;
    const uint32_t spatial_size = 49; // 7 * 7

    // Aloca a lista de saída para os 512 valores resultantes
    float *out_buf = (float *)malloc(channels * sizeof(float));
    if (!out_buf) {
        // Fallback defensivo
        ret->output.ptr = NULL;
        ret->output.len = 0;
        return;
    }

    const float *input_ptr = a->ptr;

    // Calcula a média para cada canal
    for (uint32_t c = 0; c < channels; ++c) {
        float sum = 0.0f;
        uint32_t offset = c * spatial_size;
        
        for (uint32_t i = 0; i < spatial_size; ++i) {
            sum += input_ptr[offset + i];
        }
        
        out_buf[c] = sum / (float)spatial_size;
    }

    // Preenche o contêiner de retorno
    ret->input.ptr = a->ptr;
    ret->input.len = a->len;

    ret->output.ptr = out_buf;
    ret->output.len = channels;

    ret->reply_to = result_container->reply_to;
    ret->kernels = result_container->kernels;
    ret->current_kernel = result_container->current_kernel;
    ret->request = result_container->request;
}