#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "resizeworld.h"

// Upsample nearest-neighbor 2D, canal a canal (batch=1, layout CHW
// achatado, mesma convenção de channels/h_in/w_in usada por Conv/MaxPool/
// GlobalAveragePool no resto do pipeline). h_out/w_out = round(h_in*scale_h),
// round(w_in*scale_w) — no YOLO é sempre upsample 2x fixo (FPN top-down),
// mas a fórmula vale pra qualquer scale. coordinate_transformation
// "asymmetric" (default do ONNX pra mode=nearest): src_idx = floor(dst_idx /
// scale), clampado no limite do eixo de entrada. Puro gather de memória —
// sem ganho em GPU, mesmo raciocínio de split/concat.
void exports_planner_resizeworld_plan_resize(
    resizeworld_list_f32_t *input,
    uint32_t channels,
    uint32_t h_in,
    uint32_t w_in,
    float scale_h,
    float scale_w,
    exports_planner_resizeworld_plan_data_t *result_container,
    exports_planner_resizeworld_plan_data_t *ret) {

    uint32_t h_out = (uint32_t)(h_in * scale_h + 0.5f);
    uint32_t w_out = (uint32_t)(w_in * scale_w + 0.5f);
    if (h_out < 1) h_out = 1;
    if (w_out < 1) w_out = 1;

    size_t out_size = (size_t)channels * (size_t)h_out * (size_t)w_out;
    ret->output.ptr = (float*) malloc(out_size * sizeof(float));
    ret->output.len = out_size;

    for (uint32_t c = 0; c < channels; c++) {
        size_t in_base = (size_t)c * (size_t)h_in * (size_t)w_in;
        size_t out_base = (size_t)c * (size_t)h_out * (size_t)w_out;

        for (uint32_t oy = 0; oy < h_out; oy++) {
            uint32_t iy = (uint32_t)((float) oy / scale_h);
            if (iy >= h_in) iy = h_in - 1;
            size_t row_in_base = in_base + (size_t)iy * (size_t)w_in;
            size_t row_out_base = out_base + (size_t)oy * (size_t)w_out;

            for (uint32_t ox = 0; ox < w_out; ox++) {
                uint32_t ix = (uint32_t)((float) ox / scale_w);
                if (ix >= w_in) ix = w_in - 1;
                ret->output.ptr[row_out_base + ox] = input->ptr[row_in_base + ix];
            }
        }
    }

    // Copia metadados do container
    ret->reply_to = result_container->reply_to;
    ret->current_kernel = result_container->current_kernel;
    ret->request = result_container->request;
    ret->from_step = result_container->from_step;

    // Inicializa listas auxiliares
    ret->input.ptr = NULL;
    ret->input.len = 0;
    ret->kernels.ptr = NULL;
    ret->kernels.len = 0;
}