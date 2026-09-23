#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "gemmworld.h"

void exports_planner_gemmworld_plan_gemm(uint32_t in_features, uint32_t out_features, gemmworld_list_f32_t *weights, gemmworld_list_f32_t *bias, exports_planner_gemmworld_plan_data_t *result_container, exports_planner_gemmworld_plan_data_t *ret) {


    float *out_buf = (float *)malloc(out_features * sizeof(float));
    if (!out_buf) {
        ret->output.ptr = NULL;
        ret->output.len = 0;
        return;
    }

    const float *input_ptr = result_container->input.ptr;
    const float *w_ptr = weights->ptr;
    const float *b_ptr = bias->ptr;

    // Y[i] = dot(A, B_row[i]) + C[i]
    for (uint32_t i = 0; i < out_features; ++i) {
        float sum = b_ptr[i];
        uint32_t w_offset = i * in_features;

        for (uint32_t j = 0; j < in_features; ++j) {
            sum += input_ptr[j] * w_ptr[w_offset + j];
        }

        out_buf[i] = sum;
    }

    ret->input.ptr = result_container->input.ptr;
    ret->input.len = result_container->input.len;

    ret->output.ptr = out_buf;
    ret->output.len = out_features;

    ret->reply_to = result_container->reply_to;
    ret->kernels = result_container->kernels;
    ret->current_kernel = result_container->current_kernel;
    ret->request = result_container->request;
}