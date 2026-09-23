#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "flattenworld.h"

void exports_planner_flattenworld_plan_flatten(
    flattenworld_list_f32_t *a, 
    exports_planner_flattenworld_plan_data_t *result_container, 
    exports_planner_flattenworld_plan_data_t *ret) {

    float *out_buf = (float *)malloc(a->len * sizeof(float));
    if (!out_buf) {
        ret->output.ptr = NULL;
        ret->output.len = 0;
        return;
    }

    // Apenas copia os 512 floats mantendo o vetor linear
    memcpy(out_buf, a->ptr, a->len * sizeof(float));

    ret->input.ptr = a->ptr;
    ret->input.len = a->len;

    ret->output.ptr = out_buf;
    ret->output.len = a->len;

    ret->reply_to = result_container->reply_to;
    ret->kernels = result_container->kernels;
    ret->current_kernel = result_container->current_kernel;
    ret->request = result_container->request;
}