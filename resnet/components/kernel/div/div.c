#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "divworld.h"

void exports_planner_divworld_plan_div(
    divworld_list_f32_t *a,
    divworld_list_f32_t *b,
    exports_planner_divworld_plan_data_t *result_container,
    exports_planner_divworld_plan_data_t *ret) {

    size_t size = a->len;
    size_t len_b = b->len;

    ret->output.ptr = (float*) malloc(size * sizeof(float));
    ret->output.len = size;

    #pragma omp simd
    for (size_t i = 0; i < size; i++) {
        // Broadcast por módulo (igual ao Mul): cobre elementwise
        // (len_b == size) e escalar puro (len_b == 1). O receiver já
        // garante a ORDEM correta de a/b (a/b, nunca b/a) via const_is_first.
        ret->output.ptr[i] = a->ptr[i] / b->ptr[i % len_b];
    }

    ret->reply_to = result_container->reply_to;
    ret->current_kernel = result_container->current_kernel;
    ret->request = result_container->request;
    ret->from_step = result_container->from_step;

    ret->input.ptr = NULL;
    ret->input.len = 0;
    ret->kernels.ptr = NULL;
    ret->kernels.len = 0;
}
