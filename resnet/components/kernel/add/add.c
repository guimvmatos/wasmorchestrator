#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "addworld.h"

void exports_planner_addworld_plan_add(
    addworld_list_f32_t *a, 
    addworld_list_f32_t *b, 
    exports_planner_addworld_plan_data_t *result_container, 
    exports_planner_addworld_plan_data_t *ret) {

    size_t size = a->len;

    // Aloca vetor de saída se necessário
    ret->output.ptr = (float*) malloc(size * sizeof(float));
    ret->output.len = size;

    #pragma omp simd
    for (size_t i = 0; i < size; i++) {
        ret->output.ptr[i] = a->ptr[i] + b->ptr[i];
    }

    // Copia metadados do container
    ret->reply_to = result_container->reply_to;
    ret->current_kernel = result_container->current_kernel;
    ret->request = result_container->request;

    // Inicializa listas auxiliares
    ret->input.ptr = NULL;
    ret->input.len = 0;
    ret->kernels.ptr = NULL;
    ret->kernels.len = 0;
}