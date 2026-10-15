#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "reshapeworld.h"

// Reshape é identidade pura: o buffer linear já está no layout certo (ONNX
// Reshape não move memória, só reinterpreta o shape — isso é metadado do
// grafo, resolvido pelo onnx_to_dfg.py, não algo que o kernel precise
// calcular). Então o kernel só copia input -> output sem tocar nos valores.
void exports_planner_reshapeworld_plan_reshape(
    reshapeworld_list_f32_t *input,
    exports_planner_reshapeworld_plan_data_t *result_container,
    exports_planner_reshapeworld_plan_data_t *ret) {

    size_t len = input->len;
    float *buf = (float*) malloc(len * sizeof(float));
    memcpy(buf, input->ptr, len * sizeof(float));

    ret->output.ptr = buf;
    ret->output.len = len;

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
