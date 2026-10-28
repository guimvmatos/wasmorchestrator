#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "concatworld.h"

// Concat genérico por eixo (porta direta da versão Rust): assume que todos
// os ramos têm o mesmo rank e os mesmos tamanhos em todas as dims exceto
// `axis` (garantia do próprio ONNX — senão o grafo original já não
// validaria). Puro rearranjo de memória (memcpy por fatia), sem ganho em
// GPU — mesmo raciocínio de split/resize. inputs/shapes vêm na MESMA
// ordem (input_order do onnx_to_dfg.py); shapes[i] é o shape completo
// (rank igual pra todos) já resolvido com batch=1 no lugar de dims
// desconhecidas.
void exports_planner_concatworld_plan_concat(
    concatworld_list_list_f32_t *inputs,
    concatworld_list_list_u32_t *shapes,
    uint32_t axis,
    exports_planner_concatworld_plan_data_t *result_container,
    exports_planner_concatworld_plan_data_t *ret) {

    size_t n_inputs = inputs->len;
    concatworld_list_u32_t *shape0 = &shapes->ptr[0];
    size_t rank = shape0->len;

    size_t outer = 1;
    for (size_t i = 0; i < (size_t) axis; i++) {
        outer *= (size_t) shape0->ptr[i];
    }
    size_t inner = 1;
    for (size_t i = (size_t) axis + 1; i < rank; i++) {
        inner *= (size_t) shape0->ptr[i];
    }

    size_t *axis_sizes = (size_t*) malloc(n_inputs * sizeof(size_t));
    size_t total_axis = 0;
    for (size_t i = 0; i < n_inputs; i++) {
        axis_sizes[i] = (size_t) shapes->ptr[i].ptr[axis];
        total_axis += axis_sizes[i];
    }

    size_t out_total = outer * total_axis * inner;
    float *buf = (float*) malloc(out_total * sizeof(float));

    for (size_t o = 0; o < outer; o++) {
        size_t running_offset = 0;
        for (size_t i = 0; i < n_inputs; i++) {
            size_t axis_size = axis_sizes[i];
            size_t chunk_len = axis_size * inner;
            size_t src_start = o * chunk_len;
            size_t dst_start = o * total_axis * inner + running_offset * inner;

            memcpy(buf + dst_start, inputs->ptr[i].ptr + src_start, chunk_len * sizeof(float));

            running_offset += axis_size;
        }
    }

    free(axis_sizes);

    ret->output.ptr = buf;
    ret->output.len = out_total;

    // Copia metadados do container
    ret->reply_to = result_container->reply_to;
    ret->current_kernel = result_container->current_kernel;
    ret->request = result_container->request;
    ret->from_step = result_container->from_step;
    ret->model_id = result_container->model_id;

    // Inicializa listas auxiliares
    ret->input.ptr = NULL;
    ret->input.len = 0;
    ret->kernels.ptr = NULL;
    ret->kernels.len = 0;
}