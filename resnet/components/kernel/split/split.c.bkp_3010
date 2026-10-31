#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "splitworld.h"

// Inverso do Concat: fatia `input` em `sizes.len` pedaços ao longo de `axis`.
// Puro rearranjo de memória, sem ganho em GPU. Devolve uma lista de buffers,
// um por saída, na MESMA ordem de `sizes` — quem decide pra onde cada um vai
// é o receiver, usando output_routes do grafo. Esta função não devolve
// metadados (reply_to/current_kernel/etc.) porque o WIT declara o retorno
// como list<list<f32>> puro — o receiver (Rust) reaproveita os metadados do
// próprio input_img, que não passa por aqui.
void exports_planner_splitworld_plan_split(
    splitworld_list_f32_t *input,
    splitworld_list_u32_t *shape,
    uint32_t axis,
    splitworld_list_u32_t *sizes,
    exports_planner_splitworld_plan_data_t *result_container,
    splitworld_list_list_f32_t *ret) {

    (void) result_container; // não usado: sem metadados no retorno deste kernel

    size_t rank = shape->len;

    size_t outer = 1;
    for (size_t i = 0; i < (size_t) axis; i++) {
        outer *= (size_t) shape->ptr[i];
    }
    size_t inner = 1;
    for (size_t i = (size_t) axis + 1; i < rank; i++) {
        inner *= (size_t) shape->ptr[i];
    }

    size_t total_axis = 0;
    for (size_t i = 0; i < sizes->len; i++) {
        total_axis += (size_t) sizes->ptr[i];
    }

    size_t n_outputs = sizes->len;
    ret->ptr = (splitworld_list_f32_t*) malloc(n_outputs * sizeof(splitworld_list_f32_t));
    ret->len = n_outputs;

    size_t offset = 0; // posição corrente no axis, em elementos (não floats)
    for (size_t i = 0; i < n_outputs; i++) {
        size_t size = (size_t) sizes->ptr[i];
        size_t chunk_len = size * inner;
        size_t out_total = outer * chunk_len;

        float *buf = (float*) malloc(out_total * sizeof(float));

        for (size_t o = 0; o < outer; o++) {
            size_t src_start = o * total_axis * inner + offset * inner;
            size_t dst_start = o * chunk_len;
            memcpy(buf + dst_start, input->ptr + src_start, chunk_len * sizeof(float));
        }

        ret->ptr[i].ptr = buf;
        ret->ptr[i].len = out_total;

        offset += size;
    }
}