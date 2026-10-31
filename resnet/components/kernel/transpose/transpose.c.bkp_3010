#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "transposeworld.h"

// Transpose genérico por permutação de eixos (ONNX Transpose): output_shape[i]
// = shape[perm[i]]. Pra cada elemento de saída, decompõe o índice linear em
// multi-índice de saída, remonta o multi-índice de entrada aplicando a
// permutação (in_idx[perm[i]] = out_idx[i]) e copia o valor. Puro gather de
// memória — sem FLOPs, sem ganho em GPU (mesmo raciocínio de split/concat/
// resize/reshape). Validado com teste funcional local (2D e 3D, permutações
// não-triviais) antes de entregar.
void exports_planner_transposeworld_plan_transpose(
    transposeworld_list_f32_t *input,
    transposeworld_list_u32_t *shape,
    transposeworld_list_u32_t *perm,
    exports_planner_transposeworld_plan_data_t *result_container,
    exports_planner_transposeworld_plan_data_t *ret) {

    size_t rank = shape->len;

    // output_shape[i] = shape[perm[i]]
    uint32_t *out_shape = (uint32_t*) malloc(rank * sizeof(uint32_t));
    for (size_t i = 0; i < rank; i++) {
        out_shape[i] = shape->ptr[perm->ptr[i]];
    }

    // Strides row-major (último eixo varia mais rápido)
    size_t *in_stride = (size_t*) malloc(rank * sizeof(size_t));
    size_t *out_stride = (size_t*) malloc(rank * sizeof(size_t));
    in_stride[rank - 1] = 1;
    out_stride[rank - 1] = 1;
    for (size_t i = rank - 1; i > 0; i--) {
        in_stride[i - 1] = in_stride[i] * (size_t) shape->ptr[i];
        out_stride[i - 1] = out_stride[i] * (size_t) out_shape[i];
    }

    size_t total = 1;
    for (size_t i = 0; i < rank; i++) {
        total *= (size_t) out_shape[i];
    }

    float *buf = (float*) malloc(total * sizeof(float));
    size_t *out_idx = (size_t*) malloc(rank * sizeof(size_t));

    for (size_t linear = 0; linear < total; linear++) {
        // Decompõe linear -> multi-índice de saída
        size_t rem = linear;
        for (size_t i = 0; i < rank; i++) {
            out_idx[i] = rem / out_stride[i];
            rem = rem % out_stride[i];
        }

        // Remonta multi-índice de entrada: in_idx[perm[i]] = out_idx[i]
        size_t in_linear = 0;
        for (size_t i = 0; i < rank; i++) {
            in_linear += out_idx[i] * in_stride[perm->ptr[i]];
        }

        buf[linear] = input->ptr[in_linear];
    }

    free(out_idx);
    free(in_stride);
    free(out_stride);
    free(out_shape);

    ret->output.ptr = buf;
    ret->output.len = total;

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
