#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "sliceworld.h"

// Slice genérico (ONNX Slice): corta um subconjunto de eixos do tensor,
// suportando passo (step) positivo ou negativo por eixo. `shape` é o shape
// completo do input (batch já resolvido, mesma convenção de Split/
// Transpose/Softmax). starts/ends/steps vêm em paralelo com axes (mesmo
// índice k = mesmo eixo); eixos não listados em `axes` passam inteiros
// (begin=0, end=shape[i], step=1). `axes` já vem normalizado (não-negativo)
// pelo onnx_to_dfg.py; starts/ends podem ser negativos (relativos ao fim) —
// clamp segue a spec do ONNX Slice (begin/end quando step<0 usam [-1,dim-1]
// pra permitir "até o índice 0 inclusive" numa reversão). Puro gather de
// memória — sem FLOPs, sem variante GPU (mesmo raciocínio de split/concat/
// resize/reshape/transpose). Validado com teste funcional local (1D/2D,
// eixo não-final, múltiplos eixos, step negativo/reverso, slice tipo bbox
// do decode do YOLO) antes de entregar.
void exports_planner_sliceworld_plan_slice(
    sliceworld_list_f32_t *input,
    sliceworld_list_u32_t *shape,
    sliceworld_list_s32_t *starts,
    sliceworld_list_s32_t *ends,
    sliceworld_list_u32_t *axes,
    sliceworld_list_s32_t *steps,
    exports_planner_sliceworld_plan_data_t *result_container,
    exports_planner_sliceworld_plan_data_t *ret) {

    size_t rank = shape->len;

    long *begin = (long*) malloc(rank * sizeof(long));
    long *end_  = (long*) malloc(rank * sizeof(long));
    long *stp   = (long*) malloc(rank * sizeof(long));
    for (size_t i = 0; i < rank; i++) {
        begin[i] = 0;
        end_[i]  = (long) shape->ptr[i];
        stp[i]   = 1;
    }

    size_t naxes = axes->len;
    for (size_t k = 0; k < naxes; k++) {
        uint32_t axis = axes->ptr[k];
        long dim = (long) shape->ptr[axis];
        long s = (long) starts->ptr[k];
        long e = (long) ends->ptr[k];
        long st = (k < steps->len) ? (long) steps->ptr[k] : 1;
        if (st == 0) st = 1; // guarda contra step inválido

        if (st > 0) {
            if (s < 0) s += dim;
            if (s < 0) s = 0;
            if (s > dim) s = dim;
            if (e < 0) e += dim;
            if (e < 0) e = 0;
            if (e > dim) e = dim;
        } else {
            if (s < 0) s += dim;
            if (s < -1) s = -1;
            if (s > dim - 1) s = dim - 1;
            if (e < 0) e += dim;
            if (e < -1) e = -1;
            if (e > dim - 1) e = dim - 1;
        }

        begin[axis] = s;
        end_[axis]  = e;
        stp[axis]   = st;
    }

    // Tamanho de saída por eixo (progressão aritmética begin..end com passo stp)
    size_t *out_shape = (size_t*) malloc(rank * sizeof(size_t));
    for (size_t i = 0; i < rank; i++) {
        long diff = (stp[i] > 0) ? (end_[i] - begin[i]) : (begin[i] - end_[i]);
        long s_abs = (stp[i] > 0) ? stp[i] : -stp[i];
        out_shape[i] = (diff > 0) ? (size_t) ((diff + s_abs - 1) / s_abs) : 0;
    }

    // Strides row-major de entrada e saída
    size_t *in_stride = (size_t*) malloc(rank * sizeof(size_t));
    size_t *out_stride = (size_t*) malloc(rank * sizeof(size_t));
    in_stride[rank - 1] = 1;
    out_stride[rank - 1] = 1;
    for (size_t i = rank - 1; i > 0; i--) {
        in_stride[i - 1] = in_stride[i] * (size_t) shape->ptr[i];
        out_stride[i - 1] = out_stride[i] * out_shape[i];
    }

    size_t total = 1;
    for (size_t i = 0; i < rank; i++) total *= out_shape[i];

    float *buf = (float*) malloc((total ? total : 1) * sizeof(float));
    size_t *out_idx = (size_t*) malloc(rank * sizeof(size_t));

    for (size_t linear = 0; linear < total; linear++) {
        size_t rem = linear;
        for (size_t i = 0; i < rank; i++) {
            out_idx[i] = rem / out_stride[i];
            rem = rem % out_stride[i];
        }

        size_t in_linear = 0;
        for (size_t i = 0; i < rank; i++) {
            long in_coord = begin[i] + (long) out_idx[i] * stp[i];
            in_linear += (size_t) in_coord * in_stride[i];
        }

        buf[linear] = input->ptr[in_linear];
    }

    free(out_idx);
    free(in_stride);
    free(out_stride);
    free(out_shape);
    free(begin);
    free(end_);
    free(stp);

    ret->output.ptr = buf;
    ret->output.len = total;

    ret->reply_to = result_container->reply_to;
    ret->current_kernel = result_container->current_kernel;
    ret->request = result_container->request;
    ret->from_step = result_container->from_step;

    ret->input.ptr = NULL;
    ret->input.len = 0;
    ret->kernels.ptr = NULL;
    ret->kernels.len = 0;
}
