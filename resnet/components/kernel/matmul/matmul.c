#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "matmulworld.h"

// MatMul em lote (batched) com broadcasting de batch dims, seguindo a
// mesma regra do ONNX MatMul / numpy: as duas ultimas dimensoes de cada
// shape sao as dimensoes de matriz (M,K) e (K,N); tudo antes eh batch e
// pode fazer broadcast (dimensao 1 contra qualquer tamanho, alinhado pela
// direita). lhs_shape/rhs_shape sao resolvidos estaticamente pelo
// onnx_to_dfg.py a partir do value_info do grafo (nao vem metadado de
// shape junto do buffer, igual ao Transpose/Split).
static void compute_batch_matmul(
    const float *lhs, const uint32_t *lhs_shape, size_t lhs_rank,
    const float *rhs, const uint32_t *rhs_shape, size_t rhs_rank,
    float **out_buf, size_t *out_len
) {
    size_t lhs_batch_rank = lhs_rank - 2;
    size_t rhs_batch_rank = rhs_rank - 2;
    size_t batch_rank = lhs_batch_rank > rhs_batch_rank ? lhs_batch_rank : rhs_batch_rank;

    // batch_shape[0] = dimensao de batch mais lenta (esquerda), ...
    // batch_shape[batch_rank-1] = mais rapida (direita, mais perto de M/K).
    uint32_t *batch_shape = (uint32_t*) malloc((batch_rank ? batch_rank : 1) * sizeof(uint32_t));
    for (size_t i = 0; i < batch_rank; i++) {
        long li = (long) lhs_batch_rank - 1 - (long)(batch_rank - 1 - i);
        long ri = (long) rhs_batch_rank - 1 - (long)(batch_rank - 1 - i);
        uint32_t ld = (li >= 0) ? lhs_shape[li] : 1;
        uint32_t rd = (ri >= 0) ? rhs_shape[ri] : 1;
        batch_shape[i] = (ld > rd) ? ld : rd;
    }

    size_t total_batch = 1;
    for (size_t i = 0; i < batch_rank; i++) total_batch *= (size_t) batch_shape[i];

    size_t M = (size_t) lhs_shape[lhs_rank - 2];
    size_t K = (size_t) lhs_shape[lhs_rank - 1];
    size_t N = (size_t) rhs_shape[rhs_rank - 1];

    size_t lhs_mat_size = M * K;
    size_t rhs_mat_size = K * N;
    size_t out_mat_size = M * N;

    size_t out_total = total_batch * out_mat_size;
    float *out = (float*) malloc(out_total * sizeof(float));

    // Stride (em elementos) de cada dimensao de batch dentro de lhs/rhs.
    // Dimensao broadcast (tamanho 1, sem correspondente real no outro
    // tensor) tem stride 0 — sempre le o mesmo bloco de memoria.
    size_t *lhs_bstride = (size_t*) malloc((batch_rank ? batch_rank : 1) * sizeof(size_t));
    size_t *rhs_bstride = (size_t*) malloc((batch_rank ? batch_rank : 1) * sizeof(size_t));
    {
        size_t lhs_running = lhs_mat_size;
        size_t rhs_running = rhs_mat_size;
        for (size_t idx = 0; idx < batch_rank; idx++) {
            size_t i = batch_rank - 1 - idx;
            long li = (long) lhs_batch_rank - (long) batch_rank + (long) i;
            long ri = (long) rhs_batch_rank - (long) batch_rank + (long) i;
            uint32_t ld = (li >= 0) ? lhs_shape[li] : 1;
            uint32_t rd = (ri >= 0) ? rhs_shape[ri] : 1;
            lhs_bstride[i] = (ld == 1) ? 0 : lhs_running;
            rhs_bstride[i] = (rd == 1) ? 0 : rhs_running;
            if (ld != 1) lhs_running *= ld;
            if (rd != 1) rhs_running *= rd;
        }
    }

    size_t *coord = (size_t*) malloc((batch_rank ? batch_rank : 1) * sizeof(size_t));
    for (size_t b = 0; b < total_batch; b++) {
        size_t rem = b;
        for (size_t i = 0; i < batch_rank; i++) {
            size_t dim = batch_shape[batch_rank - 1 - i];
            coord[batch_rank - 1 - i] = rem % dim;
            rem /= dim;
        }
        size_t lhs_off = 0, rhs_off = 0;
        for (size_t i = 0; i < batch_rank; i++) {
            lhs_off += coord[i] * lhs_bstride[i];
            rhs_off += coord[i] * rhs_bstride[i];
        }
        const float *lhs_mat = lhs + lhs_off;
        const float *rhs_mat = rhs + rhs_off;
        float *out_mat = out + b * out_mat_size;

        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                float acc = 0.0f;
                for (size_t k = 0; k < K; k++) {
                    acc += lhs_mat[m * K + k] * rhs_mat[k * N + n];
                }
                out_mat[m * N + n] = acc;
            }
        }
    }

    free(coord);
    free(lhs_bstride);
    free(rhs_bstride);
    free(batch_shape);

    *out_buf = out;
    *out_len = out_total;
}

void exports_planner_matmulworld_plan_matmul(
    matmulworld_list_f32_t *lhs,
    matmulworld_list_f32_t *rhs,
    matmulworld_list_u32_t *lhs_shape,
    matmulworld_list_u32_t *rhs_shape,
    exports_planner_matmulworld_plan_data_t *result_container,
    exports_planner_matmulworld_plan_data_t *ret
) {
    float *out_buf;
    size_t out_len;
    compute_batch_matmul(
        lhs->ptr, lhs_shape->ptr, lhs_shape->len,
        rhs->ptr, rhs_shape->ptr, rhs_shape->len,
        &out_buf, &out_len
    );

    ret->output.ptr = out_buf;
    ret->output.len = out_len;

    ret->reply_to = result_container->reply_to;
    ret->current_kernel = result_container->current_kernel;
    ret->request = result_container->request;
    ret->from_step = result_container->from_step;

    ret->input.ptr = NULL;
    ret->input.len = 0;
    ret->kernels.ptr = NULL;
    ret->kernels.len = 0;
}
