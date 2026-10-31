#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "mulworld.h"

void exports_planner_mulworld_plan_mul(
    mulworld_list_f32_t *a,
    mulworld_list_f32_t *b,
    exports_planner_mulworld_plan_data_t *result_container,
    exports_planner_mulworld_plan_data_t *ret) {

    size_t size = a->len;
    size_t len_b = b->len;

    // Aloca vetor de saída
    ret->output.ptr = (float*) malloc(size * sizeof(float));
    ret->output.len = size;

    #pragma omp simd
    for (size_t i = 0; i < size; i++) {
        // Broadcast por módulo: cobre elementwise (len_b == size) e
        // escalar puro (len_b == 1). NÃO cobre broadcast multi-dimensional
        // real (ex.: const shape [C,1,1] contra input [C,H,W]).
        ret->output.ptr[i] = a->ptr[i] * b->ptr[i % len_b];
    }

    // Copia metadados do container
    ret->reply_to = result_container->reply_to;
    ret->current_kernel = result_container->current_kernel;
    ret->request = result_container->request;
    ret->from_step = result_container->from_step;
    ret->model_id = result_container->model_id;

    // Inicializa listas auxiliares (evita ponteiros stale no retorno)
    ret->input.ptr = NULL;
    ret->input.len = 0;
    ret->kernels.ptr = NULL;
    ret->kernels.len = 0;

    // --- 30/10: libera os argumentos recebidos (fix de vazamento de memória) ---
    // No ABI de componentes, a função exportada é DONA dos argumentos: o
    // adapter gerado pelo wit-bindgen não os libera, e o post-return só libera
    // o valor de RETORNO. Sem isso, cada chamada vazava os buffers de entrada
    // na memória linear do kernel até estourar os 4 GiB do wasm32
    // (cabi_realloc -> abort). Não liberado aqui: reply_to vai no retorno.
    mulworld_list_f32_free(a);
    mulworld_list_f32_free(b);
    mulworld_list_f32_free(&result_container->input);
    mulworld_list_f32_free(&result_container->output);
    mulworld_list_u8_free(&result_container->kernels);
}
