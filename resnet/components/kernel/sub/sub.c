#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "subworld.h"

void exports_planner_subworld_plan_sub(
    subworld_list_f32_t *a,
    subworld_list_f32_t *b,
    exports_planner_subworld_plan_data_t *result_container,
    exports_planner_subworld_plan_data_t *ret) {

    size_t size = a->len;
    size_t len_b = b->len;

    ret->output.ptr = (float*) malloc(size * sizeof(float));
    ret->output.len = size;

    #pragma omp simd
    for (size_t i = 0; i < size; i++) {
        // Broadcast por módulo (igual ao Mul): cobre elementwise
        // (len_b == size) e escalar puro (len_b == 1). O receiver já
        // garante a ORDEM correta de a/b (a-b, nunca b-a) via const_is_first.
        ret->output.ptr[i] = a->ptr[i] - b->ptr[i % len_b];
    }

    ret->reply_to = result_container->reply_to;
    ret->current_kernel = result_container->current_kernel;
    ret->request = result_container->request;
    ret->from_step = result_container->from_step;
    ret->model_id = result_container->model_id;

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
    subworld_list_f32_free(a);
    subworld_list_f32_free(b);
    subworld_list_f32_free(&result_container->input);
    subworld_list_f32_free(&result_container->output);
    subworld_list_u8_free(&result_container->kernels);
}
