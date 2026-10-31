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
    ret->from_step = result_container->from_step;
    ret->model_id = result_container->model_id;

    // Inicializa listas auxiliares
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
    addworld_list_f32_free(a);
    addworld_list_f32_free(b);
    addworld_list_f32_free(&result_container->input);
    addworld_list_f32_free(&result_container->output);
    addworld_list_u8_free(&result_container->kernels);
}
