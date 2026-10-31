#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "dropoutworld.h"

void exports_planner_dropoutworld_plan_dropout(
    dropoutworld_list_f32_t *a, 
    exports_planner_dropoutworld_plan_data_t *result_container, 
    exports_planner_dropoutworld_plan_data_t *ret) {

    float *out_buf = (float *)malloc(a->len * sizeof(float));
    if (!out_buf) {
        ret->output.ptr = NULL;
        ret->output.len = 0;
        return;
    }

    // Modo de inferência: cópia identidade direta dos floats
    memcpy(out_buf, a->ptr, a->len * sizeof(float));

    ret->input.ptr = a->ptr;
    ret->input.len = a->len;

    ret->output.ptr = out_buf;
    ret->output.len = a->len;

    ret->reply_to = result_container->reply_to;
    ret->kernels = result_container->kernels;
    ret->current_kernel = result_container->current_kernel;
    ret->request = result_container->request;
    ret->from_step = result_container->from_step;
    ret->model_id = result_container->model_id;

    // --- 30/10: libera os argumentos recebidos (fix de vazamento de memória) ---
    // No ABI de componentes, a função exportada é DONA dos argumentos: o
    // adapter gerado pelo wit-bindgen não os libera, e o post-return só libera
    // o valor de RETORNO. Sem isso, cada chamada vazava os buffers de entrada
    // na memória linear do kernel até estourar os 4 GiB do wasm32
    // (cabi_realloc -> abort). Não liberado aqui: a vira ret->input; kernels e reply_to vão no retorno.
    dropoutworld_list_f32_free(&result_container->input);
    dropoutworld_list_f32_free(&result_container->output);
}
