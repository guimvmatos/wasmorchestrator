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
    reshapeworld_list_f32_free(input);
    reshapeworld_list_f32_free(&result_container->input);
    reshapeworld_list_f32_free(&result_container->output);
    reshapeworld_list_u8_free(&result_container->kernels);
}
