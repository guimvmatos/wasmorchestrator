#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "gapworld.h"

void exports_planner_gapworld_plan_gap(
    gapworld_list_f32_t *a, 
    exports_planner_gapworld_plan_data_t *result_container, 
    exports_planner_gapworld_plan_data_t *ret) {

    // 31/10: antes era fixo em 512 canais (só servia ResNet18/34). O mapa
    // final das ResNets com entrada 224x224 é sempre 7x7; o número de canais
    // sai do tamanho da entrada (ResNet18/34: 512, ResNet50: 2048).
    const uint32_t spatial_size = 49; // 7 * 7
    if (a->len == 0 || a->len % spatial_size != 0) {
        fprintf(stderr, "[gap] entrada com %u floats não é múltipla de 7x7\n", (unsigned)a->len);
    }
    const uint32_t channels = (uint32_t)(a->len / spatial_size);

    // Aloca a lista de saída (um valor por canal)
    float *out_buf = (float *)malloc(channels * sizeof(float));
    if (!out_buf) {
        // Fallback defensivo
        ret->output.ptr = NULL;
        ret->output.len = 0;
        return;
    }

    const float *input_ptr = a->ptr;

    // Calcula a média para cada canal
    for (uint32_t c = 0; c < channels; ++c) {
        float sum = 0.0f;
        uint32_t offset = c * spatial_size;
        
        for (uint32_t i = 0; i < spatial_size; ++i) {
            sum += input_ptr[offset + i];
        }
        
        out_buf[c] = sum / (float)spatial_size;
    }

    // Preenche o contêiner de retorno
    ret->input.ptr = a->ptr;
    ret->input.len = a->len;

    ret->output.ptr = out_buf;
    ret->output.len = channels;

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
    gapworld_list_f32_free(&result_container->input);
    gapworld_list_f32_free(&result_container->output);
}
