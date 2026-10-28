#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "softmaxworld.h"

// Softmax numericamente estavel ao longo de UM eixo de um tensor N-D
// flattened (row-major). outer = produto das dims antes do axis, inner =
// produto das dims depois do axis — mesma decomposicao outer/axis/inner
// usada pelo Concat/Split.
static void compute_softmax(
    const float *input, const uint32_t *shape, size_t rank, uint32_t axis,
    float **out_buf, size_t *out_len
) {
    size_t outer = 1;
    for (size_t i = 0; i < (size_t) axis; i++) outer *= (size_t) shape[i];
    size_t axis_size = (size_t) shape[axis];
    size_t inner = 1;
    for (size_t i = (size_t) axis + 1; i < rank; i++) inner *= (size_t) shape[i];

    size_t total = outer * axis_size * inner;
    float *buf = (float*) malloc(total * sizeof(float));

    for (size_t o = 0; o < outer; o++) {
        for (size_t in = 0; in < inner; in++) {
            size_t base = o * axis_size * inner + in;

            float maxv = input[base];
            for (size_t a = 1; a < axis_size; a++) {
                float v = input[base + a * inner];
                if (v > maxv) maxv = v;
            }

            float sum = 0.0f;
            for (size_t a = 0; a < axis_size; a++) {
                float e = expf(input[base + a * inner] - maxv);
                buf[base + a * inner] = e;
                sum += e;
            }

            for (size_t a = 0; a < axis_size; a++) {
                buf[base + a * inner] /= sum;
            }
        }
    }

    *out_buf = buf;
    *out_len = total;
}

void exports_planner_softmaxworld_plan_softmax(
    softmaxworld_list_f32_t *input,
    softmaxworld_list_u32_t *shape,
    uint32_t axis,
    exports_planner_softmaxworld_plan_data_t *result_container,
    exports_planner_softmaxworld_plan_data_t *ret
) {
    float *out_buf;
    size_t out_len;
    compute_softmax(input->ptr, shape->ptr, shape->len, axis, &out_buf, &out_len);

    ret->output.ptr = out_buf;
    ret->output.len = out_len;

    ret->reply_to = result_container->reply_to;
    ret->current_kernel = result_container->current_kernel;
    ret->request = result_container->request;
    ret->from_step = result_container->from_step;
    ret->model_id = result_container->model_id;

    ret->input.ptr = NULL;
    ret->input.len = 0;
    ret->kernels.ptr = NULL;
    ret->kernels.len = 0;
}
