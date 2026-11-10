#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "convbnworld.h"

// Conv + BatchNormalization fundidos (kernel_type 23). Mesma convolução
// agrupada do kernel conv (1) e mesma fórmula do kernel bn (2), aplicada no
// epílogo: out = (sum + bias) * a + b, com a = scale/sqrt(var+eps) e
// b = beta - mean*a. Evita gravar a saída do Conv e relê-la no BN (e a
// travessia pela fronteira do componente entre os dois passos).

#define B 1

void exports_planner_convbnworld_plan_convbn(
    uint32_t c_out,
    uint32_t c_in,
    uint32_t h_in,
    uint32_t w_in,
    uint32_t kh,
    uint32_t kw,
    uint32_t stride,
    uint32_t padding,
    uint32_t groups,
    float epsilon,
    convbnworld_list_f32_t *weights,
    convbnworld_list_f32_t *bias,
    convbnworld_list_f32_t *scale,
    convbnworld_list_f32_t *beta,
    convbnworld_list_f32_t *mean,
    convbnworld_list_f32_t *var,
    exports_planner_convbnworld_plan_data_t *img,
    exports_planner_convbnworld_plan_data_t *ret) {

    float *input_flat  = img->input.ptr;
    float *kernel_flat = weights->ptr;
    float *output_flat = img->output.ptr;
    // bias do Conv é opcional (lista vazia = sem bias), igual ao kernel conv.
    float *bias_flat = bias->len > 0 ? bias->ptr : NULL;

    if (groups == 0) groups = 1;

    // Coeficientes do BN por canal de saída (mesma conta do kernel bn).
    float *bn_a = (float *)malloc(sizeof(float) * c_out);
    float *bn_b = (float *)malloc(sizeof(float) * c_out);
    for (uint32_t co = 0; co < c_out; co++) {
        float a = scale->ptr[co] / sqrtf(var->ptr[co] + epsilon);
        bn_a[co] = a;
        bn_b[co] = beta->ptr[co] - mean->ptr[co] * a;
    }

    int out_h = ((h_in + 2 * padding - kh) / stride) + 1;
    int out_w = ((w_in + 2 * padding - kw) / stride) + 1;

    int cin_per_group = c_in / groups;
    int cout_per_group = c_out / groups;

    for (int b = 0; b < B; b++) {
        for (int co = 0; co < (int)c_out; co++) {
            int group_id = co / cout_per_group;
            int ci_offset = group_id * cin_per_group;
            float a = bn_a[co];
            float bb = bn_b[co];

            for (int i = 0; i < out_h; i++) {
                for (int j = 0; j < out_w; j++) {
                    float sum = bias_flat != NULL ? bias_flat[co] : 0.0f;

                    int in_base_h = i * stride - padding;
                    int in_base_w = j * stride - padding;

                    for (int ci = 0; ci < cin_per_group; ci++) {
                        int real_ci = ci_offset + ci;
                        for (int ki = 0; ki < (int)kh; ki++) {
                            for (int kj = 0; kj < (int)kw; kj++) {
                                int in_h = in_base_h + ki;
                                int in_w = in_base_w + kj;

                                if (in_h >= 0 && in_h < (int)h_in && in_w >= 0 && in_w < (int)w_in) {
                                    int input_idx = b * (c_in * h_in * w_in)
                                            + real_ci * (h_in * w_in)
                                            + in_h * w_in
                                            + in_w;

                                    // Peso no formato ONNX [c_out, c_in/groups, kh, kw]
                                    int kernel_idx = co * (cin_per_group * kh * kw)
                                            + ci * (kh * kw)
                                            + ki * kw
                                            + kj;

                                    sum += input_flat[input_idx] * kernel_flat[kernel_idx];
                                }
                            }
                        }
                    }
                    int out_idx = b * (c_out * out_h * out_w)
                    + co * (out_h * out_w)
                    + i * out_w
                    + j;
                    output_flat[out_idx] = sum * a + bb;
                }
            }
        }
    }
    free(bn_a);
    free(bn_b);

    *ret = *img;

    // A função exportada é dona dos argumentos (ver conv.c, fix de 30/10).
    convbnworld_list_f32_free(weights);
    convbnworld_list_f32_free(bias);
    convbnworld_list_f32_free(scale);
    convbnworld_list_f32_free(beta);
    convbnworld_list_f32_free(mean);
    convbnworld_list_f32_free(var);
}
