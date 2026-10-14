#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "convworld.h"

#define B 1

void exports_planner_convworld_plan_conv(
    uint32_t c_out, 
    uint32_t c_in, 
    uint32_t h_in, 
    uint32_t w_in, 
    uint32_t kh, 
    uint32_t kw, 
    uint32_t stride, 
    uint32_t padding, 
    uint32_t groups,
    convworld_list_f32_t *weights, 
    convworld_list_f32_t *bias,
    exports_planner_convworld_plan_data_t *img, 
    exports_planner_convworld_plan_data_t *ret) {

    // 1. Extração dos ponteiros lineares flat gerados pelo wit-bindgen
    float *input_flat  = img->input.ptr;
    float *kernel_flat = weights->ptr;
    float *output_flat = img->output.ptr;
    // bias é opcional: lista vazia (len == 0) significa "sem bias", igual ao
    // comportamento de muitas convs do ResNet (bias absorvido pelo BN seguinte).
    float *bias_flat = bias->len > 0 ? bias->ptr : NULL;

    // Fórmula universal para cálculo de saída espacial com padding duplo
    int out_h = ((h_in + 2 * padding - kh) / stride) + 1;
    int out_w = ((w_in + 2 * padding - kw) / stride) + 1;

    // Convolução agrupada (groups > 1, incluindo depthwise onde
    // groups == c_in == c_out): cada canal de saída só enxerga os canais de
    // entrada do seu próprio grupo. groups=1 é o caso clássico (VGG/ResNet),
    // onde cin_per_group == c_in e ci_offset é sempre 0 — matemática
    // idêntica ao código anterior, sem groups.
    int cin_per_group = c_in / groups;
    int cout_per_group = c_out / groups;

    for (int b = 0; b < B; b++) {
        for (int co = 0; co < c_out; co++) {
            int group_id = co / cout_per_group;
            int ci_offset = group_id * cin_per_group;

            for (int i = 0; i < out_h; i++) {
                for (int j = 0; j < out_w; j++) {
                    float sum = bias_flat != NULL ? bias_flat[co] : 0.0f;
                    
                    int in_base_h = i * stride - padding;
                    int in_base_w = j * stride - padding;
                    
                    for (int ci = 0; ci < cin_per_group; ci++) {
                        int real_ci = ci_offset + ci;
                        for (int ki = 0; ki < kh; ki++) {
                            for (int kj = 0; kj < kw; kj++) {
                                int in_h = in_base_h + ki;
                                int in_w = in_base_w + kj;
                                
                                // Tratamento de Zero-Padding genérico para qualquer borda
                                if (in_h >= 0 && in_h < h_in && in_w >= 0 && in_w < w_in) {
                                    int input_idx = b * (c_in * h_in * w_in)
                                            + real_ci * (h_in * w_in)
                                            + in_h * w_in
                                            + in_w;

                                    // Peso no formato ONNX [c_out, c_in/groups, kh, kw] —
                                    // já indexado por ci LOCAL ao grupo, sem offset.
                                    int kernel_idx = co * (cin_per_group * kh * kw) 
                                            + ci * (kh * kw) 
                                            + ki * kw 
                                            + kj;
                            
                                    sum += input_flat[input_idx] * kernel_flat[kernel_idx];
                                }
                            }
                        }
                    }
                    // Mapeamento linear seguro da saída estruturada dinamicamente [b][co][i][j]
                    int out_idx = b * (c_out * out_h * out_w)
                    + co * (out_h * out_w)
                    + i * out_w
                    + j;
                    output_flat[out_idx] = sum;
                }
            }
        }
    }
    *ret = *img;
}