#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "maxpoolworld.h"

#define B 1

void exports_planner_maxpoolworld_plan_maxpool(
    uint32_t channels, 
    uint32_t h_in, 
    uint32_t w_in, 
    uint32_t kh, 
    uint32_t kw, 
    uint32_t stride, 
    uint32_t padding, 
    exports_planner_maxpoolworld_plan_data_t *img, 
    exports_planner_maxpoolworld_plan_data_t *ret) {

    float *input_flat  = img->input.ptr;
    float *output_flat = img->output.ptr;

    uint32_t h_out = ((h_in + 2 * padding - kh) / stride) + 1;
    uint32_t w_out = ((w_in + 2 * padding - kw) / stride) + 1;

    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t c = 0; c < channels; c++) {
            uint32_t in_channel_offset  = (b * channels + c) * (h_in * w_in);
            uint32_t out_channel_offset = (b * channels + c) * (h_out * w_out);

            for (uint32_t oh = 0; oh < h_out; oh++) {
                for (uint32_t ow = 0; ow < w_out; ow++) {
                    float max_val = -INFINITY;

                    int32_t start_h = (int32_t)(oh * stride) - (int32_t)padding;
                    int32_t start_w = (int32_t)(ow * stride) - (int32_t)padding;

                    for (uint32_t fh = 0; fh < kh; fh++) {
                        for (uint32_t fw = 0; fw < kw; fw++) {
                            int32_t ih = start_h + (int32_t)fh;
                            int32_t iw = start_w + (int32_t)fw;

                            if (ih >= 0 && ih < (int32_t)h_in && iw >= 0 && iw < (int32_t)w_in) {
                                float val = input_flat[in_channel_offset + ih * w_in + iw];
                                if (val > max_val) {
                                    max_val = val;
                                }
                            }
                        }
                    }

                    output_flat[out_channel_offset + oh * w_out + ow] = max_val;
                }
            }
        }
    }

    *ret = *img;
}