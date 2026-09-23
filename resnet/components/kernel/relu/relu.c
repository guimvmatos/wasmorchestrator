#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "reluworld.h"

void exports_planner_reluworld_plan_relu(exports_planner_reluworld_plan_data_t *img, exports_planner_reluworld_plan_data_t *ret) {

    float *input_flat  = img->input.ptr;
    float *output_flat = img->output.ptr;
    size_t len = img->input.len;

    for (size_t i = 0; i < len; i++) {
        float v = input_flat[i];
        output_flat[i] = v > 0.0f ? v : 0.0f;
    }
    *ret = *img;
}