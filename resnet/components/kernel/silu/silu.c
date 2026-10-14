#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "siluworld.h"

void exports_planner_siluworld_plan_silu(
    exports_planner_siluworld_plan_data_t *img,
    exports_planner_siluworld_plan_data_t *ret) {

    float *input_flat = img->input.ptr;
    float *output_flat = img->output.ptr;
    size_t len = img->input.len;

    for (size_t i = 0; i < len; i++) {
        float x = input_flat[i];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        output_flat[i] = x * sigmoid_x;
    }

    *ret = *img;
}