#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "sigmoidworld.h"

void exports_planner_sigmoidworld_plan_sigmoid(
    exports_planner_sigmoidworld_plan_data_t *img,
    exports_planner_sigmoidworld_plan_data_t *ret) {

    float *input_flat = img->input.ptr;
    float *output_flat = img->output.ptr;
    size_t len = img->input.len;

    for (size_t i = 0; i < len; i++) {
        float x = input_flat[i];
        output_flat[i] = 1.0f / (1.0f + expf(-x));
    }

    *ret = *img;
}