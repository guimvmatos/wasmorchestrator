#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "grayscaleworld.h"

void exports_planner_grayscaleworld_plan_grayscale(exports_planner_grayscaleworld_plan_imagedata_t *img, exports_planner_grayscaleworld_plan_imagedata_t *ret){
    uint8_t *data = img->pixels.ptr;
    uint32_t width = img->width;
    uint32_t height = img->height;
    for (size_t i = 0; i < (size_t)width * height * 3; i += 3) {
        uint8_t gray = (uint8_t)(0.299f * data[i] + 0.587f * data[i+1] + 0.114f * data[i+2]);
        data[i] = data[i+1] = data[i+2] = gray;
    }
    *ret = *img;
}