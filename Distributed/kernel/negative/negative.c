#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "negativeworld.h"




void exports_planner_negativeworld_plan_invert(exports_planner_negativeworld_plan_imagedata_t *img, exports_planner_negativeworld_plan_imagedata_t *ret) {
    uint8_t *data = img->pixels.ptr;
    uint32_t width = img->width;
    uint32_t height = img->height;
    for (int i = 0; i < width * height * 3; i++) {
        data[i] = 255 - data[i];
    }
     *ret = *img;
}