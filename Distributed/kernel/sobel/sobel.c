#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "sobelworld.h"



uint8_t get_pixel(uint8_t* data, int w, int h, int x, int y) {
    if (x < 0 || x >= w || y < 0 || y >= h) return 0;
    return data[(y * w + x) * 3]; 
}

//void sobel_filter(uint8_t* input, uint8_t* output, int w, int h) 
void exports_planner_sobelworld_plan_sobel(exports_planner_sobelworld_plan_imagedata_t *img, exports_planner_sobelworld_plan_imagedata_t *ret){
    uint32_t w = img->width;
    uint32_t h = img->height;
    uint8_t* input = img->pixels.ptr;

    uint8_t* output = (uint8_t*) malloc(img->pixels.len);
    if (output == NULL) return;

    int gx_mask[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int gy_mask[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int sumX = 0;
            int sumY = 0;

            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    uint8_t pixel = get_pixel(input, w, h, x + j, y + i);
                    sumX += pixel * gx_mask[i + 1][j + 1];
                    sumY += pixel * gy_mask[i + 1][j + 1];
                }
            }

            int magnitude = abs(sumX) + abs(sumY);
            if (magnitude > 255) magnitude = 255;

            int idx = (y * w + x) * 3;
            output[idx] = output[idx + 1] = output[idx + 2] = (uint8_t)magnitude;
        }
    }
    ret->width = w;
    ret->height = h;
    ret->pixels.ptr = output;
    ret->pixels.len = img->pixels.len;
    ret->reply_to = img->reply_to;
    ret->kernels = img->kernels;
    ret->current_kernel = img->current_kernel;
}