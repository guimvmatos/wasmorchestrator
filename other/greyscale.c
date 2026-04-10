#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef B
#define B 1        
#endif
#ifndef C_IN
#define C_IN 3
#endif
#ifndef C_OUT
#define C_OUT 64
#endif
#ifndef H
#define H 224
#endif
#ifndef W
#define W 224
#endif
#ifndef KH
#define KH 3
#endif
#ifndef KW
#define KW 3
#endif
#ifndef T
#define T "conv2d3d"
#endif
#ifndef I
#define I (B*C_OUT*(H-KH)*(W-KW)*C_IN*KH*KW)
#endif

int conv2d3d(
    float input[B][C_IN][H][W], float kernel[C_OUT][C_IN][KH][KW], float output[B][C_OUT][H-KH+1][W-KW+1]) {

//uint32_t exports_planner_path_plan_conv2d3d(exports_planner_path_plan_input_t *input) {
    for (int b = 0; b < B; b++) {
        for (int co = 0; co < C_OUT; co++) {
            for (int i = 0; i <= H - KH; i++) {
                for (int j = 0; j <= W - KW; j++) {
                    float sum = 0.0f;
                    for (int ci = 0; ci < C_IN; ci++) {
                        for (int ki = 0; ki < KH; ki++) {
                            for (int kj = 0; kj < KW; kj++) {
                                sum += input[b][ci][i+ki][j+kj] * kernel[co][ci][ki][kj];
                            }
                        }
                    }
                    output[b][co][i][j] = sum;
                }
            }
        }
    }
    return 0;
}

int main() {
    //Measuring setup begin
    struct timespec start, end;
    long long e_start, e_end;
    //Measuring setup end

    //Original function begin
    float (*input4d)[C_IN][H][W] = malloc(B * sizeof *input4d);
    for (int b = 0; b < B; b++)
        for (int c = 0; c < C_IN; c++)
            for (int i = 0; i < H; i++)
                for (int j = 0; j < W; j++)
                    input4d[b][c][i][j] = (rand() % 256) / 255.0f;
    float (*kernel4d)[C_IN][KH][KW] = malloc(C_OUT * sizeof *kernel4d);
    for (int co = 0; co < C_OUT; co++)
        for (int ci = 0; ci < C_IN; ci++)
            for (int ki = 0; ki < KH; ki++)
                for (int kj = 0; kj < KW; kj++)
                    kernel4d[co][ci][ki][kj] = (rand() % 3 - 1); // {-1,0,1}
    float (*output4d)[C_OUT][H-KH+1][W-KW+1] = malloc(B * sizeof *output4d);
    //Original function end

    //Start measuring begin
    clock_gettime(CLOCK_MONOTONIC, &start);
    //Start measuring end

    //main function call
    conv2d3d(input4d, kernel4d, output4d);
    
    //Finish measuring begin
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    long long e_diff;
   
    printf("Kernel time: %.6f s\n", elapsed);
    //Finish measuring end

    
    free(input4d);
    free(kernel4d);
    free(output4d);
}