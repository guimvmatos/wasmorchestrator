#include <stdio.h>
#include <string.h>
#include "weights.h"



int main() {
    char model_name[] = "resnetv15_stage1_conv3_weight"; // Vem de um parâmetro
    int found = 0;

    // 2. Define o que o "X" deve fazer
    #define X(name) \
    if (strcmp(model_name, #name) == 0) { \
        printf("Localizado: %s\n", #name); \
        printf("Tamanho: %d\n", name ## _len); \
        printf("Primeiro valor: %f\n", name[0]); \
        found = 1; \
    }

    // 3. Expande a lista baseada nos seus nomes reais
    #include "weights.def"

    #undef X

    if (!found) printf("Peso nao encontrado.\n");
    return 0;
}