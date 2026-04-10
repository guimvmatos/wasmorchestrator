# Estágio de compilação
FROM gcc:latest AS builder
COPY negativeBench.c .
# Adicionamos a flag -static
RUN gcc -O3 negativeBench.c -lm -static -o /negativeBench

# Estágio de execução
FROM alpine:latest
COPY --from=builder /negativeBench /negativeBench
# Garante que o arquivo tenha permissão de execução
RUN chmod +x /negativeBench
ENTRYPOINT ["/negativeBench"]