# how to use
# python3 simulador.py W X Y Z   (W CPU cores unit - X % of cpu usage - Y memory usage - Z time)
# python3 simulador.py 32 50 32154 60  (will use 32 cores, at 50%. 32154 of memory -31.4GB. for 60 seconds)
# htop you can see how much of memory and cpu the server has. for each GB of memory, use 1024. So for 4GB calculate 4 * 1024 - 4096.
# for a better utilization, calculate how much of memory do you want to use as this example
# Desired: utilize 50% of memory. HTOP says that I have 62.8 in total and 7.89G in use. Half of total minus what's in use: (62.8 / 2) - 7.89 -> 23,51. Now multiply this by 1024. -> 24074,24. You can use this number in the script

import time
import sys
import multiprocessing

def estressar_cpu(porcentagem_alvo):
    # Ciclo de tempo de 100ms (0.1 segundos)
    intervalo_total = 0.1
    tempo_trabalho = intervalo_total * (porcentagem_alvo / 100.0)
    tempo_sono = intervalo_total - tempo_trabalho

    print(f"-> Worker de CPU iniciado simulando {porcentagem_alvo}%")
    while True:
        tempo_inicio = time.perf_counter()
        # Preenche o tempo de trabalho rodando um loop inútil
        while time.perf_counter() - tempo_inicio < tempo_trabalho:
            pass  # Gasta ciclo de CPU
        # Dorme o restante do ciclo para aliviar a média
        time.sleep(tempo_sono)

def iniciar_simulacao():
    if len(sys.argv) < 4:
        print("Uso: python3 simulador.py <num_cpus> <porcentagem_cpu> <memoria_mb> <tempo_segundos>")
        print("Exemplo: python3 simulador.py 2 25 400 60")
        sys.exit(1)

    num_cpus = int(sys.argv[1])
    porcentagem_cpu = int(sys.argv[2])
    memoria_mb = int(sys.argv[3])
    tempo_segundos = int(sys.argv[4])

    print(f"=== Iniciando Simulação por {tempo_segundos}s ===")
    
    # 1. ESTRESSE DE MEMÓRIA (Aloca uma string de bytes estável)
    print(f"-> Alocando {memoria_mb} MB de RAM...")
    # Cada caractere ocupa 1 byte, então multiplicamos para chegar em MB
    ancora_memoria = "X" * (memoria_mb * 1024 * 1024)
    print("-> Memória alocada e travada com sucesso!")

    # 2. ESTRESSE DE CPU (Inicia um processo independente para cada núcleo)
    processos_cpu = []
    for _ in range(num_cpus):
        p = multiprocessing.Process(target=estressar_cpu, args=(porcentagem_cpu,))
        p.daemon = True
        p.start()
        processos_cpu.append(p)

    # 3. TEMPO DE EXECUÇÃO
    print(f"\n[OK] Simulação rodando de forma estável. Monitore no htop.")
    time.sleep(tempo_segundos)
    
    # Finalização limpa
    print("\n=== Tempo esgotado! Liberando recursos... ===")
    del ancora_memoria  # Libera a memória explicitamente

if __name__ == "__main__":
    iniciar_simulacao()