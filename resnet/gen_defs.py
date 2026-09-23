import re
import os

def generate_def():
    # Procura por padrões tipo: #include "headers/nome_do_arquivo.h"
    pattern = re.compile(r'#include\s+"headers/(.+)\.h"')
    
    found_count = 0
    with open('weights.h', 'r') as f_in, open('weights.def', 'w') as f_out:
        for line in f_in:
            match = pattern.search(line)
            if match:
                var_name = match.group(1)
                f_out.write(f"X({var_name})\n")
                found_count += 1

    print(f"Sucesso! {found_count} definições geradas em weights.def")

if __name__ == "__main__":
    generate_def()