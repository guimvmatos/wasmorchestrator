import pandas as pd
import json
import math

# Carrega o teu dataset
df = pd.read_csv("temp.xlsx - Sheet1.csv")

dataset_formatado = []

for index, row in df.iterrows():
    resolution = f"{row['img_width']}x{row['img_height']}"
    pipeline = str(row['pipeline_demanda'])
    cpu_free = row['node_cpu_free_pct']
    mem_free = row['node_mem_free_pct']
    total_time = int(row['total_client_time_ms'])
    
    # 1. User Prompt (The Orchestrator asking for help)
    prompt_user = (
        f"Analyze the following image processing workload:\n"
        f"- Resolution: {resolution}\n"
        f"- Requested Pipeline: {pipeline}\n"
        f"- Edge Status: {cpu_free}% free CPU, {mem_free}% free RAM.\n"
        f"What is the expected execution behavior and your routing recommendation?"
    )
    
    # 2. Build the Kernel breakdown (only adds if the kernel actually ran)
    kernel_details = []
    if pd.notna(row['k1_total_ms']):
        kernel_details.append(f"Kernel 1 (Grayscale) took approximately {int(row['k1_total_ms'])}ms.")
    if pd.notna(row['k2_total_ms']):
        kernel_details.append(f"Kernel 2 (Sobel) took approximately {int(row['k2_total_ms'])}ms.")
    if pd.notna(row['k3_total_ms']):
        kernel_details.append(f"Kernel 3 (Negative) took approximately {int(row['k3_total_ms'])}ms.")
    
    text_kernels = " ".join(kernel_details)

    # 3. Rich Decision Logic (Chain of Thought reasoning)
    if cpu_free < 20 or mem_free < 25:
        # Edge Collapse Scenario
        reasoning = (
            f"Notice that the Edge node has critically low resources ({cpu_free}% CPU, {mem_free}% RAM). "
            f"In this scenario, the total execution time spikes to {total_time}ms. {text_kernels} "
            f"A severe bottleneck occurs due to local hardware scarcity and memory thrashing."
        )
        decision = "Total Offloading to Cloud"
        
        # Bonus: Suggesting Partial Offloading if the pipeline has the heavy Kernel 2
        if "2" in pipeline and total_time > 2000:
            decision = "Partial or Total Offloading to Cloud"
            reasoning += (
                " Since Kernel 2 (Sobel) is highly compute-intensive, an optimal strategy would be "
                "to process Kernel 1 on the Edge and offload only Kernel 2 to the Cloud."
            )

    elif total_time > 900: # SLA Threshold for reasoning
        # Moderate Overload Scenario
        reasoning = (
            f"The Edge node has moderate available resources, resulting in a total time of {total_time}ms. "
            f"This exceeds the ideal low-latency threshold. {text_kernels}"
        )
        decision = "Offload to Cloud"
    
    else:
        # Edge Idle/Abundant Resources Scenario
        reasoning = (
            f"The Edge node has abundant resources ({cpu_free}% free CPU). "
            f"The processing flows smoothly with a total time of {total_time}ms. {text_kernels}"
        )
        decision = "Process Locally on Edge"

    # 4. Assistant Response (What the AI will learn to output)
    resposta_assistant = (
        f"Performance Analysis:\n{reasoning}\n\n"
        f"Final Recommendation: {decision}."
    )
    
    # Phi-3 / Llama 3 standard format
    texto_treino = f"<|user|>\n{prompt_user}<|end|>\n<|assistant|>\n{resposta_assistant}<|end|>"
    dataset_formatado.append({"text": texto_treino})

# Save the JSONL file
with open("dataset_phi3_cot_english.jsonl", "w", encoding="utf-8") as f:
    for item in dataset_formatado:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("English Chain of Thought dataset generated successfully!")