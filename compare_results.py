"""
Load saved evaluation results and print a sorted AUROC comparison
across all classifiers and benchmarks.
"""

import json
from pathlib import Path

RESULTS = {
    "TruthfulQA": {
        "ArmoRM":                      "results/truthfulqa_armorm.json",
        "Probe Gemma2-9b (linear)":    "results/truthfulqa_probe_gemma2_9b_linear.json",
        "Probe Gemma2-9b (LoRA KL)":   "results/truthfulqa_probe_gemma2_9b_lora_kl.json",
        "Probe Llama3.1-8b (linear)":  "results/truthfulqa_probe_llama3_1_8b_linear.json",
        "Probe Llama3.1-8b (LoRA KL)": "results/truthfulqa_probe_llama3_1_8b_lora_kl.json",
        "Probe Llama3.1-8b (LoRA LM)": "results/truthfulqa_probe_llama3_1_8b_lora_lm.json",
        "Probe Qwen2.5-7b (linear)":   "results/truthfulqa_probe_qwen2_5_7b_linear.json",
        "Probe Qwen2.5-7b (LoRA KL)":  "results/truthfulqa_probe_qwen2_5_7b_lora_kl.json",
    },
    "TriviaQA": {
        "ArmoRM":                      "results/triviaqa_armorm.json",
        "Probe Gemma2-9b (linear)":    "results/triviaqa_probe_gemma2_9b_linear.json",
        "Probe Gemma2-9b (LoRA KL)":   "results/triviaqa_probe_gemma2_9b_lora_kl.json",
        "Probe Llama3.1-8b (linear)":  "results/triviaqa_probe_llama3_1_8b_linear.json",
        "Probe Llama3.1-8b (LoRA KL)": "results/triviaqa_probe_llama3_1_8b_lora_kl.json",
        "Probe Llama3.1-8b (LoRA LM)": "results/triviaqa_probe_llama3_1_8b_lora_lm.json",
        "Probe Qwen2.5-7b (linear)":   "results/triviaqa_probe_qwen2_5_7b_linear.json",
        "Probe Qwen2.5-7b (LoRA KL)":  "results/triviaqa_probe_qwen2_5_7b_lora_kl.json",
    },
    "LongFact": {
        "ArmoRM":                      "results/longfact_armorm.json",
        "Probe Gemma2-9b (linear)":    "results/longfact_probe_gemma2_9b_linear.json",
        "Probe Gemma2-9b (LoRA KL)":   "results/longfact_probe_gemma2_9b_lora_kl.json",
        "Probe Llama3.1-8b (linear)":  "results/longfact_probe_llama3_1_8b_linear.json",
        "Probe Llama3.1-8b (LoRA KL)": "results/longfact_probe_llama3_1_8b_lora_kl.json",
        "Probe Llama3.1-8b (LoRA LM)": "results/longfact_probe_llama3_1_8b_lora_lm.json",
        "Probe Qwen2.5-7b (linear)":   "results/longfact_probe_qwen2_5_7b_linear.json",
        "Probe Qwen2.5-7b (LoRA KL)":  "results/longfact_probe_qwen2_5_7b_lora_kl.json",
    },
    "HelpSteer2": {
        "ArmoRM":                      "results/helpsteer2_armorm.json",
        "Probe Gemma2-9b (linear)":    "results/helpsteer2_probe_gemma2_9b_linear.json",
        "Probe Gemma2-9b (LoRA KL)":   "results/helpsteer2_probe_gemma2_9b_lora_kl.json",
        "Probe Llama3.1-8b (linear)":  "results/helpsteer2_probe_llama3_1_8b_linear.json",
        "Probe Llama3.1-8b (LoRA KL)": "results/helpsteer2_probe_llama3_1_8b_lora_kl.json",
        "Probe Llama3.1-8b (LoRA LM)": "results/helpsteer2_probe_llama3_1_8b_lora_lm.json",
        "Probe Qwen2.5-7b (linear)":   "results/helpsteer2_probe_qwen2_5_7b_linear.json",
        "Probe Qwen2.5-7b (LoRA KL)":  "results/helpsteer2_probe_qwen2_5_7b_lora_kl.json",
    },
    "UltraFeedback": {
        "ArmoRM":                      "results/ultrafeedback_armorm.json",
        "Probe Gemma2-9b (linear)":    "results/ultrafeedback_probe_gemma2_9b_linear.json",
        "Probe Gemma2-9b (LoRA KL)":   "results/ultrafeedback_probe_gemma2_9b_lora_kl.json",
        "Probe Llama3.1-8b (linear)":  "results/ultrafeedback_probe_llama3_1_8b_linear.json",
        "Probe Llama3.1-8b (LoRA KL)": "results/ultrafeedback_probe_llama3_1_8b_lora_kl.json",
        "Probe Llama3.1-8b (LoRA LM)": "results/ultrafeedback_probe_llama3_1_8b_lora_lm.json",
        "Probe Qwen2.5-7b (linear)":   "results/ultrafeedback_probe_qwen2_5_7b_linear.json",
        "Probe Qwen2.5-7b (LoRA KL)":  "results/ultrafeedback_probe_qwen2_5_7b_lora_kl.json",
    },
}


def load_aurocs(path: str) -> dict[str, float]:
    """
    Returns a flat dict of {classifier_name: auroc}.
    ArmoRM files have auroc_aggregate + auroc_per_dimension;
    probe files have a single auroc key.
    """
    p = Path(path)
    if not p.exists():
        return {}

    with open(p) as f:
        data = json.load(f)

    aurocs = {}
    if "auroc_aggregate" in data:
        aurocs["aggregate"] = data["auroc_aggregate"]
    if "auroc_per_dimension" in data:
        aurocs.update(data["auroc_per_dimension"])
    if "auroc" in data:
        aurocs["probe"] = data["auroc"]
    if "auroc_mean" in data:
        aurocs["probe"] = data["auroc_mean"]
    if "auroc_aggregate_mean" in data:
        aurocs["aggregate"] = data["auroc_aggregate_mean"]
    if "auroc_per_dimension_mean" in data:
        for attr, val in data["auroc_per_dimension_mean"].items():
            aurocs[attr] = val
    return aurocs


def print_table(benchmark: str, rows: list[tuple[str, float]]) -> None:
    print(f"\n{'='*60}")
    print(f"  {benchmark}")
    print(f"{'='*60}")
    print(f"  {'Classifier':<47} AUROC")
    print(f"  {'-'*55}")
    for name, auroc in rows:
        print(f"  {name:<47} {auroc:.4f}")


def main():
    for benchmark, sources in RESULTS.items():
        rows = []

        for source_name, path in sources.items():
            aurocs = load_aurocs(path)
            if not aurocs:
                print(f"[{benchmark}] {source_name}: no results found at {path}")
                continue
            for classifier, auroc in aurocs.items():
                if source_name == "ArmoRM":
                    label = classifier
                elif len(aurocs) == 1:
                    label = source_name
                else:
                    label = f"{source_name} ({classifier})"
                rows.append((label, auroc))

        if not rows:
            continue

        rows.sort(key=lambda x: x[1], reverse=True)
        print_table(benchmark, rows)

    print()


if __name__ == "__main__":
    main()
