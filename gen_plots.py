"""
Generate all result plots:
  1. results/heatmaps.png  — ArmoRM attribute correlation heatmaps + label correlation bars
  2. results/auroc_bars.png — sorted AUROC bar charts for all scorers across all benchmarks
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

ATTRIBUTES = [
    'helpsteer-helpfulness', 'helpsteer-correctness', 'helpsteer-coherence',
    'helpsteer-complexity', 'helpsteer-verbosity', 'ultrafeedback-overall_score',
    'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
    'ultrafeedback-honesty', 'ultrafeedback-helpfulness', 'beavertails-is_safe',
    'prometheus-score', 'argilla-overall_quality', 'argilla-judge_lm',
    'code-complexity', 'code-style', 'code-explanation',
    'code-instruction-following', 'code-readability',
]
SHORT = [a.split("-", 1)[1] for a in ATTRIBUTES]

DATASETS = {
    "TruthfulQA":    "results/truthfulqa_armorm.json",
    "TriviaQA":      "results/triviaqa_armorm.json",
    "LongFact":      "results/longfact_armorm.json",
    "HelpSteer2":    "results/helpsteer2_armorm.json",
    "UltraFeedback": "results/ultrafeedback_armorm.json",
}
ORDINAL = {"HelpSteer2", "UltraFeedback"}

SCORERS = {
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

Path("results").mkdir(exist_ok=True)


def load_records(path):
    p = Path(path)
    if not p.exists():
        return None, None
    with open(p) as f:
        data = json.load(f)
    records = data.get("records")
    if not records:
        return None, None
    rewards = np.array([[r["rewards"][a] for a in ATTRIBUTES] for r in records])
    labels  = np.array([r["label"] for r in records])
    return rewards, labels


def load_aurocs(path):
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        data = json.load(f)
    aurocs = {}
    if "auroc_per_dimension" in data:
        aurocs.update(data["auroc_per_dimension"])
    if "auroc" in data:
        key = "aggregate" if "auroc_per_dimension" in data else "probe"
        aurocs[key] = data["auroc"]
    if "auroc_mean" in data:
        aurocs["aggregate"] = data["auroc_mean"]
    return aurocs


def corr_matrix(rewards):
    n = len(ATTRIBUTES)
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r = pearsonr(rewards[:, i], rewards[:, j])[0]
            mat[i, j] = mat[j, i] = r
    return mat


def label_corrs(rewards, labels, ordinal):
    fn = spearmanr if ordinal else pearsonr
    return np.array([fn(rewards[:, i], labels)[0] for i in range(len(ATTRIBUTES))])


datasets_loaded = {}
for name, path in DATASETS.items():
    rewards, labels = load_records(path)
    if rewards is not None:
        datasets_loaded[name] = (rewards, labels)

if datasets_loaded:
    n_ds = len(datasets_loaded)
    fig = plt.figure(figsize=(22, 6 * n_ds))
    outer = gridspec.GridSpec(n_ds, 1, hspace=0.5)

    for row, (name, (rewards, labels)) in enumerate(datasets_loaded.items()):
        ordinal = name in ORDINAL
        mat = corr_matrix(rewards)
        lc  = label_corrs(rewards, labels, ordinal)

        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row],
                                                 width_ratios=[3, 1], wspace=0.35)
        ax_heat = fig.add_subplot(inner[0])
        im = ax_heat.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax_heat.set_xticks(range(len(SHORT)))
        ax_heat.set_yticks(range(len(SHORT)))
        ax_heat.set_xticklabels(SHORT, rotation=45, ha="right", fontsize=7)
        ax_heat.set_yticklabels(SHORT, fontsize=7)
        ax_heat.set_title(f"{name} — Pearson r between reward dimensions (n={len(labels)})",
                          fontsize=10, pad=8)
        plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02)

        ax_bar = fig.add_subplot(inner[1])
        colors = ["#d73027" if v > 0 else "#4575b4" for v in lc]
        ax_bar.barh(range(len(ATTRIBUTES)), lc, color=colors)
        ax_bar.set_yticks(range(len(ATTRIBUTES)))
        ax_bar.set_yticklabels(SHORT, fontsize=7)
        ax_bar.axvline(0, color="black", linewidth=0.8)
        ax_bar.set_xlabel(f"{'Spearman' if ordinal else 'Pearson'} r with label", fontsize=8)
        ax_bar.set_title("Corr. with factuality label", fontsize=9, pad=8)
        ax_bar.invert_yaxis()

    plt.savefig("results/heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved results/heatmaps.png")
else:
    print("No ArmoRM records found — skipping heatmaps")


ARMORM_ATTRS = set(ATTRIBUTES)

benchmarks_data = {}
for benchmark, sources in SCORERS.items():
    rows = []
    for source_name, path in sources.items():
        aurocs = load_aurocs(path)
        for classifier, auroc in aurocs.items():
            if source_name == "ArmoRM":
                label = classifier
                is_probe = False
            else:
                label = source_name if classifier in ("probe", "aggregate") else f"{source_name} ({classifier})"
                is_probe = True
            rows.append((label, auroc, is_probe))
    if rows:
        rows.sort(key=lambda x: x[1], reverse=True)
        benchmarks_data[benchmark] = rows

if benchmarks_data:
    n = len(benchmarks_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 10), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (benchmark, rows) in zip(axes, benchmarks_data.items()):
        labels  = [r[0] for r in rows]
        aurocs  = [r[1] for r in rows]
        colors  = ["#e07b39" if r[2] else "#4878cf" for r in rows]

        y = range(len(labels))
        ax.barh(y, aurocs, color=colors, edgecolor="white", linewidth=0.4)
        ax.axvline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("AUROC", fontsize=9)
        ax.set_title(benchmark, fontsize=11, fontweight="bold")
        ax.set_xlim(0.4, 1.0)

    # legend
    from matplotlib.patches import Patch
    legend = [Patch(color="#4878cf", label="ArmoRM dimension"),
              Patch(color="#e07b39", label="Probe")]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig("results/auroc_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved results/auroc_bars.png")
else:
    print("No AUROC results found — skipping bar charts")
