"""
Generate all result plots:
  1. results/heatmaps.png  - ArmoRM attribute correlation heatmaps + label correlation bars
  2. results/auroc_bars.png - sorted AUROC bar charts for all scorers across all benchmarks
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
    "HelpSteer2 (correctness)":    "results/helpsteer2_armorm.json",
    "UltraFeedback (truthfulness)": "results/ultrafeedback_armorm.json",
}
ORDINAL = {"HelpSteer2 (correctness)", "UltraFeedback (truthfulness)"}

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
    "HelpSteer2 (correctness)": {
        "ArmoRM":                      "results/helpsteer2_armorm.json",
        "Probe Gemma2-9b (linear)":    "results/helpsteer2_probe_gemma2_9b_linear.json",
        "Probe Gemma2-9b (LoRA KL)":   "results/helpsteer2_probe_gemma2_9b_lora_kl.json",
        "Probe Llama3.1-8b (linear)":  "results/helpsteer2_probe_llama3_1_8b_linear.json",
        "Probe Llama3.1-8b (LoRA KL)": "results/helpsteer2_probe_llama3_1_8b_lora_kl.json",
        "Probe Llama3.1-8b (LoRA LM)": "results/helpsteer2_probe_llama3_1_8b_lora_lm.json",
        "Probe Qwen2.5-7b (linear)":   "results/helpsteer2_probe_qwen2_5_7b_linear.json",
        "Probe Qwen2.5-7b (LoRA KL)":  "results/helpsteer2_probe_qwen2_5_7b_lora_kl.json",
    },
    "UltraFeedback (truthfulness)": {
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
        ax_heat.set_title(f"{name} - Pearson r between reward dimensions (n={len(labels)})",
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

    plt.savefig("results/heatmaps.png", dpi=600, bbox_inches="tight")
    plt.close()
    print("Saved results/heatmaps.png")
else:
    print("No ArmoRM records found - skipping heatmaps")


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
    plt.savefig("results/auroc_bars.png", dpi=600, bbox_inches="tight")
    plt.close()
    print("Saved results/auroc_bars.png")
else:
    print("No AUROC results found - skipping bar charts")

LABEL_SCORERS = {
    "HelpSteer2 (correctness)":           {"ArmoRM": "results/helpsteer2_armorm.json"},
    "HelpSteer2 (helpfulness)":           {"ArmoRM": "results/helpsteer2_armorm_helpfulness.json"},
    "HelpSteer2 (coherence)":             {"ArmoRM": "results/helpsteer2_armorm_coherence.json"},
    "HelpSteer2 (complexity)":            {"ArmoRM": "results/helpsteer2_armorm_complexity.json"},
    "HelpSteer2 (verbosity)":             {"ArmoRM": "results/helpsteer2_armorm_verbosity.json"},
    "UltraFeedback (truthfulness)":       {"ArmoRM": "results/ultrafeedback_armorm.json"},
    "UltraFeedback (helpfulness)":        {"ArmoRM": "results/ultrafeedback_armorm_helpfulness.json"},
    "UltraFeedback (honesty)":            {"ArmoRM": "results/ultrafeedback_armorm_honesty.json"},
    "UltraFeedback (instruction_following)": {"ArmoRM": "results/ultrafeedback_armorm_instruction_following.json"},
}

label_benchmarks_data = {}
for benchmark, sources in LABEL_SCORERS.items():
    rows = []
    for source_name, path in sources.items():
        aurocs = load_aurocs(path)
        for classifier, auroc in aurocs.items():
            label = classifier  # ArmoRM only, so classifier = dimension name
            rows.append((label, auroc, False))
    if rows:
        rows.sort(key=lambda x: x[1], reverse=True)
        label_benchmarks_data[benchmark] = rows

if label_benchmarks_data:
    n = len(label_benchmarks_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 10), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (benchmark, rows) in zip(axes, label_benchmarks_data.items()):
        labels  = [r[0] for r in rows]
        aurocs  = [r[1] for r in rows]
        y = range(len(labels))
        ax.barh(y, aurocs, color="#4878cf", edgecolor="white", linewidth=0.4)
        ax.axvline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("AUROC", fontsize=9)
        ax.set_title(benchmark, fontsize=11, fontweight="bold")
        ax.set_xlim(0.4, 1.0)

    plt.tight_layout()
    plt.savefig("results/auroc_bars_labels.png", dpi=600, bbox_inches="tight")
    plt.close()
    print("Saved results/auroc_bars_labels.png")
else:
    print("No label variant files found - skipping label bars")

LABEL_FILES = {
    "HelpSteer2\n(correctness)":       "results/helpsteer2_armorm.json",
    "HelpSteer2\n(helpfulness)":        "results/helpsteer2_armorm_helpfulness.json",
    "HelpSteer2\n(coherence)":          "results/helpsteer2_armorm_coherence.json",
    "HelpSteer2\n(complexity)":         "results/helpsteer2_armorm_complexity.json",
    "HelpSteer2\n(verbosity)":          "results/helpsteer2_armorm_verbosity.json",
    "UltraFeedback\n(truthfulness)":    "results/ultrafeedback_armorm.json",
    "UltraFeedback\n(helpfulness)":     "results/ultrafeedback_armorm_helpfulness.json",
    "UltraFeedback\n(honesty)":         "results/ultrafeedback_armorm_honesty.json",
    "UltraFeedback\n(instruction_following)": "results/ultrafeedback_armorm_instruction_following.json",
}

label_cols = []
label_names = []
for col_name, path in LABEL_FILES.items():
    aurocs = load_aurocs(path)
    if not aurocs:
        continue
    per_dim = {k: v for k, v in aurocs.items() if k in ATTRIBUTES}
    if not per_dim:
        continue
    label_cols.append([per_dim.get(a, float("nan")) for a in ATTRIBUTES])
    label_names.append(col_name)

if label_cols:
    mat = np.array(label_cols).T  # (n_attr, n_labels)
    fig, ax = plt.subplots(figsize=(len(label_names) * 1.2 + 2, len(ATTRIBUTES) * 0.4 + 1))
    im = ax.imshow(mat, vmin=0.4, vmax=0.9, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(label_names)))
    ax.set_xticklabels(label_names, fontsize=8)
    ax.set_yticks(range(len(ATTRIBUTES)))
    ax.set_yticklabels(SHORT, fontsize=8)
    ax.set_title("ArmoRM per-dimension AUROC classifying each target label\n(shows feature entanglement)", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="AUROC")
    plt.tight_layout()
    plt.savefig("results/entanglement_heatmap.png", dpi=600, bbox_inches="tight")
    plt.close()
    print("Saved results/entanglement_heatmap.png")
else:
    print("No label variant files found - skipping entanglement heatmap")


# Plot: Mean correlation heatmap across all datasets

if datasets_loaded:
    all_mats = []
    ds_names_used = []
    for name, (rewards, labels) in datasets_loaded.items():
        all_mats.append(corr_matrix(rewards))
        ds_names_used.append(name)

    stacked = np.stack(all_mats)
    mean_mat = stacked.mean(axis=0)
    n_attr = len(ATTRIBUTES)

    # Target features (y-axis rows)
    target_short = ['truthfulness', 'correctness', 'honesty']
    target_set = set(target_short)
    target_indices = [SHORT.index(t) for t in target_short]

    # Reorder x-axis: target features first (same order), then the rest
    other_indices = [i for i in range(n_attr) if i not in target_indices]
    col_order = target_indices + other_indices
    col_labels = [SHORT[i] for i in col_order]

    # Slice rows (target only) and reorder columns
    sub_mat = mean_mat[np.ix_(target_indices, col_order)]  # shape (3, 19)

    fig, ax = plt.subplots(figsize=(16, 6.5))
    im = ax.imshow(sub_mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")

    ax.set_xticks(range(n_attr))
    ax.set_yticks(range(len(target_short)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=14)
    ax.set_yticklabels(target_short, fontsize=15, fontweight="bold")

    # Color target labels red on both axes
    target_color = "#cc0000"
    for label in ax.get_yticklabels():
        label.set_color(target_color)
    for i, label in enumerate(ax.get_xticklabels()):
        if col_labels[i] in target_set:
            label.set_color(target_color)
            label.set_fontweight("bold")

    # Annotate cells
    for ri in range(len(target_short)):
        for j in range(n_attr):
            # Skip diagonal (where row feature == column feature)
            if col_order[j] == target_indices[ri]:
                continue
            val = sub_mat[ri, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, ri, f"{val:.2f}", ha="center", va="center",
                    fontsize=12, color=color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, shrink=0.5)
    cbar.set_label("Pearson r", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig("results/mean_correlation_heatmap.png", dpi=600, bbox_inches="tight")
    plt.close()
    print("Saved results/mean_correlation_heatmap.png")
else:
    print("No datasets loaded - skipping mean correlation heatmap")


# Plot: Top-13 AUROC comparison - LongFact vs HelpSteer2

COMPARISON_BENCHMARKS = {
    "LongFact\n(entity-annotated)": SCORERS.get("LongFact", {}),
    "HelpSteer2\n(human-annotated)": SCORERS.get("HelpSteer2 (correctness)", {}),
}

comparison_data = {}
for panel_name, sources in COMPARISON_BENCHMARKS.items():
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
        comparison_data[panel_name] = rows[:13]

if comparison_data:
    n = len(comparison_data)
    fig, axes = plt.subplots(1, n, figsize=(10 * n, 8), sharey=False)
    if n == 1:
        axes = [axes]

    FACTUALITY_DIMS = {"ultrafeedback-truthfulness", "helpsteer-correctness",
                        "ultrafeedback-honesty", "truthfulness", "correctness", "honesty"}

    for ax, (benchmark, rows) in zip(axes, comparison_data.items()):
        labels_list = [r[0] for r in rows]
        aurocs_list = [r[1] for r in rows]
        colors = []
        for r in rows:
            if r[2]:  # probe
                colors.append("#e07b39")
            elif r[0] in FACTUALITY_DIMS:
                colors.append("#cc0000")
            else:
                colors.append("#4878cf")

        y = range(len(labels_list))
        bars = ax.barh(y, aurocs_list, color=colors, edgecolor="white", linewidth=0.8,
                        height=0.7)
        ax.axvline(0.5, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels_list, fontsize=18)
        ax.invert_yaxis()
        ax.set_xlabel("AUROC", fontsize=20)
        ax.set_title(benchmark, fontsize=24, fontweight="bold", pad=14)
        ax.set_xlim(0.4, 1.0)
        ax.tick_params(axis='x', labelsize=16)

        # Annotate bars with values
        for i, v in enumerate(aurocs_list):
            ax.text(v + 0.005, i, f"{v:.3f}", ha="left", va="center",
                    fontsize=14, color="#333333")

    from matplotlib.patches import Patch
    legend = [Patch(color="#cc0000", label="ArmoRM factuality dimension"),
              Patch(color="#4878cf", label="ArmoRM other dimension"),
              Patch(color="#e07b39", label="Probe")]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=18,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("Probes vs. ArmoRM: Entity-Annotated vs. Human-Annotated Factuality",
                 fontsize=26, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("results/auroc_comparison_longfact_helpsteer2.png", dpi=600, bbox_inches="tight")
    plt.close()
    print("Saved results/auroc_comparison_longfact_helpsteer2.png")
else:
    print("No comparison data found - skipping LongFact vs HelpSteer2 comparison")