"""
Poster-quality plots for LongFact ArmoRM analysis.
  1. Correlation matrix of ArmoRM dimensions, axes sorted by hierarchical clustering
  2. Bar chart of per-dimension AUROC on factuality label, sorted descending

Run:
    python plot_longfact.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

ATTRIBUTES = [
    'helpsteer-helpfulness', 'helpsteer-correctness', 'helpsteer-coherence',
    'helpsteer-complexity', 'helpsteer-verbosity', 'ultrafeedback-overall_score',
    'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
    'ultrafeedback-honesty', 'ultrafeedback-helpfulness', 'beavertails-is_safe',
    'prometheus-score', 'argilla-overall_quality', 'argilla-judge_lm',
    'code-complexity', 'code-style', 'code-explanation',
    'code-instruction-following', 'code-readability',
]

# Short display labels — drop the dataset prefix for cleanliness
SHORT = [
    'helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity',
    'overall score', 'instruction\nfollowing', 'truthfulness', 'honesty',
    'helpfulness (UF)', 'is safe', 'prometheus', 'overall quality',
    'judge LM', 'code complexity', 'code style', 'code explanation',
    'code instruction\nfollowing', 'code readability',
]

# Group colors for axis labels
FACTUALITY = {'helpsteer-correctness', 'ultrafeedback-truthfulness', 'ultrafeedback-honesty'}

def attr_color(attr):
    return '#d73027' if attr in FACTUALITY else 'black'


p = Path("results/longfact_armorm.json")
with open(p) as f:
    data = json.load(f)

records  = data["records"]
rewards  = np.array([[r["rewards"][a] for a in ATTRIBUTES] for r in records])
labels   = np.array([r["label"] for r in records])
per_dim  = data["auroc_per_dimension"]
n        = len(records)

# ── Correlation matrix ─────────────────────────────────────────────────────────
n_attr = len(ATTRIBUTES)
corr = np.ones((n_attr, n_attr))
for i in range(n_attr):
    for j in range(i + 1, n_attr):
        r = pearsonr(rewards[:, i], rewards[:, j])[0]
        corr[i, j] = corr[j, i] = r

# 3 factuality rows × 16 non-factuality columns, cols sorted by mean correlation
fact_idx    = [i for i, a in enumerate(ATTRIBUTES) if a in FACTUALITY]
nonfact_idx = [i for i, a in enumerate(ATTRIBUTES) if a not in FACTUALITY]

# sort columns by mean correlation with the 3 factuality dims (descending)
mean_corr_with_fact = corr[np.ix_(nonfact_idx, fact_idx)].mean(axis=1)
col_order = [nonfact_idx[i] for i in np.argsort(mean_corr_with_fact)[::-1]]

# full column order: non-factuality (sorted) + factuality (self-correlations)
all_col_order = fact_idx + col_order
sub = corr[np.ix_(fact_idx, all_col_order)]  # (3, 19)
row_labels = [SHORT[i] for i in fact_idx]
col_labels = [SHORT[i] for i in all_col_order]
n_fact_cols = len(fact_idx)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(13, 3.5))

im = ax1.imshow(sub, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="auto")
ax1.set_xticks(range(len(all_col_order)))
ax1.set_yticks(range(len(fact_idx)))
ax1.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=11)
for i, (tick, idx) in enumerate(zip(ax1.get_xticklabels(), all_col_order)):
    if ATTRIBUTES[idx] in FACTUALITY:
        tick.set_color("#d73027")
        tick.set_fontweight("bold")
ax1.set_yticklabels(row_labels, fontsize=12, fontweight="bold", color="#d73027")

# annotate cells with r value
for ri in range(sub.shape[0]):
    for ci in range(sub.shape[1]):
        ax1.text(ci, ri, f"{sub[ri, ci]:.2f}", ha="center", va="center",
                 fontsize=8, color="black" if abs(sub[ri, ci]) < 0.7 else "white")

# divider between non-factuality and factuality columns
ax1.axvline(n_fact_cols - 0.5, color="black", linewidth=2.0, linestyle="--", alpha=0.7)

plt.colorbar(im, ax=ax1, fraction=0.02, pad=0.02, label="Pearson r")
ax1.set_title(
    "Factuality scores show high correlation with unrelated ArmoRM objectives",
    fontsize=13, fontweight="bold", pad=12
)
plt.suptitle(f"Pearson r between ArmoRM reward scores  —  LongFact (n={n:,})",
             fontsize=10, y=1.02, color="gray")

out = Path("results/longfact_poster.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {out}")


# ── Plot 2: per-dimension AUROC + probes ──────────────────────────────────────
PROBE_FILES = {
    "Gemma2-9b (linear)":    "results/longfact_probe_gemma2_9b_linear.json",
    "Gemma2-9b (LoRA KL)":   "results/longfact_probe_gemma2_9b_lora_kl.json",
    "Llama3.1-8b (linear)":  "results/longfact_probe_llama3_1_8b_linear.json",
    "Llama3.1-8b (LoRA KL)": "results/longfact_probe_llama3_1_8b_lora_kl.json",
    "Llama3.1-8b (LoRA LM)": "results/longfact_probe_llama3_1_8b_lora_lm.json",
    "Qwen2.5-7b (linear)":   "results/longfact_probe_qwen2_5_7b_linear.json",
    "Qwen2.5-7b (LoRA KL)":  "results/longfact_probe_qwen2_5_7b_lora_kl.json",
}
probe_rows = []
for name, path in PROBE_FILES.items():
    if Path(path).exists():
        d = json.load(open(path))
        probe_rows.append((name, d["auroc"]))
probe_rows.sort(key=lambda x: x[1], reverse=True)

# ArmoRM rows
auroc_vals = [per_dim[a] for a in ATTRIBUTES]
sort_idx   = np.argsort(auroc_vals)[::-1]
auc_sorted = [auroc_vals[i] for i in sort_idx]
short_auc  = [SHORT[i] for i in sort_idx]
attr_auc   = [ATTRIBUTES[i] for i in sort_idx]
colors_auc = ['#d73027' if a in FACTUALITY else '#4878cf' for a in attr_auc]

# combined and sorted by AUROC
all_entries = (
    [(name, auroc, '#e07b39', None) for name, auroc in probe_rows] +
    [(short_auc[i], auc_sorted[i], colors_auc[i], attr_auc[i]) for i in range(len(short_auc))]
)
all_entries.sort(key=lambda x: x[1], reverse=True)
all_labels = [e[0] for e in all_entries]
all_aurocs = [e[1] for e in all_entries]
all_colors = [e[2] for e in all_entries]
all_attrs  = [e[3] for e in all_entries]

from matplotlib.patches import Patch

mid = len(all_entries) // 2
halves = [all_entries[:mid], all_entries[mid:]]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for col, (ax, half) in enumerate(zip(axes, halves)):
    lbls   = [e[0] for e in half]
    aurocs = [e[1] for e in half]
    colors = [e[2] for e in half]
    attrs  = [e[3] for e in half]
    y = range(len(lbls))
    ax.barh(y, aurocs, color=colors, edgecolor="white", linewidth=0.3)
    ax.axvline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(lbls, fontsize=10)
    for tick, attr in zip(ax.get_yticklabels(), attrs):
        if attr and attr in FACTUALITY:
            tick.set_fontweight("bold")
            tick.set_color("#d73027")
    ax.invert_yaxis()
    ax.set_xlabel("AUROC on factuality label", fontsize=11)
    ax.set_xlim(0.45, 0.90)

axes[0].set_title("RLFR probes outperform all ArmoRM dimensions\nat classifying factual vs. hallucinated claims",
                  fontsize=12, fontweight="bold", pad=10)
axes[1].set_title("(continued)", fontsize=12, pad=10)

axes[1].legend(handles=[
    Patch(color='#e07b39', label='Hallucination probe (RLFR)'),
    Patch(color='#d73027', label='ArmoRM factuality dim'),
    Patch(color='#4878cf', label='ArmoRM other dim'),
], fontsize=10, loc='lower right')

plt.suptitle(f"LongFact (n={n:,})", fontsize=10, y=1.02, color="gray")
plt.tight_layout()

out2 = Path("results/longfact_auroc.png")
plt.savefig(out2, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {out2}")
