"""
Plot dimension-vs-dimension correlation heatmaps for each dataset,
plus a "correlation with label" bar chart per dataset.
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
    "TruthfulQA": "results/truthfulqa_armorm.json",
    "TriviaQA":   "results/triviaqa_armorm.json",
    "LongFact":   "results/longfact_armorm.json",
    "HelpSteer2":    "results/helpsteer2_armorm.json",
    "UltraFeedback": "results/ultrafeedback_armorm.json",
}
ORDINAL = {"HelpSteer2", "UltraFeedback"}


def load_records(path, name):
    p = Path(path)
    if not p.exists():
        return None, None
    with open(p) as f:
        data = json.load(f)
    records = data.get("records")
    if not records:
        return None, None
    label_key = "label"
    rewards = np.array([[r["rewards"][a] for a in ATTRIBUTES] for r in records])
    labels = np.array([r[label_key] for r in records])
    return rewards, labels


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


# ── collect data ──────────────────────────────────────────────────────────────
datasets_loaded = {}
for name, path in DATASETS.items():
    rewards, labels = load_records(path, name)
    if rewards is not None:
        datasets_loaded[name] = (rewards, labels)

if not datasets_loaded:
    print("No records found — re-run evals with --save first.")
    exit(1)

n_ds = len(datasets_loaded)

# ── figure layout: one row per dataset, heatmap + bar chart ──────────────────
fig = plt.figure(figsize=(22, 6 * n_ds))
outer = gridspec.GridSpec(n_ds, 1, hspace=0.5)

for row, (name, (rewards, labels)) in enumerate(datasets_loaded.items()):
    ordinal = name in ORDINAL
    mat = corr_matrix(rewards)
    lc  = label_corrs(rewards, labels, ordinal)
    label_corr_type = "Spearman" if ordinal else "Pearson"

    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row],
                                             width_ratios=[3, 1], wspace=0.35)

    # Heatmap
    ax_heat = fig.add_subplot(inner[0])
    im = ax_heat.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax_heat.set_xticks(range(len(SHORT)))
    ax_heat.set_yticks(range(len(SHORT)))
    ax_heat.set_xticklabels(SHORT, rotation=45, ha="right", fontsize=7)
    ax_heat.set_yticklabels(SHORT, fontsize=7)
    ax_heat.set_title(f"{name} — Pearson r between reward dimensions (n={len(labels)})",
                      fontsize=10, pad=8)
    plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02)

    # Bar chart: correlation with label
    ax_bar = fig.add_subplot(inner[1])
    colors = ["#d73027" if v > 0 else "#4575b4" for v in lc]
    ax_bar.barh(range(len(ATTRIBUTES)), lc, color=colors)
    ax_bar.set_yticks(range(len(ATTRIBUTES)))
    ax_bar.set_yticklabels(SHORT, fontsize=7)
    ax_bar.axvline(0, color="black", linewidth=0.8)
    ax_bar.set_xlabel(f"{label_corr_type} r with label", fontsize=8)
    ax_bar.set_title("Corr. with factuality label", fontsize=9, pad=8)
    ax_bar.invert_yaxis()

out = Path("results/heatmaps.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
