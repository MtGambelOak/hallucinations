"""
Generate the SAE routing comparison poster figure.

Shows whether SAE latents auto-labeled as "factuality" or "safety" actually
route their dot-product contributions to the corresponding ArmoRM objectives,
measured via |dot product| between SAE decoder columns and the reward head.

Produces: poster/sae_routing_comparison.png

Usage:
    python plot_sae_routing.py
    python plot_sae_routing.py --results_dir results --output poster/sae_routing_comparison.png
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

#  Constants 

ATTRIBUTES = [
    'helpsteer-helpfulness', 'helpsteer-correctness', 'helpsteer-coherence',
    'helpsteer-complexity', 'helpsteer-verbosity', 'ultrafeedback-overall_score',
    'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
    'ultrafeedback-honesty', 'ultrafeedback-helpfulness', 'beavertails-is_safe',
    'prometheus-score', 'argilla-overall_quality', 'argilla-judge_lm',
    'code-complexity', 'code-style', 'code-explanation',
    'code-instruction-following', 'code-readability',
]

FACTUALITY_ATTRS = {
    'helpsteer-correctness', 'ultrafeedback-truthfulness', 'ultrafeedback-honesty',
}
SAFETY_ATTRS = {'beavertails-is_safe'}

FACT_MASK = np.array([a in FACTUALITY_ATTRS for a in ATTRIBUTES])
SAFETY_MASK = np.array([a in SAFETY_ATTRS for a in ATTRIBUTES])
OTHER_MASK = ~(FACT_MASK | SAFETY_MASK)

DATASET_ABBREV = {
    "helpsteer2": "hs2",
    "helpsteer2_factuality": "hs2f",
    "hh_rlhf": "hh",
    "ultrafeedback": "uf",
    "ultrafeedback_factuality": "uff",
}

# Keywords for classifying auto-interpreted SAE latent labels
FACTUALITY_KEYWORDS = [
    "factual", "correct", "truth", "accura", "honest", "literal",
    "hallucin", "verifi", "misinform", "fabricat", "translation accuracy",
]
SAFETY_KEYWORDS = [
    "safe", "harmful", "refus", "unethical", "dangerous", "offensive",
    "aggressive", "evasive", "dark humor", "moral", "ethical hesitation",
    "unfilter", "creative task", "illegal", "sensitive prompt",
    "helpful response",
]


# Helpers 

def classify_label(label: str) -> str | None:
    """Classify a latent label as 'factuality', 'safety', or None."""
    low = label.lower()
    for kw in FACTUALITY_KEYWORDS:
        if kw in low:
            return "factuality"
    for kw in SAFETY_KEYWORDS:
        if kw in low:
            return "safety"
    return None


def compute_routing(dot_products: np.ndarray, latent_idx: int) -> dict:
    """Percentage of total |dot product| going to each attribute category."""
    row = np.abs(dot_products[latent_idx])
    total = row.sum()
    if total == 0:
        return {"factuality": 0.0, "safety": 0.0, "other": 0.0, "total": 0.0}
    return {
        "factuality": float(row[FACT_MASK].sum() / total * 100),
        "safety":     float(row[SAFETY_MASK].sum() / total * 100),
        "other":      float(row[OTHER_MASK].sum() / total * 100),
        "total":      float(total),
    }


def load_entries(results_dir: Path):
    """Load all SAE analysis + label files, classify latents, return entries."""
    analysis_files = sorted(results_dir.glob("sae_analysis_rm_sae_*.json"))
    factuality_entries, safety_entries = [], []

    for analysis_path in analysis_files:
        dataset = analysis_path.stem.replace("sae_analysis_rm_sae_", "")
        abbrev = DATASET_ABBREV.get(dataset, dataset[:3])

        with open(analysis_path) as f:
            analysis = json.load(f)

        dot_products = np.array(analysis["dot_products"])
        d_sae = analysis["d_sae"]

        label_path = results_dir / f"sae_labels_rm_sae_{dataset}.json"
        if not label_path.exists():
            print(f"  No labels for {dataset}, skipping")
            continue

        with open(label_path) as f:
            labels_data = json.load(f)

        for j in range(d_sae):
            info = labels_data.get(str(j), {})
            label = info.get("label", "") if isinstance(info, dict) else str(info)
            if not label or label == "(dead latent)":
                continue

            category = classify_label(label)
            if category is None:
                continue

            routing = compute_routing(dot_products, j)
            # Truncate long labels for poster legibility
            MAX_LABEL_CHARS = 45
            short_label = label if len(label) <= MAX_LABEL_CHARS else label[:MAX_LABEL_CHARS - 1] + "…"
            entry = {
                "label": label,
                "dataset": dataset,
                "abbrev": abbrev,
                "latent_idx": j,
                "routing": routing,
                "display": f"{short_label}  ({abbrev}, l{j})",
            }

            if category == "factuality":
                factuality_entries.append(entry)
            else:
                safety_entries.append(entry)

    factuality_entries.sort(key=lambda e: e["routing"]["factuality"], reverse=True)
    safety_entries.sort(key=lambda e: e["routing"]["safety"], reverse=True)
    return factuality_entries, safety_entries


# Plot 

# Poster-scale font sizes
FONT_TITLE_SUPER = 24
FONT_TITLE_SECTION = 32
FONT_TICK = 19
FONT_ANNOT_PCT = 16
FONT_ANNOT_TOTAL = 20
FONT_AXIS_LABEL = 22
FONT_LEGEND = 18
FONT_FOOTNOTE = 12
BAR_HEIGHT = 0.6


def draw_section(ax, entries, primary_key, primary_color, secondary_key,
                 secondary_color, tertiary_key, tertiary_color, baseline_pct):
    """Draw one section (factuality or safety) of the routing comparison."""
    n = len(entries)
    if n == 0:
        ax.text(0.5, 0.5, "(no matching latents)", transform=ax.transAxes,
                ha="center", va="center", fontsize=FONT_TICK, color="gray")
        return

    y_pos = np.arange(n)
    labels = [e["display"] for e in entries]
    primary  = [e["routing"][primary_key]   for e in entries]
    second   = [e["routing"][secondary_key] for e in entries]
    tertiary = [e["routing"][tertiary_key]  for e in entries]
    totals   = [e["routing"]["total"]       for e in entries]

    # Stacked bars
    ax.barh(y_pos, primary, height=BAR_HEIGHT, color=primary_color,
            edgecolor="white", linewidth=0.5)
    ax.barh(y_pos, second, left=primary, height=BAR_HEIGHT, color=secondary_color,
            edgecolor="white", linewidth=0.5)
    left2 = [p + s for p, s in zip(primary, second)]
    ax.barh(y_pos, tertiary, left=left2, height=BAR_HEIGHT, color=tertiary_color,
            edgecolor="white", linewidth=0.5)

    # Percentage annotations
    for i in range(n):
        if primary[i] > 6:
            ax.text(primary[i] / 2, i, f"{primary[i]:.0f}%",
                    ha="center", va="center", fontsize=FONT_ANNOT_PCT,
                    fontweight="bold", color="white")
        cum = primary[i] + second[i]
        if tertiary[i] > 6:
            ax.text(cum + tertiary[i] / 2, i, f"{tertiary[i]:.0f}%",
                    ha="center", va="center", fontsize=FONT_ANNOT_PCT,
                    color="white")

    # Total |dot product| annotations (right of each bar)
    for i in range(n):
        ax.text(101.5, i, f"{totals[i]:.3f}",
                ha="left", va="center", fontsize=FONT_ANNOT_TOTAL, color="#555555")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FONT_TICK, fontfamily="monospace")
    ax.tick_params(axis='y', length=0, pad=8)
    ax.tick_params(axis='x', labelsize=FONT_TICK)
    ax.invert_yaxis()

    # Baselines
    avg = np.mean(primary)
    ax.axvline(baseline_pct, color="black", linestyle="--", linewidth=1.5,
               alpha=0.6)
    ax.axvline(avg, color="#d73027", linestyle="-.", linewidth=1.5, alpha=0.7)

    ax.set_xlim(0, 108)  # extra room for total annotations
    ax.set_xticks(range(0, 101, 20))
    ax.set_xticklabels([f"{x}%" for x in range(0, 101, 20)], fontsize=FONT_TICK)

    # Remove top/right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    parser = argparse.ArgumentParser(
        description="SAE routing comparison: do latents contribute to their 'own' ArmoRM objective?"
    )
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output", default="poster/sae_routing_comparison.png")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    factuality_entries, safety_entries = load_entries(results_dir)

    n_fact = len(factuality_entries)
    n_safe = len(safety_entries)
    print(f"Found {n_fact} factuality-labeled and {n_safe} safety-labeled SAE latents")

    if n_fact == 0 and n_safe == 0:
        print("No classified latents found. Ensure results/sae_analysis_*.json and "
              "results/sae_labels_*.json exist.")
        return

    # Colors
    BLUE   = "#3b7dd8"   # Correctness / Truthfulness / Honesty (factuality)
    ORANGE = "#e8a735"   # Safety (beavertails-is_safe)
    GRAY   = "#999999"   # All other ArmoRM objectives

    uniform_fact = len(FACTUALITY_ATTRS) / len(ATTRIBUTES) * 100  # 3/19 ≈ 15.8%
    uniform_safe = len(SAFETY_ATTRS)     / len(ATTRIBUTES) * 100  # 1/19 ≈  5.3%

    # Figure sizing (vertical layout) 
    row_height = 0.55
    top_pad = 1.6    # supertitle
    mid_pad = 0.8    # gap between sections
    bot_pad = 1.8    # footnote + xlabel (extra room to avoid overlap)
    fig_height = top_pad + row_height * max(n_fact, 1) + mid_pad + row_height * max(n_safe, 1) + bot_pad

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(22, fig_height),
        gridspec_kw={"height_ratios": [max(n_fact, 1), max(n_safe, 1)]},
    )
    # Leave right margin for the legend - generous whitespace before legend
    fig.subplots_adjust(hspace=0.18, top=1 - top_pad / fig_height,
                        bottom=bot_pad / fig_height, left=0.22, right=0.62)

    # Factuality section (top) 
    ax1.set_title(f"Factuality-labeled SAE latents  (n = {n_fact})",
                  fontsize=FONT_TITLE_SECTION, fontweight="bold", pad=14)
    draw_section(ax1, factuality_entries,
                 primary_key="factuality", primary_color=BLUE,
                 secondary_key="safety",   secondary_color=ORANGE,
                 tertiary_key="other",     tertiary_color=GRAY,
                 baseline_pct=uniform_fact)

    # Safety section (bottom) 
    ax2.set_title(f"Safety-labeled SAE latents  (n = {n_safe})",
                  fontsize=FONT_TITLE_SECTION, fontweight="bold", pad=14)
    draw_section(ax2, safety_entries,
                 primary_key="safety",     primary_color=ORANGE,
                 secondary_key="factuality", secondary_color=BLUE,
                 tertiary_key="other",     tertiary_color=GRAY,
                 baseline_pct=uniform_safe)
    ax2.set_xlabel("Share of total |dot product| contribution (%)",
                   fontsize=FONT_AXIS_LABEL, labelpad=10)

    # Supertitle 
    fig.suptitle("Do SAE latents contribute to their \"own\" ArmoRM objective?",
                 fontsize=FONT_TITLE_SUPER, fontweight="bold",
                 y=1 - 0.4 / fig_height)

    # Legend (right side) 
    legend_elements = [
        Patch(facecolor=BLUE,   label="Correctness / Truthfulness\n/ Honesty"),
        Patch(facecolor=ORANGE, label="Safety (beavertails-is_safe)"),
        Patch(facecolor=GRAY,   label="All other ArmoRM objectives"),
        Line2D([0], [0], color="black",   linestyle="--", linewidth=1.5,
               label="Uniform baseline (x/19)"),
        Line2D([0], [0], color="#d73027", linestyle="-.", linewidth=1.5,
               label="Category average"),
    ]
    fig.legend(handles=legend_elements, loc="center right", ncol=1,
               fontsize=FONT_LEGEND, bbox_to_anchor=(0.97, 0.5),
               frameon=True, edgecolor="#cccccc", fancybox=False,
               handlelength=2.0, labelspacing=1.2)

    # Footnote 
    footnote = (
        "Bars show share of total |dot product| across all 19 ArmoRM objectives.  "
        "Right-side numbers = total |dot product| magnitude.\n"
        "SAE keys:  hs2 = helpsteer2,  hs2f = hs2_factuality,  "
        "hh = hh_rlhf,  uf = ultrafeedback,  uff = uf_factuality."
    )
    fig.text(0.55, 0.3 / fig_height, footnote, ha="center",
             fontsize=FONT_FOOTNOTE, color="#666666")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()