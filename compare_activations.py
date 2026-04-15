"""
Compare ArmoRM dimension correlations across datasets.

Two views:
1. Dataset-vs-dataset: do the same dimensions rank similarly across benchmarks?
   (uses aggregated AUROCs - 19 data points per correlation)
2. Dimension-vs-dimension per dataset: within each dataset, do reward dimensions
   co-vary across examples? (uses per-example records - hundreds of data points)
"""

import json
import numpy as np
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

DATASETS = {
    "TruthfulQA":    "results/truthfulqa_armorm.json",
    "TriviaQA":      "results/triviaqa_armorm.json",
    "LongFact":      "results/longfact_armorm.json",
    "HelpSteer2":    "results/helpsteer2_armorm.json",
    "UltraFeedback": "results/ultrafeedback_armorm.json",
}

# Ordinal correctness labels (not binary)
ORDINAL_DATASETS = {"HelpSteer2", "UltraFeedback"}


def load_data(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_dim_aurocs(data: dict) -> np.ndarray | None:
    dim_aurocs = data.get("auroc_per_dimension") or data.get("auroc_per_dimension_mean")
    if dim_aurocs is None:
        return None
    return np.array([dim_aurocs[a] for a in ATTRIBUTES])


def load_records(data: dict, dataset_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Returns (rewards_matrix, labels) where rewards_matrix is (n, 19)."""
    records = data.get("records")
    if not records:
        return None
    label_key = "label"
    rewards = np.array([[r["rewards"][a] for a in ATTRIBUTES] for r in records])
    labels = np.array([r[label_key] for r in records])
    return rewards, labels


def print_correlation_matrix(datasets, matrix):
    w = 13
    header = " " * 22 + "".join(f"{d:>{w}}" for d in datasets)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(datasets):
        print(f"  {name:<20}" + "".join(f"{matrix[i,j]:>{w}.4f}" for j in range(len(datasets))))


def print_per_dimension(datasets, vectors):
    print(f"\n  {'Dimension':<45}" + "".join(f"{d:>13}" for d in datasets))
    print("  " + "-" * (45 + 13 * len(datasets)))
    for i, attr in enumerate(ATTRIBUTES):
        print(f"  {attr:<45}" + "".join(f"{v[i]:>13.4f}" for v in vectors))


def print_dim_dim_top_bottom(corr_matrix, title, n=3):
    print(f"\n{title}")
    print(f"  {'Dimension':<35} {'Top {n} correlated':<55} {'Bottom {n} correlated'}".format(n=n))
    print("  " + "-" * 120)
    for i, attr in enumerate(ATTRIBUTES):
        others = [(j, corr_matrix[i, j]) for j in range(len(ATTRIBUTES)) if j != i]
        others.sort(key=lambda x: x[1], reverse=True)
        top = ", ".join(f"{ATTRIBUTES[j].split('-',1)[1]} ({r:.2f})" for j, r in others[:n])
        bot = ", ".join(f"{ATTRIBUTES[j].split('-',1)[1]} ({r:.2f})" for j, r in others[-n:])
        print(f"  {attr:<35} {top:<55} {bot}")


def main():
    all_data = {}
    for name, path in DATASETS.items():
        d = load_data(path)
        if d is None:
            print(f"  [{name}] not found, skipping")
        else:
            all_data[name] = d

    if len(all_data) < 2:
        print("Need at least 2 datasets.")
        return

    # --- View 1: dataset-vs-dataset correlation of AUROC vectors ---
    auroc_vecs = {n: load_dim_aurocs(d) for n, d in all_data.items() if load_dim_aurocs(d) is not None}
    if len(auroc_vecs) >= 2:
        names = list(auroc_vecs.keys())
        vectors = [auroc_vecs[n] for n in names]

        n = len(names)
        matrix = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                r, _ = pearsonr(vectors[i], vectors[j])
                matrix[i, j] = matrix[j, i] = r

        print("\nDataset-vs-dataset Pearson correlation of per-dimension AUROCs:")
        print_correlation_matrix(names, matrix)

        print("\nPer-dimension AUROCs across datasets:")
        print_per_dimension(names, vectors)

    # --- View 2: per-example dimension-vs-dimension correlation, per dataset ---
    print("\n\n" + "=" * 70)
    print("Per-example dimension-vs-dimension correlations (top/bottom 3)")
    print("=" * 70)

    for name, data in all_data.items():
        result = load_records(data, name)
        if result is None:
            print(f"\n  [{name}] no records - re-run eval with --save to generate them")
            continue

        rewards, labels = result
        n_attrs = len(ATTRIBUTES)
        corr_matrix = np.ones((n_attrs, n_attrs))

        is_ordinal = name in ORDINAL_DATASETS

        for i in range(n_attrs):
            for j in range(i + 1, n_attrs):
                r = pearsonr(rewards[:, i], rewards[:, j])[0]
                corr_matrix[i, j] = corr_matrix[j, i] = r

        print_dim_dim_top_bottom(
            corr_matrix,
            f"\n[{name}] Pearson r between reward dimensions (n={len(labels)} examples)"
        )

        # Also show which dimensions correlate most with the label
        corr_fn = spearmanr if is_ordinal else pearsonr
        print(f"\n  Correlation with label ({'correctness 0-4' if is_ordinal else 'correct/incorrect'}):")
        label_corrs = []
        for i, attr in enumerate(ATTRIBUTES):
            r = corr_fn(rewards[:, i], labels)[0]
            label_corrs.append((attr, r))
        label_corrs.sort(key=lambda x: x[1], reverse=True)
        for attr, r in label_corrs:
            print(f"    {attr:<45} {r:>7.4f}")


if __name__ == "__main__":
    main()
