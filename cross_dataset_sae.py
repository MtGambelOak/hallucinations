"""
Cross-dataset SAE generalization analysis.

Runs SAEs trained on one dataset against cached activations from other datasets,
then analyzes whether the learned features generalize across preference distributions.

As promised in the intermediate report:
  "Run the SAEs we have trained on datasets (ones they were not trained on would
   be interesting) and analyze the activations of the sparse features defined by
   the SAE to see how those correlate with each other, and the activations of
   ArmoRM and the probes."

Produces:
  results/cross_dataset_sae.json             - metrics for every (SAE, dataset) pair
  results/cross_dataset_reconstruction.png   - reconstruction quality heatmap
  results/cross_dataset_feature_corr.png     - feature activation correlation matrix

Usage:
    python cross_dataset_sae.py
    python cross_dataset_sae.py --sae_dir checkpoints --activation_dir /path/to/caches
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from safetensors.torch import load_file

from sae_lens import MatryoshkaBatchTopKTrainingSAE, MatryoshkaBatchTopKTrainingSAEConfig

# Constants

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

DATASETS = [
    "helpsteer2",
    "helpsteer2_factuality",
    "ultrafeedback",
    "ultrafeedback_factuality",
    "hh_rlhf",
]


# Helpers

def load_sae(sae_path: str, device: str = "cpu") -> MatryoshkaBatchTopKTrainingSAE:
    path = Path(sae_path)
    with open(path / "cfg.json") as f:
        cfg_dict = json.load(f)
    sae_cfg = MatryoshkaBatchTopKTrainingSAEConfig(
        d_in=cfg_dict["d_in"],
        d_sae=cfg_dict["d_sae"],
        k=cfg_dict["k"],
        matryoshka_widths=cfg_dict.get("matryoshka_widths", []),
        dtype=cfg_dict.get("dtype", "float32"),
        device=device,
    )
    sae = MatryoshkaBatchTopKTrainingSAE(sae_cfg)
    weights = load_file(path / "sae_weights.safetensors")
    sae.load_state_dict(weights)
    sae.eval()
    return sae


@torch.no_grad()
def get_activations(sae, diff_vecs: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    device = next(sae.parameters()).device
    all_acts = []
    for i in range(0, len(diff_vecs), batch_size):
        batch = diff_vecs[i : i + batch_size].to(device)
        all_acts.append(sae.encode(batch).cpu())
    return torch.cat(all_acts, dim=0)


@torch.no_grad()
def compute_reconstruction(sae, diff_vecs: torch.Tensor,
                           batch_size: int = 256) -> dict:
    device = next(sae.parameters()).device
    total_mse, total_cos, ss_res, ss_tot, n = 0.0, 0.0, 0.0, 0.0, 0
    mean = diff_vecs.mean(dim=0)

    for i in range(0, len(diff_vecs), batch_size):
        batch = diff_vecs[i : i + batch_size].to(device)
        acts = sae.encode(batch)
        recon = sae.decode(acts)
        residual = batch - recon
        total_mse += (residual ** 2).sum().item()
        cos = torch.nn.functional.cosine_similarity(batch, recon, dim=-1)
        total_cos += cos.sum().item()
        ss_res += (residual ** 2).sum().item()
        ss_tot += ((batch - mean.to(device)) ** 2).sum().item()
        n += batch.shape[0]

    mse = total_mse / (n * diff_vecs.shape[1])
    cos_mean = total_cos / n
    fvu = ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mse": mse, "cosine_similarity": cos_mean, "r_squared": 1.0 - fvu}


def load_reward_head(model_id: str) -> torch.Tensor:
    import os, glob
    from safetensors import safe_open
    os.environ.setdefault("HF_HOME", "/scratch/general/vast/u1110118/huggingface")
    from huggingface_hub import snapshot_download
    snap = snapshot_download(model_id, local_files_only=True)
    for f in sorted(glob.glob(snap + "/*.safetensors")):
        with safe_open(f, framework="pt") as st:
            if "regression_layer.weight" in st.keys():
                return st.get_tensor("regression_layer.weight").float()
    raise RuntimeError("regression_layer.weight not found")


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Cross-dataset SAE generalization: run SAEs on out-of-distribution data."
    )
    parser.add_argument("--sae_dir", default="checkpoints",
                        help="Directory containing rm_sae_<dataset> subdirectories")
    parser.add_argument("--activation_dir", default="/scratch/general/vast/u1110118/hallucinations",
                        help="Directory containing <dataset>_diff.pt files")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Limit examples per dataset for faster analysis")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # Discover available SAEs and activation caches
    sae_dir = Path(args.sae_dir)
    act_dir = Path(args.activation_dir)

    available_saes = {}
    for ds in DATASETS:
        ckpt = sae_dir / f"rm_sae_{ds}"
        if ckpt.exists() and (ckpt / "cfg.json").exists():
            available_saes[ds] = str(ckpt)

    available_acts = {}
    for ds in DATASETS:
        act_path = act_dir / f"{ds}_diff.pt"
        if act_path.exists():
            available_acts[ds] = str(act_path)

    print(f"Available SAEs:        {list(available_saes.keys())}")
    print(f"Available activations: {list(available_acts.keys())}")

    if not available_saes or not available_acts:
        print("Need at least one SAE and one activation cache. "
              "Run train_rm_sae.py and cache_rm_activations.py first.")
        return

    # Load all activation caches
    cached_acts = {}
    for ds, path in available_acts.items():
        print(f"Loading {ds} activations from {path}...")
        data = torch.load(path, map_location="cpu")
        diff = data["diff"]
        if args.max_examples and len(diff) > args.max_examples:
            diff = diff[:args.max_examples]
        cached_acts[ds] = diff
        print(f"  {diff.shape[0]} vectors, d={diff.shape[1]}")

    # Run every (SAE, dataset) pair
    results = {}
    sae_names = sorted(available_saes.keys())
    act_names = sorted(available_acts.keys())

    for sae_ds in sae_names:
        print(f"\nLoading SAE trained on {sae_ds}...")
        sae = load_sae(available_saes[sae_ds], device)
        d_sae = sae.cfg.d_sae

        for act_ds in act_names:
            key = f"{sae_ds} -> {act_ds}"
            is_self = (sae_ds == act_ds)
            diff_vecs = cached_acts[act_ds]

            # Skip if dimensions don't match
            if diff_vecs.shape[1] != sae.cfg.d_in:
                print(f"  {key}: dimension mismatch ({diff_vecs.shape[1]} vs {sae.cfg.d_in}), skipping")
                continue

            # Reconstruction quality
            recon = compute_reconstruction(sae, diff_vecs)

            # Feature activations
            acts = get_activations(sae, diff_vecs)
            active_counts = (acts > 0).sum(dim=0).numpy()
            n_dead = int((active_counts == 0).sum())
            sparsity = float((acts > 0).float().mean())
            mean_act = float(acts[acts > 0].mean()) if (acts > 0).any() else 0.0

            result = {
                "sae_trained_on": sae_ds,
                "evaluated_on": act_ds,
                "is_self": is_self,
                "d_sae": d_sae,
                "n_examples": int(diff_vecs.shape[0]),
                **recon,
                "n_dead": n_dead,
                "sparsity": sparsity,
                "mean_activation": mean_act,
            }
            results[key] = result

            tag = "SELF" if is_self else "CROSS"
            print(f"  [{tag}] {key}: R²={recon['r_squared']:.4f}  "
                  f"cos={recon['cosine_similarity']:.4f}  "
                  f"dead={n_dead}/{d_sae}  sparsity={sparsity:.4f}")

    # Save JSON
    json_out = f"{args.results_dir}/cross_dataset_sae.json"
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {json_out}")

    # Plot 1: Reconstruction quality heatmap
    r2_matrix = np.full((len(sae_names), len(act_names)), np.nan)
    cos_matrix = np.full((len(sae_names), len(act_names)), np.nan)

    for i, sae_ds in enumerate(sae_names):
        for j, act_ds in enumerate(act_names):
            key = f"{sae_ds} -> {act_ds}"
            if key in results:
                r2_matrix[i, j] = results[key]["r_squared"]
                cos_matrix[i, j] = results[key]["cosine_similarity"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, matrix, title, cmap, vrange in [
        (ax1, r2_matrix, "R² (Reconstruction Quality)", "YlGn", (0, 1)),
        (ax2, cos_matrix, "Cosine Similarity", "YlGn", (0, 1)),
    ]:
        im = ax.imshow(matrix, vmin=vrange[0], vmax=vrange[1], cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(act_names)))
        ax.set_yticks(range(len(sae_names)))
        ax.set_xticklabels(act_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(sae_names, fontsize=8)
        ax.set_xlabel("Evaluated on (activation dataset)", fontsize=9)
        ax.set_ylabel("SAE trained on", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

        # Annotate cells
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isnan(matrix[i, j]):
                    color = "white" if matrix[i, j] > 0.7 else "black"
                    weight = "bold" if i == j else "normal"
                    ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                            fontsize=8, color=color, fontweight=weight)

    fig.suptitle("Cross-Dataset SAE Generalization: Do features transfer?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot1_out = f"{args.results_dir}/cross_dataset_reconstruction.png"
    plt.savefig(plot1_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot1_out}")

    # Plot 2: Feature activation correlation across datasets
    # For each SAE, compare its feature activation vectors on different datasets
    for sae_ds in sae_names:
        sae = load_sae(available_saes[sae_ds], device)
        d_sae = sae.cfg.d_sae

        act_datasets = []
        mean_acts_list = []

        for act_ds in act_names:
            diff_vecs = cached_acts[act_ds]
            if diff_vecs.shape[1] != sae.cfg.d_in:
                continue
            acts = get_activations(sae, diff_vecs)
            mean_acts = acts.mean(dim=0).numpy()  # (d_sae,)
            act_datasets.append(act_ds)
            mean_acts_list.append(mean_acts)

        if len(act_datasets) < 2:
            continue

        n = len(act_datasets)
        corr = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                r, _ = pearsonr(mean_acts_list[i], mean_acts_list[j])
                corr[i, j] = corr[j, i] = r

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(act_datasets, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(act_datasets, fontsize=9)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if abs(corr[i, j]) > 0.7 else "black")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        ax.set_title(f"Feature activation correlation\n(SAE trained on {sae_ds}, d_sae={d_sae})",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        plot_out = f"{args.results_dir}/cross_dataset_feature_corr_{sae_ds}.png"
        plt.savefig(plot_out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {plot_out}")


if __name__ == "__main__":
    main()
