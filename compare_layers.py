"""
Cross-layer SAE comparison analysis.

Compares SAEs trained on different layers of ArmoRM (final, layer 8, layer 16)
to investigate how factuality representations evolve through the model depth.

As promised in response to peer feedback:
  "Try training SAEs on other layers, not just the final layer."

cache_rm_activations.py supports --layer to cache intermediate layers.
This script loads SAEs trained on those caches and compares:
  - Reconstruction quality across layers
  - Factuality alignment (dot product with reward head)
  - Feature overlap (shared features between layers)
  - Which layer best separates factuality from other features

Produces:
  results/cross_layer_comparison.json     - metrics for each layer's SAE
  results/cross_layer_comparison.png      - comparison plots

Usage:
    python compare_layers.py
    python compare_layers.py --dataset ultrafeedback --layers -1 8 16
    python compare_layers.py --activation_dir /path/to/caches --sae_dir checkpoints
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import pearsonr
from safetensors.torch import load_file

from sae_lens import MatryoshkaBatchTopKTrainingSAE, MatryoshkaBatchTopKTrainingSAEConfig

# Constants 

ARMORM_MODEL_ID = "RLHFlow/ArmoRM-Llama3-8B-v0.1"

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
def compute_reconstruction(sae, diff_vecs: torch.Tensor, batch_size: int = 256) -> dict:
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


def compute_alignment(sae, W_head: torch.Tensor) -> dict:
    """Compute factuality alignment metrics for this SAE."""
    W_dec = sae.W_dec.detach().float()
    d_sae = W_dec.shape[0]

    # Only compute if dimensions match (final layer SAE matches reward head)
    if W_dec.shape[1] != W_head.shape[1]:
        return {"alignment_available": False}

    C = (W_dec @ W_head.T).numpy()  # (d_sae, n_attr)
    abs_C = np.abs(C)

    fact_mask = np.array([a in FACTUALITY_ATTRS for a in ATTRIBUTES])
    safety_mask = np.array([a in SAFETY_ATTRS for a in ATTRIBUTES])

    fact_total = float(abs_C[:, fact_mask].sum())
    safety_total = float(abs_C[:, safety_mask].sum())
    other_total = float(abs_C[:, ~(fact_mask | safety_mask)].sum())
    grand_total = fact_total + safety_total + other_total

    # Per-latent top attribute
    top_attrs = []
    for j in range(d_sae):
        top_i = int(np.argmax(abs_C[j]))
        top_attrs.append(ATTRIBUTES[top_i])

    n_fact_latents = sum(1 for a in top_attrs if a in FACTUALITY_ATTRS)

    return {
        "alignment_available": True,
        "factuality_fraction": fact_total / grand_total if grand_total > 0 else 0.0,
        "safety_fraction": safety_total / grand_total if grand_total > 0 else 0.0,
        "n_factuality_latents": n_fact_latents,
        "n_safety_latents": sum(1 for a in top_attrs if a in SAFETY_ATTRS),
        "top_attrs": top_attrs,
    }


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Compare SAEs trained on different layers of ArmoRM."
    )
    parser.add_argument("--dataset", default="ultrafeedback",
                        help="Dataset name to analyze")
    parser.add_argument("--layers", type=int, nargs="+", default=[-1, 8, 16],
                        help="Layers to compare (-1 = final layer)")
    parser.add_argument("--sae_dir", default="checkpoints")
    parser.add_argument("--activation_dir",
                        default="/scratch/general/vast/u1110118/hallucinations")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    act_dir = Path(args.activation_dir)
    sae_dir = Path(args.sae_dir)

    # Try to load reward head for alignment analysis
    W_head = None
    try:
        W_head = load_reward_head(ARMORM_MODEL_ID)
        print(f"Loaded reward head: {tuple(W_head.shape)}")
    except Exception as e:
        print(f"Could not load reward head (alignment analysis will be skipped): {e}")

    layer_results = []

    for layer in args.layers:
        layer_suffix = f"_layer{layer}" if layer != -1 else ""
        layer_label = f"Layer {layer}" if layer != -1 else "Final layer"

        # Find activation cache
        act_path = act_dir / f"{args.dataset}{layer_suffix}_diff.pt"
        if not act_path.exists():
            print(f"  [{layer_label}] No activation cache at {act_path}, skipping")
            continue

        # Find SAE checkpoint
        sae_path = sae_dir / f"rm_sae_{args.dataset}{layer_suffix}"
        if not sae_path.exists() or not (sae_path / "cfg.json").exists():
            # Try sweep-style naming
            sae_path = sae_dir / f"rm_sae_{args.dataset}{layer_suffix}_d32"
            if not sae_path.exists():
                print(f"  [{layer_label}] No SAE checkpoint, skipping")
                continue

        print(f"\n{'='*60}")
        print(f"  {layer_label}  (SAE: {sae_path})")
        print(f"{'='*60}")

        # Load data
        data = torch.load(act_path, map_location="cpu")
        diff_vecs = data["diff"]
        print(f"  Activations: {diff_vecs.shape}")

        sae = load_sae(str(sae_path), device)
        d_sae = sae.cfg.d_sae
        print(f"  SAE: d_sae={d_sae}, d_in={sae.cfg.d_in}")

        # Reconstruction quality
        recon = compute_reconstruction(sae, diff_vecs)
        print(f"  R²={recon['r_squared']:.4f}  cos={recon['cosine_similarity']:.4f}  "
              f"MSE={recon['mse']:.6f}")

        # Feature statistics
        acts = get_activations(sae, diff_vecs)
        active_counts = (acts > 0).sum(dim=0).numpy()
        n_dead = int((active_counts == 0).sum())
        sparsity = float((acts > 0).float().mean())
        mean_act = float(acts[acts > 0].mean()) if (acts > 0).any() else 0.0

        # Alignment (only if d_in matches reward head)
        alignment = {}
        if W_head is not None:
            alignment = compute_alignment(sae, W_head)
            if alignment.get("alignment_available"):
                print(f"  Factuality fraction: {alignment['factuality_fraction']:.4f}")
                print(f"  Factuality latents: {alignment['n_factuality_latents']}/{d_sae}")

        # Load labels if available
        label_path = Path(args.results_dir) / f"sae_labels_{sae_path.name}.json"
        labels = {}
        if label_path.exists():
            with open(label_path) as f:
                raw = json.load(f)
            labels = {int(k): v.get("label", "") if isinstance(v, dict) else str(v)
                      for k, v in raw.items()}

        result = {
            "layer": layer,
            "layer_label": layer_label,
            "sae_path": str(sae_path),
            "d_sae": d_sae,
            "d_in": sae.cfg.d_in,
            "n_examples": int(diff_vecs.shape[0]),
            **recon,
            "n_dead": n_dead,
            "n_alive": d_sae - n_dead,
            "sparsity": sparsity,
            "mean_activation": mean_act,
            **{k: v for k, v in alignment.items() if k != "top_attrs"},
            "labels": labels if labels else None,
        }
        layer_results.append(result)

    if not layer_results:
        print("\nNo layer results to compare. Ensure activation caches and SAE "
              "checkpoints exist for the requested layers.")
        return

    # Save JSON
    json_out = f"{args.results_dir}/cross_layer_comparison_{args.dataset}.json"
    with open(json_out, "w") as f:
        json.dump(layer_results, f, indent=2)
    print(f"\nSaved {json_out}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"  Cross-Layer SAE Comparison - {args.dataset}")
    print(f"{'='*80}")
    header = (f"  {'Layer':>12}  {'d_sae':>6}  {'R²':>8}  {'Cosine':>8}  "
              f"{'MSE':>10}  {'Alive':>7}  {'Sparsity':>9}")
    if any(r.get("alignment_available") for r in layer_results):
        header += f"  {'Fact%':>7}  {'#Fact':>6}"
    print(header)
    print(f"  {'-'*(len(header)-2)}")

    for r in layer_results:
        line = (f"  {r['layer_label']:>12}  {r['d_sae']:>6}  "
                f"{r['r_squared']:>8.4f}  {r['cosine_similarity']:>8.4f}  "
                f"{r['mse']:>10.6f}  {r['n_alive']:>3}/{r['d_sae']:<3}  "
                f"{r['sparsity']:>9.4f}")
        if r.get("alignment_available"):
            line += (f"  {r['factuality_fraction']:>7.4f}  "
                     f"{r['n_factuality_latents']:>6}")
        print(line)

    # Plot
    if len(layer_results) < 2:
        print("Need at least 2 layers to generate comparison plot.")
        return

    labels = [r["layer_label"] for r in layer_results]
    x = np.arange(len(labels))

    n_metrics = 4
    has_alignment = any(r.get("alignment_available") for r in layer_results)
    if has_alignment:
        n_metrics = 6

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    # R²
    ax = fig.add_subplot(gs[0, 0])
    vals = [r["r_squared"] for r in layer_results]
    ax.bar(x, vals, color="#4daf4a", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("R²"); ax.set_title("Reconstruction Quality", fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=8)

    # Cosine similarity
    ax = fig.add_subplot(gs[0, 1])
    vals = [r["cosine_similarity"] for r in layer_results]
    ax.bar(x, vals, color="#4878cf", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Cosine sim."); ax.set_title("Cosine Similarity", fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=8)

    # Dead latents
    ax = fig.add_subplot(gs[0, 2])
    alive = [r["n_alive"] for r in layer_results]
    dead = [r["n_dead"] for r in layer_results]
    ax.bar(x, alive, color="#4daf4a", edgecolor="white", label="Alive")
    ax.bar(x, dead, bottom=alive, color="#d73027", edgecolor="white", label="Dead")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Latents"); ax.set_title("Alive vs Dead Latents", fontweight="bold")
    ax.legend(fontsize=8)

    # Sparsity
    ax = fig.add_subplot(gs[1, 0])
    vals = [r["sparsity"] for r in layer_results]
    ax.bar(x, vals, color="#984ea3", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Sparsity"); ax.set_title("Feature Sparsity", fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8)

    if has_alignment:
        # Factuality fraction
        ax = fig.add_subplot(gs[1, 1])
        vals = [r.get("factuality_fraction", 0) for r in layer_results]
        baseline = len(FACTUALITY_ATTRS) / len(ATTRIBUTES)
        ax.bar(x, vals, color="#e07b39", edgecolor="white")
        ax.axhline(baseline, color="black", linestyle="--", linewidth=0.8,
                    label=f"Uniform ({baseline:.3f})")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("Fraction"); ax.set_title("Factuality Alignment", fontweight="bold")
        ax.legend(fontsize=8)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=8)

        # Number of factuality latents
        ax = fig.add_subplot(gs[1, 2])
        n_fact = [r.get("n_factuality_latents", 0) for r in layer_results]
        n_other = [r["d_sae"] - r.get("n_factuality_latents", 0) for r in layer_results]
        ax.bar(x, n_fact, color="#d73027", edgecolor="white", label="Factuality")
        ax.bar(x, n_other, bottom=n_fact, color="#cccccc", edgecolor="white", label="Other")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("Latents"); ax.set_title("Factuality vs Other Latents", fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle(f"Cross-Layer SAE Comparison - {args.dataset}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_out = f"{args.results_dir}/cross_layer_comparison_{args.dataset}.png"
    plt.savefig(plot_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_out}")


if __name__ == "__main__":
    main()
