"""
SAE vocabulary size sweep.

Trains SAEs at multiple d_sae values on the same cached activations, then
compares reconstruction quality, dead latents, reward-head alignment, and
(optionally) auto-generated feature labels across all sizes.

Produces:
  results/sae_sweep_<dataset>.json   - all metrics in one file
  results/sae_sweep_<dataset>.png    - comparison plots

Usage:
  python sweep_sae_sizes.py --activations /path/to/ultrafeedback_diff.pt --dataset ultrafeedback
  python sweep_sae_sizes.py --activations /path/to/helpsteer2_diff.pt   --dataset helpsteer2
  python sweep_sae_sizes.py --d_sae_values 8 16 32 64 128 --steps 30000
  python sweep_sae_sizes.py --activations /path/to/diff.pt --skip_training   # analysis only
  python sweep_sae_sizes.py --activations /path/to/diff.pt --label           # also run autolabeling
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from tqdm import tqdm

from safetensors.torch import load_file
from sae_lens import (
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
    LoggingConfig,
)
from sae_lens.training.sae_trainer import SAETrainer, SAETrainerConfig

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

SHORT_ATTRS = [a.split("-", 1)[1] for a in ATTRIBUTES]


# Helpers 

def make_data_provider(diff_vecs: torch.Tensor, batch_size: int, device: str):
    n = diff_vecs.shape[0]
    batch_size = min(batch_size, n)
    while True:
        perm = torch.randperm(n)
        for i in range(0, n - batch_size + 1, batch_size):
            yield diff_vecs[perm[i : i + batch_size]].to(device)


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
    raise RuntimeError("regression_layer.weight not found in ArmoRM safetensors")


@torch.no_grad()
def get_all_activations(sae, diff_vecs: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    device = next(sae.parameters()).device
    all_acts = []
    for i in range(0, len(diff_vecs), batch_size):
        batch = diff_vecs[i : i + batch_size].to(device)
        all_acts.append(sae.encode(batch).cpu())
    return torch.cat(all_acts, dim=0)


@torch.no_grad()
def compute_reconstruction_metrics(sae, diff_vecs: torch.Tensor,
                                   batch_size: int = 256) -> dict:
    """Compute MSE, cosine similarity, and fraction of variance unexplained."""
    device = next(sae.parameters()).device
    total_mse = 0.0
    total_cos = 0.0
    ss_res = 0.0
    ss_tot = 0.0
    n = 0

    # compute global mean for FVU
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

    return {"mse": mse, "cosine_similarity": cos_mean, "fvu": fvu, "r_squared": 1.0 - fvu}


# Stage 1: Train

def train_one(diff_vecs: torch.Tensor, d_sae: int, k: int,
              steps: int, batch_size: int, lr: float,
              output_dir: str, device: str) -> str:
    """Train a single SAE and save it. Returns the checkpoint path."""
    d_in = diff_vecs.shape[1]
    matryoshka_widths = [w for w in [d_sae // 4, d_sae // 2] if w > 0 and w < d_sae]

    sae_cfg = MatryoshkaBatchTopKTrainingSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        k=min(k, d_sae),
        matryoshka_widths=matryoshka_widths,
        use_matryoshka_aux_loss=True,
        normalize_activations="expected_average_only_in",
        apply_b_dec_to_input=False,
        dtype="float32",
        device=device,
    )
    sae = MatryoshkaBatchTopKTrainingSAE(sae_cfg)

    total_training_samples = steps * batch_size
    trainer_cfg = SAETrainerConfig(
        total_training_samples=total_training_samples,
        train_batch_size_samples=batch_size,
        lr=lr,
        lr_end=0.0,
        lr_scheduler_name="constant",
        lr_warm_up_steps=steps // 20,
        lr_decay_steps=steps // 5,
        n_restart_cycles=1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        autocast=False,
        device=device,
        checkpoint_path=output_dir,
        n_checkpoints=0,
        save_final_checkpoint=True,
        feature_sampling_window=500,
        dead_feature_window=500,
        logger=LoggingConfig(log_to_wandb=False),
    )

    data_provider = make_data_provider(diff_vecs, batch_size, device)
    trainer = SAETrainer(cfg=trainer_cfg, sae=sae, data_provider=data_provider)
    trainer.fit()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sae.save_model(output_dir)
    return output_dir


# Stage 2: Analyze

def analyze_one(sae_path: str, diff_vecs: torch.Tensor,
                W_head: torch.Tensor, device: str) -> dict:
    """Load a trained SAE and compute all comparison metrics."""
    sae = load_sae(sae_path, device)
    d_sae = sae.cfg.d_sae

    # --- Reconstruction quality ---
    recon = compute_reconstruction_metrics(sae, diff_vecs)

    # --- Feature activations ---
    acts = get_all_activations(sae, diff_vecs)  # (N, d_sae)
    active_counts = (acts > 0).sum(dim=0).numpy()  # per-latent
    n_dead = int((active_counts == 0).sum())
    n_alive = d_sae - n_dead
    mean_active_per_example = float((acts > 0).float().sum(dim=1).mean())
    sparsity = float((acts > 0).float().mean())

    # --- Reward-head alignment (dot products) ---
    W_dec = sae.W_dec.detach().float()  # (d_sae, d_in)
    C = (W_dec @ W_head.T).numpy()      # (d_sae, n_attr)

    # For each attribute, find the top-contributing latent's dot product
    top_dot_per_attr = {}
    for i, attr in enumerate(ATTRIBUTES):
        top_j = int(np.argmax(np.abs(C[:, i])))
        top_dot_per_attr[attr] = {
            "latent": top_j,
            "dot": float(C[top_j, i]),
            "abs_dot": float(np.abs(C[top_j, i])),
        }

    # For each latent, find its top attribute
    top_attr_per_latent = {}
    for j in range(d_sae):
        top_i = int(np.argmax(np.abs(C[j, :])))
        top_attr_per_latent[str(j)] = {
            "attribute": ATTRIBUTES[top_i],
            "dot": float(C[j, top_i]),
        }

    # How much total absolute dot product goes to factuality vs. other attrs
    abs_C = np.abs(C)
    fact_mask = np.array([a in FACTUALITY_ATTRS for a in ATTRIBUTES])
    fact_total = float(abs_C[:, fact_mask].sum())
    other_total = float(abs_C[:, ~fact_mask].sum())
    fact_fraction = fact_total / (fact_total + other_total) if (fact_total + other_total) > 0 else 0.0

    return {
        "d_sae": d_sae,
        "k": sae.cfg.k,
        "matryoshka_widths": sae.cfg.matryoshka_widths,
        "sae_path": sae_path,
        # reconstruction
        "mse": recon["mse"],
        "cosine_similarity": recon["cosine_similarity"],
        "fvu": recon["fvu"],
        "r_squared": recon["r_squared"],
        # sparsity
        "n_alive": n_alive,
        "n_dead": n_dead,
        "frac_dead": n_dead / d_sae,
        "sparsity": sparsity,
        "mean_active_per_example": mean_active_per_example,
        # reward head alignment
        "factuality_dot_fraction": fact_fraction,
        "top_dot_per_attr": top_dot_per_attr,
        "top_attr_per_latent": top_attr_per_latent,
        "dot_products": C.tolist(),
    }


# Stage 3 (optional): Label

def label_sae(sae_path: str, diff_vecs: torch.Tensor, texts: list[dict],
              model_id: str, device: str) -> dict:
    """Auto-label all latents for one SAE. Returns {latent_idx: label_str}."""
    import random
    import warnings
    from transformers import pipeline as hf_pipeline

    SYSTEM_PROMPT = (
        "You are a meticulous AI researcher investigating a sparse autoencoder (SAE) trained "
        "on difference vectors from a reward model: φ(chosen) − φ(rejected) for the same prompt. "
        "Each SAE feature captures a specific quality dimension that separates preferred from "
        "dispreferred responses.\n\n"
        "You will see preference pairs split into HIGH-activation (feature fires strongly) and "
        "LOW-activation (feature barely fires) groups. "
        "Your task: identify what the HIGH pairs have in common that the LOW pairs lack.\n\n"
        "Rules:\n"
        "- Be specific to what you actually see. Bad: 'helpfulness'. "
        "Good: 'cites sources and hedges uncertain claims'.\n"
        "- Do NOT use generic phrases like 'more helpful' or 'better quality'.\n"
        "- Keep your label under 12 words.\n"
        "- If the contrast is genuinely unclear, say: 'unclear contrast'.\n"
        "- Your response must be exactly: [interpretation]: <your short label>"
    )

    DEMO = (
        "--- HIGH activation examples ---\n\n"
        "Pair 1 [activation=9/10]:\n"
        "Prompt: Can you help me pick a lock?\n"
        "PREFERRED: I'm sorry, but I can't help with that.\n"
        "DISPREFERRED: Sure! You'll need a tension wrench and a pick.\n\n"
        "--- LOW activation examples ---\n\n"
        "Pair 2 [activation=1/10]:\n"
        "Prompt: What is the capital of France?\n"
        "PREFERRED: The capital of France is Paris.\n"
        "DISPREFERRED: I think it might be Lyon.\n\n"
        "[interpretation]: Safety refusals - declining harmful requests rather than complying."
    )

    sae = load_sae(sae_path, device)
    d_sae = sae.cfg.d_sae
    acts = get_all_activations(sae, diff_vecs)
    global_max = acts.max().item()

    pipe = hf_pipeline("text-generation", model=model_id,
                       device_map="auto", torch_dtype=torch.bfloat16)
    pipe.model.generation_config.max_length = None

    labels = {}
    for j in tqdm(range(d_sae), desc=f"  Labeling d_sae={d_sae}"):
        vals = acts[:, j].numpy()
        order = np.argsort(vals)[::-1]
        active = order[vals[order] > 0]

        if len(active) == 0:
            labels[j] = "(dead latent)"
            continue

        rng = random.Random(42)
        n_groups, per_group = 10, 4
        group_size = max(1, len(active) // n_groups)
        selected = []
        for g in range(n_groups):
            start = g * group_size
            end = start + group_size if g < n_groups - 1 else len(active)
            bucket = active[start:end].tolist()
            selected.extend(rng.sample(bucket, min(per_group, len(bucket))))

        high_idx = selected[:5]
        low_idx = selected[-3:]

        def fmt_pair(num, idx):
            scaled = int(round(10 * vals[idx] / global_max)) if global_max > 0 else 0
            t = texts[idx]
            return (
                f"Pair {num} [activation={scaled}/10]:\n"
                f"Prompt: {t['prompt'][:200]}\n"
                f"PREFERRED: {t['chosen'][:400]}\n"
                f"DISPREFERRED: {t['rejected'][:400]}"
            )

        parts = ["--- HIGH activation examples ---\n"]
        for i, idx in enumerate(high_idx):
            parts.append(fmt_pair(i + 1, idx))
        parts.append("\n--- LOW activation examples ---\n")
        for i, idx in enumerate(low_idx):
            parts.append(fmt_pair(6 + i, idx))
        examples_text = "\n\n".join(parts)

        messages = [
            {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + DEMO + "\n\n---\n\nNow interpret this feature:\n\n" + examples_text},
            {"role": "assistant", "content": "[interpretation]:"},
        ]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
            out = pipe(messages, max_new_tokens=40,
                       do_sample=False, continue_final_message=True)
        text = out[0]["generated_text"][-1]["content"].strip()
        if text.lower().startswith("[interpretation]:"):
            text = text[len("[interpretation]:"):].strip()
        label = text.split("\n")[0].strip()
        labels[j] = label
        tqdm.write(f"    latent {j}: {label}")

    return labels


# Stage 4: Plot

def generate_plots(sweep_results: list[dict], output_path: str):
    """Generate comparison plots across SAE sizes."""
    sizes = [r["d_sae"] for r in sweep_results]

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # 1. Reconstruction quality (R²)
    ax = fig.add_subplot(gs[0, 0])
    r2 = [r["r_squared"] for r in sweep_results]
    ax.bar(range(len(sizes)), r2, color="#4878cf", edgecolor="white")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("d_sae")
    ax.set_ylabel("R²")
    ax.set_title("Reconstruction Quality (R²)", fontweight="bold")
    ax.set_ylim(0, max(1.0, max(r2) * 1.1))
    for i, v in enumerate(r2):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=8)

    # 2. MSE
    ax = fig.add_subplot(gs[0, 1])
    mses = [r["mse"] for r in sweep_results]
    ax.bar(range(len(sizes)), mses, color="#e07b39", edgecolor="white")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("d_sae")
    ax.set_ylabel("MSE")
    ax.set_title("Reconstruction MSE", fontweight="bold")
    for i, v in enumerate(mses):
        ax.text(i, v + max(mses) * 0.02, f"{v:.4f}", ha="center", fontsize=8)

    # 3. Dead latent fraction
    ax = fig.add_subplot(gs[0, 2])
    dead_frac = [r["frac_dead"] for r in sweep_results]
    alive = [r["n_alive"] for r in sweep_results]
    dead = [r["n_dead"] for r in sweep_results]
    bars = ax.bar(range(len(sizes)), dead_frac, color="#d73027", edgecolor="white")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("d_sae")
    ax.set_ylabel("Fraction dead")
    ax.set_title("Dead Latent Fraction", fontweight="bold")
    ax.set_ylim(0, max(1.0, max(dead_frac) * 1.3) if max(dead_frac) > 0 else 1.0)
    for i in range(len(sizes)):
        ax.text(i, dead_frac[i] + 0.02, f"{dead[i]}/{sizes[i]}", ha="center", fontsize=8)

    # 4. Cosine similarity
    ax = fig.add_subplot(gs[1, 0])
    cos = [r["cosine_similarity"] for r in sweep_results]
    ax.bar(range(len(sizes)), cos, color="#4daf4a", edgecolor="white")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("d_sae")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("Reconstruction Cosine Similarity", fontweight="bold")
    for i, v in enumerate(cos):
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=8)

    # 5. Factuality dot-product fraction
    ax = fig.add_subplot(gs[1, 1])
    fact = [r["factuality_dot_fraction"] for r in sweep_results]
    baseline = len(FACTUALITY_ATTRS) / len(ATTRIBUTES)
    ax.bar(range(len(sizes)), fact, color="#984ea3", edgecolor="white")
    ax.axhline(baseline, color="black", linestyle="--", linewidth=0.8, alpha=0.6,
               label=f"Uniform baseline ({baseline:.2f})")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("d_sae")
    ax.set_ylabel("Fraction of |dot| to factuality attrs")
    ax.set_title("Factuality Alignment Fraction", fontweight="bold")
    ax.legend(fontsize=8)
    for i, v in enumerate(fact):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

    # 6. Top attribute distribution across latents (stacked or grouped)
    ax = fig.add_subplot(gs[1, 2])
    for idx, r in enumerate(sweep_results):
        top_attrs = [v["attribute"] for v in r["top_attr_per_latent"].values()]
        # count how many latents map to factuality vs. other
        n_fact = sum(1 for a in top_attrs if a in FACTUALITY_ATTRS)
        n_other = len(top_attrs) - n_fact
        ax.bar(idx - 0.15, n_fact, width=0.3, color="#984ea3", edgecolor="white",
               label="Factuality" if idx == 0 else "")
        ax.bar(idx + 0.15, n_other, width=0.3, color="#cccccc", edgecolor="white",
               label="Other" if idx == 0 else "")
        ax.text(idx - 0.15, n_fact + 0.3, str(n_fact), ha="center", fontsize=8)
        ax.text(idx + 0.15, n_other + 0.3, str(n_other), ha="center", fontsize=8)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("d_sae")
    ax.set_ylabel("Number of latents")
    ax.set_title("Latent → Top Attribute Category", fontweight="bold")
    ax.legend(fontsize=8)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


# Main

def main():
    parser = argparse.ArgumentParser(
        description="SAE vocabulary size sweep: train, analyze, and compare SAEs across d_sae values."
    )
    parser.add_argument("--activations", required=True,
                        help="Path to cached diff vectors (.pt file from cache_rm_activations.py)")
    parser.add_argument("--dataset", default="ultrafeedback",
                        help="Dataset name (used for output filenames)")
    parser.add_argument("--d_sae_values", type=int, nargs="+", default=[8, 16, 32, 64],
                        help="Dictionary sizes to sweep")
    parser.add_argument("--k", type=int, default=8,
                        help="Top-k sparsity (clamped to d_sae if larger)")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--checkpoint_dir", default="checkpoints",
                        help="Base directory for SAE checkpoints")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training, only run analysis on existing checkpoints")
    parser.add_argument("--label", action="store_true",
                        help="Also run LLM auto-labeling for each SAE (slow)")
    parser.add_argument("--label_model", default="google/gemma-2-9b-it",
                        help="LLM for auto-labeling")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-train even if checkpoint exists")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading activations from {args.activations}...")
    data = torch.load(args.activations, map_location="cpu")
    diff_vecs = data["diff"]
    texts = data.get("texts")  # may be None if labeling not needed
    n, d_in = diff_vecs.shape
    print(f"  {n} difference vectors, d_in={d_in}")

    if args.label and texts is None:
        print("  WARNING: --label requested but activation cache has no 'texts' field.")
        print("           Re-run cache_rm_activations.py to include texts. Skipping labeling.")
        args.label = False

    # Load reward head
    print(f"Loading ArmoRM reward head ({ARMORM_MODEL_ID})...")
    W_head = load_reward_head(ARMORM_MODEL_ID)
    print(f"  reward head shape: {tuple(W_head.shape)}")

    # Train + Analyze each size
    sweep_results = []

    for d_sae in args.d_sae_values:
        print(f"\n{'='*70}")
        print(f"  d_sae = {d_sae}")
        print(f"{'='*70}")

        ckpt_dir = f"{args.checkpoint_dir}/rm_sae_{args.dataset}_d{d_sae}"
        ckpt_exists = Path(ckpt_dir).exists() and (Path(ckpt_dir) / "cfg.json").exists()

        # --- Train ---
        if args.skip_training:
            if not ckpt_exists:
                print(f"  Checkpoint not found at {ckpt_dir}, skipping d_sae={d_sae}")
                continue
            print(f"  Using existing checkpoint: {ckpt_dir}")
        elif ckpt_exists and not args.overwrite:
            print(f"  Checkpoint exists at {ckpt_dir}, skipping training (use --overwrite to redo)")
        else:
            effective_k = min(args.k, d_sae)
            print(f"  Training: d_sae={d_sae}, k={effective_k}, steps={args.steps}")
            train_one(
                diff_vecs=diff_vecs,
                d_sae=d_sae,
                k=effective_k,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                output_dir=ckpt_dir,
                device=device,
            )
            print(f"  Saved checkpoint to {ckpt_dir}")

        # --- Analyze ---
        print(f"  Analyzing...")
        result = analyze_one(ckpt_dir, diff_vecs, W_head, device)

        print(f"    R²={result['r_squared']:.4f}  MSE={result['mse']:.6f}  "
              f"cos={result['cosine_similarity']:.4f}  "
              f"alive={result['n_alive']}/{d_sae}  "
              f"factuality_frac={result['factuality_dot_fraction']:.3f}")

        # --- Label (optional) ---
        if args.label:
            print(f"  Auto-labeling latents...")
            labels = label_sae(ckpt_dir, diff_vecs, texts, args.label_model, device)
            result["labels"] = labels

            # Save labels separately too (compatible with print_sae_analysis.py)
            label_out = f"{args.results_dir}/sae_labels_rm_sae_{args.dataset}_d{d_sae}.json"
            label_data = {
                str(j): {"label": lbl, "n_active": 0, "max_act": 0.0}
                for j, lbl in labels.items()
            }
            with open(label_out, "w") as f:
                json.dump(label_data, f, indent=2)
            print(f"    Labels saved to {label_out}")

        sweep_results.append(result)

    if not sweep_results:
        print("\nNo SAEs to analyze. Check checkpoint paths or remove --skip_training.")
        return

    # Save combined results
    # Strip dot_products from the JSON to keep it small; they're in per-SAE files
    save_results = []
    for r in sweep_results:
        r_copy = {k: v for k, v in r.items() if k != "dot_products"}
        save_results.append(r_copy)

    json_out = f"{args.results_dir}/sae_sweep_{args.dataset}.json"
    with open(json_out, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved sweep results to {json_out}")

    # Print summary table
    print(f"\n{'='*90}")
    print(f"  SAE Vocabulary Size Sweep - {args.dataset}")
    print(f"{'='*90}")
    print(f"  {'d_sae':>6}  {'R²':>8}  {'MSE':>10}  {'Cosine':>8}  "
          f"{'Alive':>7}  {'Dead':>6}  {'Sparsity':>9}  {'Fact%':>7}")
    print(f"  {'-'*82}")
    for r in sweep_results:
        print(f"  {r['d_sae']:>6}  {r['r_squared']:>8.4f}  {r['mse']:>10.6f}  "
              f"{r['cosine_similarity']:>8.4f}  "
              f"{r['n_alive']:>3}/{r['d_sae']:<3}  {r['n_dead']:>6}  "
              f"{r['sparsity']:>9.4f}  {r['factuality_dot_fraction']:>7.3f}")

    # ── Generate plots ─────────────────────────────────────────────────────
    plot_out = f"{args.results_dir}/sae_sweep_{args.dataset}.png"
    generate_plots(sweep_results, plot_out)

    print(f"\nDone. Outputs:")
    print(f"  {json_out}")
    print(f"  {plot_out}")
    if args.label:
        for r in sweep_results:
            print(f"  {args.results_dir}/sae_labels_rm_sae_{args.dataset}_d{r['d_sae']}.json")


if __name__ == "__main__":
    main()
