import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from safetensors.torch import load_file
from safetensors import safe_open

from sae_lens import MatryoshkaBatchTopKTrainingSAE, MatryoshkaBatchTopKTrainingSAEConfig

SCRATCH     = Path("/scratch/general/vast/u1110118/hallucinations")
CHECKPOINTS = Path("checkpoints")
ARMORM_PATH = Path("/scratch/general/vast/u1110118/huggingface/hub/models--RLHFlow--ArmoRM-Llama3-8B-v0.1/snapshots")

ATTRIBUTES = [
    'helpsteer-helpfulness', 'helpsteer-correctness', 'helpsteer-coherence',
    'helpsteer-complexity', 'helpsteer-verbosity', 'ultrafeedback-overall_score',
    'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
    'ultrafeedback-honesty', 'ultrafeedback-helpfulness', 'beavertails-is_safe',
    'prometheus-score', 'argilla-overall_quality', 'argilla-judge_lm',
    'code-complexity', 'code-style', 'code-explanation',
    'code-instruction-following', 'code-readability',
]

MAX_SAMPLES = 10_000

SAE_TO_CACHE = {
    "rm_sae_helpsteer2":               "helpsteer2_diff.pt",
    "rm_sae_helpsteer2_factuality":    "helpsteer2_factuality_diff.pt",
    "rm_sae_ultrafeedback":            "ultrafeedback_diff.pt",
    "rm_sae_ultrafeedback_factuality": "ultrafeedback_factuality_diff.pt",
    "rm_sae_hh_rlhf":                  "hh_rlhf_diff.pt",
}


def load_reward_head():
    snap = sorted(ARMORM_PATH.iterdir())[-1]
    for f in sorted(snap.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as st:
            if "regression_layer.weight" in st.keys():
                W = st.get_tensor("regression_layer.weight").float()
                b = st.get_tensor("regression_layer.bias").float() if "regression_layer.bias" in st.keys() else torch.zeros(W.shape[0])
                return W, b
    raise RuntimeError("regression_layer.weight not found")


def load_sae(sae_path: Path) -> MatryoshkaBatchTopKTrainingSAE:
    with open(sae_path / "cfg.json") as f:
        cfg_dict = json.load(f)
    cfg = MatryoshkaBatchTopKTrainingSAEConfig(
        d_in=cfg_dict["d_in"],
        d_sae=cfg_dict["d_sae"],
        k=cfg_dict["k"],
        matryoshka_widths=cfg_dict.get("matryoshka_widths", []),
        dtype=cfg_dict.get("dtype", "float32"),
        device="cpu",
    )
    sae = MatryoshkaBatchTopKTrainingSAE(cfg)
    weights = load_file(str(sae_path / "sae_weights.safetensors"))
    sae.load_state_dict(weights, strict=False)
    sae.eval()
    return sae


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def load_labels(sae_name: str) -> dict:
    p = Path(f"results/sae_labels_{sae_name}.json")
    if not p.exists():
        return {}
    with open(p) as f:
        raw = json.load(f)
    return {int(k): v["label"] for k, v in raw.items()}


def analyze(sae_name: str, sae: MatryoshkaBatchTopKTrainingSAE,
            diff: torch.Tensor, W_head: torch.Tensor, batch_size: int = 512):
    N = len(diff)
    all_actual = []
    all_recon  = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            chunk = diff[start:start + batch_size]
            actual_chunk = (chunk @ W_head.T).numpy()
            out = sae(chunk)
            recon = out[0] if isinstance(out, tuple) else (out.sae_out if hasattr(out, 'sae_out') else out)
            recon_chunk = (recon @ W_head.T).numpy()
            all_actual.append(actual_chunk)
            all_recon.append(recon_chunk)

    actual      = np.concatenate(all_actual, axis=0)   # (N, 19)
    recon_scores = np.concatenate(all_recon,  axis=0)  # (N, 19)

    print(f"\n{'='*65}")
    print(f"  {sae_name}  (n={N})")
    print(f"{'='*65}")
    print(f"\n  R² (reconstruction fidelity per reward dimension):")
    print(f"  {'Attribute':<45}  R²")
    print(f"  {'-'*55}")
    r2s = []
    for i, attr in enumerate(ATTRIBUTES):
        rv = r2(actual[:, i], recon_scores[:, i])
        r2s.append(rv)
        print(f"  {attr:<45}  {rv:+.4f}")
    print(f"  {'-'*55}")
    print(f"  {'mean R²':<45}  {np.nanmean(r2s):+.4f}")
    return {"r2_per_dim": dict(zip(ATTRIBUTES, [float(v) for v in r2s])), "mean_r2": float(np.nanmean(r2s)), "n": N}



def main():
    results = {}
    print("Loading ArmoRM reward head...")
    W_head, _ = load_reward_head()

    for sae_name, cache_file in SAE_TO_CACHE.items():
        sae_path = CHECKPOINTS / sae_name
        cache_path = SCRATCH / cache_file
        if not sae_path.exists():
            print(f"  skipping {sae_name} (checkpoint not found)")
            continue
        if not cache_path.exists():
            print(f"  skipping {sae_name} (cache {cache_file} not found)")
            continue

        print(f"\nLoading SAE {sae_name}...")
        sae = load_sae(sae_path)

        print(f"Loading cache {cache_file}...")
        data = torch.load(cache_path, map_location="cpu", mmap=True)
        diff = data["diff"]
        if len(diff) > MAX_SAMPLES:
            diff = diff[:MAX_SAMPLES].float()
            print(f"  Subsampled to {MAX_SAMPLES} examples")
        else:
            diff = diff.float()

        results[sae_name] = analyze(sae_name, sae, diff, W_head)

    Path("results/sae_r2.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to results/sae_r2.json")
    plot_r2(results)


def plot_r2(results: dict):
    names = list(results.keys())
    n_sae = len(names)
    fig, axes = plt.subplots(n_sae + 1, 1, figsize=(12, 4 * (n_sae + 1)))
    attrs_short = [a.replace("ultrafeedback-", "uf-").replace("helpsteer-", "hs-")
                    .replace("beavertails-", "bt-").replace("prometheus-", "prom-")
                    .replace("argilla-", "arg-").replace("code-", "code-") for a in ATTRIBUTES]

    all_r2s = []
    for ax, name in zip(axes[:-1], names):
        r2s = [results[name]["r2_per_dim"][a] for a in ATTRIBUTES]
        all_r2s.append(r2s)
        bars = ax.bar(range(len(ATTRIBUTES)), r2s)
        ax.set_xticks(range(len(ATTRIBUTES)))
        ax.set_xticklabels(attrs_short, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("R²")
        ax.set_title(name)
        ax.axhline(np.mean(r2s), color="red", linestyle="--", linewidth=0.8, label=f"mean={np.mean(r2s):.3f}")
        ax.legend(fontsize=8)

    # average across all SAEs
    avg_r2s = np.mean(all_r2s, axis=0)
    axes[-1].bar(range(len(ATTRIBUTES)), avg_r2s)
    axes[-1].set_xticks(range(len(ATTRIBUTES)))
    axes[-1].set_xticklabels(attrs_short, rotation=45, ha="right", fontsize=8)
    axes[-1].set_ylim(0, 1)
    axes[-1].set_ylabel("R²")
    axes[-1].set_title("Average across all SAEs")
    axes[-1].axhline(np.mean(avg_r2s), color="red", linestyle="--", linewidth=0.8, label=f"mean={np.mean(avg_r2s):.3f}")
    axes[-1].legend(fontsize=8)

    fig.suptitle("R² per Reward Dimension", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = "results/sae_r2.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
