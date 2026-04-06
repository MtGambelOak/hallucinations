"""
Compute how much each SAE latent direction contributes to each ArmoRM
reward-head direction.
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file

from sae_lens import MatryoshkaBatchTopKTrainingSAE, MatryoshkaBatchTopKTrainingSAEConfig

ARMORM_MODEL_ID = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
SCRATCH = Path("/scratch/general/vast/u1110118/hallucinations")

ATTRIBUTES = [
    'helpsteer-helpfulness', 'helpsteer-correctness', 'helpsteer-coherence',
    'helpsteer-complexity', 'helpsteer-verbosity', 'ultrafeedback-overall_score',
    'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
    'ultrafeedback-honesty', 'ultrafeedback-helpfulness', 'beavertails-is_safe',
    'prometheus-score', 'argilla-overall_quality', 'argilla-judge_lm',
    'code-complexity', 'code-style', 'code-explanation',
    'code-instruction-following', 'code-readability',
]

DATASETS = [
    "helpsteer2_factuality",
    "helpsteer2",
    "ultrafeedback_factuality",
    "ultrafeedback",
    "hh_rlhf",
]


def load_sae(sae_path: str) -> MatryoshkaBatchTopKTrainingSAE:
    path = Path(sae_path)
    with open(path / "cfg.json") as f:
        cfg_dict = json.load(f)
    sae_cfg = MatryoshkaBatchTopKTrainingSAEConfig(
        d_in=cfg_dict["d_in"],
        d_sae=cfg_dict["d_sae"],
        k=cfg_dict["k"],
        matryoshka_widths=cfg_dict.get("matryoshka_widths", []),
        dtype=cfg_dict.get("dtype", "float32"),
        device="cpu",
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
                w = st.get_tensor("regression_layer.weight").float()
                print(f"  Loaded regression_layer.weight {tuple(w.shape)} from {os.path.basename(f)}")
                return w
    raise RuntimeError("regression_layer.weight not found in ArmoRM safetensors")


def analyze_one(sae_path: str, W_head: torch.Tensor, output: str):
    print(f"\nLoading SAE from {sae_path}...")
    sae = load_sae(sae_path)
    W_dec = sae.W_dec.detach().float()  # (d_sae, d_in)
    d_sae, d_in = W_dec.shape
    print(f"  d_sae={d_sae}, d_in={d_in}")

    # raw dot products: C[k,j] = w_j · d_k
    C = (W_dec @ W_head.T).numpy()  # (d_sae, n_attr)

    print(f"  Dot product range:  [{C.min():.3f}, {C.max():.3f}]")

    results = {
        "sae_path":   sae_path,
        "attributes": ATTRIBUTES,
        "d_sae":      d_sae,
        "n_attr":     len(ATTRIBUTES),
        "dot_products": C.tolist(),  # w_j · d_k
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f)
    print(f"  Saved to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae_path", default=None,
                        help="Single SAE path (default: analyze all available)")
    parser.add_argument("--output",   default=None,
                        help="Output path (only with --sae_path)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print(f"Loading ArmoRM reward head ({ARMORM_MODEL_ID})...")
    W_head = load_reward_head(ARMORM_MODEL_ID)

    if args.sae_path:
        output = args.output or f"results/sae_analysis_{Path(args.sae_path).name}.json"
        if Path(output).exists() and not args.overwrite:
            print(f"Skipping — {output} already exists (use --overwrite to redo)")
            return
        analyze_one(args.sae_path, W_head, output)
    else:
        for dataset in DATASETS:
            sae_path = f"checkpoints/rm_sae_{dataset}"
            if not Path(sae_path).exists():
                print(f"Skipping {dataset} — no checkpoint at {sae_path}")
                continue
            output = f"results/sae_analysis_rm_sae_{dataset}.json"
            if Path(output).exists() and not args.overwrite:
                print(f"Skipping {dataset} — {output} already exists (use --overwrite to redo)")
                continue
            analyze_one(sae_path, W_head, output)


if __name__ == "__main__":
    main()
