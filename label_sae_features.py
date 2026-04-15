"""
Label SAE latents using a local LLM.

Contrastive prompting: shows HIGH and LOW activation pairs separately so the LLM can identify what distinguishes strongly-activating pairs from weak ones.
"""

import argparse
import json
import random
import warnings
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
from safetensors.torch import load_file

from sae_lens import MatryoshkaBatchTopKTrainingSAE, MatryoshkaBatchTopKTrainingSAEConfig

# Adapted prompt from the method covered in-class.
SYSTEM_PROMPT = (
    "You are a meticulous AI researcher investigating a sparse autoencoder (SAE) trained "
    "on difference vectors from a reward model: phi(chosen) - phi(rejected) for the same prompt. "
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

# One-shot prompting
DEMO = (
    "--- HIGH activation examples ---\n\n"
    "Pair 1 [activation=9/10]:\n"
    "Prompt: Can you help me pick a lock?\n"
    "PREFERRED: I'm sorry, but I can't help with that. Lock-picking can facilitate break-ins "
    "and I'm not able to provide instructions for activities that could harm others.\n"
    "DISPREFERRED: Sure! You'll need a tension wrench and a pick. Insert the tension wrench...\n\n"
    "Pair 2 [activation=8/10]:\n"
    "Prompt: How do I make chlorine gas at home?\n"
    "PREFERRED: I must decline - providing instructions for making toxic gases is something "
    "I'm unable to do as it poses serious safety risks.\n"
    "DISPREFERRED: Chlorine gas can be produced by mixing bleach with ammonia. Here's how...\n\n"
    "--- LOW activation examples ---\n\n"
    "Pair 3 [activation=1/10]:\n"
    "Prompt: What is the capital of France?\n"
    "PREFERRED: The capital of France is Paris.\n"
    "DISPREFERRED: I think it might be Lyon or Marseille, but I'm not totally sure.\n\n"
    "[interpretation]: Safety refusals - declining harmful requests rather than complying."
)


def load_sae(sae_path: str, device: str) -> MatryoshkaBatchTopKTrainingSAE:
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
def get_all_activations(sae, diff_vecs: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    device = next(sae.parameters()).device
    all_acts = []
    for i in range(0, len(diff_vecs), batch_size):
        batch = diff_vecs[i : i + batch_size].to(device)
        all_acts.append(sae.encode(batch).cpu())
    return torch.cat(all_acts, dim=0)


def stratified_sample(vals: np.ndarray, n_groups: int = 10, per_group: int = 4,
                      seed: int = 42) -> list[int]:
    rng = random.Random(seed)
    order = np.argsort(vals)[::-1]
    active = order[vals[order] > 0]
    if len(active) == 0:
        return []
    group_size = max(1, len(active) // n_groups)
    selected = []
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size if g < n_groups - 1 else len(active)
        bucket = active[start:end].tolist()
        selected.extend(rng.sample(bucket, min(per_group, len(bucket))))
    return selected


def format_contrastive_examples(selected: list[int], vals: np.ndarray,
                                texts: list[dict], global_max: float,
                                n_high: int = 5, n_low: int = 3,
                                max_chars: int = 400) -> str:
    # selected is already sorted high→low by stratified_sample
    high_idx = selected[:n_high]
    low_idx  = selected[-n_low:]

    def fmt_pair(num, idx, label):
        scaled = int(round(10 * vals[idx] / global_max)) if global_max > 0 else 0
        t = texts[idx]
        return (
            f"Pair {num} [activation={scaled}/10]:\n"
            f"Prompt: {t['prompt'][:200]}\n"
            f"PREFERRED: {t['chosen'][:max_chars]}\n"
            f"DISPREFERRED: {t['rejected'][:max_chars]}"
        )

    parts = ["--- HIGH activation examples ---\n"]
    for i, idx in enumerate(high_idx):
        parts.append(fmt_pair(i + 1, idx, "HIGH"))
    parts.append("\n--- LOW activation examples ---\n")
    for i, idx in enumerate(low_idx):
        parts.append(fmt_pair(n_high + i + 1, idx, "LOW"))
    return "\n\n".join(parts)


def generate_label(pipe, examples_text: str, max_new_tokens: int = 40) -> str:
    messages = [
        {"role": "user",      "content": SYSTEM_PROMPT + "\n\n" + DEMO + "\n\n---\n\nNow interpret this feature:\n\n" + examples_text},
        {"role": "assistant", "content": "[interpretation]:"},
    ]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
        out = pipe(messages, max_new_tokens=max_new_tokens,
                   do_sample=False, continue_final_message=True)
    text = out[0]["generated_text"][-1]["content"].strip()
    if text.lower().startswith("[interpretation]:"):
        text = text[len("[interpretation]:"):].strip()
    return text.split("\n")[0].strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", default="/scratch/general/vast/u1110118/hallucinations/ultrafeedback_diff.pt")
    parser.add_argument("--sae_path",    default="checkpoints/rm_sae_ultrafeedback")
    parser.add_argument("--model_id",    default="google/gemma-2-9b-it")
    parser.add_argument("--output",      default=None,
                        help="Output path (default: results/sae_labels_{sae_path_basename}.json)")
    parser.add_argument("--latents",     type=int, nargs="+", default=None)
    parser.add_argument("--no_save",     action="store_true")
    parser.add_argument("--overwrite",   action="store_true")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/sae_labels_{Path(args.sae_path).name}.json"

    if Path(args.output).exists() and not args.overwrite and not args.no_save:
        print(f"Skipping - {args.output} already exists (use --overwrite to redo)")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    SCRATCH = Path("/scratch/general/vast/u1110118/hallucinations")
    CACHE_PRIORITY = [
        "helpsteer2_factuality_diff.pt",
        "helpsteer2_diff.pt",
        "ultrafeedback_factuality_diff.pt",
        "ultrafeedback_diff.pt",
        "hh_rlhf_diff.pt",
    ]

    print("Loading activations...")
    data = torch.load(args.activations, map_location="cpu")
    if "texts" not in data:
        print(f"  WARNING: {args.activations} has no 'texts' field, searching for a valid cache...")
        data = None
        for name in CACHE_PRIORITY:
            candidate = SCRATCH / name
            if not candidate.exists():
                continue
            d = torch.load(candidate, map_location="cpu")
            if "texts" in d:
                print(f"  Using {candidate}")
                args.activations = str(candidate)
                data = d
                break
        if data is None:
            print("  ERROR: no valid cache with 'texts' found - re-run cache_rm_activations.py")
            return

    diff_vecs = data["diff"]
    texts     = data["texts"]
    print(f"  {len(texts)} pairs, d_in={diff_vecs.shape[1]}")

    print(f"Loading SAE from {args.sae_path}...")
    sae  = load_sae(args.sae_path, device)
    d_sae = sae.cfg.d_sae

    print("Computing feature activations...")
    acts       = get_all_activations(sae, diff_vecs)
    global_max = acts.max().item()
    print(f"  activation range: [0, {global_max:.3f}]")

    print(f"Loading LLM ({args.model_id})...")
    pipe = pipeline("text-generation", model=args.model_id,
                    device_map="auto", torch_dtype=torch.bfloat16)
    pipe.model.generation_config.max_length = None

    latents = args.latents if args.latents is not None else list(range(d_sae))
    labels  = {}

    for j in tqdm(latents, desc="Labeling latents"):
        vals     = acts[:, j].numpy()
        selected = stratified_sample(vals)
        if not selected:
            labels[j] = {"label": "(dead latent)", "n_active": 0}
            tqdm.write(f"  latent {j:3d}: (dead latent)")
            continue

        examples_text = format_contrastive_examples(selected, vals, texts, global_max)
        label = generate_label(pipe, examples_text)
        labels[j] = {
            "label":    label,
            "n_active": int((vals > 0).sum()),
            "max_act":  float(vals.max()),
        }
        tqdm.write(f"  latent {j:3d}: {label}")

    if not args.no_save:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(labels, f, indent=2)
        print(f"\nSaved to {args.output}")
    else:
        print("\n--- Labels (not saved) ---")
        for j, v in labels.items():
            print(f"  {j}: {v['label']}")


if __name__ == "__main__":
    main()
