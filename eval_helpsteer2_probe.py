"""
Evaluate obalcells hallucination probes on nvidia/HelpSteer2 using AUROC.

NOTE: No entity spans are available, so we span-max over the entire response.
This is a reasonable proxy for short answers but may be noisy for long ones.

Score = 1 - max(hallucination_prob over all response tokens).
Labels are binarized from the correctness annotation (0-4) at thresholds >=1..4,
and AUROC is averaged across thresholds.
"""

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent / "demos"))
from probe_tutorial import download_probe_from_hf


PROBE_REPO_ID = "obalcells/hallucination-probes"
PROBE_SUBFOLDER = ""


def load_probe(probe_dir: Path) -> tuple[nn.Linear, dict]:
    config_path = probe_dir / "probe_config.json"
    weights_path = probe_dir / "probe_head.bin"
    with open(config_path) as f:
        config = json.load(f)
    probe = nn.Linear(config["hidden_size"], 1)
    state_dict = torch.load(weights_path, map_location="cpu")
    probe.load_state_dict(state_dict)
    probe = probe.to(torch.bfloat16)
    probe.eval()
    return probe, config


@torch.no_grad()
def score_example(
    prompt: str,
    response: str,
    llm,
    tokenizer,
    probe: nn.Linear,
    probe_layer: int,
) -> float:
    """Returns 1 - max hallucination prob over all response tokens."""
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt",
        truncation=True, max_length=2048,
        add_generation_prompt=True,
    )
    if not isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids["input_ids"]
    response_start = prompt_ids.shape[1]

    full_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt},
         {"role": "assistant", "content": response}],
        return_tensors="pt",
        truncation=True, max_length=2048,
    )
    if not isinstance(full_ids, torch.Tensor):
        full_ids = full_ids["input_ids"]

    llm_device = next(llm.parameters()).device
    full_ids = full_ids.to(llm_device)
    outputs = llm(full_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[probe_layer + 1]  # +1: index 0 is embedding layer

    response_hidden = hidden[0, response_start:, :].to(probe.weight.device)
    logits = probe(response_hidden).squeeze(-1)
    hal_probs = torch.sigmoid(logits).cpu().float()

    return 1.0 - hal_probs.max().item()


def threshold_auroc(scores: np.ndarray, correctness: np.ndarray, threshold: int) -> float | None:
    labels = (correctness >= threshold).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None
    return roc_auc_score(labels, scores)


def main(args):
    probe_dir = Path(args.probe_dir)
    if not probe_dir.exists() or not (probe_dir / "probe_config.json").exists():
        print(f"Downloading probe '{args.probe_id}' from {PROBE_REPO_ID}...")
        download_probe_from_hf(
            repo_id=PROBE_REPO_ID,
            probe_id=args.probe_id,
            local_folder=probe_dir,
            hf_repo_subfolder_prefix=PROBE_SUBFOLDER,
        )

    print(f"Loading probe from {probe_dir}")
    probe, config = load_probe(probe_dir)
    probe_layer = config["layer_idx"]
    print(f"  Probe layer: {probe_layer}, hidden size: {config['hidden_size']}")

    print(f"Loading LLM: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    llm.eval()
    probe = probe.to(next(llm.parameters()).device)

    print(f"Loading nvidia/HelpSteer2 (split={args.split})")
    ds = load_dataset("nvidia/HelpSteer2", split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"  {len(ds)} examples")

    scores, correctness_all = [], []
    for row in tqdm(ds, desc="Scoring"):
        s = score_example(row["prompt"], row["response"], llm, tokenizer, probe, probe_layer)
        scores.append(s)
        correctness_all.append(row["correctness"])

    scores = np.array(scores)
    correctness_all = np.array(correctness_all)

    print(f"\nAUROC by threshold:")
    per_threshold = {}
    for t in range(1, 5):
        auroc = threshold_auroc(scores, correctness_all, t)
        if auroc is not None:
            per_threshold[f">={t}"] = auroc
            print(f"  correctness >= {t}:  {auroc:.4f}  "
                  f"({(correctness_all >= t).sum()} pos / {(correctness_all < t).sum()} neg)")
    mean_auroc = np.mean(list(per_threshold.values()))
    print(f"  Mean AUROC:         {mean_auroc:.4f}")

    if args.save:
        results = {
            "model_id": args.model_id,
            "probe_id": args.probe_id,
            "probe_layer": probe_layer,
            "split": args.split,
            "n_examples": len(scores),
            "auroc_per_threshold": per_threshold,
            "auroc_mean": mean_auroc,
        }
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="google/gemma-2-9b-it")
    parser.add_argument("--probe_id", default="gemma2_9b_linear")
    parser.add_argument("--probe_dir", default="probes/gemma2_9b_linear")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save", default="results/helpsteer2_probe.json")
    args = parser.parse_args()
    main(args)
