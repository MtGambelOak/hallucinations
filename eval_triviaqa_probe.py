"""
Evaluate obalcells hallucination probes on obalcells/triviaqa-balanced using AUROC.

Pipeline:
  1. Run LLM on prompt + completion, collect hidden states at probe layer
  2. Find exact_answer in the completion via string search -> character offsets
  3. span_max = max hallucination prob over tokens in that span
  4. Correctness score = 1 - span_max  (higher = more truthful)
  5. Compute AUROC over examples with correct (1) vs incorrect (0) labels
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


def find_span_token_indices(
    offset_mapping: list[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> list[int]:
    return [
        i for i, (ts, te) in enumerate(offset_mapping)
        if te > char_start and ts < char_end and te > ts
    ]


@torch.no_grad()
def score_example(
    prompt: str,
    completion: str,
    exact_answer: str,
    llm,
    tokenizer,
    probe: nn.Linear,
    probe_layer: int,
) -> float | None:
    """
    Returns 1 - span_max_hallucination_prob, or None if exact_answer not found.
    """
    char_start = completion.lower().find(exact_answer.lower())
    if char_start == -1:
        return None
    char_end = char_start + len(exact_answer)

    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt", tokenize=True,
        truncation=True, max_length=2048, add_generation_prompt=True,
    )
    if not isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids["input_ids"]
    response_start = prompt_ids.shape[1]

    full_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt},
         {"role": "assistant", "content": completion}],
        return_tensors="pt", tokenize=True,
        truncation=True, max_length=2048,
    )
    if not isinstance(full_ids, torch.Tensor):
        full_ids = full_ids["input_ids"]

    llm_device = next(llm.parameters()).device
    full_ids = full_ids.to(llm_device)
    outputs = llm(full_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[probe_layer + 1]  # (1, seq_len, hidden_size); +1 because index 0 is embedding

    response_hidden = hidden[0, response_start:, :].to(probe.weight.device)
    logits = probe(response_hidden).squeeze(-1)
    hal_probs = torch.sigmoid(logits).cpu().float()  # (n_response_tokens,)

    # Map char offsets to response token indices
    enc = tokenizer(completion, return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = enc["offset_mapping"][:hal_probs.shape[0]]  # clamp to match

    token_indices = find_span_token_indices(offset_mapping, char_start, char_end)
    if not token_indices:
        return None

    span_max = hal_probs[token_indices].max().item()
    return 1.0 - span_max


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

    print(f"Loading dataset: obalcells/triviaqa-balanced [{args.subset}] (split={args.split})")
    ds = load_dataset("obalcells/triviaqa-balanced", args.subset, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"  {len(ds)} examples, columns: {list(ds.features.keys())}")

    all_scores, all_labels = [], []
    skipped = 0

    for row in tqdm(ds, desc="Scoring"):
        prompt = row["conversation"][0]["content"]
        completion = row["completions"][0]
        exact_answer = row["exact_answer"]
        label = 1 if row["label"] == "S" else 0

        score = score_example(prompt, completion, exact_answer, llm, tokenizer, probe, probe_layer)
        if score is None:
            skipped += 1
            continue

        all_scores.append(score)
        all_labels.append(label)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    if skipped:
        print(f"  (Skipped {skipped} examples where exact_answer not found in completion)")

    print(f"\nTotal examples scored: {len(all_labels)}")
    print(f"  Correct:   {all_labels.sum()}")
    print(f"  Incorrect: {(1 - all_labels).sum()}")

    auroc = roc_auc_score(all_labels, all_scores)
    print(f"\nAUROC: {auroc:.4f}")
    print(f"Mean score (correct):   {all_scores[all_labels == 1].mean():.4f}")
    print(f"Mean score (incorrect): {all_scores[all_labels == 0].mean():.4f}")

    if args.save:
        results = {
            "model_id": args.model_id,
            "probe_id": args.probe_id,
            "probe_layer": probe_layer,
            "split": args.split,
            "n_scored": len(all_labels),
            "n_skipped": skipped,
            "n_correct": int(all_labels.sum()),
            "n_incorrect": int((1 - all_labels).sum()),
            "auroc": auroc,
            "mean_score_correct": float(all_scores[all_labels == 1].mean()),
            "mean_score_incorrect": float(all_scores[all_labels == 0].mean()),
        }
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--probe_id", default="llama3_1_8b_linear")
    parser.add_argument("--probe_dir", default="probes/llama3_1_8b_linear")
    parser.add_argument("--subset", default="Meta-Llama-3.1-8B-Instruct",
                        choices=["Meta-Llama-3.1-8B-Instruct", "Llama-3.3-70B-Instruct"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save", default="results/triviaqa_probe.json")
    args = parser.parse_args()
    main(args)
