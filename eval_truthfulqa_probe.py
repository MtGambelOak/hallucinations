"""
Evaluate obalcells hallucination probes on TruthfulQA using AUROC.

Pipeline (per paper):
  1. Run LLM (Gemma 2 9B) on prompt + response, collect hidden states
  2. Slice hidden states to response tokens only
  3. Pass each token's hidden state through the linear probe -> per-token hallucination prob
  4. Truthfulness score = 1 - max(token probs)  [higher = more truthful]
  5. Compute AUROC over correct (1) vs incorrect (0) answers
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

# Reuse download utility from probe_tutorial
import sys
sys.path.insert(0, str(Path(__file__).parent / "demos"))
from probe_tutorial import download_probe_from_hf


PROBE_REPO_ID = "obalcells/hallucination-probes"
# Subfolder prefix for linear probes in that repo
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


def get_response_token_ids(
    tokenizer,
    question: str,
    answer: str,
) -> tuple[torch.Tensor, int]:
    """
    Tokenize prompt+answer using chat template and return (input_ids, response_start_idx).
    response_start_idx is the index of the first response token.
    """
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        return_tensors="pt", tokenize=True,
        truncation=True, max_length=2048, add_generation_prompt=True,
    )
    if not isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids["input_ids"]
    full_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": question},
         {"role": "assistant", "content": answer}],
        return_tensors="pt", tokenize=True,
        truncation=True, max_length=2048,
    )
    if not isinstance(full_ids, torch.Tensor):
        full_ids = full_ids["input_ids"]
    response_start = prompt_ids.shape[1]
    return full_ids, response_start


@torch.no_grad()
def score_pair(
    question: str,
    answer: str,
    llm,
    tokenizer,
    probe: nn.Linear,
    probe_layer: int,
    device: torch.device,
) -> float:
    """
    Returns truthfulness score = 1 - max_hallucination_prob over response tokens.
    """
    input_ids, response_start = get_response_token_ids(tokenizer, question, answer)

    if response_start >= input_ids.shape[1]:
        # Answer tokenized to nothing (shouldn't happen but guard anyway)
        return 0.5

    input_ids = input_ids.to(device)
    outputs = llm(input_ids, output_hidden_states=True)

    # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden_size)
    hidden = outputs.hidden_states[probe_layer + 1]  # (1, seq_len, hidden_size); +1 because index 0 is embedding
    response_hidden = hidden[0, response_start:, :]  # (n_response_tokens, hidden_size)

    response_hidden = response_hidden.to(probe.weight.device)
    logits = probe(response_hidden).squeeze(-1)          # (n_response_tokens,)
    hallucination_probs = torch.sigmoid(logits).cpu()    # (n_response_tokens,)

    max_hal_prob = hallucination_probs.max().item()
    return 1.0 - max_hal_prob


def collect_pairs(dataset):
    pairs = []
    for row in dataset:
        question = row["question"]
        correct, incorrect = row["correct_answers"], row["incorrect_answers"]
        n = min(len(correct), len(incorrect))
        pairs += [(question, a, 1) for a in correct[:n]]
        pairs += [(question, a, 0) for a in incorrect[:n]]
    return pairs


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

    probe_device = next(llm.parameters()).device
    probe = probe.to(probe_device)

    print("Loading TruthfulQA...")
    ds = load_dataset("truthful_qa", "generation", split="validation")
    pairs = collect_pairs(ds)
    if args.max_samples:
        pairs = pairs[:args.max_samples]
    print(f"  {len(pairs)} pairs ({sum(l for *_, l in pairs)} correct, "
          f"{sum(1-l for *_, l in pairs)} incorrect)")

    scores, labels = [], []
    for question, answer, label in tqdm(pairs, desc="Scoring"):
        s = score_pair(question, answer, llm, tokenizer, probe, probe_layer, probe_device)
        scores.append(s)
        labels.append(label)

    scores = np.array(scores)
    labels = np.array(labels)

    auroc = roc_auc_score(labels, scores)
    print(f"\nAUROC: {auroc:.4f}")
    print(f"Mean score (correct):   {scores[labels == 1].mean():.4f}")
    print(f"Mean score (incorrect): {scores[labels == 0].mean():.4f}")

    if args.save:
        results = {
            "model_id": args.model_id,
            "probe_id": args.probe_id,
            "probe_layer": probe_layer,
            "split": args.split,
            "auroc": auroc,
            "n_pairs": len(pairs),
            "n_correct": int(labels.sum()),
            "n_incorrect": int((1 - labels).sum()),
            "mean_score_correct": float(scores[labels == 1].mean()),
            "mean_score_incorrect": float(scores[labels == 0].mean()),
        }
        import json as _json
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            _json.dump(results, f, indent=2)
        print(f"Results saved to {args.save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="google/gemma-2-9b-it")
    parser.add_argument("--probe_id", default="gemma2_9b_linear",
                        help="Probe subdirectory name in the HF repo")
    parser.add_argument("--probe_dir", default="probes/gemma2_9b_linear",
                        help="Local path to save/load the probe")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save", default="results/truthfulqa_probe.json")
    args = parser.parse_args()
    main(args)
