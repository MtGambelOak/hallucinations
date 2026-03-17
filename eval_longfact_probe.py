"""
Evaluate obalcells hallucination probes on obalcells/longfact-annotations using AUROC.

Pipeline (per paper):
  1. Run LLM on prompt + response, collect hidden states at probe layer
  2. For each annotated entity, find its token span via character offset mapping
  3. span_max = max hallucination prob over tokens in that span
  4. Correctness score = 1 - span_max  (higher = more truthful)
  5. Compute AUROC over entities with Supported (1) vs Not Supported (0) labels
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


def find_entity_token_indices(
    offset_mapping: list[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> list[int]:
    """Return indices (within the response token sequence) that overlap [char_start, char_end)."""
    return [
        i for i, (ts, te) in enumerate(offset_mapping)
        if te > char_start and ts < char_end and te > ts  # skip padding/special tokens
    ]


@torch.no_grad()
def score_example(
    conversation: list[dict],
    annotations: list[dict],
    llm,
    tokenizer,
    probe: nn.Linear,
    probe_layer: int,
) -> list[dict]:
    """
    Returns a list of entity-level results:
        {"score": float, "label": int, "span": str}
    where score = 1 - span_max_hallucination_prob.
    """
    # Extract prompt and response text
    # Assumes last message is the assistant response
    assert conversation[-1]["role"] == "assistant"
    response_text = conversation[-1]["content"]

    # Tokenize prompt portion to find response_start index in full sequence
    prompt_messages = conversation[:-1]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    if not isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids["input_ids"]
    response_start = prompt_ids.shape[1]

    # Tokenize full conversation
    full_ids = tokenizer.apply_chat_template(
        conversation,
        return_tensors="pt",
    )
    if not isinstance(full_ids, torch.Tensor):
        full_ids = full_ids["input_ids"]
    full_ids = full_ids.to(probe.weight.device)

    # Run LLM
    llm_device = next(llm.parameters()).device
    full_ids = full_ids.to(llm_device)
    outputs = llm(full_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[probe_layer + 1]  # (1, seq_len, hidden_size); +1 because index 0 is embedding

    # Compute per-token hallucination probs for response tokens
    response_hidden = hidden[0, response_start:, :].to(probe.weight.device)
    logits = probe(response_hidden).squeeze(-1)       # (n_response_tokens,)
    hal_probs = torch.sigmoid(logits).cpu().float()   # (n_response_tokens,)

    # Tokenize response text alone to get character offset mapping
    enc = tokenizer(
        response_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offset_mapping = enc["offset_mapping"]

    # Sanity check: offset_mapping length should match response token count
    # (may differ by 1 if apply_chat_template adds a BOS to the response; clamp if so)
    n_response_tokens = hal_probs.shape[0]
    if len(offset_mapping) != n_response_tokens:
        # Trim or pad offset_mapping to match; typically off by trailing special tokens
        offset_mapping = offset_mapping[:n_response_tokens]

    results = []
    for ann in annotations:
        if ann["index"] is None or ann["span"] is None:
            continue
        char_start = ann["index"]
        char_end = char_start + len(ann["span"])
        label = 1 if ann["label"] == "Supported" else 0

        token_indices = find_entity_token_indices(offset_mapping, char_start, char_end)
        if not token_indices:
            # Entity span didn't map to any tokens — skip
            continue

        span_hal_probs = hal_probs[token_indices]
        span_max = span_hal_probs.max().item()
        score = 1.0 - span_max

        results.append({"score": score, "label": label, "span": ann["span"]})

    return results


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

    print(f"Loading dataset: obalcells/longfact-annotations [{args.subset}] (split={args.split})")
    ds = load_dataset("obalcells/longfact-annotations", args.subset, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"  {len(ds)} examples, columns: {list(ds.features.keys())}")

    all_scores, all_labels = [], []
    skipped_entities = 0

    for row in tqdm(ds, desc="Scoring"):
        conversation = row["conversation"]
        annotations = row["annotations"]

        entity_results = score_example(
            conversation, annotations, llm, tokenizer, probe, probe_layer
        )
        for r in entity_results:
            all_scores.append(r["score"])
            all_labels.append(r["label"])
        skipped_entities += len(annotations) - len(entity_results)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    if skipped_entities:
        print(f"  (Skipped {skipped_entities} entities with no token overlap)")

    print(f"\nTotal entities: {len(all_labels)}")
    print(f"  Supported:     {all_labels.sum()}")
    print(f"  Not Supported: {(1 - all_labels).sum()}")

    auroc = roc_auc_score(all_labels, all_scores)
    print(f"\nAUROC: {auroc:.4f}")
    print(f"Mean score (supported):     {all_scores[all_labels == 1].mean():.4f}")
    print(f"Mean score (not supported): {all_scores[all_labels == 0].mean():.4f}")

    if args.save:
        results = {
            "model_id": args.model_id,
            "probe_id": args.probe_id,
            "probe_layer": probe_layer,
            "split": args.split,
            "n_entities": len(all_labels),
            "n_supported": int(all_labels.sum()),
            "n_not_supported": int((1 - all_labels).sum()),
            "auroc": auroc,
            "mean_score_supported": float(all_scores[all_labels == 1].mean()),
            "mean_score_not_supported": float(all_scores[all_labels == 0].mean()),
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
    parser.add_argument("--subset", default="gemma-2-9b-it",
                        choices=["Llama-3.3-70B-Instruct", "Meta-Llama-3.1-8B-Instruct",
                                 "Mistral-Small-24B-Instruct-2501", "Qwen2.5-7B-Instruct",
                                 "gemma-2-9b-it"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save", default="results/longfact_probe.json")
    args = parser.parse_args()
    main(args)
