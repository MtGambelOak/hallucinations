"""
Unified hallucination evaluation script.

Usage:
  python eval.py --scorer armorm --dataset truthfulqa
  python eval.py --scorer armorm --dataset triviaqa --subset Meta-Llama-3.1-8B-Instruct
  python eval.py --scorer armorm --dataset helpsteer2
  python eval.py --scorer armorm --dataset longfact  --subset gemma-2-9b-it

  python eval.py --scorer probe --dataset truthfulqa --model_id google/gemma-2-9b-it --probe_id gemma2_9b_linear
  python eval.py --scorer probe --dataset triviaqa   --subset Meta-Llama-3.1-8B-Instruct
  python eval.py --scorer probe --dataset longfact   --subset gemma-2-9b-it
  python eval.py --scorer probe --dataset helpsteer2

NOTE: LongFact + ArmoRM scores each response once and broadcasts to its entities.
      HelpSteer2/UltraFeedback + Probe uses whole-response max (no entity spans available).
"""

import argparse
import json
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datasets import load_dataset as hf_load
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "demos"))
from probe_tutorial import download_probe_from_hf


ATTRIBUTES = [
    'helpsteer-helpfulness', 'helpsteer-correctness', 'helpsteer-coherence',
    'helpsteer-complexity', 'helpsteer-verbosity', 'ultrafeedback-overall_score',
    'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
    'ultrafeedback-honesty', 'ultrafeedback-helpfulness', 'beavertails-is_safe',
    'prometheus-score', 'argilla-overall_quality', 'argilla-judge_lm',
    'code-complexity', 'code-style', 'code-explanation',
    'code-instruction-following', 'code-readability',
]

PROBE_REPO_ID = "obalcells/hallucination-probes"

ORDINAL_DATASETS = {"helpsteer2", "ultrafeedback"}

DEFAULT_SPLITS   = {"truthfulqa": "validation", "triviaqa": "test", "longfact": "test", "helpsteer2": "validation", "ultrafeedback": "train"}
DEFAULT_SUBSETS  = {"triviaqa": "Meta-Llama-3.1-8B-Instruct", "longfact": "gemma-2-9b-it"}
DEFAULT_MODEL_ID = {"armorm": "RLHFlow/ArmoRM-Llama3-8B-v0.1", "probe": "google/gemma-2-9b-it"}


# ── Dataset loaders ────────────────────────────────────────────────────────────
# Each returns a list of dicts with at least {"prompt", "response"}.
# Binary datasets add "label" (0/1); ordinal adds "correctness" (0-4).
# TriviaQA adds "exact_answer"; LongFact adds "entities" (list of span dicts).

def load_truthfulqa(split, subset=None):
    ds = hf_load("truthful_qa", "generation", split=split)
    items = []
    for row in ds:
        q = row["question"]
        correct, incorrect = row["correct_answers"], row["incorrect_answers"]
        n = min(len(correct), len(incorrect))
        items += [{"prompt": q, "response": a, "label": 1} for a in correct[:n]]
        items += [{"prompt": q, "response": a, "label": 0} for a in incorrect[:n]]
    return items


def load_triviaqa(split, subset):
    ds = hf_load("obalcells/triviaqa-balanced", subset, split=split)
    return [
        {
            "prompt":        row["conversation"][0]["content"],
            "response":      row["completions"][0],
            "label":         1 if row["label"] == "S" else 0,
            "exact_answer":  row["exact_answer"],
        }
        for row in ds
    ]


def load_longfact(split, subset):
    ds = hf_load("obalcells/longfact-annotations", subset, split=split)
    items = []
    for row in ds:
        conv = row["conversation"]
        assert conv[-1]["role"] == "assistant"
        entities = [
            {"span": a["span"], "index": a["index"], "label": 1 if a["label"] == "Supported" else 0}
            for a in row["annotations"]
            if a.get("index") is not None and a.get("span") is not None
        ]
        if entities:
            items.append({
                "prompt":   conv[0]["content"],
                "response": conv[-1]["content"],
                "entities": entities,
            })
    return items


def load_helpsteer2(split, subset=None):
    ds = hf_load("nvidia/HelpSteer2", split=split)
    return [{"prompt": r["prompt"], "response": r["response"], "correctness": r["correctness"]} for r in ds]


def load_ultrafeedback(split, subset=None):
    ds = hf_load("openbmb/UltraFeedback", split=split)
    items = []
    for row in ds:
        prompt = row["instruction"]
        for completion in row["completions"]:
            try:
                rating = completion["annotations"]["truthfulness"]["Rating"]
                correctness = int(rating)
            except (KeyError, TypeError, ValueError):
                continue  # skip N/A or missing ratings
            items.append({
                "prompt":      prompt,
                "response":    completion["response"],
                "correctness": correctness,
            })
    return items


LOADERS = {
    "truthfulqa":    load_truthfulqa,
    "triviaqa":      load_triviaqa,
    "longfact":      load_longfact,
    "helpsteer2":    load_helpsteer2,
    "ultrafeedback": load_ultrafeedback,
}


# ── AUROC ──────────────────────────────────────────────────────────────────────

def compute_auroc(scores, labels, dataset):
    if dataset in ORDINAL_DATASETS:
        per_thresh = {}
        for t in range(int(labels.min()) + 1, int(labels.max()) + 1):
            binary = (labels >= t).astype(int)
            if 0 < binary.sum() < len(binary):
                per_thresh[f">={t}"] = float(roc_auc_score(binary, scores))
        return {"auroc_per_threshold": per_thresh, "auroc_mean": float(np.mean(list(per_thresh.values())))}
    return {"auroc": float(roc_auc_score(labels, scores))}


def auroc_scalar(auroc_results):
    """Return a single representative AUROC value for printing."""
    return auroc_results.get("auroc") or auroc_results.get("auroc_mean")


# ── ArmoRM ─────────────────────────────────────────────────────────────────────

class ArmoRM:
    def __init__(self, model_id):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.device = self.model.device

    @torch.no_grad()
    def score(self, prompt, response):
        msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        ids = self.tokenizer.apply_chat_template(msgs, return_tensors="pt", truncation=True, max_length=4096)
        if not isinstance(ids, torch.Tensor):
            ids = ids["input_ids"]
        out = self.model(ids.to(self.device))
        return {"score": out.score.float().item(), "rewards": out.rewards.cpu().float().squeeze(0).tolist()}


def run_armorm(items, dataset, rm):
    scores, rewards_all, labels = [], [], []
    for item in tqdm(items, desc="ArmoRM"):
        result = rm.score(item["prompt"], item["response"])
        if "entities" in item:
            # LongFact: broadcast one response score to all entities
            for ent in item["entities"]:
                scores.append(result["score"])
                rewards_all.append(result["rewards"])
                labels.append(ent["label"])
        else:
            scores.append(result["score"])
            rewards_all.append(result["rewards"])
            labels.append(item.get("label", item.get("correctness")))
    return np.array(scores), np.array(rewards_all), np.array(labels)


def dim_aurocs(rewards_all, labels, dataset):
    out = {}
    for i, attr in enumerate(ATTRIBUTES):
        dim = rewards_all[:, i]
        if dataset in ORDINAL_DATASETS:
            vals = [
                roc_auc_score((labels >= t).astype(int), dim)
                for t in range(int(labels.min()) + 1, int(labels.max()) + 1)
                if 0 < (labels >= t).sum() < len(labels)
            ]
            out[attr] = float(np.mean(vals)) if vals else None
        else:
            out[attr] = float(roc_auc_score(labels, dim))
    return out


# ── Probe ──────────────────────────────────────────────────────────────────────

def load_probe(probe_dir):
    with open(probe_dir / "probe_config.json") as f:
        config = json.load(f)
    probe = nn.Linear(config["hidden_size"], 1)
    probe.load_state_dict(torch.load(probe_dir / "probe_head.bin", map_location="cpu"))
    return probe.to(torch.bfloat16).eval(), config


@torch.no_grad()
def get_hal_probs(prompt, response, llm, tokenizer, probe, probe_layer):
    """Single forward pass; returns per-response-token hallucination probabilities."""
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt", truncation=True, max_length=2048, add_generation_prompt=True,
    )
    if not isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids["input_ids"]
    response_start = prompt_ids.shape[1]

    full_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        return_tensors="pt", truncation=True, max_length=2048,
    )
    if not isinstance(full_ids, torch.Tensor):
        full_ids = full_ids["input_ids"]

    hidden = llm(full_ids.to(next(llm.parameters()).device), output_hidden_states=True).hidden_states[probe_layer + 1]
    response_hidden = hidden[0, response_start:, :].to(probe.weight.device)
    return torch.sigmoid(probe(response_hidden).squeeze(-1)).cpu().float()


def span_max(hal_probs, offset_mapping, char_start, char_end):
    idxs = [i for i, (ts, te) in enumerate(offset_mapping) if te > char_start and ts < char_end and te > ts]
    return hal_probs[idxs].max().item() if idxs else None


def run_probe(items, dataset, llm, tokenizer, probe, probe_layer):
    scores, labels = [], []
    skipped = 0

    for item in tqdm(items, desc="Probe"):
        prompt, response = item["prompt"], item["response"]

        try:
            hal_probs = get_hal_probs(prompt, response, llm, tokenizer, probe, probe_layer)
        except Exception:
            skipped += 1
            continue

        enc = tokenizer(response, return_offsets_mapping=True, add_special_tokens=False)
        offset_mapping = enc["offset_mapping"][:hal_probs.shape[0]]

        if dataset == "longfact":
            for ent in item["entities"]:
                m = span_max(hal_probs, offset_mapping, ent["index"], ent["index"] + len(ent["span"]))
                if m is None:
                    skipped += 1
                    continue
                scores.append(1.0 - m)
                labels.append(ent["label"])

        elif dataset == "triviaqa":
            ans = item["exact_answer"]
            char_start = response.lower().find(ans.lower())
            if char_start == -1:
                skipped += 1
                continue
            m = span_max(hal_probs, offset_mapping, char_start, char_start + len(ans))
            if m is None:
                skipped += 1
                continue
            scores.append(1.0 - m)
            labels.append(item["label"])

        else:  # truthfulqa / helpsteer2: max over entire response
            scores.append(1.0 - hal_probs.max().item())
            labels.append(item.get("label", item.get("correctness")))

    if skipped:
        print(f"  Skipped {skipped} items")
    return np.array(scores), np.array(labels)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    loader = LOADERS[args.dataset]
    load_kwargs = {} if args.dataset in ("truthfulqa", "helpsteer2") else {"subset": args.subset}
    items = loader(args.split, **load_kwargs)
    if args.max_samples:
        items = items[:args.max_samples]
    print(f"Loaded {len(items)} items from {args.dataset}")

    if args.scorer == "armorm":
        print(f"Loading ArmoRM: {args.model_id}")
        rm = ArmoRM(args.model_id)
        scores, rewards_all, labels = run_armorm(items, args.dataset, rm)

        auroc_results = compute_auroc(scores, labels, args.dataset)
        per_dim = dim_aurocs(rewards_all, labels, args.dataset)

        print(f"\nAggregate AUROC: {auroc_scalar(auroc_results):.4f}")
        if "auroc_per_threshold" in auroc_results:
            for k, v in auroc_results["auroc_per_threshold"].items():
                print(f"  {k}: {v:.4f}")
        print("\nPer-dimension AUROCs (sorted):")
        for attr, v in sorted(per_dim.items(), key=lambda x: x[1] or 0, reverse=True):
            if v is not None:
                print(f"  {attr:<45} {v:.4f}")

        results = {
            "scorer": "armorm", "model_id": args.model_id,
            "dataset": args.dataset, "subset": args.subset, "split": args.split,
            "n": len(labels), **auroc_results, "auroc_per_dimension": per_dim,
            "records": [
                {"label": int(labels[i]), "score": float(scores[i]),
                 "rewards": {a: float(rewards_all[i, j]) for j, a in enumerate(ATTRIBUTES)}}
                for i in range(len(labels))
            ],
        }

    elif args.scorer == "probe":
        probe_dir = Path(args.probe_dir)
        if not probe_dir.exists() or not (probe_dir / "probe_config.json").exists():
            print(f"Downloading probe '{args.probe_id}'...")
            download_probe_from_hf(PROBE_REPO_ID, args.probe_id, probe_dir)

        probe, config = load_probe(probe_dir)
        probe_layer = config["layer_idx"]
        print(f"  Probe layer: {probe_layer}  |  Loading LLM: {args.model_id}")

        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        llm = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto", torch_dtype=torch.bfloat16).eval()
        probe = probe.to(next(llm.parameters()).device)

        scores, labels = run_probe(items, args.dataset, llm, tokenizer, probe, probe_layer)

        auroc_results = compute_auroc(scores, labels, args.dataset)
        print(f"\nAUROC: {auroc_scalar(auroc_results):.4f}")
        if "auroc_per_threshold" in auroc_results:
            for k, v in auroc_results["auroc_per_threshold"].items():
                print(f"  {k}: {v:.4f}")

        results = {
            "scorer": "probe", "model_id": args.model_id,
            "probe_id": args.probe_id, "probe_layer": probe_layer,
            "dataset": args.dataset, "subset": args.subset, "split": args.split,
            "n": len(labels), **auroc_results,
        }

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer",      required=True, choices=["armorm", "probe"])
    parser.add_argument("--dataset",     required=True, choices=list(LOADERS))
    parser.add_argument("--model_id",    default=None)
    parser.add_argument("--probe_id",    default="gemma2_9b_linear")
    parser.add_argument("--probe_dir",   default=None)
    parser.add_argument("--subset",      default=None)
    parser.add_argument("--split",       default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save",        default=None)
    args = parser.parse_args()

    if args.model_id is None:
        args.model_id = DEFAULT_MODEL_ID[args.scorer]
    if args.split is None:
        args.split = DEFAULT_SPLITS[args.dataset]
    if args.subset is None and args.dataset in DEFAULT_SUBSETS:
        args.subset = DEFAULT_SUBSETS[args.dataset]
    if args.probe_dir is None:
        args.probe_dir = f"probes/{args.probe_id}"
    if args.save is None:
        suffix = f"_{args.subset}" if args.subset else ""
        args.save = f"results/{args.dataset}{suffix}_{args.scorer}.json"

    main(args)
