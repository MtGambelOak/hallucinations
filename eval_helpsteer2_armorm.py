"""
Evaluate ArmoRM (RLHFlow/ArmoRM-Llama3-8B-v0.1) on nvidia/HelpSteer2 using AUROC.

Each example has a human-annotated correctness score (0-4).
We compute AUROC at each binarization threshold (correctness >= t for t in 1..4)
and report the mean across thresholds.
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


ATTRIBUTES = [
    'helpsteer-helpfulness', 'helpsteer-correctness', 'helpsteer-coherence',
    'helpsteer-complexity', 'helpsteer-verbosity', 'ultrafeedback-overall_score',
    'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
    'ultrafeedback-honesty', 'ultrafeedback-helpfulness', 'beavertails-is_safe',
    'prometheus-score', 'argilla-overall_quality', 'argilla-judge_lm',
    'code-complexity', 'code-style', 'code-explanation',
    'code-instruction-following', 'code-readability',
]


class ArmoRMPipeline:
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16,
                 truncation=True, trust_remote_code=True, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def score(self, question: str, answer: str) -> dict:
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        )
        input_ids = tokenized if isinstance(tokenized, torch.Tensor) else tokenized["input_ids"]
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
        return {
            "score": output.score.float().item(),
            "rewards": output.rewards.cpu().float().squeeze(0).tolist(),
        }


def threshold_auroc(scores: np.ndarray, correctness: np.ndarray, threshold: int) -> float | None:
    """AUROC for binary label: correctness >= threshold. Returns None if only one class present."""
    labels = (correctness >= threshold).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None
    return roc_auc_score(labels, scores)


def main(args):
    print(f"Loading model: {args.model_id}")
    rm = ArmoRMPipeline(args.model_id)

    print(f"Loading nvidia/HelpSteer2 (split={args.split})")
    ds = load_dataset("nvidia/HelpSteer2", split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"  {len(ds)} examples, columns: {list(ds.features.keys())}")

    scores, rewards_all, correctness_all = [], [], []
    for row in tqdm(ds, desc="Scoring"):
        result = rm.score(row["prompt"], row["response"])
        scores.append(result["score"])
        rewards_all.append(result["rewards"])
        correctness_all.append(row["correctness"])

    scores = np.array(scores)
    rewards_all = np.array(rewards_all)       # (n, 19)
    correctness_all = np.array(correctness_all)

    # Per-threshold and mean AUROC for aggregate score
    print(f"\nAggregate score — AUROC by threshold:")
    agg_aurocs = {}
    for t in range(1, 5):
        auroc = threshold_auroc(scores, correctness_all, t)
        if auroc is not None:
            agg_aurocs[f">={t}"] = auroc
            print(f"  correctness >= {t}:  {auroc:.4f}  "
                  f"({(correctness_all >= t).sum()} pos / {(correctness_all < t).sum()} neg)")
    mean_agg = np.mean(list(agg_aurocs.values()))
    print(f"  Mean AUROC:         {mean_agg:.4f}")

    # Per-dimension mean AUROC (averaged across thresholds)
    dim_mean_aurocs = {}
    for i, attr in enumerate(ATTRIBUTES):
        dim_scores = rewards_all[:, i]
        per_thresh = [threshold_auroc(dim_scores, correctness_all, t) for t in range(1, 5)]
        valid = [a for a in per_thresh if a is not None]
        if valid:
            dim_mean_aurocs[attr] = np.mean(valid)

    print(f"\nPer-dimension mean AUROCs (sorted):")
    for attr, auroc in sorted(dim_mean_aurocs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {attr:<45} {auroc:.4f}")

    if args.save:
        results = {
            "model_id": args.model_id,
            "split": args.split,
            "n_examples": len(scores),
            "auroc_aggregate_per_threshold": agg_aurocs,
            "auroc_aggregate_mean": mean_agg,
            "auroc_per_dimension_mean": dim_mean_aurocs,
            "records": [
                {"correctness": int(correctness_all[i]), "score": float(scores[i]),
                 "rewards": {a: float(rewards_all[i, j]) for j, a in enumerate(ATTRIBUTES)}}
                for i in range(len(scores))
            ],
        }
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="RLHFlow/ArmoRM-Llama3-8B-v0.1")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save", default="results/helpsteer2_armorm.json")
    args = parser.parse_args()
    main(args)
