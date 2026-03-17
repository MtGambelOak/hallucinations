"""
Evaluate ArmoRM (RLHFlow/ArmoRM-Llama3-8B-v0.1) on TruthfulQA using AUROC.

For each question, we score both correct and incorrect answers with ArmoRM's
preference score, then compute AUROC over the full dataset (correct=1, incorrect=0).
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


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
        """
        Returns:
            score:   aggregate preference score (scalar)
            rewards: per-objective reward values (list of 19 floats)
        """
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


def load_truthfulqa(split="validation"):
    ds = load_dataset("truthful_qa", "generation", split=split)
    return ds


def collect_pairs(dataset):
    """
    Returns lists of (question, answer, label) tuples.
    label=1 for correct answers, label=0 for incorrect answers.
    Caps pairs per question to min(n_correct, n_incorrect) to keep balanced.
    """
    pairs = []
    for row in dataset:
        question = row["question"]
        correct = row["correct_answers"]
        incorrect = row["incorrect_answers"]
        n = min(len(correct), len(incorrect))
        pairs += [(question, a, 1) for a in correct[:n]]
        pairs += [(question, a, 0) for a in incorrect[:n]]
    return pairs


def main(args):
    print(f"Loading model: {args.model_id}")
    rm = ArmoRMPipeline(args.model_id)

    print("Loading TruthfulQA dataset...")
    ds = load_truthfulqa(split=args.split)
    print(f"  {len(ds)} questions loaded")

    # Inspect columns on first row
    print(f"  Columns: {list(ds.features.keys())}")
    print(f"  Sample row keys: {list(ds[0].keys())}")

    pairs = collect_pairs(ds)
    if args.max_samples:
        pairs = pairs[:args.max_samples]
    print(f"  {len(pairs)} (question, answer) pairs ({sum(l for *_, l in pairs)} correct, "
          f"{sum(1-l for *_, l in pairs)} incorrect)")

    ATTRIBUTES = [
        'helpsteer-helpfulness', 'helpsteer-correctness', 'helpsteer-coherence',
        'helpsteer-complexity', 'helpsteer-verbosity', 'ultrafeedback-overall_score',
        'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
        'ultrafeedback-honesty', 'ultrafeedback-helpfulness', 'beavertails-is_safe',
        'prometheus-score', 'argilla-overall_quality', 'argilla-judge_lm',
        'code-complexity', 'code-style', 'code-explanation',
        'code-instruction-following', 'code-readability',
    ]

    scores, rewards_all, labels = [], [], []
    for question, answer, label in tqdm(pairs, desc="Scoring"):
        result = rm.score(question, answer)
        scores.append(result["score"])
        rewards_all.append(result["rewards"])
        labels.append(label)

    scores = np.array(scores)
    rewards_all = np.array(rewards_all)  # (n_pairs, 19)
    labels = np.array(labels)

    auroc = roc_auc_score(labels, scores)
    print(f"\nAggregate score AUROC: {auroc:.4f}")
    print(f"  Mean score (correct):   {scores[labels == 1].mean():.4f}")
    print(f"  Mean score (incorrect): {scores[labels == 0].mean():.4f}")

    dim_aurocs = {}
    for i, attr in enumerate(ATTRIBUTES):
        dim_aurocs[attr] = roc_auc_score(labels, rewards_all[:, i])

    print(f"\nPer-dimension AUROCs (sorted):")
    for attr, dim_auroc in sorted(dim_aurocs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {attr:<45} {dim_auroc:.4f}")

    if args.save:
        import json
        results = {
            "model_id": args.model_id,
            "split": args.split,
            "n_pairs": len(pairs),
            "n_correct": int(labels.sum()),
            "n_incorrect": int((1 - labels).sum()),
            "auroc_aggregate": auroc,
            "auroc_per_dimension": dim_aurocs,
            "records": [
                {"label": int(labels[i]), "score": float(scores[i]),
                 "rewards": {a: float(rewards_all[i, j]) for j, a in enumerate(ATTRIBUTES)}}
                for i in range(len(labels))
            ],
        }
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="RLHFlow/ArmoRM-Llama3-8B-v0.1")
    parser.add_argument("--split", default="validation",
                        help="Dataset split to use")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save", default="results/truthfulqa_armorm.json",
                        help="Path to save JSON results")
    args = parser.parse_args()
    main(args)
