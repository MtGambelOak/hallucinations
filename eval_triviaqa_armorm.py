"""
Evaluate ArmoRM (RLHFlow/ArmoRM-Llama3-8B-v0.1) on obalcells/triviaqa-balanced using AUROC.

One score per example: ArmoRM scores the (prompt, completion) pair directly.
Label: correct (1) vs incorrect (0).
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


def main(args):
    print(f"Loading model: {args.model_id}")
    rm = ArmoRMPipeline(args.model_id)

    print(f"Loading dataset: obalcells/triviaqa-balanced [{args.subset}] (split={args.split})")
    ds = load_dataset("obalcells/triviaqa-balanced", args.subset, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"  {len(ds)} examples, columns: {list(ds.features.keys())}")

    scores, rewards_all, labels = [], [], []
    for row in tqdm(ds, desc="Scoring"):
        prompt = row["conversation"][0]["content"]
        completion = row["completions"][0]
        label = 1 if row["label"] == "S" else 0

        result = rm.score(prompt, completion)
        scores.append(result["score"])
        rewards_all.append(result["rewards"])
        labels.append(label)

    scores = np.array(scores)
    rewards_all = np.array(rewards_all)  # (n_examples, 19)
    labels = np.array(labels)

    print(f"\nTotal: {len(labels)}  Correct: {labels.sum()}  Incorrect: {(1-labels).sum()}")

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
        results = {
            "model_id": args.model_id,
            "subset": args.subset,
            "split": args.split,
            "n_examples": len(labels),
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
    parser.add_argument("--subset", default="Meta-Llama-3.1-8B-Instruct",
                        choices=["Meta-Llama-3.1-8B-Instruct", "Llama-3.3-70B-Instruct"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save", default="results/triviaqa_armorm.json")
    args = parser.parse_args()
    main(args)
