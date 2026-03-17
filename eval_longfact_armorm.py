"""
Evaluate ArmoRM on obalcells/longfact-annotations using response-level AUROC.

Each response is labeled hallucinated (0) if any entity is "Not Supported" or
"Insufficient Information", else clean (1). ArmoRM scores the full (prompt, response)
pair via its standard last-token reward heads.
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
    "helpsteer-helpfulness", "helpsteer-correctness", "helpsteer-coherence",
    "helpsteer-complexity", "helpsteer-verbosity", "ultrafeedback-overall_score",
    "ultrafeedback-instruction_following", "ultrafeedback-truthfulness",
    "ultrafeedback-honesty", "ultrafeedback-helpfulness", "beavertails-is_safe",
    "prometheus-score", "argilla-overall_quality", "argilla-judge_lm",
    "code-complexity", "code-style", "code-explanation",
    "code-instruction-following", "code-readability",
]


@torch.no_grad()
def score_response(conversation, model, tokenizer):
    full_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    if not isinstance(full_ids, torch.Tensor):
        full_ids = full_ids["input_ids"]
    full_ids = full_ids.to(model.device)
    output = model(full_ids)
    return {
        "score": output.score.float().item(),
        "rewards": output.rewards[0].float().cpu().tolist(),
    }


def main(args):
    print(f"Loading ArmoRM: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval()

    print(f"Loading dataset: obalcells/longfact-annotations [{args.subset}] split={args.split}")
    ds = load_dataset("obalcells/longfact-annotations", args.subset, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"  {len(ds)} examples")

    agg_scores, rewards_all, labels = [], [], []

    for row in tqdm(ds, desc="Scoring"):
        has_hallucination = any(
            a["label"] in ("Not Supported", "Insufficient Information")
            for a in row["annotations"]
        )
        label = 0 if has_hallucination else 1

        result = score_response(row["conversation"], model, tokenizer)
        agg_scores.append(result["score"])
        rewards_all.append(result["rewards"])
        labels.append(label)

    agg_scores = np.array(agg_scores)
    rewards_all = np.array(rewards_all)  # (n, 19)
    labels = np.array(labels)

    print(f"\nTotal responses: {len(labels)} ({labels.sum()} clean, {(1-labels).sum()} hallucinated)")

    agg_auroc = roc_auc_score(labels, agg_scores)
    print(f"\nAggregate score AUROC: {agg_auroc:.4f}")

    dim_aurocs = {}
    for i, attr in enumerate(ATTRIBUTES):
        dim_aurocs[attr] = roc_auc_score(labels, rewards_all[:, i])

    print(f"\nPer-dimension AUROCs (sorted):")
    for attr, auroc in sorted(dim_aurocs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {attr:<45} {auroc:.4f}")

    if args.save:
        results = {
            "model_id": args.model_id,
            "subset": args.subset,
            "split": args.split,
            "n_responses": len(labels),
            "auroc_aggregate": agg_auroc,
            "auroc_per_dimension": dim_aurocs,
            "records": [
                {"label": int(labels[i]), "score": float(agg_scores[i]),
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
                        choices=["Llama-3.3-70B-Instruct", "Meta-Llama-3.1-8B-Instruct",
                                 "Mistral-Small-24B-Instruct-2501", "Qwen2.5-7B-Instruct",
                                 "gemma-2-9b-it"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save", default="results/longfact_armorm.json")
    args = parser.parse_args()
    main(args)
