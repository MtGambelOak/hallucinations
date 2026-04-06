"""
Cache ArmoRM hidden-state difference vectors for SAE training.
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "RLHFlow/ArmoRM-Llama3-8B-v0.1"

def load_ultrafeedback_pairs():
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    pairs = []
    for row in ds:
        # chosen/rejected are lists of {"role": ..., "content": ...} turns
        prompt = row["prompt"]
        chosen_response  = row["chosen"][-1]["content"]
        rejected_response = row["rejected"][-1]["content"]
        pairs.append({"prompt": prompt, "chosen": chosen_response, "rejected": rejected_response})
    return pairs


def load_hh_rlhf_pairs():
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    pairs = []
    for row in ds:
        def extract_last_assistant(text):
            parts = text.split("\n\nAssistant:")
            return parts[-1].strip() if len(parts) > 1 else text.strip()

        def extract_prompt(text):
            parts = text.split("\n\nAssistant:")
            return parts[0].strip() if len(parts) > 1 else ""

        pairs.append({
            "prompt":   extract_prompt(row["chosen"]),
            "chosen":   extract_last_assistant(row["chosen"]),
            "rejected": extract_last_assistant(row["rejected"]),
        })
    return pairs


def load_helpsteer2_pairs():
    from collections import defaultdict
    ds = load_dataset("nvidia/HelpSteer2", split="train")
    RATING_KEYS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]

    by_prompt = defaultdict(list)
    for row in ds:
        scores = [row[k] for k in RATING_KEYS if k in row]
        if not scores:
            continue
        by_prompt[row["prompt"]].append({
            "response": row["response"],
            "score":    sum(scores) / len(scores),
        })

    pairs = []
    for prompt, responses in by_prompt.items():
        if len(responses) < 2:
            continue
        responses.sort(key=lambda x: x["score"])
        chosen   = responses[-1]["response"]
        rejected = responses[0]["response"]
        if chosen == rejected:
            continue
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return pairs


def load_ultrafeedback_factuality_pairs():
    from collections import defaultdict
    ds = load_dataset("openbmb/UltraFeedback", split="train")
    by_prompt = defaultdict(list)
    for row in ds:
        prompt = row["instruction"]
        for completion in row["completions"]:
            try:
                rating = completion["annotations"]["truthfulness"]["Rating"]
                score = int(rating)
            except (KeyError, TypeError, ValueError):
                continue
            by_prompt[prompt].append({"response": completion["response"], "score": score})

    pairs = []
    for prompt, responses in by_prompt.items():
        if len(responses) < 2:
            continue
        responses.sort(key=lambda x: x["score"])
        chosen, rejected = responses[-1]["response"], responses[0]["response"]
        if chosen == rejected or responses[-1]["score"] == responses[0]["score"]:
            continue
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return pairs


def load_helpsteer2_factuality_pairs():
    from collections import defaultdict
    ds = load_dataset("nvidia/HelpSteer2", split="train")
    by_prompt = defaultdict(list)
    for row in ds:
        by_prompt[row["prompt"]].append({
            "response": row["response"],
            "score":    row["correctness"],
        })

    pairs = []
    for prompt, responses in by_prompt.items():
        if len(responses) < 2:
            continue
        responses.sort(key=lambda x: x["score"])
        chosen, rejected = responses[-1]["response"], responses[0]["response"]
        if chosen == rejected or responses[-1]["score"] == responses[0]["score"]:
            continue
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return pairs


LOADERS = {
    "ultrafeedback":            load_ultrafeedback_pairs,
    "ultrafeedback_factuality": load_ultrafeedback_factuality_pairs,
    "hh_rlhf":                  load_hh_rlhf_pairs,
    "helpsteer2":               load_helpsteer2_pairs,
    "helpsteer2_factuality":    load_helpsteer2_factuality_pairs,
}


@torch.no_grad()
def get_hidden_state(model, tokenizer, prompt, response):
    msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    ids = tokenizer.apply_chat_template(
        msgs, return_tensors="pt", truncation=True, max_length=4096
    )
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    out = model(ids.to(model.device))
    return out.hidden_state.cpu().float().squeeze(0)  # (d_model,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ultrafeedback", choices=list(LOADERS.keys()))
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    scratch = Path("/scratch/general/vast/u1110118/hallucinations")
    output = Path(args.output or scratch / f"{args.dataset}_diff.pt")
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} pairs...")
    pairs = LOADERS[args.dataset]()
    if args.max_examples:
        pairs = pairs[:args.max_examples]
    print(f"  {len(pairs)} pairs")

    checkpoint = output.with_suffix(".ckpt.pt")

    # resume from checkpoint if one exists
    if checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location="cpu")
        diff_vecs    = list(ckpt["diff"])
        chosen_vecs  = list(ckpt["chosen"])
        rejected_vecs = list(ckpt["rejected"])
        texts        = ckpt["texts"]
        start        = len(texts)
        print(f"  Resuming from checkpoint at example {start}")
    else:
        diff_vecs, chosen_vecs, rejected_vecs, texts = [], [], [], []
        start = 0

    print(f"Loading ArmoRM ({MODEL_ID})...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model.eval()

    checkpoint_every = 1000

    for i, pair in enumerate(tqdm(pairs[start:], desc="Extracting", initial=start, total=len(pairs))):
        h_c = get_hidden_state(model, tokenizer, pair["prompt"], pair["chosen"])
        h_r = get_hidden_state(model, tokenizer, pair["prompt"], pair["rejected"])
        diff_vecs.append(h_c - h_r)
        chosen_vecs.append(h_c)
        rejected_vecs.append(h_r)
        texts.append({"prompt": pair["prompt"], "chosen": pair["chosen"], "rejected": pair["rejected"]})

        if (i + 1) % checkpoint_every == 0:
            torch.save({
                "diff":     torch.stack(diff_vecs),
                "chosen":   torch.stack(chosen_vecs),
                "rejected": torch.stack(rejected_vecs),
                "texts":    texts,
            }, checkpoint)

    data = {
        "diff":     torch.stack(diff_vecs),
        "chosen":   torch.stack(chosen_vecs),
        "rejected": torch.stack(rejected_vecs),
        "dataset":  args.dataset,
        "texts":    texts,
    }

    torch.save(data, output)
    if checkpoint.exists():
        checkpoint.unlink()  # clean up checkpoint once final file is saved
    print(f"\nSaved to {output}")
    print(f"  diff:     {data['diff'].shape}  (mean norm {data['diff'].norm(dim=-1).mean():.3f})")
    print(f"  chosen:   {data['chosen'].shape}")
    print(f"  rejected: {data['rejected'].shape}")


if __name__ == "__main__":
    main()
