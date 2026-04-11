# Why Don't LLMs Use What They Know?

**An Analysis of the Encoding of Factuality in Reinforcement Learning**

Lucas Pearce and Christian Rogers

---

## Overview

RLHF is the dominant approach for aligning LLMs, yet even top models hallucinate frequently. Recent work (RLFR; Prasad et al., 2026) shows that using activation probes as reward signals reduces hallucinations by 58% over classical RLHF, and prior work demonstrates that models encode factuality internally (Orgad et al., 2025). This raises a natural question: **why does RLHF fail to leverage these internal representations?**

This project investigates two hypotheses:

1. **RLHF reward models predict factuality less reliably than hallucination probes**, and their "correctness" signals are entangled with shallow features like helpfulness and fluency.
2. **Sparse Autoencoder (SAE) features learned from reward model internals** do not isolate factuality. instead, non-factuality features dominate the directions that contribute to the model's correctness scores.

We use [ArmoRM](https://arxiv.org/abs/2406.12845) (a multi-attribute reward model with 19 interpretable reward dimensions) as our primary reward model, and compare it against pretrained hallucination probes from [Obeso et al., 2025](https://arxiv.org/abs/2509.03531) across five evaluation benchmarks.

## Key Findings

- **Probes outperform ArmoRM on entity-annotated factuality benchmarks** (LongFact, TriviaQA, TruthfulQA), while ArmoRM performs better on human-feedback datasets (HelpSteer2, UltraFeedback).
- **ArmoRM's reward dimensions are highly correlated** - factuality-related dimensions (correctness, truthfulness, honesty) co-vary strongly with unrelated dimensions like helpfulness.
- **SAE features labeled as factuality-related do not significantly contribute to ArmoRM's factuality scores** when measured by dot product with the reward head, suggesting the reward model does not cleanly separate factuality from other qualities.

## Repository Structure

```
.
├── eval.py                      # Unified evaluation: ArmoRM & probes on all benchmarks
├── cache_rm_activations.py      # Extract ArmoRM hidden-state difference vectors
├── train_rm_sae.py              # Train Matryoshka Top-K SAEs on cached activations
├── analyze_sae_directions.py    # Dot products between SAE decoder and reward head
├── label_sae_features.py        # LLM-based contrastive labeling of SAE latents
├── print_sae_analysis.py        # Generate text report from SAE analysis + labels
├── compare_activations.py       # Cross-dataset dimension correlation analysis
├── compare_results.py           # Aggregate AUROC leaderboard across all scorers
├── gen_plots.py                 # Generate heatmaps and AUROC bar charts
├── demos/
│   ├── armorm1.py               # ArmoRM usage demo
│   ├── armorm2.py               # ArmoRM usage demo (alternative)
│   └── probe_tutorial.py        # Hallucination probe utilities (used by eval.py)
├── results/                     # Evaluation outputs (JSON + plots)
├── checkpoints/                 # Trained SAE checkpoints
├── requirements.txt
└── README.md
```

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (recommended: 40GB+ VRAM for ArmoRM + probe inference)
- Access to Hugging Face models and datasets

### Installation

```bash
pip install -r requirements.txt
```

## Pipeline

The pipeline has four stages. Each stage produces artifacts consumed by later stages.

### Stage 1: Evaluation: ArmoRM vs. Probes

Run ArmoRM on a benchmark:

```bash
python eval.py --scorer armorm --dataset truthfulqa
python eval.py --scorer armorm --dataset triviaqa
python eval.py --scorer armorm --dataset longfact
python eval.py --scorer armorm --dataset helpsteer2
python eval.py --scorer armorm --dataset ultrafeedback
```

Run a hallucination probe on the same benchmarks:

```bash
python eval.py --scorer probe --dataset truthfulqa --probe_id gemma2_9b_linear
python eval.py --scorer probe --dataset triviaqa --probe_id gemma2_9b_linear
python eval.py --scorer probe --dataset longfact --probe_id gemma2_9b_linear
```

Additional probe variants: `gemma2_9b_lora_kl`, `llama3_1_8b_linear`, `llama3_1_8b_lora_kl`, `llama3_1_8b_lora_lm`, `qwen2_5_7b_linear`, `qwen2_5_7b_lora_kl`.

Results are saved to `results/` as JSON files containing AUROC scores, per-dimension breakdowns, and per-example records.

### Stage 2: Cache Activations & Train SAEs

Cache hidden-state difference vectors from ArmoRM:

```bash
python cache_rm_activations.py --dataset ultrafeedback
python cache_rm_activations.py --dataset helpsteer2
python cache_rm_activations.py --dataset helpsteer2_factuality
python cache_rm_activations.py --dataset hh_rlhf
```

Train a Matryoshka SAE on the cached vectors:

```bash
python train_rm_sae.py \
    --activations /path/to/ultrafeedback_diff.pt \
    --d_sae 32 --k 8 \
    --matryoshka_widths 8 16 \
    --steps 30000 \
    --output checkpoints/rm_sae_ultrafeedback
```

### Stage 3: Analyze & Label SAE Features

Compute alignment between SAE latent directions and ArmoRM's 19 reward-head directions:

```bash
python analyze_sae_directions.py --sae_path checkpoints/rm_sae_ultrafeedback
```

Auto-label SAE latents using contrastive prompting with a local LLM:

```bash
python label_sae_features.py \
    --activations /path/to/ultrafeedback_diff.pt \
    --sae_path checkpoints/rm_sae_ultrafeedback \
    --model_id google/gemma-2-9b-it
```

Generate a combined text report:

```bash
python print_sae_analysis.py
```

### Stage 4: Compare & Visualize

Print the AUROC leaderboard across all scorers and benchmarks:

```bash
python compare_results.py
```

Analyze inter-dimension and inter-dataset correlations:

```bash
python compare_activations.py
```

Generate all plots:

```bash
python gen_plots.py
```

This produces `results/heatmaps.png` (reward dimension correlation heatmaps with factuality label correlations) and `results/auroc_bars.png` (sorted AUROC comparison across all scorers).

## Datasets

| Dataset | Type | Labels | Source |
|---|---|---|---|
| TruthfulQA | QA | Binary (correct/incorrect) | Lin et al., 2022 |
| TriviaQA | QA | Binary (supported/unsupported) | Obeso et al., 2025 |
| LongFact | Long-form generation | Entity-level binary | Obeso et al., 2025 |
| HelpSteer2 | Human feedback | Ordinal correctness (0-4) | Wang et al., 2024 |
| UltraFeedback | Human feedback | Ordinal truthfulness (1-5) | Cui et al., 2024 |

## References

Fan, A., et al. (2026). HalluHard: A Challenging Benchmark for Hallucination Detection. https://arxiv.org/abs/2602.01031

Hosking, T., Blunsom, P., & Bartolo, M. (2024). Human Feedback is not Gold Standard. ICLR 2024. https://arxiv.org/abs/2309.16349

Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. ACL 2022. https://arxiv.org/abs/2109.07958

Obeso, O., Arditi, A., Ferrando, J., Freeman, J., Holmes, C., & Nanda, N. (2025). Real-Time Detection of Hallucinated Entities in Long-Form Generation. https://arxiv.org/abs/2509.03531

Orgad, H., Toker, M., Gekhman, Z., Reichart, R., Szpektor, I., Kotek, H., & Belinkov, Y. (2025). LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations. https://arxiv.org/abs/2410.02707

Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. NeurIPS 2022. https://arxiv.org/abs/2203.02155

Prasad, A.V., Watts, C., Merullo, J., Gala, D., Lewis, O., McGrath, T., & Lubana, E.S. (2026). Features as Rewards: Scalable Supervision for Open-Ended Tasks via Interpretability. https://arxiv.org/abs/2602.10067

Wang, H., Xiong, W., Xie, T., Zhao, H., & Zhang, T. (2024). Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts (ArmoRM). EMNLP 2024. https://arxiv.org/abs/2406.12845

Wang, Z., et al. (2024). HelpSteer2: Open-source dataset for training top-performing reward models. https://arxiv.org/abs/2406.08673
