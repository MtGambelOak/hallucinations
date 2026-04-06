"""
Train a Top-k Matryoshka SAE on ArmoRM hidden-state difference vectors.

Reconstructs x_i = h(chosen) - h(rejected).
"""

import argparse
import torch
from pathlib import Path

from sae_lens import (
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
    LoggingConfig,
)
# External libraries my beloved
from sae_lens.training.sae_trainer import SAETrainer, SAETrainerConfig


def make_data_provider(diff_vecs: torch.Tensor, batch_size: int, device: str):
    n = diff_vecs.shape[0]
    batch_size = min(batch_size, n)
    while True:
        perm = torch.randperm(n)
        for i in range(0, n - batch_size + 1, batch_size):
            yield diff_vecs[perm[i : i + batch_size]].to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", default="/scratch/general/vast/u1110118/hallucinations/ultrafeedback_diff.pt")
    parser.add_argument("--d_sae",  type=int,   default=32)
    parser.add_argument("--k",      type=int,   default=8)
    parser.add_argument("--matryoshka_widths", type=int, nargs="+", default=[8, 16],
                        help="Intermediate prefix widths")
    parser.add_argument("--steps",  type=int,   default=30_000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr",     type=float, default=5e-4)
    parser.add_argument("--output", default="checkpoints/rm_sae")
    parser.add_argument("--wandb_project", default="rm_sae")
    parser.add_argument("--no_wandb", action="store_true", default=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading activations from {args.activations}...")
    data = torch.load(args.activations, map_location="cpu")
    diff_vecs = data["diff"]  # (N, d_model)
    d_in = diff_vecs.shape[1]
    n = diff_vecs.shape[0]
    print(f"  {n} difference vectors, d_in={d_in}")

    if args.matryoshka_widths is not None:
        matryoshka_widths = [w for w in args.matryoshka_widths if w < args.d_sae]
    else:
        matryoshka_widths = [args.d_sae // 4, args.d_sae // 2]

    sae_cfg = MatryoshkaBatchTopKTrainingSAEConfig(
        d_in=d_in,
        d_sae=args.d_sae,
        k=args.k,
        matryoshka_widths=matryoshka_widths,
        use_matryoshka_aux_loss=True,
        normalize_activations="expected_average_only_in",
        apply_b_dec_to_input=False,
        dtype="float32",
        device=device,
    )
    sae = MatryoshkaBatchTopKTrainingSAE(sae_cfg)

    total_training_samples = args.steps * args.batch_size
    trainer_cfg = SAETrainerConfig(
        total_training_samples=total_training_samples,
        train_batch_size_samples=args.batch_size,
        lr=args.lr,
        lr_end=0.0,
        lr_scheduler_name="constant",
        lr_warm_up_steps=args.steps // 20,
        lr_decay_steps=args.steps // 5,
        n_restart_cycles=1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        autocast=False,
        device=device,
        checkpoint_path=args.output,
        n_checkpoints=3,
        save_final_checkpoint=True,
        feature_sampling_window=500,
        dead_feature_window=500,
        logger=LoggingConfig(
            log_to_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_log_frequency=30,
            eval_every_n_wandb_logs=20,
        ),
    )

    data_provider = make_data_provider(diff_vecs, args.batch_size, device)
    trainer = SAETrainer(cfg=trainer_cfg, sae=sae, data_provider=data_provider)

    print(f"\nTraining: d_in={d_in}, d_sae={args.d_sae}, k={args.k}, "
          f"matryoshka_widths={matryoshka_widths}, steps={args.steps}")
    print(f"  {n} examples, batch_size={args.batch_size} "
          f"→ ~{total_training_samples / n:.0f} passes through data\n")

    trainer.fit()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    sae.save_model(args.output)
    print(f"\nSAE saved to {args.output}")


if __name__ == "__main__":
    main()
