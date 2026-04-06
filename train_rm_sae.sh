#!/bin/bash
#SBATCH --job-name=train_rm_sae
#SBATCH --account=marasovic
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-students-grn
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_rm_sae_%j.out
#SBATCH --error=logs/train_rm_sae_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cs6966
source ~/hallucinations/setup_session.sh

cd ~/hallucinations
mkdir -p logs checkpoints

SCRATCH=/scratch/general/vast/u1110118/hallucinations
SAE_ARGS="--steps 30000 --batch_size 512 --lr 5e-4 --no_wandb"

for DATASET in helpsteer2_factuality helpsteer2 ultrafeedback_factuality ultrafeedback hh_rlhf; do
    CACHE=$SCRATCH/${DATASET}_diff.pt
    if [ ! -f "$CACHE" ]; then
        echo "Skipping $DATASET — cache not found at $CACHE"
        continue
    fi
    echo "========================================"
    echo "Training SAE on $DATASET"
    echo "========================================"
    python train_rm_sae.py \
        --activations $CACHE \
        --output checkpoints/rm_sae_${DATASET} \
        $SAE_ARGS
done

echo "Done."
