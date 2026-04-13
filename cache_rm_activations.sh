#!/bin/bash
#SBATCH --job-name=cache_rm_acts
#SBATCH --account=marasovic
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-students-grn
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cache_rm_%j.out
#SBATCH --error=logs/cache_rm_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cs6966
source ~/hallucinations/setup_session.sh

cd ~/hallucinations
mkdir -p logs /scratch/general/vast/u1110118/hallucinations

SCRATCH=/scratch/general/vast/u1110118/hallucinations

for DATASET in helpsteer2_factuality helpsteer2 ultrafeedback_factuality ultrafeedback hh_rlhf; do
    # skip final layer if already cached
    if [ ! -f "$SCRATCH/${DATASET}_diff.pt" ]; then
        echo "Caching $DATASET (final layer)..."
        python cache_rm_activations.py --dataset $DATASET \
            --output $SCRATCH/${DATASET}_diff.pt
    fi

    for LAYER in 8 16; do
        if [ ! -f "$SCRATCH/${DATASET}_layer${LAYER}_diff.pt" ]; then
            echo "Caching $DATASET (layer $LAYER)..."
            python cache_rm_activations.py --dataset $DATASET --layer $LAYER \
                --output $SCRATCH/${DATASET}_layer${LAYER}_diff.pt
        fi
    done
done

echo "Done."
