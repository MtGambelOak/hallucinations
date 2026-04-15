#!/bin/bash
#SBATCH --job-name=sweep_and_layers
#SBATCH --account=rai
#SBATCH --partition=rai-gpu-grn
#SBATCH --qos=rai-gpu-grn
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH -o slurmjob-%j.out-%N
#SBATCH -e slurmjob-%j.err-%N

set -euo pipefail

source ~/hallucinations/venv/bin/activate
source ~/hallucinations/setup_session.sh

cd ~/hallucinations
mkdir -p logs results checkpoints

# Configure scratch paths
# Try both team members' scratch dirs; use whichever has the caches.
SCRATCH_A=/scratch/general/vast/u1110118/hallucinations
SCRATCH_B=/scratch/general/vast/u1493630/hallucinations

# Pick the scratch dir that has activation caches
if [ -f "$SCRATCH_A/helpsteer2_diff.pt" ]; then
    SCRATCH=$SCRATCH_A
elif [ -f "$SCRATCH_B/helpsteer2_diff.pt" ]; then
    SCRATCH=$SCRATCH_B
else
    echo "ERROR: No activation caches found at either scratch path."
    echo "  Checked: $SCRATCH_A"
    echo "  Checked: $SCRATCH_B"
    exit 1
fi
echo "Using activation caches from: $SCRATCH"

DATASETS="helpsteer2_factuality helpsteer2 ultrafeedback_factuality ultrafeedback hh_rlhf"
SWEEP_ARGS="--d_sae_values 8 16 32 64 --k 8 --steps 30000 --batch_size 512 --lr 5e-4"
SAE_ARGS="--steps 30000 --batch_size 512 --lr 5e-4 --no_wandb"


# ========================================================================
# PART 1: SAE vocabulary size sweep (peer feedback commitment)
# ========================================================================
echo ""
echo "========================================"
echo "PART 1: SAE vocabulary size sweep"
echo "========================================"

for DATASET in $DATASETS; do
    CACHE=$SCRATCH/${DATASET}_diff.pt
    if [ ! -f "$CACHE" ]; then
        echo "  Skipping $DATASET - cache not found at $CACHE"
        continue
    fi
    echo "----------------------------------------"
    echo "  Sweep: $DATASET"
    echo "----------------------------------------"
    python sweep_sae_sizes.py \
        --activations $CACHE \
        --dataset $DATASET \
        $SWEEP_ARGS
done

echo ""
echo "========================================"
echo "PART 1 COMPLETE: Sweep results in results/sae_sweep_*.{json,png}"
echo "========================================"


# ========================================================================
# PART 2: Cross-layer SAE comparison (peer feedback commitment)
# ========================================================================
echo ""
echo "========================================"
echo "PART 2: Cross-layer SAE comparison"
echo "========================================"

# Step 2a: Cache intermediate layer activations if they don't exist
NEED_CACHE=false
for DATASET in $DATASETS; do
    for LAYER in 8 16; do
        if [ ! -f "$SCRATCH/${DATASET}_layer${LAYER}_diff.pt" ]; then
            NEED_CACHE=true
            break 2
        fi
    done
done

if $NEED_CACHE; then
    echo ""
    echo "  Intermediate layer caches not found. Generating them..."
    echo "  (This requires ArmoRM inference - one-time cost.)"
    echo ""
    for DATASET in $DATASETS; do
        for LAYER in 8 16; do
            OUTPUT=$SCRATCH/${DATASET}_layer${LAYER}_diff.pt
            if [ -f "$OUTPUT" ]; then
                echo "  $DATASET layer $LAYER - cache exists, skipping"
                continue
            fi
            echo "  Caching $DATASET (layer $LAYER)..."
            python cache_rm_activations.py --dataset $DATASET --layer $LAYER \
                --output $OUTPUT
        done
    done
else
    echo "  All intermediate layer caches found."
fi

# Step 2b: Train SAEs on intermediate layers if checkpoints don't exist
echo ""
echo "  Training SAEs on intermediate layers..."
for DATASET in $DATASETS; do
    for LAYER in 8 16; do
        CACHE=$SCRATCH/${DATASET}_layer${LAYER}_diff.pt
        CKPT=checkpoints/rm_sae_${DATASET}_layer${LAYER}

        if [ ! -f "$CACHE" ]; then
            echo "  Skipping $DATASET layer $LAYER - cache not found"
            continue
        fi
        if [ -d "$CKPT" ] && [ -f "$CKPT/cfg.json" ]; then
            echo "  $DATASET layer $LAYER - SAE checkpoint exists, skipping"
            continue
        fi

        echo "  Training SAE: $DATASET layer $LAYER..."
        python train_rm_sae.py \
            --activations $CACHE \
            --output $CKPT \
            $SAE_ARGS
    done
done

# Step 2c: Run cross-layer comparison
echo ""
echo "  Running cross-layer comparison..."
for DATASET in ultrafeedback helpsteer2 hh_rlhf; do
    echo "  compare_layers: $DATASET"
    python compare_layers.py --dataset $DATASET --activation_dir $SCRATCH
done

echo ""
echo "========================================"
echo "PART 2 COMPLETE: Results in results/cross_layer_comparison_*.{json,png}"
echo "========================================"

echo ""
echo "========================================"
echo "ALL DONE"
echo "========================================"
echo "Sweep results:       results/sae_sweep_*.json  results/sae_sweep_*.png"
echo "Cross-layer results: results/cross_layer_comparison_*.json  results/cross_layer_comparison_*.png"
