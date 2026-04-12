#!/bin/bash
#SBATCH --job-name=hallucinations_master
#SBATCH --account=rai
#SBATCH --partition=rai-gpu-grn
#SBATCH --qos=rai-gpu-grn
#SBATCH --time=24:00:00
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

SCRATCH=/scratch/general/vast/u1493630/hallucinations
mkdir -p $SCRATCH

# ════════════════════════════════════════════════════════════════════════
# STAGE 1: ArmoRM evaluations
# ════════════════════════════════════════════════════════════════════════
echo "========================================"
echo "STAGE 1a: ArmoRM evals"
echo "========================================"
python eval.py --scorer armorm --dataset longfact      --save results/longfact_armorm.json
python eval.py --scorer armorm --dataset triviaqa      --save results/triviaqa_armorm.json
python eval.py --scorer armorm --dataset truthfulqa    --save results/truthfulqa_armorm.json
python eval.py --scorer armorm --dataset helpsteer2    --save results/helpsteer2_armorm.json
python eval.py --scorer armorm --dataset ultrafeedback --max_samples 10000 --save results/ultrafeedback_armorm.json

# ════════════════════════════════════════════════════════════════════════
# STAGE 1b: Probe evaluations
# ════════════════════════════════════════════════════════════════════════
DATASETS="longfact triviaqa truthfulqa helpsteer2"
UF_DATASETS="ultrafeedback"

run_probe() {
    local probe_id=$1
    local model_id=${2:-}
    local model_arg=""
    if [ -n "$model_id" ]; then
        model_arg="--model_id $model_id"
    fi

    local safe_probe=$(echo "$probe_id" | sed 's/_lambda_kl_0_05/_lora_kl/' | sed 's/_lambda_lm_0_01/_lora_lm/')

    for ds in $DATASETS; do
        python eval.py --scorer probe --dataset $ds \
            --probe_id $probe_id $model_arg \
            --save results/${ds}_probe_${safe_probe}.json
    done
    for ds in $UF_DATASETS; do
        python eval.py --scorer probe --dataset $ds --max_samples 10000 \
            --probe_id $probe_id $model_arg \
            --save results/${ds}_probe_${safe_probe}.json
    done
}

echo "========================================"
echo "STAGE 1b: Probe evals — Gemma2-9b linear"
echo "========================================"
run_probe gemma2_9b_linear

echo "========================================"
echo "STAGE 1b: Probe evals — Gemma2-9b LoRA (KL)"
echo "========================================"
run_probe gemma2_9b_lora_lambda_kl_0_05

echo "========================================"
echo "STAGE 1b: Probe evals — Llama3.1-8b linear"
echo "========================================"
run_probe llama3_1_8b_linear meta-llama/Meta-Llama-3.1-8B-Instruct

echo "========================================"
echo "STAGE 1b: Probe evals — Llama3.1-8b LoRA (KL)"
echo "========================================"
run_probe llama3_1_8b_lora_lambda_kl_0_05 meta-llama/Meta-Llama-3.1-8B-Instruct

echo "========================================"
echo "STAGE 1b: Probe evals — Llama3.1-8b LoRA (LM)"
echo "========================================"
run_probe llama3_1_8b_lora_lambda_lm_0_01 meta-llama/Meta-Llama-3.1-8B-Instruct

echo "========================================"
echo "STAGE 1b: Probe evals — Qwen2.5-7b linear"
echo "========================================"
run_probe qwen2_5_7b_linear Qwen/Qwen2.5-7B-Instruct

echo "========================================"
echo "STAGE 1b: Probe evals — Qwen2.5-7b LoRA (KL)"
echo "========================================"
run_probe qwen2_5_7b_lora_lambda_kl_0_05 Qwen/Qwen2.5-7B-Instruct

# ════════════════════════════════════════════════════════════════════════
# STAGE 2: Cache ArmoRM hidden-state difference vectors
# ════════════════════════════════════════════════════════════════════════
echo "========================================"
echo "STAGE 2: Cache RM activations"
echo "========================================"
for DATASET in helpsteer2_factuality helpsteer2 ultrafeedback_factuality ultrafeedback hh_rlhf; do
    OUTPUT=$SCRATCH/${DATASET}_diff.pt
    if [ -f "$OUTPUT" ]; then
        echo "  $DATASET — cache exists, skipping"
        continue
    fi
    echo "  Caching $DATASET..."
    python cache_rm_activations.py --dataset $DATASET --output $OUTPUT
done

# ════════════════════════════════════════════════════════════════════════
# STAGE 3: SAE vocabulary size sweep (train + analyze + plot)
# ════════════════════════════════════════════════════════════════════════
echo "========================================"
echo "STAGE 3: SAE vocabulary size sweep"
echo "========================================"
SWEEP_ARGS="--d_sae_values 8 16 32 64 --k 8 --steps 30000 --batch_size 512 --lr 5e-4"

for DATASET in helpsteer2_factuality helpsteer2 ultrafeedback_factuality ultrafeedback hh_rlhf; do
    CACHE=$SCRATCH/${DATASET}_diff.pt
    if [ ! -f "$CACHE" ]; then
        echo "  Skipping $DATASET — cache not found"
        continue
    fi
    echo "  Sweep: $DATASET"
    python sweep_sae_sizes.py \
        --activations $CACHE \
        --dataset $DATASET \
        $SWEEP_ARGS
done

# ════════════════════════════════════════════════════════════════════════
# STAGE 4: SAE direction analysis
# ════════════════════════════════════════════════════════════════════════
echo "========================================"
echo "STAGE 4: SAE direction analysis"
echo "========================================"
python analyze_sae_directions.py --overwrite
python print_sae_analysis.py

# ════════════════════════════════════════════════════════════════════════
# STAGE 5: Auto-label SAE features
# ════════════════════════════════════════════════════════════════════════
echo "========================================"
echo "STAGE 5: Label SAE features"
echo "========================================"
for DATASET in helpsteer2_factuality helpsteer2 ultrafeedback_factuality ultrafeedback hh_rlhf; do
    CACHE=$SCRATCH/${DATASET}_diff.pt
    SAE=checkpoints/rm_sae_${DATASET}
    if [ ! -f "$CACHE" ]; then
        echo "  Skipping $DATASET — cache not found"
        continue
    fi
    if [ ! -d "$SAE" ]; then
        echo "  Skipping $DATASET — SAE checkpoint not found at $SAE"
        continue
    fi
    echo "  Labeling: $DATASET"
    python label_sae_features.py \
        --activations $CACHE \
        --sae_path $SAE \
        --model_id google/gemma-2-9b-it
done

# ════════════════════════════════════════════════════════════════════════
# STAGE 6: Comparison tables and plots
# ════════════════════════════════════════════════════════════════════════
echo "========================================"
echo "STAGE 6: Comparison & plots"
echo "========================================"
python compare_results.py
python compare_activations.py
python gen_plots.py

echo "========================================"
echo "ALL DONE"
echo "========================================"
echo "Results in: results/"
echo "Checkpoints in: checkpoints/"