#!/bin/bash
#SBATCH --account=rai
#SBATCH --partition=rai-gpu-grn
#SBATCH --qos=rai-gpu-grn
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1
#SBATCH -o slurmjob-%j.out-%N
#SBATCH -e slurmjob-%j.err-%N

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cd ~/hallucinations
mkdir -p logs results checkpoints

SCRATCH=/scratch/general/vast/u1493630/hallucinations
SWEEP_ARGS="--d_sae_values 8 16 32 64 --k 8 --steps 30000 --batch_size 512 --lr 5e-4"

for DATASET in helpsteer2_factuality helpsteer2 ultrafeedback_factuality ultrafeedback hh_rlhf; do
    CACHE=$SCRATCH/${DATASET}_diff.pt
    if [ ! -f "$CACHE" ]; then
        echo "Skipping $DATASET - cache not found at $CACHE"
        continue
    fi
    echo "========================================"
    echo "Sweep: $DATASET"
    echo "========================================"
    python sweep_sae_sizes.py \
        --activations $CACHE \
        --dataset $DATASET \
        $SWEEP_ARGS
done

echo "========================================"
echo "Running analysis and direction alignment"
echo "========================================"
python analyze_sae_directions.py --overwrite
python print_sae_analysis.py

echo "Done."
