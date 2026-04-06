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

python cache_rm_activations.py --dataset helpsteer2_factuality \
    --output $SCRATCH/helpsteer2_factuality_diff.pt

python cache_rm_activations.py --dataset helpsteer2 \
    --output $SCRATCH/helpsteer2_diff.pt

python cache_rm_activations.py --dataset ultrafeedback_factuality \
    --output $SCRATCH/ultrafeedback_factuality_diff.pt

python cache_rm_activations.py --dataset ultrafeedback \
    --output $SCRATCH/ultrafeedback_diff.pt

python cache_rm_activations.py --dataset hh_rlhf \
    --output $SCRATCH/hh_rlhf_diff.pt

echo "Done."
