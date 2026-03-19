#!/bin/bash
#SBATCH --job-name=hallucinations_evals
#SBATCH --account=marasovic
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-students-grn
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --exclude=grn075
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/evals_%j.out
#SBATCH --error=logs/evals_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cs6966
source ~/hallucinations/setup_session.sh

cd ~/hallucinations
mkdir -p logs results

echo "========================================"
echo "Probe evals"
echo "========================================"
python eval.py --scorer probe --dataset longfact   --save results/longfact_probe.json
python eval.py --scorer probe --dataset triviaqa   --probe_id llama3_1_8b_linear --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --save results/triviaqa_probe.json
python eval.py --scorer probe --dataset truthfulqa --save results/truthfulqa_probe.json
python eval.py --scorer probe --dataset helpsteer2 --save results/helpsteer2_probe.json

echo "========================================"
echo "ArmoRM evals"
echo "========================================"
python eval.py --scorer armorm --dataset longfact   --save results/longfact_armorm.json
python eval.py --scorer armorm --dataset triviaqa   --save results/triviaqa_armorm.json
python eval.py --scorer armorm --dataset truthfulqa --save results/truthfulqa_armorm.json
python eval.py --scorer armorm --dataset helpsteer2 --save results/helpsteer2_armorm.json

echo "========================================"
echo "Comparison"
echo "========================================"
python compare_results.py
python compare_activations.py
python plot_heatmaps.py

echo "Done."
