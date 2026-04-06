#!/bin/bash
#SBATCH --job-name=hallucinations_evals
#SBATCH --account=marasovic
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-students-grn
#SBATCH --gres=gpu:rtxpr6000bl:1
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
echo "ArmoRM evals"
echo "========================================"
python eval.py --scorer armorm --dataset longfact      --save results/longfact_armorm.json
python eval.py --scorer armorm --dataset triviaqa      --save results/triviaqa_armorm.json
python eval.py --scorer armorm --dataset truthfulqa    --save results/truthfulqa_armorm.json
python eval.py --scorer armorm --dataset helpsteer2    --save results/helpsteer2_armorm.json
python eval.py --scorer armorm --dataset ultrafeedback --max_samples 10000 --save results/ultrafeedback_armorm.json

echo "========================================"
echo "Probe evals — Gemma2-9b linear"
echo "========================================"
python eval.py --scorer probe --dataset longfact      --probe_id gemma2_9b_linear --save results/longfact_probe_gemma2_9b_linear.json
python eval.py --scorer probe --dataset triviaqa      --probe_id gemma2_9b_linear --save results/triviaqa_probe_gemma2_9b_linear.json
python eval.py --scorer probe --dataset truthfulqa    --probe_id gemma2_9b_linear --save results/truthfulqa_probe_gemma2_9b_linear.json
python eval.py --scorer probe --dataset helpsteer2    --probe_id gemma2_9b_linear --save results/helpsteer2_probe_gemma2_9b_linear.json
python eval.py --scorer probe --dataset ultrafeedback --max_samples 10000 --probe_id gemma2_9b_linear --save results/ultrafeedback_probe_gemma2_9b_linear.json

echo "========================================"
echo "Probe evals — Gemma2-9b LoRA (KL)"
echo "========================================"
python eval.py --scorer probe --dataset longfact      --probe_id gemma2_9b_lora_lambda_kl_0_05 --save results/longfact_probe_gemma2_9b_lora_kl.json
python eval.py --scorer probe --dataset triviaqa      --probe_id gemma2_9b_lora_lambda_kl_0_05 --save results/triviaqa_probe_gemma2_9b_lora_kl.json
python eval.py --scorer probe --dataset truthfulqa    --probe_id gemma2_9b_lora_lambda_kl_0_05 --save results/truthfulqa_probe_gemma2_9b_lora_kl.json
python eval.py --scorer probe --dataset helpsteer2    --probe_id gemma2_9b_lora_lambda_kl_0_05 --save results/helpsteer2_probe_gemma2_9b_lora_kl.json
python eval.py --scorer probe --dataset ultrafeedback --max_samples 10000 --probe_id gemma2_9b_lora_lambda_kl_0_05 --save results/ultrafeedback_probe_gemma2_9b_lora_kl.json

echo "========================================"
echo "Probe evals — Llama3.1-8b linear"
echo "========================================"
python eval.py --scorer probe --dataset longfact      --probe_id llama3_1_8b_linear --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --save results/longfact_probe_llama3_1_8b_linear.json
python eval.py --scorer probe --dataset triviaqa      --probe_id llama3_1_8b_linear --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --save results/triviaqa_probe_llama3_1_8b_linear.json
python eval.py --scorer probe --dataset truthfulqa    --probe_id llama3_1_8b_linear --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --save results/truthfulqa_probe_llama3_1_8b_linear.json
python eval.py --scorer probe --dataset helpsteer2    --probe_id llama3_1_8b_linear --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --save results/helpsteer2_probe_llama3_1_8b_linear.json
python eval.py --scorer probe --dataset ultrafeedback --max_samples 10000 --probe_id llama3_1_8b_linear --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --save results/ultrafeedback_probe_llama3_1_8b_linear.json

echo "========================================"
echo "Probe evals — Llama3.1-8b LoRA (KL)"
echo "========================================"
python eval.py --scorer probe --dataset longfact      --probe_id llama3_1_8b_lora_lambda_kl_0_05 --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --save results/longfact_probe_llama3_1_8b_lora_kl.json
python eval.py --scorer probe --dataset triviaqa      --probe_id llama3_1_8b_lora_lambda_kl_0_05 --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --save results/triviaqa_probe_llama3_1_8b_lora_kl.json
python eval.py --scorer probe --dataset truthfulqa    --probe_id llama3_1_8b_lora_lambda_kl_0_05 --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --save results/truthfulqa_probe_llama3_1_8b_lora_kl.json
python eval.py --scorer probe --dataset helpsteer2    --probe_id llama3_1_8b_lora_lambda_kl_0_05 --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --save results/helpsteer2_probe_llama3_1_8b_lora_kl.json
python eval.py --scorer probe --dataset ultrafeedback --max_samples 10000 --probe_id llama3_1_8b_lora_lambda_kl_0_05 --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --save results/ultrafeedback_probe_llama3_1_8b_lora_kl.json

echo "========================================"
echo "Probe evals — Llama3.1-8b LoRA (LM)"
echo "========================================"
python eval.py --scorer probe --dataset longfact      --probe_id llama3_1_8b_lora_lambda_lm_0_01 --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --save results/longfact_probe_llama3_1_8b_lora_lm.json
python eval.py --scorer probe --dataset triviaqa      --probe_id llama3_1_8b_lora_lambda_lm_0_01 --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --save results/triviaqa_probe_llama3_1_8b_lora_lm.json
python eval.py --scorer probe --dataset truthfulqa    --probe_id llama3_1_8b_lora_lambda_lm_0_01 --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --save results/truthfulqa_probe_llama3_1_8b_lora_lm.json
python eval.py --scorer probe --dataset helpsteer2    --probe_id llama3_1_8b_lora_lambda_lm_0_01 --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --save results/helpsteer2_probe_llama3_1_8b_lora_lm.json
python eval.py --scorer probe --dataset ultrafeedback --max_samples 10000 --probe_id llama3_1_8b_lora_lambda_lm_0_01 --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --save results/ultrafeedback_probe_llama3_1_8b_lora_lm.json

echo "========================================"
echo "Probe evals — Qwen2.5-7b linear"
echo "========================================"
python eval.py --scorer probe --dataset longfact      --probe_id qwen2_5_7b_linear --model_id Qwen/Qwen2.5-7B-Instruct --save results/longfact_probe_qwen2_5_7b_linear.json
python eval.py --scorer probe --dataset triviaqa      --probe_id qwen2_5_7b_linear --model_id Qwen/Qwen2.5-7B-Instruct --save results/triviaqa_probe_qwen2_5_7b_linear.json
python eval.py --scorer probe --dataset truthfulqa    --probe_id qwen2_5_7b_linear --model_id Qwen/Qwen2.5-7B-Instruct --save results/truthfulqa_probe_qwen2_5_7b_linear.json
python eval.py --scorer probe --dataset helpsteer2    --probe_id qwen2_5_7b_linear --model_id Qwen/Qwen2.5-7B-Instruct --save results/helpsteer2_probe_qwen2_5_7b_linear.json
python eval.py --scorer probe --dataset ultrafeedback --max_samples 10000 --probe_id qwen2_5_7b_linear --model_id Qwen/Qwen2.5-7B-Instruct --save results/ultrafeedback_probe_qwen2_5_7b_linear.json

echo "========================================"
echo "Probe evals — Qwen2.5-7b LoRA (KL)"
echo "========================================"
python eval.py --scorer probe --dataset longfact      --probe_id qwen2_5_7b_lora_lambda_kl_0_05 --model_id Qwen/Qwen2.5-7B-Instruct --save results/longfact_probe_qwen2_5_7b_lora_kl.json
python eval.py --scorer probe --dataset triviaqa      --probe_id qwen2_5_7b_lora_lambda_kl_0_05 --model_id Qwen/Qwen2.5-7B-Instruct --save results/triviaqa_probe_qwen2_5_7b_lora_kl.json
python eval.py --scorer probe --dataset truthfulqa    --probe_id qwen2_5_7b_lora_lambda_kl_0_05 --model_id Qwen/Qwen2.5-7B-Instruct --save results/truthfulqa_probe_qwen2_5_7b_lora_kl.json
python eval.py --scorer probe --dataset helpsteer2    --probe_id qwen2_5_7b_lora_lambda_kl_0_05 --model_id Qwen/Qwen2.5-7B-Instruct --save results/helpsteer2_probe_qwen2_5_7b_lora_kl.json
python eval.py --scorer probe --dataset ultrafeedback --max_samples 10000 --probe_id qwen2_5_7b_lora_lambda_kl_0_05 --model_id Qwen/Qwen2.5-7B-Instruct --save results/ultrafeedback_probe_qwen2_5_7b_lora_kl.json

echo "========================================"
echo "Comparison"
echo "========================================"
python compare_results.py
python compare_activations.py
python plot_heatmaps.py

echo "Done."
