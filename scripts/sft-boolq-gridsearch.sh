#!/usr/bin/env bash
# ---- logging ----
LOG_DIR=~/pda-llm/logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_sft_boolq.log"

exec > >(tee -a "$LOG_FILE") 2>&1 

GPU_IDS=(0 1)                        

for GPU_ID in "${GPU_IDS[@]}"; do
  sudo nvidia-smi -i "$GPU_ID" -c EXCLUSIVE_PROCESS
done
if [ -f ~/pda-llm/.env ]; then
  set -a
  source ~/pda-llm/.env
  set +a
fi

echo "OPENAI_API_KEY: $OPENAI_API_KEY"

for safety_ratio_tol in -0.2 -0.1 -0.3; do
  # for model_name_or_path in "huggyllama/llama-7b" "meta-llama/Llama-3.1-8B" "meta-llama/Llama-2-7b-hf" "Qwen/Qwen3-8B"  ; do
  for model_name_or_path in "huggyllama/llama-7b" "meta-llama/Llama-3.1-8B"; do
    for epochs in 3; do
      for resilient_coeff in 1; do
        # for algo in "l2" "l1" "erm" "dual" "penalty"; do
        for algo in "l2" "l1" "erm"; do
            bash ./sft-boolq.sh  --algo  "$algo" --resilient_coeff "$resilient_coeff" --safety_ratio_tol "$safety_ratio_tol" --epochs "$epochs" --model_name_or_path "$model_name_or_path"
            python ~/pda-llm/safe_rlhf/evaluate/boolq/eval.py --model ~/pda-llm/output/sft-boolq/"${model_name_or_path//\//_}"/run-"${algo}"-"${epochs}"-"${resilient_coeff}"-"${safety_ratio_tol}" --output_path ~/pda-llm/safe_rlhf/evaluate/boolq/outputs/"${model_name_or_path//\//_}"/run-"${algo}"-"${epochs}"-"${resilient_coeff}"-"${safety_ratio_tol}"
            lm_eval --model hf --model_args pretrained=~/pda-llm/output/sft-boolq/"${model_name_or_path//\//_}"/run-"${algo}"-"${epochs}"-"${resilient_coeff}"-"${safety_ratio_tol}",backend=peft  --tasks openbookqa,piqa,boolq  --device cuda:0  --batch_size 8 --output_path ~/pda-llm/safe_rlhf/evaluate/boolq/outputs/"${model_name_or_path//\//_}"/run-"${algo}"-"${epochs}"-"${resilient_coeff}"-"${safety_ratio_tol}"/helpfulness.json --write_out --log_samples
        done
      done
    done
  done
done
