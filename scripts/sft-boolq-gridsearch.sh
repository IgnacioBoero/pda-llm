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

for epochs in 3 5; do
  for resilient_coeff in 1; do
    for safety_ratio_tol in -0.7 -0.35; do
      for algo in "l2" "l1" "erm" "dual" "penalty"; do
          bash ./sft-boolq.sh  --algo  "$algo" --resilient_coeff "$resilient_coeff" --safety_ratio_tol "$safety_ratio_tol" --epochs "$epochs"
          python ~/pda-llm/safe_rlhf/evaluate/boolq/eval.py --model ~/pda-llm/output/sft-boolq/run-"${algo}"-"${epochs}"-"${resilient_coeff}"-"${safety_ratio_tol}" --output_path ~/pda-llm/safe_rlhf/evaluate/boolq/outputs/run-"${algo}"-"${epochs}"-"${resilient_coeff}"-"${safety_ratio_tol}"
          lm_eval --model hf --model_args pretrained=~/pda-llm/output/sft-boolq/run-"${algo}"-"${epochs}"-"${resilient_coeff}"-"${safety_ratio_tol}",backend=peft  --tasks openbookqa,piqa,boolq  --device cuda:0  --batch_size 8 --output_path ~/pda-llm/safe_rlhf/evaluate/boolq/outputs/run-"${algo}"-"${epochs}"-"${resilient_coeff}"-"${safety_ratio_tol}"/helpfulness.json --write_out --log_samples
      done
    done
  done
done
