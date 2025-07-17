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

for epochs in 4; do
  for resilient_coeff in 1 10 100; do
    for safety_ratio_tol in -0.7; do
      for algo in "l2" "l1" "erm" "dual" "penalty"; do
          bash ./sft-boolq.sh  --algo  "$algo" --resilient_coeff "$resilient_coeff" --safety_ratio_tol "$safety_ratio_tol" --epochs "$epochs"
      done
    done
  done
done
