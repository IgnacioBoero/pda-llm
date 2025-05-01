#!/usr/bin/env bash
for num_safety_samples in  "300" "2000"; do
  for resilient_coeff in 1 10 0.1; do
    for safety_ratio_tol in 14 25; do
      bash ./sft-safe.sh  --safe_sft true --num_safety_samples "$num_safety_samples" --resilient_coeff "$resilient_coeff" --safety_ratio_tol "$safety_ratio_tol" 
    done
  done
  bash ./sft-safe.sh  --safe_sft false --num_safety_samples "$num_safety_samples"
done