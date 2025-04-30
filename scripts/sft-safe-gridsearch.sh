#!/usr/bin/env bash
# for resilient_coeff in 1 10 100; do
#   for safety_ratio_tol in 14 25; do
#     for num_safety_samples in "100" "300" "500" "1000" "2000"; do
#       bash ./sft-safe.sh  --resilient_coeff "$resilient_coeff" --safety_ratio_tol "$safety_ratio_tol" --num_safety_samples "$num_safety_samples"
#     done
#   done
# done

for num_safety_samples in "100" "300" "500" "1000" "1500" "2000"; do
  bash ./sft-safe.sh --num_safety_samples "$num_safety_samples"
done