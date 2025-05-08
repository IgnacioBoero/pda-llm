#!/usr/bin/env bash

bash ./sft-tools.sh  --important_sft false

for safety_ratio_tol in 50 100; do
  bash ./sft-tools.sh  --important_sft true --safety_ratio_tol "$safety_ratio_tol" --epochs 4
done

