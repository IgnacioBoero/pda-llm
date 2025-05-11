#!/usr/bin/env bash

for max_length in 2048 4096; do
  for lora_r in 4 16; do 
    for safety_ratio_tol in 50 100; do
      bash ./sft-tools.sh  --important_sft true --safety_ratio_tol "$safety_ratio_tol" --epochs 4 --max_length "$max_length" --lora_r "$lora_r"
    done
  done
done


