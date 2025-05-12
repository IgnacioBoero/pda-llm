#!/usr/bin/env bash


for dataset in "tools"; do
  for important_sft in  false true; do
    for safety_ratio_tol in 180; do
      # Activate the "pda" conda environment for the bash script
      source ~/miniconda3/etc/profile.d/conda.sh
      conda activate pda
      bash ./sft-tools.sh \
        --dataset="${dataset}" \
        --important_sft="${important_sft}" \
        --safety_ratio_tol="${safety_ratio_tol}"
    done
  done
done

