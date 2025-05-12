#!/usr/bin/env bash


for dataset in "tools:0.01" "tools-original:0.01"; do
  for important_sft in  false true; do
    for safety_ratio_tol in 180; do
      # Activate the "pda" conda environment for the bash script
      source ~/miniconda3/etc/profile.d/conda.sh
      conda activate pda
      bash ./sft-tools.sh \
        --dataset="${dataset}" \
        --important_sft="${important_sft}" \
        --safety_ratio_tol="${safety_ratio_tol}"
      python ./merge_adapters.py  --adapter_id ~/pda-llm/output/sft-tools/run-"${dataset}"-"${important_sft}" --target_dir ~/pda-llm/output/sft-tools/run-"${dataset}"-"${important_sft}"-merged
      # Switch to the "BFCL" conda environment for bfcl commands
      conda activate bfcl
      bfcl generate   --model Team-ACE/ToolACE-2-8B --test-category simple,live_simple  --backend vllm   --num-gpus 2   --gpu-memory-utilization 0.8   --local-model-path ~/pda-llm/output/sft-tools/run-"${dataset}"-"${important_sft}"-merged --result-dir ./results_"${dataset}"_"${important_sft}"_"${safety_ratio_tol}"/ 
      bfcl evaluate --model Team-ACE/ToolACE-2-8B --result-dir ./results_"${dataset}"_"${important_sft}"_"${safety_ratio_tol}" --score-dir score__"${dataset}"_"${important_sft}"_"${safety_ratio_tol}"
    done
  done
done

