#!/usr/bin/env bash
#
# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

if [ -z "${BASH_VERSION}" ]; then
	echo "Please use bash to run this script." >&2
	exit 1
fi

set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"
export WANDB_MODE=online
export WANDB_ENTITY="alelab"

MODEL_NAME_OR_PATH="huggyllama/llama-7b"
timestamp="$(date +%Y%m%d-%H%M%S)"
unset HOSTFILE
ZERO_STAGE=0
OFFLOAD="none"
SAFE_SFT=False
# GRIDSEARCH PARAMS
SAFETY_RATIO_TOL=10
RESILIENT_COEFF=1
LEARNING_RATE=1e-4
NUM_SAFETY_SAMPLES="100"
EPOCHS=6
while [[ "$#" -gt 0 ]]; do
	arg="$1"
	shift
	case "${arg}" in
		--model_name_or_path)
			MODEL_NAME_OR_PATH="$1"
			shift
			;;
		--model_name_or_path=*)
			MODEL_NAME_OR_PATH="${arg#*=}"
			;;
		--output_dir)
			OUTPUT_DIR="$1"
			shift
			;;
		--output_dir=*)
			OUTPUT_DIR="${arg#*=}"
			;;
		--hostfile)
			HOSTFILE="$1"
			shift
			;;
		--hostfile=*)
			HOSTFILE="${arg#*=}"
			;;
		--zero_stage)
			ZERO_STAGE="$1"
			shift
			;;
		--zero_stage=*)
			ZERO_STAGE="${arg#*=}"
			;;
		--offload)
			OFFLOAD="$1"
			shift
			;;
		--offload=*)
			OFFLOAD="${arg#*=}"
			;;
		--safety_ratio_tol)
			SAFETY_RATIO_TOL="$1"
			shift
			;;
		--safety_ratio_tol=*)
			SAFETY_RATIO_TOL="${arg#*=}"
			;;
		--resilient_coeff)
			RESILIENT_COEFF="$1"
			shift
			;;
		--resilient_coeff=*)
			RESILIENT_COEFF="${arg#*=}"
			;;
		--learning_rate)
			LEARNING_RATE="$1"
			shift
			;;
		--learning_rate=*)
			LEARNING_RATE="${arg#*=}"
			;;
		--epochs)
			EPOCHS="$1"
			shift
			;;
		--epochs=*)
			EPOCHS="${arg#*=}"
			;;
		--safe_sft)
			SAFE_SFT="$1"
			shift
			;;
		--safe_sft=*)
			SAFE_SFT="${arg#*=}"
			;;
		--num_safety_samples)
			NUM_SAFETY_SAMPLES="$1"
			shift
			;;
		--num_safety_samples=*)
			NUM_SAFETY_SAMPLES="${arg#*=}"
			;;
		*)
			echo "Unknown parameter passed: '${arg}'" >&2
			exit 1
			;;
	esac
done

OUTPUT_DIR="${ROOT_DIR}/output/sft-safe/${NUM_SAFETY_SAMPLES}/run-${SAFE_SFT}-${NUM_SAFETY_SAMPLES}-${RESILIENT_COEFF}-${SAFETY_RATIO_TOL}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
	echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

cp -f "$0" "${OUTPUT_DIR}/script.sh"

export WANDB_API_KEY="6a71e7fad84fe1aa8f6ccaa01e4e02fcf4c7ffb4"
if [[ -z "${WANDB_API_KEY}" ]]; then
	export WANDB_MODE="offline"
fi

MASTER_PORT_START=10000
MASTER_PORT_END=65535
MASTER_PORT="$(
	comm -23 \
		<(seq "${MASTER_PORT_START}" "${MASTER_PORT_END}" | sort) \
		<(ss -Htan | awk '{ print $4 }' | awk -F ':' '{ print $NF }' | sort -u) |
		shuf | head -n 1
)"

DEEPSPEED_ARGS=()
if [[ -n "${HOSTFILE+x}" ]]; then
	DEEPSPEED_ARGS+=("--hostfile" "${HOSTFILE}")
fi
DEEPSPEED_ARGS+=("--master_port" "${MASTER_PORT}")

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)
echo "--------- Environment sanity check ---------"
echo "shell:        $0 running under bash $BASH_VERSION"
echo "conda env:    ${CONDA_DEFAULT_ENV:-<none>}"
echo "python:       $(which python)"
python - <<'PY'
import sys, os
print("sys.executable :", sys.executable)
print("python version :", sys.version.split()[0])
print("CONDA_PREFIX   :", os.environ.get("CONDA_PREFIX"))
PY
echo "deepspeed:    $(command -v deepspeed)"
echo "--------------------------------------------"

CUDA_VISIBLE_DEVICES=0,1 deepspeed "${DEEPSPEED_ARGS[@]}" \
	--module safe_rlhf.algorithms.safe_ft \
	--train_datasets "SAFE-ALPACA/${NUM_SAFETY_SAMPLES}" \
	--model_name_or_path "${MODEL_NAME_OR_PATH}" \
	--cache_dir "${ROOT_DIR}/cache/sft-${NUM_SAFETY_SAMPLES}" \
	--safe_sft "${SAFE_SFT}"	 \
	--max_length 1024 \
	--trust_remote_code True \
	--epochs "${EPOCHS}"  \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
	--gradient_accumulation_steps 128 \
	--gradient_checkpointing \
	--learning_rate  "${LEARNING_RATE}" \
	--lr_scheduler_type cosine \
	--lr_warmup_ratio 0.03 \
	--weight_decay 0.0 \
	--seed 42 \
	--output_dir "${OUTPUT_DIR}" \
	--log_type wandb \
	--log_project SAFE-SFT-v3 \
	--zero_stage "${ZERO_STAGE}" \
	--offload "${OFFLOAD}" \
	--safety_ratio_tol "${SAFETY_RATIO_TOL}" \
	--resilient_coeff "${RESILIENT_COEFF}" \
	--lora_r "4" \
	--lora_alpha "16" \
	--lora_dropout "0.05" \
	--bf16 False \
	--fp16 True \
	--tf32 True
