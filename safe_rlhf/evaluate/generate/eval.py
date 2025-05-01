
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

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from tqdm import tqdm

from safe_rlhf.configs.constants import PROMPT_INPUT
from safe_rlhf.models import load_pretrained_models
from safe_rlhf.utils import to_device

from peft import AutoPeftModelForCausalLM

import numpy as np

def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate outputs for given inputs',
    )

    # Model
    parser.add_argument(
        '--model',
        type=str,
        help='the name or path of the model.',
        required=True,
    )

    # Input path
    parser.add_argument(
        '--problem_path',
        type=str,
        default='./data',
        help='The path to the input file.',
    )
    # # Logging
    # parser.add_argument(
    #     '--output_dir',
    #     type=str,
    #     default=None,
    #     help='Where to store the eval output.',
    # )

    return parser.parse_args()


def generate_answer(problems: list[str], model_name_or_path: str) -> list[str]:
    model, tokenizer = load_pretrained_models(
        model_name_or_path,
        auto_device_mapping=True,
        auto_model_type = AutoPeftModelForCausalLM,
        trust_remote_code=True,
    )
    answers = []
    for problem in tqdm(problems):
        prompt = PROMPT_INPUT.format(input=problem)
        input_ids = to_device(
            tokenizer(prompt, return_tensors='pt'),
            device=('cuda' if torch.cuda.is_available() else None),
        )
        output_ids = model.generate(
            **input_ids,
            max_length=tokenizer.model_max_length,
        )
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)[len(prompt) :]
        answers.append(answer)
    return answers


def evaluate_harmfulness(
    user_texts: list[str],
    assistant_texts: list[str],
    setup: str =  "redteam-osst",
) -> list[float]:
    """Evaluate the harmfulness of the outputs."""
    from HarmfulnessRewardModel import AbsoluteHarmfulnessPredictor

    harmfulness_predictor = AbsoluteHarmfulnessPredictor(setup, device="cuda:0")
    harmfulness_scores = harmfulness_predictor.predict(user_texts, assistant_texts)
    return np.array(harmfulness_scores)


def main() -> None:
    """The main function."""
    args = parse_arguments()

    # get files ending in json of args.problem_path
    files = [os.path.join(args.problem_path, f) for f in os.listdir(args.problem_path) if f.endswith('.json')]
    for file in files:

        # Open file with instructions
        with open(file, encoding='utf-8') as f:
            problems = json.load(f)
        print(f'Generating answers for {file}')

        # Generate answers
        answer = generate_answer(problems['instructions'], args.model)
        print('Generating answers finished.')
        results = {
                'instructions': problems['instructions'],
                'outputs': answer,
            }
        # # Save answers
        # answer_out_dir = f'outputs/{args.model.split('/')[-1]}/answers/'
        # if not os.path.exists(answer_out_dir):
        #     os.makedirs(answer_out_dir)
        # answer_output_file = file.split('/')[-1].split('.')[0]
        # answer_output_file = os.path.join(answer_out_dir, f'{answer_output_file}_answers.json')
        # with open(answer_output_file, mode='w', encoding='utf-8') as f:
        #     json.dump(results, f, indent=4, ensure_ascii=False)
        # print(f'Saved answers to {answer_out_dir}.')


        # Evaluate harmfulness
        print('Evaluating harmfulness...')
        harmfulness_scores = evaluate_harmfulness(
            problems['instructions'],
            answer,
        )
        print('Evaluating harmfulness finished.')
        # # Save harmfulness scores
        # scores_out_dir = f'outputs/{args.model.split("/")[-1]}/scores/'
        # if not os.path.exists(scores_out_dir):
        #     os.makedirs(scores_out_dir)
        # scores_output_file = file.split('/')[-1].split('.')[0]
        # scores_output_file = os.path.join(scores_out_dir, f'{scores_output_file}_scores.npy')
        # np.save(scores_output_file, harmfulness_scores)
        # print(f'Saved harmfulness scores to {scores_output_file}.')
        
        dict_out = {
            'mean_harmfulness_scores': np.mean(harmfulness_scores),
            'instructions': problems['instructions'],
            'outputs': answer,
            'harmfulness_scores': harmfulness_scores.tolist(),
        }
        # Save results
        result_out_dir = f'outputs/{args.model.split("/")[-1]}/'
        if not os.path.exists(result_out_dir):
            os.makedirs(result_out_dir)
        result_output_file = file.split('/')[-1].split('.')[0]
        result_output_file = os.path.join(result_out_dir, f'{result_output_file}_results.json')
        with open(result_output_file, mode='w', encoding='utf-8') as f:
            json.dump(dict_out, f, indent=4, ensure_ascii=False)
        print(f'Saved results to {result_out_dir}.')


if __name__ == '__main__':
    main()