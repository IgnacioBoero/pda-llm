
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
    # print(f'Generating answers with {model_name_or_path}')
    for problem in tqdm(problems):
        prompt = PROMPT_INPUT.format(input=problem)
        input_ids = to_device(
            tokenizer(prompt, return_tensors='pt'),
            device=('cuda' if torch.cuda.is_available() else None),
        )
        output_ids = model.generate(
            **input_ids,
            max_length=512,
        )
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)[len(prompt) :]
        answers.append(answer)
    return answers



def main() -> None:
    """The main function."""
    args = parse_arguments()

    # get files ending in json of args.problem_path
    files = [os.path.join(args.problem_path, f) for f in os.listdir(args.problem_path) if f.endswith('.json')]
    for file in files:
        with open(file, encoding='utf-8') as f:
            problems = json.load(f)
        print(f'Generating answers for {file}')

        answer = generate_answer(problems['instructions'], args.model)

        print('Generating answers finished.')

        results = {
                'instructions': problems['instructions'],
                'outputs': answer,
            }

        out_dir = 'outputs/{}/'.format(args.model)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        output_file = file.split('/')[-1].split('.')[0]
        output_file = os.path.join(out_dir, f'{output_file}_eval.json')
        with open(output_file, mode='w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()