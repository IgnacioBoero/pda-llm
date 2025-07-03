
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
import openai    
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent   


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
        type=Path,
        default= BASE_DIR / 'data',
        help='The path to the input file.',
    )


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



def evaluate_openai_harm(
    texts: list[str],
    model: str = "omni-moderation-latest",
    sleep_sec: float = 0.5,
) -> np.ndarray:
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise RuntimeError("Set the OPENAI_API_KEY environment variable first.")

    client = openai.OpenAI(api_key=api_key)
    harm_scores: list[float] = []

    for txt in tqdm(texts, desc="OpenAI moderation"):
        try:
            resp = client.moderations.create(input=txt, model=model)
            scores: dict[str, float] = resp.results[0].category_scores
            scores_dict = scores.model_dump(exclude_none=True)   # → ordinary dict
            harm_scores.append(max(scores_dict.values()))
        except Exception as err:
            # In case of failure, fall back to the most conservative score
            print(f"[moderation-error] {err!r} — setting score to 1.0")
            harm_scores.append(1.0)
        time.sleep(sleep_sec)       # simple rate-limit buffer

    return np.array(harm_scores)


def main() -> None:
    """The main function."""
    args = parse_arguments()

    # list all .json problem files
    files = [os.path.join(args.problem_path, f) 
             for f in os.listdir(args.problem_path) if f.endswith('.json')]

    for file in files:
        with open(file, encoding="utf-8") as f:
            problems = json.load(f)
        print(f"Generating answers for {file}")

        # ---------- 1) GENERATE & SAVE IMMEDIATELY ----------
        answer = generate_answer(problems["instructions"], args.model)
        print("Generating answers finished.")

        # build output path once so we can reuse it
        result_out_dir = Path(BASE_DIR) / "outputs" / args.model.split("/")[-1]
        result_out_dir.mkdir(parents=True, exist_ok=True)
        result_output_file = result_out_dir / f"{Path(file).stem}_results.json"

        # save answers only
        with open(result_output_file, "w", encoding="utf-8") as f_out:
            json.dump(
                {
                    "instructions": problems["instructions"],
                    "outputs": answer,
                },
                f_out,
                indent=4,
                ensure_ascii=False,
            )
        print(f"Saved answers to {result_output_file}.")

        # ---------- 2) SCORE ----------
        print("Evaluating harmfulness…")
        harmfulness_scores = evaluate_harmfulness(
            problems["instructions"], answer
        )
        print("Evaluating harmfulness finished.")

        print("Evaluating OpenAI harm…")
        openai_harm_scores = evaluate_openai_harm(answer)
        print("Evaluating OpenAI harm finished.")

        # ---------- 3) RE-OPEN FILE & AUGMENT ----------
        with open(result_output_file, "r+", encoding="utf-8") as f_out:
            existing = json.load(f_out)
            existing.update(
                {
                    "mean_harmfulness_scores": float(np.mean(harmfulness_scores)),
                    "mean_openai_harm_score": float(np.mean(openai_harm_scores)),
                    "harmfulness_scores": harmfulness_scores.tolist(),
                    "openai_harm_scores": openai_harm_scores.tolist(),
                }
            )
            f_out.seek(0)
            json.dump(existing, f_out, indent=4, ensure_ascii=False)
            f_out.truncate()
        print(f"Augmented {result_output_file} with harm scores.\n")

if __name__ == '__main__':
    main()