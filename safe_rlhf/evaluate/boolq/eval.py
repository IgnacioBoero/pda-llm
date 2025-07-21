
from tqdm import tqdm
import argparse
import json
import os
import time
from safe_rlhf.configs.constants import PROMPT_INPUT
from safe_rlhf.models import load_pretrained_models
from safe_rlhf.utils import to_device

from peft import AutoPeftModelForCausalLM

import numpy as np
import openai    
from pathlib import Path
from safe_rlhf.datasets import SupervisedBoolQDataset

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
    
    # Output path
    parser.add_argument(
        '--output_path',
        type=Path,
        default= BASE_DIR / 'outputs' / 'default',
        help='The path to the output file.',
    )


    return parser.parse_args()


def generate_answer(model_name_or_path: str, val=True) -> list[str]:
    model, tokenizer = load_pretrained_models(
        model_name_or_path,
        auto_device_mapping=True,
        auto_model_type = AutoPeftModelForCausalLM,
        trust_remote_code=True,
        model_max_length = 2048,  # Set a max length to avoid issues with long inputs
    )
    if val:
        ds ="boolq/val"
    else:
        ds = "boolq/train"
    train_dataset = SupervisedBoolQDataset(
        [
            (ds, 1.0),

            # If you have multiple, you can specify [("Foo", 0.6), ("Bar", 0.4), ...]
        ],
        tokenizer=tokenizer,
        lazy_tokenization=False,
        seed=42,
    )
    answers = []
    for sample in tqdm(train_dataset):

        input_ids = sample['input_ids'][:-2].to(model.device)
        output_ids = model.generate(
            input_ids.unsqueeze(0),
            max_new_tokens=2,
        )
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Find Answer: in the string answer and return the text after as answer and before as prompt
        response = response.split("Answer:")
        try:
            answer = response[1]
            prompt = response[0]
        except:
            breakpoint()
        
        real_answer = ' yes' if sample['is_true'] else ' no'
        if answer.startswith(real_answer):
            correct = True
        else:
            correct = False
        if answer == real_answer:
            correct_and_finished = True
        else:
            correct_and_finished = False
        answers.append({ 'prompt': prompt, 'answer':answer, 'real_answer':real_answer, 'correct': correct, 'correct_and_finished': correct_and_finished })
    avg_correct = np.mean([a['correct'] for a in answers])
    avg_correct_and_finished = np.mean([a['correct_and_finished'] for a in answers])
    return {'avg_correct': avg_correct, 'avg_correct_and_finished': avg_correct_and_finished, 'answers': answers}




def main() -> None:
    """The main function."""
    args = parse_arguments()
    result_out_dir = args.output_path
    result_out_dir.mkdir(parents=True, exist_ok=True)
    
    
    
    out_dict_val = generate_answer(args.model, val=True)
    result_output_file_val = result_out_dir / f"answers_val.json"
    with open(result_output_file_val, "w", encoding="utf-8") as f_out:
        json.dump(
            out_dict_val,
            f_out,
            indent=4,
            ensure_ascii=False,
        )

    out_dict_train = generate_answer(args.model, val=False)
    result_output_file_train = result_out_dir / f"answers_train.json"
    with open(result_output_file_train, "w", encoding="utf-8") as f_out:
        json.dump(
            out_dict_train,
            f_out,
            indent=4,
            ensure_ascii=False,
        )
        

if __name__ == '__main__':
    main()