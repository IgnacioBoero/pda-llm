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

from typing import Callable
from typing_extensions import TypedDict  # Python 3.10+

import torch

from safe_rlhf.configs import IGNORE_INDEX, PROMPT_ASSISTANT, PROMPT_BEGIN, PROMPT_USER
from safe_rlhf.datasets.base import CollatorBase, RawSample, TokenizedDataset
from safe_rlhf.datasets.utils import format_prompt, right_padding


__all__ = [
    'MaskedBiasDataset',
    'MaskedBiasCollator',
    'MaskedBiasSample',
    'MaskedBiasBatch',
]


class MaskedBiasSample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)
    labels: torch.LongTensor  # size = (L,)
    safe: bool  # size = (1,)


class MaskedBiasBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    labels: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)


class MaskedBiasDataset(TokenizedDataset):
    def preprocess(self, raw_sample: RawSample) -> MaskedBiasSample:
        MASK = self.tokenizer.mask_token
        input = raw_sample['input'].replace("_", MASK)
        if not isinstance(input, str):
            raise ValueError(f'Unsupported type of `input`: {type(input)}. Expected: str.')

        input_ids = self.tokenizer(input, truncation=True, max_length=128)["input_ids"]
        mask_pos = input_ids.index(self.tokenizer.mask_token_id)
        labels0 = self.tokenizer.convert_tokens_to_ids(raw_sample["answer"])
        labels1 = self.tokenizer.convert_tokens_to_ids(raw_sample["other_answer"])
        if labels0 == self.tokenizer.unk_token_id:
            print(f'Warning: `answer` {raw_sample["answer"]} is not in the tokenizer vocabulary. Using {self.tokenizer.unk_token} instead.')
        if labels1 == self.tokenizer.unk_token_id:
            print(f'Warning: `other_answer` {raw_sample["other_answer"]} is not in the tokenizer vocabulary. Using {self.tokenizer.unk_token} instead.')
        correct = raw_sample["correct"]
        if correct == raw_sample["answer"]:
            label_correct = labels0
            label_incorrect = labels1
        elif correct == raw_sample["other_answer"]:
            label_correct = labels1
            label_incorrect = labels0
        else:
            raise ValueError(f'Invalid `correct` value: {correct}. Expected one of: {raw_sample["answer"]}, {raw_sample["other_answer"]}.')
        return {'input_ids': input_ids, 'label_correct': label_correct, 'label_incorrect':label_incorrect, 'mask_pos':mask_pos, 'is_bias': raw_sample['is_safe']}

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return MaskedBiasCollator(self.tokenizer.pad_token_id)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a tokenized data sample by index."""
        data = self.data[index]
        if data is self._SENTINEL:
            raw_sample = self.rawdata[index]
            data = self.preprocess(raw_sample)
            self.data[index] = data
        # Add the index to the data dictionary
        data['index'] = index
        return data


class MaskedBiasCollator(CollatorBase):
    def __call__(self, samples: list[MaskedBiasSample]) -> MaskedBiasBatch:
        input_ids = right_padding(
            [sample['input_ids'] for sample in samples],
            padding_value=self.pad_token_id,
        )
        label_correct = right_padding(
            [sample['label_correct'] for sample in samples],
            padding_value=IGNORE_INDEX,
        )
        label_incorrect = right_padding(
            [sample['label_incorrect'] for sample in samples],
            padding_value=IGNORE_INDEX,
        )
        is_bias = torch.tensor(
            [sample['is_bias'] for sample in samples],
            dtype=torch.bool,
        )
        mask_pos = torch.tensor(
            [sample['mask_pos'] for sample in samples],
            dtype=torch.long,
        )
        
        index_list = [s['index'] for s in samples]
        indexes = torch.tensor(index_list, dtype=torch.long)

        return {
            'input_ids': input_ids,  # size = (B, L)
            'label_correct': label_correct,  # size = (B, L)
            'label_incorrect': label_incorrect,  # size = (B, L)
            'mask_pos': mask_pos,  # size = (B,)
            'is_bias': is_bias,  # size = (B,)
            'index': indexes,
        }
