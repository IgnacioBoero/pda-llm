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
    'SupervisedToolsDataset',
    'SupervisedToolsCollator',
    'SupervisedToolsSample',
    'SupervisedToolsBatch',
]


class SupervisedToolsSample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)
    labels: torch.LongTensor  # size = (L,)
    important: bool  # size = (1,)


class SupervisedToolsBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    labels: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)


class SupervisedToolsDataset(TokenizedDataset):
    def preprocess(self, raw_sample: RawSample) -> SupervisedToolsSample:

        messages = [
            {'role': 'system', 'content': raw_sample['system']},
        ]
        for m in raw_sample['dialogue']:
            messages.append({'role': m['from'], 'content': m['value']})
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt")[0]
        if (
            input_ids[-1] != self.tokenizer.eos_token_id
            and len(input_ids) < self.tokenizer.model_max_length
        ):
            input_ids = torch.cat([
                input_ids,
                torch.tensor([self.tokenizer.eos_token_id], dtype=input_ids.dtype, device=input_ids.device)
            ])
        labels = input_ids.clone()
        labels[: len(self.tokenizer.apply_chat_template([{'role': 'system', 'content': raw_sample['system']}]))] = IGNORE_INDEX
        return {'input_ids': input_ids, 'labels': labels, 'important': raw_sample['is_important']}

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return SupervisedToolsCollator(self.tokenizer.pad_token_id)

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


class SupervisedToolsCollator(CollatorBase):
    def __call__(self, samples: list[SupervisedToolsSample]) -> SupervisedToolsBatch:
        input_ids = right_padding(
            [sample['input_ids'] for sample in samples],
            padding_value=self.pad_token_id,
        )
        labels = right_padding(
            [sample['labels'] for sample in samples],
            padding_value=IGNORE_INDEX,
        )
        attention_mask = input_ids.ne(self.pad_token_id)
        important = torch.tensor(
            [sample['important'] for sample in samples],
            dtype=torch.bool,
        )
        index_list = [s['index'] for s in samples]
        indexes = torch.tensor(index_list, dtype=torch.long)

        return {
            'input_ids': input_ids,  # size = (B, L)
            'labels': labels,  # size = (B, L)
            'attention_mask': attention_mask,  # size = (B, L)
            'important': important,  # size = (B,)
            'index': indexes,
        }
