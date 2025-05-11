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


def build_assistant_mask(input_ids,
                         bos_ass=(128006, 78191, 128007),
                         eot=128009):
    """
    Build a 0/1 mask that marks assistant-message content (plus the EOT)
    in a Llama-3.1 token sequence whose chat template lacks {% generation %}.

    Parameters
    ----------
    input_ids : list[int]              flat token sequence
    bos_ass   : tuple[int,int,int]     tokens that start an assistant header
    eot       : int                    token that ends a turn

    Returns
    -------
    list[int] – mask of equal length (1 = assistant token, 0 = other)
    """
    ids  = list(input_ids)           # make sure we can slice
    mask = [0] * len(ids)
    i, b = 0, len(bos_ass)

    while i <= len(ids) - b:
        # Detect the assistant-header triple
        if tuple(ids[i:i+b]) == bos_ass:
            start = i + b
            try:                     # look for the next end-of-turn
                end = ids.index(eot, start)
                last = end + 1       # include the eot itself
            except ValueError:       # truncated sequence
                last = len(ids)

            for j in range(start, last):
                mask[j] = 1
            i = last                 # resume scanning after this span
        else:
            i += 1

    return torch.tensor(mask)
class SupervisedToolsDataset(TokenizedDataset):
    def preprocess(self, raw_sample: RawSample) -> SupervisedToolsSample:

        messages = [
            {'role': 'system', 'content': raw_sample['system']},
        ]
        for m in raw_sample['dialogue']:
            messages.append({'role': m['from'], 'content': m['value']})
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length,   
        )[0]
        labels = input_ids.clone()
    
        response_mask = build_assistant_mask(input_ids)
        labels[response_mask == 0] = IGNORE_INDEX
        
        return {'input_ids': input_ids, 'labels': labels, 'important': raw_sample['is_important'], 'response_mask': response_mask}

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
        response_mask = right_padding([s['response_mask'] for s in samples], padding_value=1)

        return {
            'input_ids': input_ids,  # size = (B, L)
            'labels': labels,  # size = (B, L)
            'attention_mask': attention_mask,  # size = (B, L)
            'important': important,  # size = (B,)
            'index': indexes,
            'response_mask': response_mask
        }
