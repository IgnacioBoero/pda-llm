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
"""Stanford Alpaca dataset for supervised instruction fine-tuning."""


from __future__ import annotations

from typing import ClassVar

from datasets import load_dataset
from safe_rlhf.datasets.base import RawDataset, RawSample


__all__ = ['BiasDataset', 'BiasTrainDataset', 'BiasValDataset']


class BiasDataset(RawDataset):
    SPLIT: ClassVar[str]
    PATH: ClassVar[str]
    NAME: str = 'bias'

    def __init__(self, path: str | None = None) -> None:

        self.data = load_dataset(path or self.PATH, split=self.SPLIT)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        input = data['sentence']
        answer = 'Ġ' + data['option1']
        other_answer = 'Ġ' + data['option2']
        is_safe = data["gender_bias"]
        correct = 'Ġ' + data["answer"]

        return RawSample(
            input=input,
            answer=answer,
            other_answer=other_answer,
            is_safe=is_safe,
            correct=correct,
        )

    def __len__(self) -> int:
        return len(self.data)


class BiasTrainDataset(BiasDataset):
    NAME: str = 'bias/train'
    PATH: str = 'iboero16/winogrande_bias'
    SPLIT: str = 'train'


class BiasValDataset(BiasDataset):
    NAME: str = 'bias/val'
    PATH: str = 'iboero16/winogrande_bias'
    SPLIT: str = 'eval'