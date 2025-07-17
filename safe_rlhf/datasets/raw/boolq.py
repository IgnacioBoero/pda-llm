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


__all__ = ['BoolQDataset', 'BoolQTrainDataset', 'BoolQValDataset']


class BoolQDataset(RawDataset):
    SPLIT: ClassVar[str]
    PATH: ClassVar[str]
    NAME: str = 'boolq'
    ALIASES: tuple[str, ...] = ('boolq-google',)

    def __init__(self, path: str | None = None) -> None:

        self.data = load_dataset(path or self.PATH, split=self.SPLIT)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        answer = " Yes" if data["answer"] else " No"
        input = data['question']
        context = data['passage']
        is_safe = data["answer"]

        return RawSample(input=input, answer=answer, context=context, is_safe=is_safe)

    def __len__(self) -> int:
        return len(self.data)


class BoolQTrainDataset(BoolQDataset):
    NAME: str = 'boolq/train'
    PATH: str = 'google/boolq'
    SPLIT: str = 'train'


class BoolQValDataset(BoolQDataset):
    NAME: str = 'boolq/val'
    PATH: str = 'google/boolq'
    SPLIT: str = 'validation'