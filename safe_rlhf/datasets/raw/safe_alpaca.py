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


__all__ = ['AlpacaDataset']


class SafeAlpacaDataset(RawDataset):
    SPLIT: ClassVar[str]
    PATH: ClassVar[str]

    NAME: str = 'safe-alpaca'
    ALIASES: tuple[str, ...] = ('safe-stanford-alpaca',)

    def __init__(self, path: str | None = None) -> None:

        self.data = load_dataset(path or self.PATH, split=self.SPLIT)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        input = (  # pylint: disable=redefined-builtin
            ' '.join((data['instruction'], data['input'])) if data['input'] else data['instruction']
        )
        answer = data['output']
        is_safe = data['safety_label']
        return RawSample(input=input, answer=answer, is_safe=is_safe)

    def __len__(self) -> int:
        return len(self.data)


class SafeAlpaca100Dataset(SafeAlpacaDataset):
    NAME: str = 'SAFE-ALPACA/100'
    PATH: str = 'iboero16/SAFE-ALPACA'
    SPLIT: str = 'safe_alpaca_100'


class SafeAlpaca300Dataset(SafeAlpacaDataset):
    NAME: str = 'SAFE-ALPACA/300'
    PATH: str = 'iboero16/SAFE-ALPACA'
    SPLIT: str = 'safe_alpaca_300'


class SafeAlpaca500Dataset(SafeAlpacaDataset):
    NAME: str = 'SAFE-ALPACA/500'
    PATH: str = 'iboero16/SAFE-ALPACA'
    SPLIT: str = 'safe_alpaca_500'


class SafeAlpaca1000Dataset(SafeAlpacaDataset):
    NAME: str = 'SAFE-ALPACA/1000'
    PATH: str = 'iboero16/SAFE-ALPACA'
    SPLIT: str = 'safe_alpaca_1000'

class SafeAlpaca1500Dataset(SafeAlpacaDataset):
    NAME: str = 'SAFE-ALPACA/1500'
    PATH: str = 'iboero16/SAFE-ALPACA'
    SPLIT: str = 'safe_alpaca_1500'

class SafeAlpaca2000Dataset(SafeAlpacaDataset):
    NAME: str = 'SAFE-ALPACA/2000'
    PATH: str = 'iboero16/SAFE-ALPACA'
    SPLIT: str = 'safe_alpaca_2000'
