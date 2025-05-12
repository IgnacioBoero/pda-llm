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


__all__ = ['ToolsDataset', 'ToolsDatasetOriginal']


class ToolsDataset(RawDataset):
    NAME: str = 'tools'
    PATH: str = 'iboero16/TOOLS-MT'
    SPLIT: str = 'train'
    ALIASES: tuple[str, ...] = ('tools-ACETOOLS',)

    def __init__(self, path: str | None = None) -> None:

        self.data = load_dataset(path or self.PATH, split=self.SPLIT)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        system = data['system']
        dialogue = data['conversations']
        is_important = data['is_multitool']
        return RawSample(system=system, dialogue=dialogue, is_important=is_important)

    def __len__(self) -> int:
        return len(self.data)



class ToolsDatasetOriginal(RawDataset):
    NAME: str = 'tools-original'
    PATH: str = 'Team-ACE/ToolACE'
    SPLIT: str = 'train'
    ALIASES: tuple[str, ...] = ('tools-ACETOOLS-original',)

    def __init__(self, path: str | None = None) -> None:

        self.data = load_dataset(path or self.PATH, split=self.SPLIT)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        system = data['system']
        dialogue = data['conversations']
        is_important = False
        return RawSample(system=system, dialogue=dialogue, is_important=is_important)

    def __len__(self) -> int:
        return len(self.data)