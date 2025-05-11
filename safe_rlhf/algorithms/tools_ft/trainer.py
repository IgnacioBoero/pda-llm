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
import os
from typing import Any

import deepspeed
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from safe_rlhf.datasets import SupervisedToolsDataset
from safe_rlhf.models import load_pretrained_models
from safe_rlhf.trainers import SupervisedToolsTrainer
from safe_rlhf.utils import gather_log_probabilities, get_all_reduce_mean


class ToolsSFTTrainer(SupervisedToolsTrainer):
    TRAINING_TYPE = 'tools-sft'
    DATASET_TYPE = SupervisedToolsDataset

    model: deepspeed.DeepSpeedEngine
    reference_model: deepspeed.DeepSpeedEngine

    ds_train_config: dict[str, Any]
    ds_eval_config: dict[str, Any]

    def __init__(
        self,
        args: argparse.Namespace,
        ds_train_config: dict[str, Any],
        ds_eval_config: dict[str, Any],
    ) -> None:
        """Initialize trainer."""
        self.args = args
        self.ds_train_config = ds_train_config
        self.ds_eval_config = ds_eval_config
        super().__init__(args, ds_train_config)

    @staticmethod
    def compute_log_probs(
        model: AutoModelForCausalLM,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Compute log probabilities of given sequences."""
        logits = model(input_ids, attention_mask=attention_mask).logits
        return gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])

    def loss(  # pylint: disable=too-many-locals
        self,
        input_ids: torch.LongTensor,  # size = (B, L)
        attention_mask: torch.BoolTensor,  # size = (B, L)
        labels: torch.LongTensor,  # size = (B, L)
        important: torch.LongTensor,  # size = (B)
        ref_log_probs: torch.Tensor,  # size = (B, L)
        response_mask: torch.BoolTensor,  # size = (B, L)
    ) -> dict[str, torch.Tensor]:
        """Loss function for the pdalignment algorithm.

        Args:
            better_input_ids (torch.LongTensor): The input ids of the better answer.
            better_attention_mask (torch.BoolTensor): The attention mask of the better answer.
            worse_input_ids (torch.LongTensor): The input ids of the worse answer.
            worse_attention_mask (torch.BoolTensor): The attention mask of the worse answer.

        Returns:
            dict[str, torch.Tensor]: loss, reward, better sample reward, worse sample reward
        """
        logs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        loss = logs.loss
        if self.args.important_sft:
            sequence_log_probs = gather_log_probabilities(
                        logs.logits[:, :-1], input_ids[:, 1:]
                    )
            log_probs = (sequence_log_probs * attention_mask[:, 1:]* response_mask[:,1:]).sum(dim=1)
            log_ratio = log_probs - ref_log_probs
            loss_safety = (
                self.args.resilient_coeff
                / 2
                * torch.clamp(-log_ratio + self.args.safety_ratio_tol, 0, None) ** 2
            )  * important
            loss = loss + loss_safety.sum()

        # Total loss
        loss = loss.mean()

        return {
            'loss': loss,
            'important': important,
        }

    def train_step(
        self,
        input_ids: torch.LongTensor,  # size = (B, L)
        attention_mask: torch.BoolTensor,  # size = (B, L)
        labels: torch.LongTensor,  # size = (B, L)
        important: torch.LongTensor,  # size = (B, L)
        index: torch.LongTensor,  # size = (B, L)
        response_mask: torch.BoolTensor,  # size = (B, L)
    ) -> dict[str, Any]:
        """Perform a single training step.

        Args:
            better_input_ids (torch.LongTensor): The input ids of the better answer.
            better_attention_mask (torch.BoolTensor): The attention mask of the better answer.
            worse_input_ids (torch.LongTensor): The input ids of the worse answer.
            worse_attention_mask (torch.BoolTensor): The attention mask of the worse answer.
            better_safe (torch.BoolTensor): The safety of the better answer.
            worse_safe (torch.BoolTensor): The safety of the worse answer.
            index (torch.LongTensor): The index of the batch.
        Returns:
            dict[str, Any]: training loss, reward, learning rate
        """
        batch_ref_sequence_log_probs = self.baseline_logprobs[index]
        loss_dict = self.loss(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            important=important,
            ref_log_probs=batch_ref_sequence_log_probs,
            response_mask=response_mask,
        )
        loss = loss_dict['loss']
        self.model.backward(loss)
        self.model.step()

        loss = get_all_reduce_mean(loss)

        return {
            'train/loss': loss.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
        }
