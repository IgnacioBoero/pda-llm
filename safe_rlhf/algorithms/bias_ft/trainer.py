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

from safe_rlhf.datasets import MaskedBiasDataset
from safe_rlhf.models import load_pretrained_models
from safe_rlhf.trainers import MaskedBiasTrainer
from safe_rlhf.utils import gather_log_probabilities, get_all_reduce_mean


class BiasTrainer(MaskedBiasTrainer):
    TRAINING_TYPE = 'bias-ft'
    DATASET_TYPE = MaskedBiasDataset

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
        mask_pos: torch.BoolTensor,  # size = (B, L)
        labels_correct: torch.LongTensor,  # size = (B, L)
        labels_incorrect: torch.LongTensor,  # size = (B, L)
        index: torch.LongTensor,  # size = (B,)
        is_bias: torch.Tensor,  # size = (B,)
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
        is_not_bias = ~is_bias

        logits = self.model(
                input_ids=input_ids,
            ).logits

        bsz, _, vocab = logits.shape
        
        mask_logits = logits[torch.arange(bsz), mask_pos]
        logprobs = F.log_softmax_probs(mask_logits, dim=-1)
        pc = logprobs[torch.arange(bsz), labels_correct]
        pi = logprobs[torch.arange(bsz), labels_incorrect]
        
        
        log_probs = pc - pi
        slack = log_probs * is_bias.float()
        objective = - log_probs * is_not_bias.float()

        if self.args.algorithm == "l2":
            loss = (
                self.args.resilient_coeff
                / 2
                * slack ** 2
            )  + objective
            
        elif self.args.algorithm == "l1":
            loss = (
                self.args.resilient_coeff
                / 2
                * slack.abs()
            ) + objective

        elif self.args.algorithm == "dual":
            # Update duals before computing the loss
            self.update_duals(
                index=index,
                slack=slack,
            )
            batch_duals = self.dual_vars[index].detach()
            loss = (
                batch_duals
                * (slack)
            ) + objective
            
        # elif self.args.algorithm == "penalty":
        #     loss = (
        #         self.args.resilient_coeff
        #         * (slack)
        #     ) + objective

        # Total loss
        loss = loss.sum()

        return {
            'loss': loss,
            'objective': objective,
            'constraint_value': slack,
            }

    def update_duals(self,index,slack):
        self.dual_vars[index] = self.dual_vars[index] + (1 / self.args.resilient_coeff) * slack,
    

    def train_step(
        self,
        input_ids: torch.LongTensor,  # size = (B, L)
        mask_pos: torch.BoolTensor,  # size = (B, L)
        label_correct: torch.LongTensor,  # size = (B, L)
        label_incorrect: torch.LongTensor,  # size = (B, L)
        index: torch.LongTensor,  # size = (B, L)
        is_bias: torch.Tensor,  # size = (B, L)
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
        

        loss_dict = self.loss(
            input_ids=input_ids,
            mask_pos=mask_pos,
            label_correct=label_correct,
            label_incorrect=label_incorrect,
            index=index,
            is_bias=is_bias,
        )
        loss = loss_dict['loss']
        self.model.backward(loss)
        self.model.step()

        loss = get_all_reduce_mean(loss)

        return {
            'train/loss': loss.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
        }
