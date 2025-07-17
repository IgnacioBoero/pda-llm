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
"""Trainer base class for supervised training."""

from __future__ import annotations
import wandb
import abc
import argparse
import os
from typing import Any, ClassVar

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, get_scheduler
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from safe_rlhf.configs import ADAM_BETAS
from safe_rlhf.datasets import TokenizedDataset
from safe_rlhf.models import load_pretrained_models
from safe_rlhf.trainers.base import TrainerBase
from safe_rlhf.utils import get_optimizer_grouped_parameters, is_main_process, to_device


class SupervisedBoolQTrainer(TrainerBase):
    """Trainer base class for supervised training.

    Abstract methods:
        loss: Compute supervised training loss.
        train_step: Perform a single training step.
    """

    TRAINING_TYPE: ClassVar[str] = 'supervised'
    DATASET_TYPE: ClassVar[type[TokenizedDataset]]
    MODEL_TYPE = AutoModelForCausalLM

    model: deepspeed.DeepSpeedEngine
    ds_config: dict[str, Any]

    extra_model_kwargs: dict[str, Any] | None = None
    extra_tokenizer_kwargs: dict[str, Any] | None = None

    def __init__(self, args: argparse.Namespace, ds_config: dict[str, Any]) -> None:
        """Initialize trainer."""
        self.args = args
        self.ds_config = ds_config
        self.global_step = 0

        self.init_models()
        dist.barrier()
        self.init_datasets()
        dist.barrier()

        self.init_engines()
        dist.barrier()
    
        self.init_duals()
        dist.barrier()


        self.init_logger()

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if self.ds_config is not None and self.ds_config['zero_optimization']['stage'] == 3:
            self.dstchf = HfDeepSpeedConfig(self.ds_config)

        self.model, self.tokenizer = load_pretrained_models(
            self.args.model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='right',
            auto_model_type=self.MODEL_TYPE,
            trust_remote_code=self.args.trust_remote_code,
            auto_model_kwargs=self.extra_model_kwargs,
            auto_tokenizer_kwargs=self.extra_tokenizer_kwargs,
        )
        self.model = get_peft_model(
            self.model,
            LoraConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "down_proj",
                    "up_proj",
                    "lm_head",
                ],
            ),
        )

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        train_dataset = self.DATASET_TYPE(
            self.args.train_datasets,
            tokenizer=self.tokenizer,
            lazy_tokenization=False,
        )
        
        if self.args.need_eval:
            if self.args.eval_datasets is None and self.args.eval_split_ratio is not None:
                train_dataset, eval_dataset = train_dataset.split_train_test(
                    self.args.eval_split_ratio,
                    seed=42,
                    stratify_key="safe"
                )
            elif self.args.eval_datasets is not None and self.args.eval_split_ratio is None:
                eval_dataset = self.DATASET_TYPE(
                    self.args.eval_datasets,
                    tokenizer=self.tokenizer,
                )
            else:
                raise ValueError('Either `eval_datasets` or `eval_split_ratio` should be provided.')

            self.eval_dataloader = DataLoader(
                eval_dataset,
                collate_fn=eval_dataset.get_collator(),
                sampler=DistributedSampler(eval_dataset, shuffle=True),
                batch_size=self.args.per_device_eval_batch_size,
            )
        else:
            self.eval_dataloader = None
        
        self.train_dataloader = DataLoader(
            train_dataset,
            collate_fn=train_dataset.get_collator(),
            sampler=DistributedSampler(train_dataset, shuffle=True),
            batch_size=self.args.per_device_train_batch_size,
        )


    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        self.args.num_update_steps_per_epoch = (
            len(self.train_dataloader) + self.args.gradient_accumulation_steps - 1
        ) // self.args.gradient_accumulation_steps
        self.args.total_training_steps = self.args.epochs * self.args.num_update_steps_per_epoch

        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            self.model,
            self.args.weight_decay,
        )
        if (
            self.ds_config['zero_optimization'].get('offload_optimizer', {}).get('device', 'none')
            != 'none'
        ):
            optimizer = DeepSpeedCPUAdam(
                optimizer_grouped_parameters,
                lr=self.args.lr,
                betas=ADAM_BETAS,
            )
        else:
            optimizer = FusedAdam(
                optimizer_grouped_parameters,
                lr=self.args.lr,
                betas=ADAM_BETAS,
            )

        num_warmup_steps = int(self.args.lr_warmup_ratio * self.args.total_training_steps)
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.args.total_training_steps,
        )

        self.model, *_ = deepspeed.initialize(
            model=self.model,
            optimizer=optimizer,
            args=self.args,
            config=self.ds_config,
            lr_scheduler=lr_scheduler,
            dist_init_required=True,
        )

        # if self.args.gradient_checkpointing:
        #     self.model.gradient_checkpointing_enable()
    def init_duals(self) -> None:
        """Initialize dual variables for safety constraints."""
        num_samples = len(self.train_dataloader.dataset)
        self.dual_vars = torch.zeros(num_samples, device=self.args.device, requires_grad=False)

    @abc.abstractmethod
    def loss(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Compute supervised training loss."""
        raise NotImplementedError

    @abc.abstractmethod
    def train_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Perform a single training step."""
        raise NotImplementedError

    def train(self) -> None:
        """Train the model."""
        
        if self.args.need_eval:
            self.logger.print('\n***** Evaluating at the beginning *****')
            self.logger.log(self.eval(), step=0)
            

        self.logger.print('***** Running training *****')

        progress_bar = tqdm(
            total=self.args.epochs * len(self.train_dataloader),
            desc=f'Training 1/{self.args.epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process(),
        )

        for epoch in range(self.args.epochs):
            self.model.train()

            for batch in self.train_dataloader:
                info = self.train_step(**to_device(batch, self.args.device))
                torch.cuda.empty_cache()

                self.global_step += 1
                progress_bar.set_description(
                    f'Training {epoch + 1}/{self.args.epochs} epoch '
                    f'(loss {info["train/loss"]:.4f})',
                )
                progress_bar.update(1)

                info['train/epoch'] = self.global_step / len(self.train_dataloader)
                self.logger.log(info, step=self.global_step)
                
            if self.args.need_eval and self.args.eval_strategy == 'epoch':
                self.logger.print(
                    f'\n***** Evaluating at epoch {epoch + 1}/{self.args.epochs} *****',
                )
                self.logger.log(self.eval(), step=self.global_step)

            self.model.tput_timer.update_epoch_count()

    def eval(self):
        constraint_slacks = []
        loss = []
        table = wandb.Table(columns=["data_index"] + ["value"])
        self.model.eval()


        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = to_device(batch, self.args.device)
                index = batch['index']
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                is_true = batch['is_true']


                loss_dict = self.loss(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    index=index,
                    is_true=is_true,
                )
                loss.append(loss_dict['objective'])

                constraint_slack = loss_dict['constraint_value']
                constraint_slacks.append(constraint_slack)
                table.add_data(index, constraint_slack.cpu().to(torch.float16))
        constraint_slacks = torch.stack(constraint_slacks, dim=0) if len(constraint_slacks) > 0 else torch.tensor([0.0])

        # Evaluate on train dataloader
        train_constraint_slacks = []
        train_loss = []
        train_table = wandb.Table(columns=["data_index"] + ["value"])
        dual_var_table = wandb.Table(columns=["data_index"] + ["value"])

        with torch.no_grad():
            for batch in self.train_dataloader:
                batch = to_device(batch, self.args.device)
                index = batch['index']
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']


                loss_dict = self.loss(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    index=index,
                    is_true=is_true,
                )
                train_loss.append(loss_dict['objective'])
                

                train_constraint_slack = loss_dict['constraint_value']
                train_constraint_slacks.append(train_constraint_slack)
                train_table.add_data(index, train_constraint_slack.cpu().to(torch.float16))
                dual_var_table.add_data(
                    index,
                    self.dual_vars[index].cpu().to(torch.float16),
                )
                
        train_constraint_slacks = torch.stack(train_constraint_slacks, dim=0) if len(train_constraint_slacks) > 0 else torch.tensor([0.0])

        return {
            'eval/mean_constraint_slack': constraint_slacks.mean().item() if len(constraint_slacks) > 0 else 0.0,
            'eval/min_constraint_slack': constraint_slacks.min().item() if len(constraint_slacks) > 0 else 0.0,
            'eval/max_constraint_slack': constraint_slacks.max().item() if len(constraint_slacks) > 0 else 0.0,
            'eval/hist_constraint_slack': wandb.Histogram(constraint_slacks.cpu().to(torch.float16)),
            'eval/table': table,
            'eval/obj': torch.stack(loss).mean().item() if len(loss) > 0 else 0.0,
            'train/mean_constraint_slack': train_constraint_slacks.mean().item() if len(train_constraint_slacks) > 0 else 0.0,
            'train/min_constraint_slack': train_constraint_slacks.min().item() if len(train_constraint_slacks) > 0 else 0.0,
            'train/max_constraint_slack': train_constraint_slacks.max().item() if len(train_constraint_slacks) > 0 else 0.0,
            'train/hist_constraint_slack': wandb.Histogram(train_constraint_slacks.cpu().to(torch.float16)),
            'train/table': train_table,
            'train/obj': torch.stack(train_loss).mean().item() if len(train_loss) > 0 else 0.0,
            'train/dual_vars': dual_var_table,
        }

    def set_train(self, mode: bool = True) -> None:
        """Set training mode for model."""
        if mode:
            self.model.train()
        else:
            self.model.eval()

