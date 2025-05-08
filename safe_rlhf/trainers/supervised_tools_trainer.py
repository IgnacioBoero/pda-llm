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


class SupervisedToolsTrainer(TrainerBase):
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
        print("calculating baseline ...")
        if dist.get_rank() == 0:
            self.init_baseline()
        else:
            self.baseline_logprobs = torch.zeros(
                (len(self.train_dataloader.dataset)),
                dtype=self.model.dtype,
            )
            self.baseline_logprobs = to_device(self.baseline_logprobs, self.args.device)
        dist.broadcast(
            self.baseline_logprobs,
            src=0,
        )

        self.init_engines()
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
        )

        # if self.args.need_eval:
        #     if self.args.eval_datasets is None and self.args.eval_split_ratio is not None:
        #         train_dataset, eval_dataset = train_dataset.split_train_test(
        #             split_ratio=self.args.eval_split_ratio,
        #         )
        #     elif self.args.eval_datasets is not None and self.args.eval_split_ratio is None:
        #         eval_dataset = self.DATASET_TYPE(
        #             self.args.eval_datasets,
        #             tokenizer=self.tokenizer,
        #         )
        #     else:
        #         raise ValueError('Either `eval_datasets` or `eval_split_ratio` should be provided.')

        #     self.eval_dataloader = DataLoader(
        #         eval_dataset,
        #         collate_fn=eval_dataset.get_collator(),
        #         sampler=DistributedSampler(eval_dataset, shuffle=True),
        #         batch_size=self.args.per_device_eval_batch_size,
        #     )
        # else:
        #     self.eval_dataloader = None
        
        self.train_dataloader = DataLoader(
            train_dataset,
            collate_fn=train_dataset.get_collator(),
            sampler=DistributedSampler(train_dataset, shuffle=True),
            batch_size=self.args.per_device_train_batch_size,
        )

    def init_baseline(self) -> None:
        """Initialize baseline log probabilities with caching functionality."""

        os.makedirs(self.args.cache_dir, exist_ok=True)
        baseline_cache_path = os.path.join(self.args.cache_dir, "cached_baseline_logprobs.pt")
        if os.path.exists(baseline_cache_path) and not self.args.recompute_baseline:
            print(f"Loading cached baseline logprobs from {baseline_cache_path}")
            self.baseline_logprobs = torch.load(baseline_cache_path, map_location=self.args.device)
            print("Loaded cached baseline logprobs successfully")
        else:
            # Assert only one process is computing baseline logprobs
            if dist.is_initialized() and dist.get_rank() != 0:
                print("Only one process should compute baseline logprobs.")
                dist.barrier()
                return
            print("Computing baseline logprobs...")

            baseline_dataloader = DataLoader(
                self.train_dataloader.dataset,
                collate_fn=self.train_dataloader.dataset.get_collator(),
                batch_size=4,
            )

            self.baseline_logprobs = torch.zeros(
                (len(baseline_dataloader.dataset)),
                dtype=self.model.dtype,
            )
            self.baseline_logprobs = to_device(self.baseline_logprobs, self.args.device)

            reference_model, _ = load_pretrained_models(
                self.args.model_name_or_path,
                model_max_length=self.args.max_length,
                padding_side='right',
                auto_model_type=AutoModelForCausalLM,
                trust_remote_code=self.args.trust_remote_code,
            )
            reference_model.requires_grad_(False)
            reference_model.eval()
            reference_model.to(self.args.device)

            with torch.no_grad():
                for batch in tqdm(baseline_dataloader, desc='Computing baseline logprobs'):
                    batch = to_device(batch, self.args.device)
                    logprobs = (
                        self.compute_log_probs(
                            reference_model,
                            batch["input_ids"],
                            batch["attention_mask"],
                        )
                        * batch["attention_mask"][:, 1:]
                    )

                    self.baseline_logprobs[batch['index']] = logprobs.sum(dim=1)

            # Save computed baseline logprobs
            print(f"Saving computed baseline logprobs to {baseline_cache_path}")
            torch.save(self.baseline_logprobs, baseline_cache_path)
            print("Saved baseline logprobs successfully")

            # Free up memory
            del reference_model
            torch.cuda.empty_cache()
        return

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
        self.logger.print('***** Running training *****')

        progress_bar = tqdm(
            total=self.args.epochs * len(self.train_dataloader),
            desc=f'Training 1/{self.args.epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process(),
        )

        if self.args.need_eval:
            self.logger.print('\n***** Evaluating at the beginning *****')
            self.logger.log(self.eval(), step=0)

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

                if self.global_step % self.args.save_interval == 0:
                    self.logger.print(f'Saving checkpoint at step {self.global_step} ...')
                    self.model.save_checkpoint(self.args.output_dir, tag=self.global_step)
                    self.logger.print('Checkpoint saved.')

                if (
                    self.args.need_eval
                    and self.args.eval_strategy == 'steps'
                    and self.global_step % self.args.eval_interval == 0
                ):
                    self.logger.print(f'\n***** Evaluating at step {self.global_step} *****')
                    self.logger.log(self.eval(), step=self.global_step)

            if self.args.need_eval and self.args.eval_strategy == 'epoch':
                self.logger.print(
                    f'\n***** Evaluating at epoch {epoch + 1}/{self.args.epochs} *****',
                )
                self.logger.log(self.eval(), step=self.global_step)

            self.model.tput_timer.update_epoch_count()

    def eval(self):
        safe_log_ratios = []
        table = wandb.Table(columns=["data_index"] + ["value"])
        self.model.eval()
        self.eval_dataloader = DataLoader(
                    self.train_dataloader.dataset,
                    collate_fn=self.train_dataloader.dataset.get_collator(),
                    batch_size=1,
                )
        with torch.no_grad():
            for batch in self.eval_dataloader:

                is_important = batch['important']
                if is_important:
                    batch = to_device(batch, self.args.device)
                    index = batch['index']
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    ref_log_probs = self.baseline_logprobs[index]
                    sequence_log_probs = self.compute_log_probs(  # size = (2 * B, L - 1)
                        self.model.module,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ) * attention_mask[:, 1:]

                    log_ratio = sequence_log_probs.sum(dim=1) - ref_log_probs
                    safe_log_ratios.append(log_ratio)
                    table.add_data(index, log_ratio)
        safe_log_ratios = torch.cat(safe_log_ratios, dim=0)
        safe_log_ratios = safe_log_ratios.cpu().numpy()
        # log all log ratios values as a table
            

        return {
            'eval/mean_important_log_ratio': safe_log_ratios.mean(),
            'eval/min_log_ratio': safe_log_ratios.min(),
            'eval/max_log_ratio': safe_log_ratios.max(),
            'eval/hist_log_ratio': wandb.Histogram(safe_log_ratios),
            'eval/table': table,    
        }

    def set_train(self, mode: bool = True) -> None:
        """Set training mode for model."""
        if mode:
            self.model.train()
            # if self.args.gradient_checkpointing:
            #     self.model.gradient_checkpointing_enable()
        else:
            self.model.eval()
            # if self.args.gradient_checkpointing:
            #     self.model.gradient_checkpointing_disable()
