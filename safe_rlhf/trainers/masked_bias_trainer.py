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


class MaskedBiasTrainer(TrainerBase):
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
        print("Models initialized.")
        self.init_datasets()
        dist.barrier()
        print("Datasets initialized.")
        self.init_engines()
        dist.barrier()
        print("Engines initialized.")
        self.init_duals()
        dist.barrier()
        print("Dual variables initialized.")
        self.init_logger()
        print("Logger initialized.")
        
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
        # self.model = get_peft_model(
        #     self.model,
        #     LoraConfig(
        #         r=self.args.lora_r,
        #         lora_alpha=self.args.lora_alpha,
        #         lora_dropout=self.args.lora_dropout,
        #         target_modules=[
        #             "q_proj",
        #             "k_proj",
        #             "v_proj",
        #             "o_proj",
        #             "gate_proj",
        #             "down_proj",
        #             "up_proj",
        #             "lm_head",
        #         ],
        #     ),
        # # )
        # self.args.yes_token = self.tokenizer.encode(' yes', add_special_tokens=False)[-1]
        # self.args.no_token = self.tokenizer.encode(' no', add_special_tokens=False)[-1]
        # print(f"Using YES token: {self.args.yes_token}, NO token: {self.args.no_token}")
    
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
                    lazy_tokenization=False,
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

    def eval(self) -> dict[str, Any]:

        device = self.args.device
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        def ddp_reduce_stats(flat_tensor: torch.Tensor):
            """Return global (mean, min, max) over all ranks."""
            if flat_tensor.numel() == 0:
                # create neutral elements
                local_sum = torch.zeros(1, device=device)
                local_cnt = torch.zeros(1, device=device)
                local_min = torch.tensor([float('inf')], device=device)
                local_max = torch.tensor([float('-inf')], device=device)
            else:
                local_sum = torch.tensor([flat_tensor.sum()], device=device)
                local_cnt = torch.tensor([flat_tensor.numel()], device=device)
                local_min = torch.tensor([flat_tensor.min()], device=device)
                local_max = torch.tensor([flat_tensor.max()], device=device)

            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_cnt, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_min, op=dist.ReduceOp.MIN)
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX)

            if local_cnt.item() > 0:
                mean = (local_sum / local_cnt).item()
                gmin = local_min.item()
                gmax = local_max.item()
            else:
                mean = gmin = gmax = 0.0
            return mean, gmin, gmax

        def ddp_mean(values: list[torch.Tensor]) -> float:
            """Compute global mean over list of scalar tensors (each scalar) on all ranks."""
            if len(values) == 0:
                local_sum = torch.zeros(1, device=device)
                local_cnt = torch.zeros(1, device=device)
            else:
                stacked = torch.stack([v.detach().float() for v in values])
                local_sum = torch.tensor([stacked.sum()], device=device)
                local_cnt = torch.tensor([stacked.numel()], device=device)
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_cnt, op=dist.ReduceOp.SUM)
            return (local_sum / local_cnt).item() if local_cnt.item() > 0 else 0.0
        
        def gather_rows_all(local_rows):
            world_size = dist.get_world_size()
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local_rows)
            flat = []
            for part in gathered:
                if part:
                    flat.extend(part)
            return flat


        self.model.eval()


        eval_rows: list[tuple[int, float]] = []
        eval_constraint_vals: list[torch.Tensor] = []
        eval_loss_vals: list[torch.Tensor] = []

        if self.eval_dataloader is not None:
            eval_pbar = tqdm(
            total=len(self.eval_dataloader),
            desc="Eval (eval split)",
            disable=not is_main_process(),
            leave=False,
            )
            with torch.no_grad():
                for batch in self.eval_dataloader:
                    batch = to_device(batch, device)
                    index = batch['index']                 # shape (B,)
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

                    obj = loss_dict['objective'].detach().float()
                    constraint_slack = loss_dict['constraint_value'].detach().float()  # shape (B,) or scalar

                    # Normalize to 1D vector of per-sample values
                    constraint_slack_flat = constraint_slack.view(-1)
                    # Add per-sample rows
                    for i_val, c_val in zip(index.tolist(), constraint_slack_flat.tolist()):
                        eval_rows.append((int(i_val), float(c_val)))

                    eval_constraint_vals.append(constraint_slack_flat)
                    eval_loss_vals.append(obj)
                    if is_main_process():
                        eval_pbar.set_postfix(
                            obj=float(obj),
                            last_slack=float(constraint_slack_flat.mean().item())
                                if constraint_slack_flat.numel() > 0 else 0.0
                        )
                    eval_pbar.update(1)
            if is_main_process():
                eval_pbar.close()
            if len(eval_constraint_vals) > 0:
                eval_constraint_tensor = torch.cat(eval_constraint_vals, dim=0)
            else:
                eval_constraint_tensor = torch.zeros(0, device=device)
        else:
            # No eval dataloader
            eval_constraint_tensor = torch.zeros(0, device=device)

        # ---------------------------
        # TRAIN SPLIT (for monitoring)
        # ---------------------------
        train_rows: list[tuple[int, float]] = []
        dual_rows: list[tuple[int, float]] = []
        train_constraint_vals: list[torch.Tensor] = []
        train_loss_vals: list[torch.Tensor] = []
        train_pbar = tqdm(
            total=len(self.train_dataloader),
            desc="Eval (train split)",
            disable=not is_main_process(),
            leave=False,
        )
        with torch.no_grad():
            for batch in self.train_dataloader:
                batch = to_device(batch, device)
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

                obj = loss_dict['objective'].detach().float()
                c_slack = loss_dict['constraint_value'].detach().float()
                c_slack_flat = c_slack.view(-1)

                for i_val, c_val in zip(index.tolist(), c_slack_flat.tolist()):
                    train_rows.append((int(i_val), float(c_val)))
                    # dual vars indexed by i_val
                    dual_rows.append((int(i_val), float(self.dual_vars[i_val].item())))

                train_constraint_vals.append(c_slack_flat)
                train_loss_vals.append(obj)
                if is_main_process():
                    train_pbar.set_postfix(
                        obj=float(obj),
                        last_slack=float(c_slack_flat.mean().item())
                            if c_slack_flat.numel() > 0 else 0.0
                    )
                train_pbar.update(1)
        if is_main_process():
            train_pbar.close()
        if len(train_constraint_vals) > 0:
            train_constraint_tensor = torch.cat(train_constraint_vals, dim=0)
        else:
            train_constraint_tensor = torch.zeros(0, device=device)

        # ---------------------------
        # GLOBAL SCALARS
        # ---------------------------
        print(f"Eval split: {len(eval_rows)} rows, train split: {len(train_rows)} rows")
        eval_mean, eval_min, eval_max = ddp_reduce_stats(eval_constraint_tensor)
        print(f"Eval split: mean {eval_mean:.4f}, min {eval_min:.4f}, max {eval_max:.4f}")
        train_mean, train_min, train_max = ddp_reduce_stats(train_constraint_tensor)
        print(f"Train split: mean {train_mean:.4f}, min {train_min:.4f}, max {train_max:.4f}")
        eval_obj_mean = ddp_mean(eval_loss_vals)
        print(f"Eval split: objective mean {eval_obj_mean:.4f}")
        train_obj_mean = ddp_mean(train_loss_vals)
        print(f"Train split: objective mean {train_obj_mean:.4f}")
        result: dict[str, Any] = {
            'eval/mean_constraint_slack': eval_mean,
            'eval/min_constraint_slack': eval_min,
            'eval/max_constraint_slack': eval_max,
            'train/mean_constraint_slack': train_mean,
            'train/min_constraint_slack': train_min,
            'train/max_constraint_slack': train_max,
            'eval/obj': eval_obj_mean,
            'train/obj': train_obj_mean,
        }


        # ---- Gather per-rank rows on *all* ranks ----

        if self.eval_dataloader is not None:
            all_eval_rows  = gather_rows_all(eval_rows)
        else:
            all_eval_rows  = []
        all_train_rows = gather_rows_all(train_rows)
        all_dual_rows  = gather_rows_all(dual_rows)

        # ---- Only rank 0 constructs W&B artifacts ----
        if is_main_process():
            all_eval_rows.sort(key=lambda x: x[0])
            all_train_rows.sort(key=lambda x: x[0])
            all_dual_rows.sort(key=lambda x: x[0])

            if self.eval_dataloader is not None:
                eval_table = wandb.Table(columns=["data_index", "value"])
                for i, v in all_eval_rows:
                    eval_table.add_data(i, v)
                result['eval/table'] = eval_table
                result['eval/hist_constraint_slack'] = (
                    wandb.Histogram([v for _, v in all_eval_rows]) if all_eval_rows else None
                )

            train_table = wandb.Table(columns=["data_index", "value"])
            for i, v in all_train_rows:
                train_table.add_data(i, v)
            result['train/table'] = train_table

            dual_table = wandb.Table(columns=["data_index", "value"])
            for i, v in all_dual_rows:
                dual_table.add_data(i, v)
            result['train/dual_vars'] = dual_table

            result['train/hist_constraint_slack'] = (
                wandb.Histogram([v for _, v in all_train_rows]) if all_train_rows else None
            )
        # Optional sync (not strictly necessary because all_gather already synchronizes)
        dist.barrier()
        return result

    def set_train(self, mode: bool = True) -> None:
        """Set training mode for model."""
        if mode:
            self.model.train()
        else:
            self.model.eval()

