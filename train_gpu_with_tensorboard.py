#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/train_gpu_with_tensorboard.py

"""
Training script for Llama model with Infini-Attention, with TensorBoard integration.

Usage:
```
python train_gpu_with_tensorboard.py --config-file custom_infini_config_gpu.yaml --tensorboard-dir tensorboard_logs
```
"""
import argparse
import sys
import os
import time
from typing import Dict, cast
from datetime import datetime

# Add the 'src' directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure infini attention
from dataclasses import dataclass, field

@dataclass
class InfiniAttentionConfig:
    segment_length: int = 64
    turn_on_memory: bool = True
    balance_init_type: str = 'zeros'
    balance_act_type: str = 'orig_sigmoid'
    balance_factor_lr: float = 0.001  # Added missing required parameter

@dataclass
class Config:
    infini_attention: InfiniAttentionConfig = field(default_factory=InfiniAttentionConfig)

# Import nanotron modules
from nanotron import constants
constants.CONFIG = Config()

from nanotron.config import (
    DataArgs,
    DatasetStageArgs,
    PretrainDatasetsArgs,
)
from nanotron.dataloader import (
    clm_process,
    dummy_infinite_data_generator,
    get_datasets,
    get_train_dataloader,
)
from nanotron.logging import log_rank
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.trainer import DistributedTrainer
from nanotron.utils import (
    main_rank_first,
)
from nanotron import logging
from torch.utils.data import DataLoader

# TensorBoard imports
from torch.utils.tensorboard import SummaryWriter

try:
    from huggingface_hub import __version__ as hf_hub_version
    from transformers import AutoTokenizer
    from transformers import __version__ as tf_version
except ImportError:
    hf_hub_version = None
    tf_version = None

logger = logging.get_logger(__name__)

# Custom callback for TensorBoard integration
class TensorBoardCallback:
    def __init__(self, log_dir, trainer):
        """
        Initialize TensorBoard callback.
        
        Args:
            log_dir: Directory where TensorBoard logs will be saved
            trainer: The DistributedTrainer instance
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        self.trainer = trainer
        self.start_time = time.time()
        
        # Log hyperparameters
        hparams = {
            "model/hidden_size": trainer.model_config.hidden_size,
            "model/num_hidden_layers": trainer.model_config.num_hidden_layers,
            "model/num_attention_heads": trainer.model_config.num_attention_heads,
            "optim/learning_rate": trainer.config.optimizer.learning_rate_scheduler.learning_rate,
            "optim/weight_decay": trainer.config.optimizer.weight_decay,
            "optim/warmup_steps": trainer.config.optimizer.learning_rate_scheduler.lr_warmup_steps,
            "training/micro_batch_size": trainer.micro_batch_size,
            "training/sequence_length": trainer.sequence_length,
            "infini/segment_length": constants.CONFIG.infini_attention.segment_length,
            "infini/turn_on_memory": constants.CONFIG.infini_attention.turn_on_memory,
        }
        self.writer.add_hparams(hparams, {})
        
        print(f"TensorBoard initialized at {log_dir}")
        print(f"View with: tensorboard --logdir={log_dir}")

    def on_step_end(self, step, loss, learning_rate=None, throughput=None):
        """Log metrics at the end of each training step."""
        # Log loss
        self.writer.add_scalar('training/loss', loss, step)
        
        # Log learning rate if available
        if learning_rate is not None:
            self.writer.add_scalar('training/learning_rate', learning_rate, step)
        
        # Log throughput (samples/sec) if available
        if throughput is not None:
            self.writer.add_scalar('training/throughput', throughput, step)
            
        # Log elapsed time
        elapsed_time = time.time() - self.start_time
        self.writer.add_scalar('training/elapsed_time_hours', elapsed_time / 3600, step)
        
        # Ensure logs are written to disk
        self.writer.flush()
        
    def on_validation_end(self, step, val_loss, val_perplexity):
        """Log validation metrics."""
        self.writer.add_scalar('validation/loss', val_loss, step)
        self.writer.add_scalar('validation/perplexity', val_perplexity, step)
        self.writer.flush()
        
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()

def get_dataloader_from_data_stage(trainer: DistributedTrainer, data: DataArgs):
    """Returns a dataloader for training."""

    # First, we need to know which ranks to feed the dataloader to
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)

    # Case 1: Dummy data generator
    if data.dataset is None:
        log_rank("Using dummy data generator", logger=logger, level=logging.INFO, rank=0)
        dataloader = dummy_infinite_data_generator(
            micro_batch_size=trainer.micro_batch_size,
            sequence_length=trainer.sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            vocab_size=trainer.model_config.vocab_size,
            seed=data.seed,
            parallel_context=trainer.parallel_context,
        )()

    # Case 2: HuggingFace datasets
    elif isinstance(data.dataset, PretrainDatasetsArgs):
        log_rank("Using `datasets` library", logger=logger, level=logging.INFO, rank=0)
        tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path
        log_rank(
            f"Loading tokenizer from {tokenizer_path} and transformers/hf_hub versions {tf_version, hf_hub_version}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        # We need to the 1st device to process dataset and cache it, then other devices load from cache
        with main_rank_first(trainer.parallel_context.world_pg):
            # We load the raw dataset
            raw_dataset = get_datasets(
                hf_dataset_or_datasets=data.dataset.hf_dataset_or_datasets,
                hf_dataset_config_name=data.dataset.hf_dataset_config_name,
                splits=data.dataset.hf_dataset_splits,
            )["train"]

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            # We apply the Causal Language Modeling preprocessing
            train_dataset = clm_process(
                raw_dataset=raw_dataset,
                tokenizer=tokenizer,
                text_column_name=data.dataset.text_column_name,
                dataset_processing_num_proc_per_process=data.dataset.dataset_processing_num_proc_per_process,
                dataset_overwrite_cache=data.dataset.dataset_overwrite_cache,
                sequence_length=trainer.sequence_length,
            )

            # We load the processed dataset on the ranks requiring it
            dataloader = get_train_dataloader(
                train_dataset=train_dataset,
                sequence_length=trainer.sequence_length,
                parallel_context=trainer.parallel_context,
                input_pp_rank=input_pp_rank,
                output_pp_rank=output_pp_rank,
                micro_batch_size=trainer.micro_batch_size,
                consumed_train_samples=trainer.consumed_train_samples,
                dataloader_num_workers=data.num_loading_workers,
                seed_worker=data.seed,
                dataloader_drop_last=True,
            )
            # Check if we have enough samples for train_steps
            total_tokens_dataset = len(dataloader.dataset) * trainer.sequence_length
            num_tokens_needed_for_training = (
                (trainer.config.tokens.train_steps - trainer.start_iteration_step)
                * trainer.global_batch_size
                * trainer.sequence_length
            )

            if num_tokens_needed_for_training <= total_tokens_dataset:
                print("Dataset is sufficient for the requested number of training steps")
            else:
                print(
                    f"Dataset is smaller than needed for steps ({total_tokens_dataset} < {num_tokens_needed_for_training}), "
                    f"Training will loop through the dataset multiple times."
                )
    else:
        raise ValueError(f"Unhandled case of `self.config.data.dataset`. Got: {data.dataset}")

    return dataloader


def get_dataloader(trainer: DistributedTrainer) -> Dict[str, DataLoader]:
    sorted_stages = sorted(trainer.config.data_stages, key=lambda stage: stage.start_training_step)
    dataloaders = {}
    for idx, stage in enumerate(sorted_stages):
        # NOTE: we only create the dataloader for the first stage,
        # then we lazy initialize the dataloader for the other stages
        stage = cast(DatasetStageArgs, stage)
        dataloader = (
            get_dataloader_from_data_stage(trainer, stage.data)
            if idx == 0
            else lambda stage=stage: get_dataloader_from_data_stage(trainer, stage.data)
        )
        dataloaders[stage.name] = dataloader
    return dataloaders


class CustomDistributedTrainer(DistributedTrainer):
    """Extended DistributedTrainer with TensorBoard integration."""
    
    def __init__(self, config_file, tensorboard_dir):
        super().__init__(config_file)
        self.tensorboard_callback = TensorBoardCallback(tensorboard_dir, self)
        
    def train(self, dataloader):
        """Override train method to include TensorBoard logging."""
        last_log_time = time.time()
        
        for step, batch, loss in super().train(dataloader):
            current_time = time.time()
            
            # Calculate throughput (samples per second)
            elapsed = current_time - last_log_time
            if elapsed > 0:  # Avoid division by zero
                throughput = self.micro_batch_size / elapsed
            else:
                throughput = 0
                
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"] if hasattr(self, "optimizer") else None
            
            # Log to TensorBoard
            self.tensorboard_callback.on_step_end(step, loss, current_lr, throughput)
            
            # Update time for next iteration
            last_log_time = current_time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    parser.add_argument("--tensorboard-dir", type=str, default="tensorboard_logs", help="Directory for TensorBoard logs")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file
    
    # Create tensorboard directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_dir = os.path.join(args.tensorboard_dir, f"infini_llama_{timestamp}")
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    print(f"Starting training with config: {config_file}")
    print(f"TensorBoard logs will be saved to: {tensorboard_dir}")
    
    # Initialize the trainer
    trainer = CustomDistributedTrainer(config_file, tensorboard_dir)
    
    # Get dataloader
    dataloader = get_dataloader(trainer)
    
    # Train
    try:
        trainer.train(dataloader)
    finally:
        # Close tensorboard writer
        if hasattr(trainer, "tensorboard_callback"):
            trainer.tensorboard_callback.close()
