#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/run_train_cpu.py

"""
Modified Nanotron training script for CPU-only mode.

Usage:
```
python run_train_cpu.py --config-file custom_infini_config.yaml
```
"""
import argparse
import sys
import os
from typing import Dict, cast

# Add the 'src' directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up infini attention configuration
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

# Import the rest of the necessary modules
from nanotron import constants, logging
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
from torch.utils.data import DataLoader

# Import optional dependencies
try:
    from huggingface_hub import __version__ as hf_hub_version
    from transformers import AutoTokenizer
    from transformers import __version__ as tf_version
except ImportError:
    hf_hub_version = None
    tf_version = None

logger = logging.get_logger(__name__)

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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    print("Setting up Infini Attention configuration...")
    print(f"Segment length: {constants.CONFIG.infini_attention.segment_length}")
    print(f"Turn on memory: {constants.CONFIG.infini_attention.turn_on_memory}")
    print(f"Balance init type: {constants.CONFIG.infini_attention.balance_init_type}")
    print(f"Balance act type: {constants.CONFIG.infini_attention.balance_act_type}")
    
    print("Loading trainer and data...")
    # Force CPU mode
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Load trainer and data
    trainer = DistributedTrainer(config_file)
    dataloader = get_dataloader(trainer)

    # Train
    print("Starting training...")
    trainer.train(dataloader)
