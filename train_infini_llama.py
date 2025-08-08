#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/train_infini_llama.py

"""
Consolidated Training Script for Infini-Attention Llama Model.
This script integrates both CPU and GPU training with proper Flash Attention support.

Features:
- Auto-detection of Flash Attention compatibility
- Fallback to standard attention when Flash Attention fails
- TensorBoard integration for monitoring training
- Support for both CPU and GPU training
- Proper distributed environment setup

Usage:
```
# For GPU training with Flash Attention:
python train_infini_llama.py --config-file custom_infini_config_gpu.yaml

# For GPU training without Flash Attention:
python train_infini_llama.py --config-file custom_infini_config_gpu.yaml --disable-flash-attn

# For CPU-only training:
python train_infini_llama.py --config-file custom_infini_config_cpu.yaml --cpu-only

# With TensorBoard integration:
python train_infini_llama.py --config-file custom_infini_config_gpu.yaml --tensorboard-dir tensorboard_logs

# With distributed training (using torchrun):
torchrun --nproc_per_node=8 train_infini_llama.py --config-file custom_infini_config_gpu.yaml
```
"""

import argparse
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, cast

# Add the 'src' directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Check for Flash Attention errors early
def check_flash_attention():
    """Check if Flash Attention is available and compatible with the current environment."""
    if os.environ.get("DISABLE_FLASH_ATTN", "0") != "1":
        try:
            # Try to import Flash Attention to catch issues early
            import flash_attn
            print("Flash Attention imported successfully.")
            return True
        except ImportError as e:
            print(f"Warning: Could not import Flash Attention: {e}")
            print("Automatically disabling Flash Attention.")
            os.environ["DISABLE_FLASH_ATTN"] = "1"
            return False
    return False

# Set up infini attention configuration
from dataclasses import dataclass, field

@dataclass
class InfiniAttentionConfig:
    segment_length: int = 64
    turn_on_memory: bool = True
    balance_init_type: str = 'zeros'
    balance_act_type: str = 'orig_sigmoid'
    balance_factor_lr: float = 0.001  # Required parameter
    logging: bool = False  # Required parameter
    logging_interval: int = 100  # Required parameter
    log_grad: bool = False  # Required parameter
    log_segment_acts: bool = False  # Required parameter

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
# Import GPU-accelerated data processing functions
from nanotron.gpu_dataloader import (
    gpu_clm_process,
    get_gpu_train_dataloader,
)
from nanotron import logging
from nanotron.logging import log_rank
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.trainer import DistributedTrainer
from nanotron.utils import (
    main_rank_first,
)
from torch.utils.data import DataLoader

# TensorBoard imports
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Training will continue without TensorBoard logging.")

# Optional imports
try:
    from huggingface_hub import __version__ as hf_hub_version
    from transformers import AutoTokenizer
    from transformers import __version__ as tf_version
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    hf_hub_version = None
    tf_version = None
    TRANSFORMERS_AVAILABLE = False
    print("Transformers library not available. Will use dummy data if no other dataset is specified.")

logger = logging.get_logger(__name__)


def apply_flash_attention_patch():
    """Apply the Flash Attention patch to handle failures gracefully."""
    patch_script = os.path.join(os.path.dirname(__file__), "patch_flash_attention.py")
    if os.path.exists(patch_script):
        print("Applying Flash Attention patch to handle potential Flash Attention failures...")
        try:
            import subprocess
            result = subprocess.run([sys.executable, patch_script], 
                                   capture_output=True, text=True, check=False)
            if result.returncode == 0:
                print("Flash Attention patch applied successfully")
            else:
                print(f"Warning: Flash Attention patch failed: {result.stderr}")
        except Exception as e:
            print(f"Error applying Flash Attention patch: {e}")
            print("Continuing with training...")


# TensorBoard Callback for monitoring training
class TensorBoardCallback:
    def __init__(self, log_dir, trainer):
        """
        Initialize TensorBoard callback.
        
        Args:
            log_dir: Directory where TensorBoard logs will be saved
            trainer: The DistributedTrainer instance
        """
        if not TENSORBOARD_AVAILABLE:
            print("TensorBoard not available. Logging will be disabled.")
            self.writer = None
            return
            
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
            "infini/flash_attention": os.environ.get("DISABLE_FLASH_ATTN", "0") != "1",
        }
        self.writer.add_hparams(hparams, {})
        
        print(f"TensorBoard initialized at {log_dir}")
        print(f"View with: tensorboard --logdir={log_dir}")

    def on_step_end(self, step, loss, learning_rate=None, throughput=None):
        """Log metrics at the end of each training step."""
        if not TENSORBOARD_AVAILABLE or self.writer is None:
            return
            
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
        if not TENSORBOARD_AVAILABLE or self.writer is None:
            return
            
        self.writer.add_scalar('validation/loss', val_loss, step)
        self.writer.add_scalar('validation/perplexity', val_perplexity, step)
        self.writer.flush()
        
    def close(self):
        """Close the TensorBoard writer."""
        if TENSORBOARD_AVAILABLE and self.writer is not None:
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
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library required for using HuggingFace datasets")
            
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


def get_gpu_accelerated_dataloader_from_data_stage(trainer: DistributedTrainer, data: DataArgs, gpu_device: str = "cuda:0"):
    """Returns a dataloader for training with GPU-accelerated data processing.
    
    This version uses GPU acceleration for text chunking and processing, which significantly
    speeds up the data preparation pipeline.
    
    Args:
        trainer: The distributed trainer instance
        data: Dataset configuration
        gpu_device: The GPU device to use for data processing
    """
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
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library required for using HuggingFace datasets")
            
        log_rank("Using `datasets` library with GPU acceleration", logger=logger, level=logging.INFO, rank=0)
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

            # We apply the GPU-accelerated Causal Language Modeling preprocessing
            print(f"Using GPU-accelerated text processing on {gpu_device}...")
            train_dataset = gpu_clm_process(
                raw_dataset=raw_dataset,
                tokenizer=tokenizer,
                text_column_name=data.dataset.text_column_name,
                dataset_processing_num_proc_per_process=data.dataset.dataset_processing_num_proc_per_process,
                dataset_overwrite_cache=data.dataset.dataset_overwrite_cache,
                sequence_length=trainer.sequence_length,
                device=gpu_device,
                batch_size=2048,  # Process this many texts at a time on GPU
            )

            # We load the processed dataset on the ranks requiring it with GPU acceleration
            dataloader = get_gpu_train_dataloader(
                train_dataset=train_dataset,
                sequence_length=trainer.sequence_length,
                parallel_context=trainer.parallel_context,
                input_pp_rank=input_pp_rank,
                output_pp_rank=output_pp_rank,
                micro_batch_size=trainer.micro_batch_size,
                consumed_train_samples=trainer.consumed_train_samples,
                dataloader_num_workers=data.num_loading_workers,
                seed_worker=data.seed,
                gpu_device=gpu_device,
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


def get_gpu_accelerated_dataloader(trainer: DistributedTrainer, gpu_device: str = "cuda:0") -> Dict[str, DataLoader]:
    """Returns a dictionary of dataloaders with GPU-accelerated data processing.
    
    Args:
        trainer: The distributed trainer instance
        gpu_device: The GPU device to use for data processing
    """
    sorted_stages = sorted(trainer.config.data_stages, key=lambda stage: stage.start_training_step)
    dataloaders = {}
    for idx, stage in enumerate(sorted_stages):
        # NOTE: we only create the dataloader for the first stage,
        # then we lazy initialize the dataloader for the other stages
        stage = cast(DatasetStageArgs, stage)
        dataloader = (
            get_gpu_accelerated_dataloader_from_data_stage(trainer, stage.data, gpu_device)
            if idx == 0
            else lambda stage=stage: get_gpu_accelerated_dataloader_from_data_stage(trainer, stage.data, gpu_device)
        )
        dataloaders[stage.name] = dataloader
    return dataloaders


class CustomDistributedTrainer(DistributedTrainer):
    """Extended DistributedTrainer with TensorBoard integration."""
    
    def __init__(self, config_file, tensorboard_dir=None):
        super().__init__(config_file)
        self.tensorboard_callback = None
        if tensorboard_dir:
            self.tensorboard_callback = TensorBoardCallback(tensorboard_dir, self)
        
    def train(self, dataloader):
        """Override train method to include TensorBoard logging."""
        last_log_time = time.time()
        
        try:
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
                
                # Log to TensorBoard if available
                if self.tensorboard_callback:
                    self.tensorboard_callback.on_step_end(step, loss, current_lr, throughput)
                
                # Update time for next iteration
                last_log_time = current_time
        except Exception as e:
            print(f"Error during training: {str(e)}")
            traceback.print_exc()
            # Close TensorBoard writer before propagating the error
            if hasattr(self, "tensorboard_callback") and self.tensorboard_callback:
                self.tensorboard_callback.close()
            raise
        finally:
            # Ensure TensorBoard is closed properly
            if hasattr(self, "tensorboard_callback") and self.tensorboard_callback:
                self.tensorboard_callback.close()


def init_distributed_environment():
    """Initialize the distributed environment if environment variables are not properly set.
    This allows the script to be run directly without using torchrun or similar launcher.
    """
    if "WORLD_SIZE" not in os.environ or "RANK" not in os.environ:
        print("Initializing single-GPU/CPU environment for training...")
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
    
    # Print environment details for debugging
    print(f"Distributed training environment:")
    print(f"  RANK: {os.environ.get('RANK', 'Not set')}")
    print(f"  WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"  LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
    print(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'Not set')}")
    print(f"  MASTER_PORT: {os.environ.get('MASTER_PORT', 'Not set')}")


def get_args():
    parser = argparse.ArgumentParser(description="Infini-Attention Llama Training Script")
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    parser.add_argument("--tensorboard-dir", type=str, default=None, 
                       help="Directory for TensorBoard logs. If not provided, TensorBoard logging is disabled.")
    parser.add_argument("--disable-flash-attn", action="store_true", 
                       help="Disable Flash Attention and use standard attention")
    parser.add_argument("--cpu-only", action="store_true", 
                       help="Force CPU-only training even if GPUs are available")
    parser.add_argument("--gpu-device", type=str, default="cuda:0",
                       help="GPU device to use for data processing and training (default: cuda:0)")
    parser.add_argument("--use-gpu-dataloader", action="store_true",
                       help="Use GPU-accelerated data processing for faster text chunking")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose logging")
    return parser.parse_args()


def setup_environment(args):
    """Set up the training environment based on command-line arguments."""
    # Set environment variable to disable Flash Attention if requested
    if args.disable_flash_attn:
        os.environ["DISABLE_FLASH_ATTN"] = "1"
        print("Flash Attention has been disabled, using standard attention implementation")
    
    # Force CPU-only mode if requested
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Forcing CPU-only training mode")
        # Disable GPU dataloader in CPU-only mode
        args.use_gpu_dataloader = False
    
    # Set CUDA device for training if specified
    if not args.cpu_only and "cuda" in args.gpu_device:
        gpu_id = args.gpu_device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"Setting primary GPU device to: {args.gpu_device}")
    
    # Apply Flash Attention patch if not in CPU-only mode
    if not args.cpu_only and not args.disable_flash_attn:
        # First check if Flash Attention is available
        flash_available = check_flash_attention()
        if flash_available:
            apply_flash_attention_patch()
    
    # Initialize distributed environment
    init_distributed_environment()
    
    # Print configuration details
    print("\nInfini Attention configuration:")
    print(f"  Segment length: {constants.CONFIG.infini_attention.segment_length}")
    print(f"  Turn on memory: {constants.CONFIG.infini_attention.turn_on_memory}")
    print(f"  Balance init type: {constants.CONFIG.infini_attention.balance_init_type}")
    print(f"  Balance act type: {constants.CONFIG.infini_attention.balance_act_type}")
    print(f"  Balance factor LR: {constants.CONFIG.infini_attention.balance_factor_lr}")
    print(f"  Flash Attention enabled: {os.environ.get('DISABLE_FLASH_ATTN', '0') == '0'}")
    
    if args.use_gpu_dataloader:
        print(f"  GPU-accelerated data processing: Enabled (device: {args.gpu_device})")
    else:
        print("  GPU-accelerated data processing: Disabled")
    print("")


def main():
    args = get_args()
    
    # Set up the environment
    setup_environment(args)
    
    config_file = args.config_file
    print(f"Using configuration file: {config_file}")
    
    # Create tensorboard directory with timestamp if needed
    tensorboard_dir = None
    if args.tensorboard_dir:
        if TENSORBOARD_AVAILABLE:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_dir = os.path.join(args.tensorboard_dir, f"infini_llama_{timestamp}")
            os.makedirs(tensorboard_dir, exist_ok=True)
            print(f"TensorBoard logs will be saved to: {tensorboard_dir}")
        else:
            print("Warning: TensorBoard not available, but --tensorboard-dir was specified.")
            print("Training will continue without TensorBoard logging.")
    
    print("\nInitializing trainer...")
    # Initialize the trainer
    trainer = CustomDistributedTrainer(config_file, tensorboard_dir)
    
    print("Preparing data loaders...")
    # Get dataloader, using GPU-accelerated version if requested
    if args.use_gpu_dataloader and not args.cpu_only:
        print(f"Using GPU-accelerated data processing on {args.gpu_device}...")
        dataloader = get_gpu_accelerated_dataloader(trainer, gpu_device=args.gpu_device)
    else:
        dataloader = get_dataloader(trainer)
    
    print("\nStarting training...")
    # Train
    try:
        trainer.train(dataloader)
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
