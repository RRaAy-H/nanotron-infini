#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/train_infini_llama_fixed.py

"""
Training script for Infini-Llama models using preprocessed data.

This script handles the training process for Infini-Llama models, loading preprocessed
datasets and managing the training loop. It supports both CPU and GPU training with
proper Flash Attention support when available.

Usage:
```
# Basic usage with preprocessed data:
python train_infini_llama_fixed.py --config-file custom_tiny_infini_config.yaml --data-dir /data1/dataset/HuggingFaceFW/processed/tiny

# For GPU training with Flash Attention:
python train_infini_llama_fixed.py --config-file custom_tiny_infini_config.yaml --data-dir /data1/dataset/HuggingFaceFW/processed/tiny

# For GPU training without Flash Attention:
python train_infini_llama_fixed.py --config-file custom_tiny_infini_config.yaml --data-dir /data1/dataset/HuggingFaceFW/processed/tiny --disable-flash-attn

# For CPU-only training:
python train_infini_llama_fixed.py --config-file custom_tiny_infini_config.yaml --data-dir /data1/dataset/HuggingFaceFW/processed/tiny --cpu-only

# With TensorBoard integration:
python train_infini_llama_fixed.py --config-file custom_tiny_infini_config.yaml --data-dir /data1/dataset/HuggingFaceFW/processed/tiny --tensorboard-dir tensorboard_logs
```
"""

import argparse
import os
import sys
import time
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, cast, List

# Add the project root and src directories to Python path (important for nanotron imports)
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, os.path.join(str(root_dir), 'src'))

# Print debugging information
print("Python version:", sys.version)
print("Python path:", sys.path)
print("Current directory:", os.getcwd())
print("Root directory:", root_dir)

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
    balance_factor_lr: float = 0.001
    logging: bool = False
    logging_interval: int = 100
    log_grad: bool = False
    log_segment_acts: bool = False

@dataclass
class Config:
    infini_attention: InfiniAttentionConfig = field(default_factory=InfiniAttentionConfig)

# Import nanotron modules - using try/except for better error reporting
try:
    print("Importing nanotron.constants...")
    from nanotron import constants
    constants.CONFIG = Config()
    
    print("Importing nanotron.config...")
    from nanotron.config import (
        DataArgs,
        DatasetStageArgs,
        PretrainDatasetsArgs,
        ModelArgs,
        TokensArgs,
        OptimizerArgs,
        GeneralArgs,
        Config as NanotronConfig,
    )
    
    print("Importing nanotron.dataloader...")
    from nanotron.dataloader import (
        dummy_infinite_data_generator,
        get_train_dataloader,
        DataCollatorForCLM,
    )
    
    # Import GPU-accelerated data processing functions if available
    try:
        print("Importing nanotron.gpu_dataloader...")
        from nanotron.gpu_dataloader import (
            get_gpu_train_dataloader,
            GPUDataCollatorForCLM,
        )
        HAS_GPU_DATALOADER = True
    except ImportError as e:
        print(f"GPU dataloader error: {e}")
        HAS_GPU_DATALOADER = False
        print("GPU dataloader not available, falling back to CPU processing.")
    
    print("Importing nanotron.logging...")
    from nanotron import logging
    
    print("Importing nanotron.models.llama...")
    from nanotron.models.llama import LlamaForTraining
    
    print("Importing nanotron.helpers...")
    from nanotron.helpers import (
        init_optimizer_and_grad_accumulator,
        lr_scheduler_builder,
    )
    
    print("Importing nanotron.parallel...")
    from nanotron.parallel import ParallelContext
    
    print("Importing nanotron.parallel.pipeline_parallel.block...")
    from nanotron.parallel.pipeline_parallel.block import PipelineBlock
    
    print("Importing nanotron.parallel.pipeline_parallel.p2p...")
    from nanotron.parallel.pipeline_parallel.p2p import P2P
    
    print("Importing nanotron.parallel.tensor_parallel.enum...")
    from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
    
    print("Importing nanotron.trainer...")
    from nanotron.trainer import DistributedTrainer
    
    # Commented out problematic imports
    # from nanotron.utils import rank_print, import_module
    
    # Tensorboard imports
    try:
        print("Importing torch.utils.tensorboard...")
        from torch.utils.tensorboard import SummaryWriter
        HAS_TENSORBOARD = True
    except ImportError as e:
        print(f"TensorBoard error: {e}")
        HAS_TENSORBOARD = False
        print("TensorBoard not available. Training metrics will not be logged to TensorBoard.")
    
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this script from the project root or add the project root to PYTHONPATH.")
    sys.exit(1)

logger = logging.get_logger(__name__)

def load_preprocessed_data(data_dir: str):
    """Load preprocessed data from the specified directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory {data_dir} does not exist. Please preprocess data first.")
        
    metadata_path = data_path / "metadata.json"
    if not metadata_path.exists():
        # Try to work with just the dataset without metadata
        print(f"Warning: Metadata file not found in {data_dir}. Continuing without metadata.")
        metadata = {"tokenizer": "meta-llama/Llama-2-7b-hf"}
    else:
        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
    # Load dataset
    try:
        import datasets
        dataset_path = data_path
        if not dataset_path.exists():
            raise ValueError(f"Training dataset not found at {data_dir}")
            
        print(f"Loading dataset from {dataset_path}...")
        train_dataset = datasets.load_from_disk(str(dataset_path))
        print("Dataset loaded successfully")
        logger.info(f"Loaded preprocessed dataset from {dataset_path}")
        
        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer_path = data_path / "tokenizer"
        if tokenizer_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            logger.info(f"Loaded tokenizer from {tokenizer_path}")
        else:
            tokenizer_name = metadata.get("tokenizer", "meta-llama/Llama-2-7b-hf")
            print(f"Loading tokenizer from {tokenizer_name}...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info(f"Loaded tokenizer from {tokenizer_name}")
            
        return train_dataset, tokenizer, metadata
        
    except Exception as e:
        logger.error(f"Error loading preprocessed data: {e}")
        import traceback
        print(traceback.format_exc())
        raise

def create_and_configure_trainer(config_file: str, disable_infini_attn: bool = False) -> DistributedTrainer:
    """Create and configure the trainer based on the config file."""
    # Create trainer
    print(f"Creating trainer with config {config_file}...")
    trainer = DistributedTrainer(config_file)
    print("Trainer created successfully")
    
    # Disable Infini-Attention if requested
    if disable_infini_attn:
        logger.info("Disabling Infini-Attention for baseline model training")
        # This is implementation-dependent, but you might need to set a flag in the trainer
        # or modify the configuration before model initialization
        trainer.config.infini_attention.turn_on_memory = False
    
    return trainer

def setup_tensorboard(log_dir: str):
    """Set up TensorBoard writer if available."""
    if not HAS_TENSORBOARD:
        return None
        
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard logs will be saved to {log_dir}")
    
    return writer

def main():
    parser = argparse.ArgumentParser(description="Train Infini-Llama model from preprocessed data")
    parser.add_argument("--config-file", type=str, required=True,
                        help="Path to the configuration file")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing preprocessed data")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Use CPU only for training")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force CPU usage even if GPU is available")
    parser.add_argument("--disable-flash-attn", action="store_true",
                        help="Disable Flash Attention even if available")
    parser.add_argument("--disable-infini-attn", action="store_true",
                        help="Disable Infini-Attention and run with standard attention only (baseline)")
    parser.add_argument("--tensorboard-dir", type=str, default=None,
                        help="Directory for TensorBoard logs (disabled if not specified)")
    parser.add_argument("--gpu-device", type=str, default="cuda:0",
                        help="GPU device to use for training (default: cuda:0)")
    parser.add_argument("--use-gpu-dataloader", action="store_true",
                        help="Use GPU-accelerated data processing for faster text chunking")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()
    
    print(f"Arguments: {args}")
    
    # Set environment variables based on arguments
    if args.disable_flash_attn:
        os.environ["DISABLE_FLASH_ATTN"] = "1"
        
    # Configure logging
    if args.verbose:
        os.environ["NANOTRON_LOG_LEVEL"] = "debug"
    
    # Check if Flash Attention is available
    has_flash_attn = check_flash_attention()
    
    # Set device
    if args.cpu_only:
        device = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        if ":" in args.gpu_device:
            device_id = args.gpu_device.split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
            device = args.gpu_device
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
            device = f"cuda:{args.gpu_device}"
    
    print(f"Using device: {device}")
    
    # Load configuration
    config_path = Path(args.config_file)
    if not config_path.exists():
        logger.error(f"Configuration file {args.config_file} does not exist.")
        sys.exit(1)
        
    # Set random seed for reproducibility
    if args.seed is not None:
        from nanotron.random import set_random_seed
        print(f"Setting random seed to {args.seed}")
        set_random_seed(args.seed)
    
    # Create trainer
    trainer = create_and_configure_trainer(args.config_file, args.disable_infini_attn)
    
    # Set up TensorBoard
    tb_writer = None
    if args.tensorboard_dir is not None:
        tb_writer = setup_tensorboard(args.tensorboard_dir)
    
    # Load preprocessed data
    print(f"Loading data from {args.data_dir}...")
    train_dataset, tokenizer, metadata = load_preprocessed_data(args.data_dir)
    print(f"Dataset loaded with {len(train_dataset)} examples")
    
    # Get the input and output pipeline parallel ranks
    from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
    print("Getting input and output pipeline parallel ranks...")
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)
    
    # Configure data loading
    print("Configuring data collator...")
    data_collator = DataCollatorForCLM(
        sequence_length=trainer.config.tokens.sequence_length,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        parallel_context=trainer.parallel_context,
    )
    
    # Use GPU dataloader if requested and available
    if args.use_gpu_dataloader and HAS_GPU_DATALOADER and torch.cuda.is_available() and not args.cpu_only:
        print("Using GPU-accelerated data collator")
        logger.info("Using GPU-accelerated data collator")
        try:
            from nanotron.gpu_dataloader import GPUDataCollatorForCLM
            data_collator = GPUDataCollatorForCLM(
                sequence_length=trainer.config.tokens.sequence_length,
                input_pp_rank=input_pp_rank,
                output_pp_rank=output_pp_rank,
                parallel_context=trainer.parallel_context,
                device=device,
            )
        except Exception as e:
            print(f"Error initializing GPU data collator: {e}")
            print("Falling back to CPU data collator")
            # Continue with the already initialized CPU data collator
    
    # Create train dataloader
    print("Creating train dataloader...")
    train_loader = get_train_dataloader(
        train_dataset=train_dataset,
        sequence_length=trainer.config.tokens.sequence_length,
        parallel_context=trainer.parallel_context,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        micro_batch_size=trainer.config.tokens.micro_batch_size,
        consumed_train_samples=trainer.consumed_train_samples,
        dataloader_num_workers=trainer.config.data_stages[0].data.num_loading_workers,
        seed_worker=trainer.config.data_stages[0].data.seed,
    )
    
    # Initialize and train the model
    start_time = time.time()
    
    try:
        # Train the model
        print("Starting training...")
        trainer.train(train_loader)
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"Training failed: {e}")
        print(traceback.format_exc())
        sys.exit(1)
        
    finally:
        # Close TensorBoard writer
        if tb_writer is not None:
            tb_writer.close()
            
    # Report training time
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")
    print(f"Training completed in {total_time:.2f} seconds")
    
    # Save final model
    try:
        model_type = "baseline" if args.disable_infini_attn else "infini"
        output_dir = Path("models") / f"{model_type}_llama_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving final model to {output_dir}")
        print(f"Saving final model to {output_dir}")
        
        # Save model
        trainer.save(
            config=trainer.config,
            model=trainer.model,
            optimizer=trainer.optimizer,
            lr_scheduler=trainer.lr_scheduler,
            path=output_dir,
        )
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    main()
