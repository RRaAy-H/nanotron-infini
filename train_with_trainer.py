#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train Infini-Llama model using Nanotron's DistributedTrainer.
This script handles training with automatic Flash Attention compatibility detection.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Union, Iterator

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from nanotron.trainer import DistributedTrainer
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.dataloader import get_datasets, clm_process, get_train_dataloader
from nanotron import distributed as dist
from nanotron import logging as nanotron_logging

# Configure logging
logger = nanotron_logging.get_logger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

def detect_flash_attention_compatibility():
    """Check if Flash Attention is compatible with the current system."""
    try:
        import ctypes
        import re
        
        # Get GLIBC version
        libc = ctypes.CDLL("libc.so.6")
        glibc_version_str = ctypes.c_char_p(libc.gnu_get_libc_version()).value.decode()
        glibc_version = tuple(map(int, glibc_version_str.split(".")))
        logger.info(f"Detected GLIBC version: {glibc_version_str}")
        
        # Try importing Flash Attention
        try:
            import flash_attn
            logger.info(f"Flash Attention version: {flash_attn.__version__}")
            return True, None
        except ImportError as e:
            logger.warning(f"Flash Attention not installed: {e}")
            return False, str(e)
        except OSError as e:
            if "GLIBC" in str(e):
                match = re.search(r"GLIBC_([0-9.]+)", str(e))
                if match:
                    required_version = match.group(1)
                    logger.warning(
                        f"Flash Attention requires GLIBC {required_version} but found {glibc_version_str}"
                    )
                return False, str(e)
            else:
                logger.warning(f"Flash Attention loading error: {e}")
                return False, str(e)
    except Exception as e:
        logger.warning(f"Error checking Flash Attention compatibility: {e}")
        return False, str(e)

def disable_flash_attention():
    """Disable Flash Attention by setting environment variables."""
    os.environ["NANOTRON_FORCE_DISABLE_FLASH_ATTN"] = "1"
    logger.info("Flash Attention disabled due to compatibility issues")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Infini-Llama with Nanotron Trainer")
    parser.add_argument(
        "--config-file", 
        type=str, 
        required=True,
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        required=True,
        help="Directory containing the training data (parquet files)"
    )
    parser.add_argument(
        "--tokenizer-path", 
        type=str, 
        default="meta-llama/Llama-2-7b-hf",
        help="Path or name of the tokenizer to use"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Directory to save checkpoints and logs"
    )
    parser.add_argument(
        "--auto-detect-flash-attn", 
        action="store_true",
        help="Automatically detect Flash Attention compatibility"
    )
    parser.add_argument(
        "--disable-flash-attn", 
        action="store_true",
        help="Forcibly disable Flash Attention regardless of compatibility"
    )
    parser.add_argument(
        "--micro-batch-size", 
        type=int, 
        default=None,
        help="Override micro batch size from config"
    )
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=4,
        help="Number of workers for data loading"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Run in debug mode with additional logging"
    )
    return parser.parse_args()

def find_parquet_files(data_dir):
    """Find all parquet files in the given directory."""
    data_dir = Path(data_dir)
    parquet_files = list(data_dir.glob("**/*.parquet"))
    if not parquet_files:
        logger.error(f"No parquet files found in {data_dir}")
        sys.exit(1)
    return parquet_files

def main():
    """Main training function."""
    args = parse_args()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        nanotron_logging.set_rank_logger_level(level=logging.DEBUG)
        logger.debug("Debug mode enabled - verbose logging activated")
    
    # Check Flash Attention compatibility or disable it if requested
    if args.disable_flash_attn:
        logger.info("Flash Attention explicitly disabled via command line option")
        disable_flash_attention()
    elif args.auto_detect_flash_attn:
        is_compatible, error_msg = detect_flash_attention_compatibility()
        if not is_compatible:
            logger.warning(f"Flash Attention is not compatible: {error_msg}")
            disable_flash_attention()
    
    # Initialize the trainer
    logger.info(f"Initializing trainer with config: {args.config_file}")
    trainer = DistributedTrainer(args.config_file)
    
    # Get model parameters for data loading
    input_pp_rank = trainer.unwrapped_model.input_pp_rank
    output_pp_rank = trainer.unwrapped_model.output_pp_rank
    sequence_length = trainer.config.tokens.sequence_length
    
    # Override micro_batch_size if provided as command line argument
    if args.micro_batch_size is not None:
        logger.info(f"Overriding micro batch size from {trainer.config.tokens.micro_batch_size} to {args.micro_batch_size}")
        trainer.config.tokens.micro_batch_size = args.micro_batch_size
    
    micro_batch_size = trainer.config.tokens.micro_batch_size
    
    # Set output directory if provided
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        trainer.config.checkpoints.checkpoint_dir = args.output_dir
    
    # Load tokenizer
    try:
        logger.info(f"Loading tokenizer from {args.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        sys.exit(1)
    
    # Check for parquet files
    parquet_files = find_parquet_files(args.data_dir)
    logger.info(f"Found {len(parquet_files)} parquet files in {args.data_dir}")
    
    # Load datasets and create dataloaders
    current_rank = dist.get_rank(trainer.parallel_context.world_pg)
    world_size = dist.get_world_size(trainer.parallel_context.world_pg)
    
    # Only load data on rank 0 if distributed
    if current_rank == 0 or world_size == 1:
        logger.info("Loading datasets from parquet files")
        
        # Load parquet dataset
        dataset_dict = get_datasets(args.data_dir, splits="train")
        
        # Find the text column name in the dataset
        text_column_name = None
        for column in dataset_dict["train"].column_names:
            if column.lower() in ["text", "content", "document", "input_text"]:
                text_column_name = column
                break
                
        if text_column_name is None:
            logger.warning(f"No text column found in dataset. Available columns: {dataset_dict['train'].column_names}")
            text_column_name = dataset_dict["train"].column_names[0]
            logger.info(f"Using column '{text_column_name}' as text column")
        
        # Process the dataset for causal language modeling
        logger.info(f"Processing dataset for causal language modeling with sequence length {sequence_length}")
        train_dataset = clm_process(
            raw_dataset=dataset_dict["train"],
            tokenizer=tokenizer,
            text_column_name=text_column_name,
            dataset_processing_num_proc_per_process=args.num_workers,
            dataset_overwrite_cache=False,
            sequence_length=sequence_length
        )
        
        # Create the dataloader
        logger.info(f"Creating dataloader with batch size {micro_batch_size}")
        train_dataloader = get_train_dataloader(
            train_dataset=train_dataset,
            sequence_length=sequence_length,
            parallel_context=trainer.parallel_context,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            micro_batch_size=micro_batch_size,
            consumed_train_samples=trainer.consumed_train_samples,
            dataloader_num_workers=args.num_workers,
            seed_worker=trainer.config.general.seed
        )
    else:
        # Create an empty dataloader for other ranks
        logger.info(f"Creating empty dataloader for rank {current_rank}")
        train_dataloader = get_train_dataloader(
            train_dataset=None,
            sequence_length=sequence_length,
            parallel_context=trainer.parallel_context,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            micro_batch_size=micro_batch_size,
            consumed_train_samples=trainer.consumed_train_samples,
            dataloader_num_workers=0,
            seed_worker=trainer.config.general.seed
        )
    
    # Start training
    logger.info("Starting training")
    trainer.train({"train": train_dataloader})
    
    logger.info("Training completed!")
    
if __name__ == "__main__":
    main()
