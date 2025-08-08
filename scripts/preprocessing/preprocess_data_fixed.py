#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/preprocessing/preprocess_data_fixed.py

"""
Data preprocessing script for Infini-Llama training (fixed version).

This script handles the preprocessing of datasets for Infini-Llama training,
including tokenization, chunking, and saving processed data to disk.
It uses GPU acceleration for faster processing when available.

Usage:
```
python preprocess_data_fixed.py --config-file custom_infini_config_gpu.yaml --output-dir processed_data
python preprocess_data_fixed.py --config-file custom_infini_config_gpu.yaml --output-dir processed_data --gpu-id 1
python preprocess_data_fixed.py --config-file custom_infini_config_gpu.yaml --output-dir processed_data --no-gpu
```
"""
import os
import sys
import time
import torch
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add the project root to Python path to access the nanotron module
root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))

# Import project modules
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

try:
    # Import required modules
    from nanotron import constants
    constants.CONFIG = Config()
    
    from nanotron.config import (
        DataArgs,
        DatasetStageArgs,
        PretrainDatasetsArgs,
    )
    from nanotron.dataloader import (
        clm_process,
        get_datasets,
    )
    from nanotron import logging
    from nanotron.utils import main_rank_first
    from nanotron.parallel import ParallelContext
    from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
    from nanotron.trainer import DistributedTrainer

    # Import GPU-accelerated data processing if available
    try:
        from src.nanotron.gpu_dataloader import gpu_clm_process
        HAS_GPU_DATALOADER = True
    except ImportError:
        HAS_GPU_DATALOADER = False
        print("GPU data processing not available, falling back to CPU processing.")
    
    # Import transformers
    try:
        from huggingface_hub import __version__ as hf_hub_version
        from transformers import AutoTokenizer
        from transformers import __version__ as tf_version
        HAS_TRANSFORMERS = True
    except ImportError:
        HAS_TRANSFORMERS = False
        print("Transformers library not available. This is required for data preprocessing.")
        
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this script from the project root or add the project root to PYTHONPATH.")
    sys.exit(1)

logger = logging.get_logger(__name__)


def check_gpu_availability(gpu_id: Optional[int] = None) -> bool:
    """Check if the specified GPU is available."""
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU for processing.")
        return False
    
    if gpu_id is not None:
        gpu_count = torch.cuda.device_count()
        if gpu_id >= gpu_count:
            print(f"GPU {gpu_id} not available. Total GPUs: {gpu_count}")
            print(f"Using GPU 0 instead.")
            return 0 < gpu_count
    
    return torch.cuda.is_available()


def setup_environment(gpu_id: Optional[int] = None, use_gpu: bool = True):
    """Set up the environment for data preprocessing."""
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    # Set up GPU environment if requested and available
    if use_gpu and gpu_id is not None:
        if check_gpu_availability(gpu_id):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"Using GPU {gpu_id} for data preprocessing")
        else:
            print("GPU not available. Using CPU for processing.")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif not use_gpu:
        print("GPU processing disabled. Using CPU for processing.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def load_config(config_file: str) -> DistributedTrainer:
    """Load configuration from file."""
    print(f"Loading configuration from {config_file}")
    try:
        # We create a trainer instance to access configuration but won't use it for training
        trainer = DistributedTrainer(config_file)
        return trainer
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def preprocess_dataset(
    trainer: DistributedTrainer,
    config_file_path: str,  # Added parameter for config file path
    output_dir: str,
    use_gpu: bool = True,
    gpu_device: str = "cuda:0",
    batch_size: int = 2048
) -> str:
    """
    Preprocess the dataset and save to disk.
    
    Args:
        trainer: The trainer instance with configuration
        config_file_path: Path to the config file used to create the trainer
        output_dir: Directory to save preprocessed data
        use_gpu: Whether to use GPU for preprocessing
        gpu_device: GPU device to use (e.g. "cuda:0")
        batch_size: Batch size for GPU processing
        
    Returns:
        Path to the preprocessed dataset
    """
    if not HAS_TRANSFORMERS:
        print("Error: Transformers library is required for data preprocessing.")
        sys.exit(1)
    
    start_time = time.time()
    print("Starting data preprocessing...")
    
    # Get the dataset configuration from the first stage
    sorted_stages = sorted(trainer.config.data_stages, key=lambda stage: stage.start_training_step)
    if not sorted_stages:
        print("Error: No data stages found in configuration")
        sys.exit(1)
    
    data_stage = sorted_stages[0]
    data_args = data_stage.data
    
    if not isinstance(data_args.dataset, PretrainDatasetsArgs):
        print("Error: Only PretrainDatasetsArgs are supported for preprocessing")
        sys.exit(1)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"preprocessed_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    
    # Save metadata about the preprocessing
    metadata = {
        "timestamp": timestamp,
        "config_file": config_file_path,  # Use passed config file path instead
        "sequence_length": trainer.sequence_length,
        "dataset_source": data_args.dataset.hf_dataset_or_datasets,
        "tokenizer": trainer.config.tokenizer.tokenizer_name_or_path,
        "use_gpu": use_gpu,
        "gpu_device": gpu_device if use_gpu else "cpu"
    }
    
    with open(os.path.join(output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created output directory: {output_path}")
    print(f"Using tokenizer: {trainer.config.tokenizer.tokenizer_name_or_path}")
    
    # We need a dummy parallel context for some functions
    parallel_context = ParallelContext(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        expert_parallel_size=1,
    )
    
    # Create a dummy context for main_rank_first
    with main_rank_first(parallel_context.world_pg):
        # Load dataset
        print(f"Loading raw dataset from {data_args.dataset.hf_dataset_or_datasets}")
        raw_dataset = get_datasets(
            hf_dataset_or_datasets=data_args.dataset.hf_dataset_or_datasets,
            hf_dataset_config_name=data_args.dataset.hf_dataset_config_name,
            splits=data_args.dataset.hf_dataset_splits,
        )["train"]
        
        print(f"Raw dataset loaded with {len(raw_dataset)} examples")
        
        # Load tokenizer
        print(f"Loading tokenizer {trainer.config.tokenizer.tokenizer_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            trainer.config.tokenizer.tokenizer_name_or_path,
            revision=trainer.config.tokenizer.tokenizer_revision,
        )
        
        # Process the dataset
        if use_gpu and HAS_GPU_DATALOADER:
            print(f"Using GPU for text processing with device {gpu_device}")
            train_dataset = gpu_clm_process(
                raw_dataset=raw_dataset,
                tokenizer=tokenizer,
                text_column_name=data_args.dataset.text_column_name,
                dataset_processing_num_proc_per_process=data_args.dataset.dataset_processing_num_proc_per_process,
                dataset_overwrite_cache=data_args.dataset.dataset_overwrite_cache,
                sequence_length=trainer.sequence_length,
                device=gpu_device,
                batch_size=batch_size,
            )
        else:
            print("Using CPU for text processing")
            train_dataset = clm_process(
                raw_dataset=raw_dataset,
                tokenizer=tokenizer,
                text_column_name=data_args.dataset.text_column_name,
                dataset_processing_num_proc_per_process=data_args.dataset.dataset_processing_num_proc_per_process,
                dataset_overwrite_cache=data_args.dataset.dataset_overwrite_cache,
                sequence_length=trainer.sequence_length,
            )
        
        # Save the processed dataset
        dataset_path = os.path.join(output_path, "processed_dataset")
        print(f"Saving processed dataset to {dataset_path}")
        train_dataset.save_to_disk(dataset_path)
        
        # Save a reference file to easily find the latest preprocessing
        latest_path = os.path.join(output_dir, "latest_preprocessed")
        with open(latest_path, "w") as f:
            f.write(output_path)
    
    elapsed_time = time.time() - start_time
    print(f"Data preprocessing completed in {elapsed_time:.2f} seconds")
    print(f"Processed dataset saved to {dataset_path}")
    print(f"Total examples after preprocessing: {len(train_dataset)}")
    
    return output_path


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Data preprocessing for Infini-Llama training")
    parser.add_argument("--config-file", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--output-dir", type=str, default="preprocessed_data", help="Directory to save preprocessed data")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use for preprocessing (default: 0)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration for preprocessing")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for GPU processing (default: 2048)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = get_args()
    
    # Configure logging level
    if args.verbose:
        os.environ["NANOTRON_LOG_LEVEL"] = "debug"
        print("Verbose logging enabled")
    
    # Set up environment
    setup_environment(gpu_id=args.gpu_id, use_gpu=not args.no_gpu)
    
    # Load configuration
    trainer = load_config(args.config_file)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Preprocess dataset
    gpu_device = f"cuda:{args.gpu_id}" if not args.no_gpu and torch.cuda.is_available() else "cpu"
    output_path = preprocess_dataset(
        trainer=trainer,
        config_file_path=args.config_file,  # Pass the config file path to the function
        output_dir=args.output_dir,
        use_gpu=not args.no_gpu,
        gpu_device=gpu_device,
        batch_size=args.batch_size
    )
    
    print("\nPreprocessing completed successfully!")
    print(f"Output directory: {output_path}")
    print("You can now train using the preprocessed data with:")
    print(f"  python scripts/training/train_with_preprocessed.py --config-file {args.config_file} --data-dir {output_path}")


if __name__ == "__main__":
    main()
