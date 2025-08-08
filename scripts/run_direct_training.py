#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/run_direct_training.py

"""
Direct training script for Infini-Llama that uses DistributedTrainer directly.
This script avoids the import issues with the other training scripts.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add project root and src directories to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Set up argument parsing
parser = argparse.ArgumentParser(description="Train Infini-Llama model using DistributedTrainer")
parser.add_argument("--config-file", type=str, required=True, help="Path to the config file")
parser.add_argument("--data-dir", type=str, required=True, help="Path to preprocessed data directory")
parser.add_argument("--disable-infini-attn", action="store_true", help="Disable Infini-Attention for baseline model")
parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use (default: 0)")
parser.add_argument("--tensorboard-dir", type=str, default=None, help="Directory for TensorBoard logs")
parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
parser.add_argument("--use-gpu-dataloader", action="store_true", help="Use GPU-accelerated dataloader")
args = parser.parse_args()

# Configure the environment
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# Configure logging level
if args.verbose:
    os.environ["NANOTRON_LOG_LEVEL"] = "debug"
    print("Verbose logging enabled")

# Import after setting environment variables
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

# Set up the Infini-Attention configuration
from nanotron import constants
constants.CONFIG = Config()

# If disabling Infini-Attention, update configuration
if args.disable_infini_attn:
    print("Disabling Infini-Attention (running baseline model)")
    constants.CONFIG.infini_attention.turn_on_memory = False

# Import trainer
from nanotron.trainer import DistributedTrainer
from datasets import load_from_disk
from transformers import AutoTokenizer

def main():
    print(f"Creating trainer with config: {args.config_file}")
    # Create trainer
    trainer = DistributedTrainer(args.config_file)
    
    print(f"Loading dataset from {args.data_dir}")
    # Load dataset
    dataset_path = Path(args.data_dir) / "processed_dataset"
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)
    
    train_dataset = load_from_disk(str(dataset_path))
    
    # Load tokenizer
    tokenizer_path = Path(args.data_dir) / "tokenizer"
    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    else:
        # Try to get tokenizer from metadata or use default
        metadata_path = Path(args.data_dir) / "metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            tokenizer_name = metadata.get("tokenizer", "meta-llama/Llama-2-7b-hf")
        else:
            tokenizer_name = "meta-llama/Llama-2-7b-hf"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Set up data collator
    from nanotron.dataloader import DataCollatorForCLM, get_train_dataloader
    from nanotron.parallel import ParallelContext

    # Create a simple parallel context for single GPU training
    parallel_context = ParallelContext(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        expert_parallel_size=1
    )
    
    # Initialize data collator with proper parameters
    data_collator = DataCollatorForCLM(
        sequence_length=trainer.config.tokens.sequence_length,
        input_pp_rank=0,
        output_pp_rank=0,
        parallel_context=parallel_context
    )
    
    # Create dataloader
    train_loader = get_train_dataloader(
        train_dataset=train_dataset,
        batch_size=trainer.config.tokens.micro_batch_size,
        collate_fn=data_collator,
    )
    
    # Set up TensorBoard if requested
    if args.tensorboard_dir:
        os.makedirs(args.tensorboard_dir, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)
            print(f"TensorBoard logging enabled at {args.tensorboard_dir}")
        except ImportError:
            print("TensorBoard not available. Install torch.utils.tensorboard for logging.")
            tb_writer = None
    else:
        tb_writer = None
    
    # Train model
    print("Starting training")
    try:
        trainer.train(
            train_dataloader=train_loader,
            tokenizer=tokenizer,
            device=f"cuda:0" if torch.cuda.is_available() else "cpu",
        )
        print("Training completed successfully!")
    
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        if tb_writer:
            tb_writer.close()
    
    # Save final model
    if trainer.is_main_process:
        model_type = "baseline" if args.disable_infini_attn else "infini"
        from datetime import datetime
        output_dir = Path("models") / f"{model_type}_llama_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving final model to {output_dir}")
        trainer.save_model(output_dir)

if __name__ == "__main__":
    main()
