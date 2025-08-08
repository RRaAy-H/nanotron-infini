#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/run_infini_llama.py

"""
Main entry point for Infini-Llama training pipeline.

This script provides a unified interface to preprocess data and train the model,
either as a single pipeline or as separate steps.

Usage:
```
# Run the full pipeline (preprocess + train):
python run_infini_llama.py --config-file custom_infini_config_gpu.yaml --output-dir processed_data

# Preprocess data only:
python run_infini_llama.py --config-file custom_infini_config_gpu.yaml --output-dir processed_data --preprocess-only

# Train only (using previously preprocessed data):
python run_infini_llama.py --config-file custom_infini_config_gpu.yaml --data-dir processed_data --train-only

# With TensorBoard integration:
python run_infini_llama.py --config-file custom_infini_config_gpu.yaml --output-dir processed_data --tensorboard-dir tensorboard_logs
```
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Infini-Llama Training Pipeline")
    parser.add_argument("--config-file", type=str, required=True,
                        help="Path to the configuration file")
    parser.add_argument("--output-dir", type=str, default="processed_data",
                        help="Directory to save preprocessed data (default: processed_data)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing preprocessed data (defaults to --output-dir if not specified)")
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Only run preprocessing step")
    parser.add_argument("--train-only", action="store_true",
                        help="Only run training step (requires preprocessed data)")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Use CPU only for training")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force CPU usage even if GPU is available")
    parser.add_argument("--disable-flash-attn", action="store_true",
                        help="Disable Flash Attention even if available")
    parser.add_argument("--tensorboard-dir", type=str, default=None,
                        help="Directory for TensorBoard logs (disabled if not specified)")
    parser.add_argument("--gpu-device", type=str, default="0",
                        help="GPU device ID to use for training (default: 0)")
    parser.add_argument("--use-gpu-dataloader", action="store_true",
                        help="Use GPU-accelerated data processing for faster text chunking")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set data directory if not explicitly specified
    if args.data_dir is None:
        args.data_dir = args.output_dir
    
    # Set up paths to scripts
    script_dir = Path(__file__).resolve().parent
    preprocess_script = script_dir / "preprocessing" / "preprocess_data.py"
    train_script = script_dir / "training" / "train_infini_llama.py"
    
    # Validate script paths
    for script in [preprocess_script, train_script]:
        if not script.exists():
            print(f"Error: Script {script} does not exist.")
            sys.exit(1)
    
    # Run preprocessing if requested
    if not args.train_only:
        print(f"Running preprocessing step...")
        preprocess_cmd = [
            sys.executable,
            str(preprocess_script),
            "--config-file", args.config_file,
            "--output-dir", args.output_dir,
        ]
        
        # Add GPU options
        if args.cpu_only:
            preprocess_cmd.extend(["--no-gpu"])
        elif args.gpu_device:
            preprocess_cmd.extend(["--gpu-id", args.gpu_device])
            
        # Execute preprocessing
        print(f"Executing: {' '.join(preprocess_cmd)}")
        preprocess_result = subprocess.run(preprocess_cmd)
        
        if preprocess_result.returncode != 0:
            print("Error during preprocessing. Aborting.")
            sys.exit(1)
            
        print("Preprocessing completed successfully.")
        
    # Run training if requested
    if not args.preprocess_only:
        print(f"Running training step...")
        train_cmd = [
            sys.executable,
            str(train_script),
            "--config-file", args.config_file,
            "--data-dir", args.data_dir,
        ]
        
        # Add options
        if args.cpu_only:
            train_cmd.extend(["--cpu-only"])
        if args.force_cpu:
            train_cmd.extend(["--force-cpu"])
        if args.disable_flash_attn:
            train_cmd.extend(["--disable-flash-attn"])
        if args.tensorboard_dir:
            train_cmd.extend(["--tensorboard-dir", args.tensorboard_dir])
        if args.gpu_device:
            train_cmd.extend(["--gpu-device", f"cuda:{args.gpu_device}"])
        if args.use_gpu_dataloader:
            train_cmd.extend(["--use-gpu-dataloader"])
        if args.seed != 42:
            train_cmd.extend(["--seed", str(args.seed)])
            
        # Execute training
        print(f"Executing: {' '.join(train_cmd)}")
        train_result = subprocess.run(train_cmd)
        
        if train_result.returncode != 0:
            print("Error during training. Aborting.")
            sys.exit(1)
            
        print("Training completed successfully.")
    
    print("All tasks completed.")

if __name__ == "__main__":
    main()
