#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/verify_config.py

"""
Verify configuration for Infini-Llama training.
This script checks that all paths and configurations are correct.

Usage:
```
python verify_config.py --config-file custom_infini_config_gpu.yaml
```
"""
import argparse
import sys
import os
import yaml
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description="Verify configuration for Infini-Llama training")
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML config file")
    return parser.parse_args()

def main():
    args = get_args()
    config_file = args.config_file
    
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)
    
    # Load config
    with open(config_file, "r") as f:
        try:
            config = yaml.safe_load(f)
            print(f"Successfully loaded config file: {config_file}")
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            sys.exit(1)
    
    # Check checkpoints directory
    checkpoints_path = config.get("checkpoints", {}).get("checkpoints_path", "")
    if checkpoints_path:
        # Replace environment variables if any
        checkpoints_path = os.path.expandvars(checkpoints_path)
        checkpoints_dir = Path(checkpoints_path)
        print(f"Checkpoints directory: {checkpoints_path}")
        if not checkpoints_dir.exists():
            print(f"  - Creating checkpoints directory: {checkpoints_path}")
            try:
                os.makedirs(checkpoints_dir, exist_ok=True)
            except Exception as e:
                print(f"  - Error creating checkpoints directory: {e}")
        else:
            print(f"  - Checkpoints directory exists")
    else:
        print("Warning: No checkpoints path specified in config")
    
    # Check dataset path
    dataset_path = None
    for stage in config.get("data_stages", []):
        if stage.get("data", {}).get("dataset", {}).get("hf_dataset_or_datasets"):
            dataset_path = stage["data"]["dataset"]["hf_dataset_or_datasets"]
            # Replace environment variables if any
            dataset_path = os.path.expandvars(dataset_path)
            break
    
    if dataset_path:
        print(f"Dataset path: {dataset_path}")
        if os.path.exists(dataset_path):
            print(f"  - Dataset path exists")
        else:
            print(f"  - Error: Dataset path does not exist")
            print(f"  - Please ensure the dataset is available at: {dataset_path}")
    else:
        print("Warning: No dataset path specified in config")
    
    # Check tokenizer
    tokenizer = config.get("tokenizer", {}).get("tokenizer_name_or_path", "")
    if tokenizer:
        print(f"Tokenizer: {tokenizer}")
        if tokenizer.startswith("/"):
            # It's a local path
            if os.path.exists(tokenizer):
                print(f"  - Tokenizer path exists")
            else:
                print(f"  - Warning: Local tokenizer path does not exist: {tokenizer}")
        else:
            # It's a HuggingFace model ID
            print(f"  - Will attempt to download tokenizer from HuggingFace Hub")
            print(f"  - Note: Some models like 'meta-llama/Llama-2-7b-hf' require authentication")
    else:
        print("Warning: No tokenizer specified in config")
    
    # Check infini_attention parameters
    infini_attention = config.get("infini_attention", {})
    if infini_attention:
        print("Infini-Attention configuration:")
        segment_length = infini_attention.get("segment_length")
        turn_on_memory = infini_attention.get("turn_on_memory")
        balance_init_type = infini_attention.get("balance_init_type")
        balance_act_type = infini_attention.get("balance_act_type")
        balance_factor_lr = infini_attention.get("balance_factor_lr")
        
        print(f"  - segment_length: {segment_length}")
        print(f"  - turn_on_memory: {turn_on_memory}")
        print(f"  - balance_init_type: {balance_init_type}")
        print(f"  - balance_act_type: {balance_act_type}")
        print(f"  - balance_factor_lr: {balance_factor_lr}")
        
        if balance_factor_lr is None:
            print("  - Warning: Missing required parameter 'balance_factor_lr'")
    else:
        print("Warning: No infini_attention section in config")
    
    # Check training parameters
    tokens = config.get("tokens", {})
    if tokens:
        print("Training parameters:")
        print(f"  - micro_batch_size: {tokens.get('micro_batch_size')}")
        print(f"  - sequence_length: {tokens.get('sequence_length')}")
        print(f"  - train_steps: {tokens.get('train_steps')}")
        print(f"  - val_check_interval: {tokens.get('val_check_interval')}")
    
    print("\nConfiguration verification complete.")
    
if __name__ == "__main__":
    main()
