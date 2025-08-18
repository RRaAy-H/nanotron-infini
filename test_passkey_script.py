#!/usr/bin/env python3
"""
Test the passkey finetuning script with debugging to understand the dataloader issue.
This script creates a controlled environment to test the actual run_passkey_finetune_300m.sh workflow.

Usage: python test_passkey_script.py [checkpoint_path]
"""

import sys
import os
import subprocess
import tempfile
import shutil
from pathlib import Path

def create_fake_checkpoint(path):
    """Create a fake checkpoint directory"""
    print(f"Creating fake checkpoint at {path}...")
    os.makedirs(path, exist_ok=True)
    
    # Create minimal required files
    files = [
        "model_weights.bin",
        "config.json", 
        "optimizer_states.bin",
        "lr_scheduler.bin"
    ]
    
    for file in files:
        with open(os.path.join(path, file), 'w') as f:
            f.write(f"fake {file} content")
    
    print(f"✓ Created fake checkpoint with {len(files)} files")

def create_minimal_dataset():
    """Create minimal dataset for testing"""
    print("Creating minimal test dataset...")
    
    try:
        import pandas as pd
        
        # Create minimal passkey-style data
        data = []
        for i in range(10):
            # Simple passkey format
            prompt = f"Find the passkey in the following text: {'x' * 1000} The passkey is: KEY{i:03d} {'x' * 1000}"
            data.append({"prompt": prompt})
        
        df = pd.DataFrame(data)
        df.to_parquet("test_passkey_data.parquet")
        print("✓ Created test_passkey_data.parquet with 10 examples")
        return "test_passkey_data.parquet"
        
    except ImportError:
        print("✗ pandas not available, creating empty placeholder")
        with open("test_passkey_data.parquet", 'w') as f:
            f.write("")
        return "test_passkey_data.parquet"

def modify_config_for_testing(original_config, checkpoint_path, dataset_path):
    """Modify config for testing"""
    print("Creating test configuration...")
    
    # Read original config
    import yaml
    with open(original_config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for testing
    config['general']['step'] = 30000  # Simulate resume
    config['general']['project'] = 'debug_test'
    config['general']['run'] = 'debug_dataloader_test'
    
    # Make model smaller for faster testing
    config['model']['model_config']['hidden_size'] = 64
    config['model']['model_config']['intermediate_size'] = 256
    config['model']['model_config']['num_hidden_layers'] = 2
    config['model']['model_config']['num_attention_heads'] = 2
    config['model']['model_config']['num_key_value_heads'] = 2
    config['model']['model_config']['vocab_size'] = 1000
    
    # Reduce training steps
    config['tokens']['train_steps'] = 30002  # Just 2 steps
    config['tokens']['micro_batch_size'] = 1
    config['tokens']['sequence_length'] = 1024  # Smaller sequence
    
    # Use single GPU
    config['parallelism']['dp'] = 1
    
    # Update paths
    config['checkpoints']['resume_checkpoint_path'] = checkpoint_path
    config['checkpoints']['checkpoints_path'] = './debug_test_checkpoints'
    config['data_stages'][0]['data']['data_files'] = dataset_path
    
    # Increase logging
    config['infini_attention']['logging_interval'] = 1
    config['logging']['iteration_step_info_interval'] = 1
    
    # Save test config
    test_config = "debug_passkey_test.yaml"
    with open(test_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ Created test config: {test_config}")
    return test_config

def run_training_test(config_file):
    """Run training test to reproduce the error"""
    print(f"\nRunning training test with {config_file}...")
    
    # Run with debugging
    cmd = f"python3 debug_train.py --config-file {config_file}"
    
    print(f"Command: {cmd}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        print(f"\nReturn code: {result.returncode}")
        
        # Check for the specific error
        if "TypeError: 'NoneType' object is not an iterator" in result.stderr or "TypeError: 'NoneType' object is not an iterator" in result.stdout:
            print("\n✓ REPRODUCED: TypeError 'NoneType' object is not an iterator")
            return True, result.stdout, result.stderr
        elif "CRITICAL - No dataloader found" in result.stdout:
            print("\n✓ IDENTIFIED: Dataloader is None (would cause TypeError)")
            return True, result.stdout, result.stderr
        elif result.returncode != 0:
            print(f"\n? Different error occurred (return code {result.returncode})")
            return False, result.stdout, result.stderr
        else:
            print("\n✗ No error reproduced")
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        print("\n⚠ Test timed out (>120s)")
        return False, "", "TIMEOUT"
    except Exception as e:
        print(f"\n✗ Test execution failed: {e}")
        return False, "", str(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", nargs='?', default=None,
                       help="Checkpoint path (will create fake one if not provided)")
    parser.add_argument("--original-config", type=str, 
                       default="passkey_finetune_300m_simple_config.yaml",
                       help="Original config file to base test on")
    args = parser.parse_args()
    
    print("PASSKEY SCRIPT DEBUG TEST")
    print("=" * 60)
    print(f"Original config: {args.original_config}")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 60)
    
    try:
        # Step 1: Set up checkpoint
        if args.checkpoint_path and os.path.exists(args.checkpoint_path):
            checkpoint_path = args.checkpoint_path
            print(f"Using existing checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = "./fake_test_checkpoint_30000"
            create_fake_checkpoint(checkpoint_path)
        
        # Step 2: Create dataset
        dataset_path = create_minimal_dataset()
        
        # Step 3: Create test config
        test_config = modify_config_for_testing(args.original_config, checkpoint_path, dataset_path)
        
        # Step 4: Run the test
        success, stdout, stderr = run_training_test(test_config)
        
        # Step 5: Save results
        timestamp = os.popen("date +%Y%m%d_%H%M%S").read().strip()
        
        with open(f"passkey_debug_output_{timestamp}.log", 'w') as f:
            f.write("PASSKEY SCRIPT DEBUG TEST RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Config: {test_config}\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Dataset: {dataset_path}\n")
            f.write(f"Success: {success}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("STDOUT:\n")
            f.write(stdout)
            f.write("\n\nSTDERR:\n")
            f.write(stderr)
        
        print("\n" + "=" * 60)
        print("PASSKEY DEBUG TEST COMPLETED")
        print("=" * 60)
        
        if success:
            print("✓ Test completed - error reproduced")
        else:
            print("✗ Test completed - error may not be reproduced")
        
        print(f"Results saved to: passkey_debug_output_{timestamp}.log")
        print(f"Test config saved to: {test_config}")
        
        print("\nCleanup commands:")
        print(f"  rm -rf {checkpoint_path}")
        print(f"  rm -f {dataset_path}")
        print(f"  rm -f {test_config}")
        print(f"  rm -rf debug_test_checkpoints")
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()