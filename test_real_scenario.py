#!/usr/bin/env python3
"""
Test the real scenario that's causing the error.
This script simulates running the actual finetuning but with controlled conditions.

Usage: python test_real_scenario.py [checkpoint_path]
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path

def setup_test_environment():
    """Set up a minimal test environment"""
    print("Setting up test environment...")
    
    # Create fake checkpoint directory if needed
    fake_checkpoint = "./fake_checkpoint_30000"
    if not os.path.exists(fake_checkpoint):
        os.makedirs(fake_checkpoint, exist_ok=True)
        # Create minimal checkpoint files
        with open(f"{fake_checkpoint}/model_weights.bin", 'w') as f:
            f.write("fake checkpoint data")
        with open(f"{fake_checkpoint}/config.json", 'w') as f:
            f.write('{"fake": "config"}')
        print(f"✓ Created fake checkpoint at {fake_checkpoint}")
    
    # Create fake dataset if needed  
    fake_dataset = "./fake_passkey_data.parquet"
    if not os.path.exists(fake_dataset):
        # Create a minimal parquet file using pandas if available
        try:
            import pandas as pd
            df = pd.DataFrame({
                'prompt': ['fake prompt text ' * 100] * 10  # 10 fake entries
            })
            df.to_parquet(fake_dataset)
            print(f"✓ Created fake dataset at {fake_dataset}")
        except ImportError:
            # Create an empty file as placeholder
            with open(fake_dataset, 'w') as f:
                f.write("")
            print(f"✓ Created placeholder dataset at {fake_dataset}")
    
    return fake_checkpoint, fake_dataset

def create_debug_config(checkpoint_path, dataset_path):
    """Create debug configuration that reproduces the issue"""
    config_content = f"""general:
  benchmark_csv_path: null
  consumed_train_samples: null
  ignore_sanity_checks: true
  project: debug_real_test
  run: debug_real_scenario
  seed: 42
  step: 30000  # THIS IS KEY: Resume from step 30000

# Infini-Attention Configuration
infini_attention:
  segment_length: 512
  turn_on_memory: true
  balance_factor_lr: 0.01
  balance_act_type: hard_sigmoid
  balance_init_type: zeros
  logging: true
  logging_interval: 1
  log_grad: false
  log_segment_acts: false
  balance_factor_weight_decay: 0.0

# Model Configuration (minimal for testing)
model:
  ddp_bucket_cap_mb: 25
  dtype: bfloat16
  init_method:
    std: 0.03125
  make_vocab_size_divisible_by: 1
  model_config:
    bos_token_id: 1
    eos_token_id: 2
    hidden_act: silu
    initializer_range: 0.02
    hidden_size: 64
    intermediate_size: 256
    num_hidden_layers: 2
    is_llama_config: true
    max_position_embeddings: 2048
    num_attention_heads: 2
    num_key_value_heads: 2
    pad_token_id: null
    pretraining_tp: 1
    rms_norm_eps: 1.0e-05
    rope_scaling: null
    tie_word_embeddings: false
    use_cache: true
    vocab_size: 1000

# Optimizer Configuration
optimizer:
  accumulate_grad_in_fp32: true
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_eps: 1.0e-08
  torch_adam_is_fused: false
  clip_grad: 1.0
  learning_rate_scheduler:
    learning_rate: 0.0001
    lr_decay_starting_step: null
    lr_decay_steps: null
    lr_decay_style: cosine
    lr_warmup_steps: 1
    lr_warmup_style: linear
    min_decay_lr: 0.00001
  weight_decay: 0.01
  zero_stage: 0

# Parallelism Configuration
parallelism:
  dp: 1
  tp: 1
  expert_parallel_size: 1
  pp: 1
  pp_engine: 1f1b
  tp_linear_async_communication: false
  tp_mode: ALL_REDUCE

# Tokenizer Configuration
tokenizer:
  tokenizer_max_length: null
  tokenizer_name_or_path: lvwerra/the-tokenizer-v1
  tokenizer_revision: null

# Training Tokens Configuration
tokens:
  batch_accumulation_per_replica: 1
  limit_test_batches: 0
  limit_val_batches: 0
  micro_batch_size: 1
  sequence_length: 1024
  train_steps: 30002  # Train from 30001 to 30002
  val_check_interval: -1

# Data Configuration - THE CRITICAL PART
data_stages:
  - name: "Test Passkey Finetune"
    start_training_step: 1  # Stage starts at 1
    data:
      dataset: null  # Use dummy data to avoid file dependencies
      num_loading_workers: 1
      seed: 42

# Checkpoints Configuration
checkpoints:
  checkpoint_interval: 1
  checkpoints_path: ./debug_real_checkpoints
  checkpoints_path_is_shared_file_system: false
  resume_checkpoint_path: {checkpoint_path}
  save_initial_state: false

# Logging Configuration
logging:
  iteration_step_info_interval: 1
  log_level: info
  log_level_replica: info

# Profiler (disabled)
profiler: null"""
    
    config_file = "debug_real_scenario.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✓ Created debug config: {config_file}")
    return config_file

def run_real_scenario_test(config_file):
    """Run the real scenario test"""
    print("\nTesting real scenario reproduction...")
    
    # Test with single process first (no torchrun)
    print("\n--- Test 1: Single process (no distributed) ---")
    cmd = f"python3 run_train.py --config-file {config_file}"
    success, stdout, stderr = run_command_with_output(cmd)
    
    if "TypeError: 'NoneType' object is not an iterator" in stderr:
        print("✓ REPRODUCED: TypeError with single process")
        return True
    elif success:
        print("✗ No error with single process - may need distributed setup")
    else:
        print("✗ Different error occurred")
        print(f"Error output: {stderr}")
    
    return False

def run_command_with_output(cmd):
    """Run command and return success, stdout, stderr"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT"
    except Exception as e:
        return False, "", str(e)

def cleanup_test_files():
    """Clean up test files"""
    print("\nCleaning up test files...")
    files_to_remove = [
        "debug_real_scenario.yaml",
        "fake_passkey_data.parquet",
        "dataloader_debug.json"
    ]
    
    dirs_to_remove = [
        "fake_checkpoint_30000",
        "debug_real_checkpoints",
        "debug_test_checkpoints"
    ]
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"  Removed {file}")
    
    for dir in dirs_to_remove:
        if os.path.exists(dir):
            shutil.rmtree(dir)
            print(f"  Removed {dir}/")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, 
                       default="passkey_finetune_300m_simple_config.yaml",
                       help="Path to the original YAML config file")
    parser.add_argument("checkpoint_path", nargs='?', 
                       default="./fake_checkpoint_30000",
                       help="Checkpoint path to use for testing")
    parser.add_argument("--cleanup", action="store_true", help="Clean up test files and exit")
    args = parser.parse_args()
    
    if args.cleanup:
        cleanup_test_files()
        return
    
    print("REAL SCENARIO DEBUG TEST")
    print("=" * 60)
    print(f"Original config: {args.config_file}")
    print(f"Checkpoint path: {args.checkpoint_path}")
    print("=" * 60)
    
    try:
        # Step 1: Set up test environment
        fake_checkpoint, fake_dataset = setup_test_environment()
        checkpoint_to_use = args.checkpoint_path if os.path.exists(args.checkpoint_path) else fake_checkpoint
        
        # Step 2: Create debug config
        debug_config = create_debug_config(checkpoint_to_use, fake_dataset)
        
        # Step 3: Run debugging tests
        results = []
        
        # Test with comprehensive debugger
        print("\n" + "="*60)
        print("RUNNING COMPREHENSIVE DEBUG")
        print("="*60)
        success, stdout, stderr = run_command_with_output(
            f"python3 debug_dataloader_comprehensive.py --config-file {debug_config} --mode reproduce"
        )
        results.append(("Comprehensive Debug", success, stdout, stderr))
        
        # Test with state inspector
        print("\n" + "="*60)
        print("RUNNING STATE INSPECTION")
        print("="*60)
        success, stdout, stderr = run_command_with_output(
            f"python3 inspect_dataloader_state.py --config-file {debug_config}"
        )
        results.append(("State Inspection", success, stdout, stderr))
        
        # Test actual reproduction
        print("\n" + "="*60)
        print("RUNNING ERROR REPRODUCTION")
        print("="*60)
        success, stdout, stderr = run_command_with_output(
            f"python3 reproduce_error.py --config-file {debug_config}"
        )
        results.append(("Error Reproduction", success, stdout, stderr))
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"real_scenario_debug_{timestamp}.txt"
        generate_test_report(results, report_file)
        
        print("\n" + "=" * 60)
        print("REAL SCENARIO TEST COMPLETED")
        print("=" * 60)
        
        error_reproduced = any("TypeError: 'NoneType' object is not an iterator" in stderr 
                              for _, _, _, stderr in results)
        
        if error_reproduced:
            print("✓ Successfully reproduced the TypeError")
            print("✓ Debug information captured in report")
        else:
            print("✗ Could not reproduce the TypeError")
            print("This may indicate the issue is environment-specific")
        
        print(f"\nDetailed report: {report_file}")
        print("\nTo clean up test files, run:")
        print(f"  python3 {sys.argv[0]} --cleanup")
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()