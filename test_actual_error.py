#!/usr/bin/env python3
"""
Script to test the actual error scenario that's happening.
This simulates resuming from step 30000 with the exact same config.

Run this with torchrun to reproduce the error:
torchrun --nproc_per_node=4 test_actual_error.py
"""

import argparse
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_resume_test_config():
    """Create config that reproduces the exact error scenario"""
    config_content = """general:
  benchmark_csv_path: null
  consumed_train_samples: null
  ignore_sanity_checks: true
  project: debug_resume_test
  run: debug_resume_error
  seed: 42
  step: 30000  # CRITICAL: This simulates resuming from step 30000

# Infini-Attention Configuration (minimal for speed)
infini_attention:
  segment_length: 256
  turn_on_memory: true
  balance_factor_lr: 0.01
  balance_act_type: hard_sigmoid
  balance_init_type: zeros
  logging: true
  logging_interval: 1
  log_grad: false
  log_segment_acts: false
  balance_factor_weight_decay: 0.0

# Model Configuration (tiny)
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
    hidden_size: 32
    intermediate_size: 128
    num_hidden_layers: 1
    is_llama_config: true
    max_position_embeddings: 1024
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
    learning_rate: 0.001
    lr_decay_starting_step: null
    lr_decay_steps: null
    lr_decay_style: cosine
    lr_warmup_steps: 1
    lr_warmup_style: linear
    min_decay_lr: 0.0001
  weight_decay: 0.01
  zero_stage: 0

# Parallelism Configuration
parallelism:
  dp: 4
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
  sequence_length: 512
  train_steps: 30002  # Train steps 30001, 30002 (after resuming from 30000)
  val_check_interval: -1

# Data Configuration - THE KEY PART THAT CAUSES THE ERROR
data_stages:
  - name: "Test Resume Stage"
    start_training_step: 1  # Stage starts at 1, but we resume from 30000!
    data:
      dataset: null  # Use dummy data
      num_loading_workers: 1
      seed: 42

# Checkpoints Configuration
checkpoints:
  checkpoint_interval: 1
  checkpoints_path: ./debug_resume_checkpoints
  checkpoints_path_is_shared_file_system: false
  resume_checkpoint_path: null  # No actual resume needed for test
  save_initial_state: false

# Logging Configuration
logging:
  iteration_step_info_interval: 1
  log_level: info
  log_level_replica: info

# Profiler (disabled)
profiler: null"""
    
    config_file = "debug_resume_test.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    return config_file

def add_detailed_debug_patches():
    """Add very detailed debug patches to understand the exact failure"""
    from nanotron.trainer import DistributedTrainer
    import torch.distributed as dist
    
    # Store original methods
    original_update_dataloader = DistributedTrainer._update_dataloader_based_on_training_stages
    original_training_step = DistributedTrainer.training_step
    
    def debug_update_dataloader(self, dataloaders):
        """Extremely detailed debug version"""
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        print(f"[RANK{rank}] === _update_dataloader_based_on_training_stages ===")
        print(f"[RANK{rank}] iteration_step: {self.iteration_step}")
        print(f"[RANK{rank}] current_dataloader before: {type(self.current_dataloader) if self.current_dataloader else None}")
        print(f"[RANK{rank}] dataloaders type: {type(dataloaders)}")
        
        if isinstance(dataloaders, dict):
            print(f"[RANK{rank}] dataloaders keys: {list(dataloaders.keys())}")
        
        if hasattr(self.config, 'data_stages') and self.config.data_stages:
            print(f"[RANK{rank}] data_stages:")
            for i, stage in enumerate(self.config.data_stages):
                print(f"[RANK{rank}]   Stage {i}: '{stage.name}' starts at {stage.start_training_step}")
        
        # Track the dataloader selection logic step by step
        print(f"[RANK{rank}] --- Entering stage selection loop ---")
        dataloader = None
        for stage_id, stage in enumerate(self.config.data_stages):
            matches = stage.start_training_step == self.iteration_step
            print(f"[RANK{rank}] Stage {stage_id} '{stage.name}': start={stage.start_training_step}, current={self.iteration_step}, matches={matches}")
            
            if matches:
                print(f"[RANK{rank}] MATCH FOUND! Setting dataloader from stage '{stage.name}'")
                dataloader = dataloaders[stage.name]
                if callable(dataloader):
                    print(f"[RANK{rank}] Dataloader is callable, calling it...")
                    dataloader = dataloader()
                print(f"[RANK{rank}] Dataloader set to: {type(dataloader)}")
                break
        
        if dataloader is None:
            print(f"[RANK{rank}] *** CRITICAL: No matching stage found! dataloader is None ***")
            print(f"[RANK{rank}] This means no stage has start_training_step == {self.iteration_step}")
            print(f"[RANK{rank}] This will cause current_dataloader to remain None")
            print(f"[RANK{rank}] Which will trigger TypeError in training_step!")
        
        # Call original method to get the actual result
        result = original_update_dataloader(self, dataloaders)
        
        print(f"[RANK{rank}] current_dataloader after: {type(self.current_dataloader) if self.current_dataloader else None}")
        print(f"[RANK{rank}] === End _update_dataloader_based_on_training_stages ===")
        
        return result
    
    def debug_training_step(self, dataloader):
        """Debug training_step to catch the exact error"""
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        print(f"[RANK{rank}] === training_step ===")
        print(f"[RANK{rank}] dataloader parameter: {type(dataloader) if dataloader else None}")
        print(f"[RANK{rank}] dataloader is None: {dataloader is None}")
        print(f"[RANK{rank}] iteration_step: {self.iteration_step}")
        print(f"[RANK{rank}] n_micro_batches_per_batch: {self.n_micro_batches_per_batch}")
        
        if dataloader is None:
            print(f"[RANK{rank}] *** FATAL ERROR DETECTED ***")
            print(f"[RANK{rank}] dataloader is None in training_step!")
            print(f"[RANK{rank}] About to execute: train_batches = (next(dataloader) for _ in range({self.n_micro_batches_per_batch}))")
            print(f"[RANK{rank}] This will cause: TypeError: 'NoneType' object is not an iterator")
            print(f"[RANK{rank}] *** This is the exact error from the traceback! ***")
        
        # Call original method (this will trigger the error if dataloader is None)
        return original_training_step(self, dataloader)
    
    # Apply patches
    DistributedTrainer._update_dataloader_based_on_training_stages = debug_update_dataloader
    DistributedTrainer.training_step = debug_training_step
    
    print("Detailed debug patches applied")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, help="Optional config file (will create test config if not provided)")
    args = parser.parse_args()
    
    try:
        # Create or use config
        if args.config_file and os.path.exists(args.config_file):
            config_file = args.config_file
            print(f"Using provided config: {config_file}")
        else:
            config_file = create_resume_test_config()
            print(f"Created test config: {config_file}")
        
        # Apply debug patches before importing
        add_detailed_debug_patches()
        
        # Import and run
        from nanotron.trainer import DistributedTrainer
        from run_train import get_dataloader
        
        print("Creating DistributedTrainer...")
        trainer = DistributedTrainer(config_file)
        
        print("Creating dataloaders...")
        dataloader = get_dataloader(trainer)
        
        print("Starting training (this should reproduce the error)...")
        trainer.train(dataloader)
        
        print("Training completed unexpectedly (no error occurred)")
        
    except Exception as e:
        print(f"\\n*** CAUGHT EXCEPTION ***")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        
        if "TypeError" in str(e) and "'NoneType' object is not an iterator" in str(e):
            print("*** SUCCESS: Reproduced the exact error! ***")
        else:
            print("Different error occurred")
        
        # Print full traceback for analysis
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if 'config_file' in locals() and config_file.startswith('debug_'):
            try:
                os.remove(config_file)
                print(f"Cleaned up: {config_file}")
            except:
                pass

if __name__ == "__main__":
    main()