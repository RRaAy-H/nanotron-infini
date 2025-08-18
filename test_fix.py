#!/usr/bin/env python3
"""
Simple test to verify the dataloader fix works.

USAGE:
  torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29401 test_fix.py

This creates a minimal test scenario that reproduces the original error condition
and verifies the fix works correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_test_config():
    """Create minimal test config"""
    config_content = '''general:
  project: test_fix
  run: test_dataloader_fix
  seed: 42
  step: 30000  # Simulate resuming from step 30000
  ignore_sanity_checks: true

infini_attention:
  segment_length: 256
  turn_on_memory: true
  balance_factor_lr: 0.01
  balance_act_type: hard_sigmoid
  balance_init_type: zeros
  logging: false
  logging_interval: 1

model:
  dtype: bfloat16
  init_method:
    std: 0.03125
  model_config:
    bos_token_id: 1
    eos_token_id: 2
    hidden_act: silu
    hidden_size: 32
    intermediate_size: 128
    num_hidden_layers: 1
    is_llama_config: true
    max_position_embeddings: 1024
    num_attention_heads: 2
    num_key_value_heads: 2
    rms_norm_eps: 1.0e-05
    vocab_size: 1000

optimizer:
  accumulate_grad_in_fp32: true
  adam_beta1: 0.9
  adam_beta2: 0.95
  clip_grad: 1.0
  learning_rate_scheduler:
    learning_rate: 0.001
    lr_warmup_steps: 1
    lr_warmup_style: linear
  weight_decay: 0.01
  zero_stage: 0

parallelism:
  dp: 4
  tp: 1
  pp: 1
  pp_engine: 1f1b

tokenizer:
  tokenizer_name_or_path: lvwerra/the-tokenizer-v1

tokens:
  micro_batch_size: 1
  sequence_length: 512
  train_steps: 30003  # Just 3 steps
  batch_accumulation_per_replica: 1

data_stages:
  - name: "Test Stage"
    start_training_step: 1  # Starts at 1, but we resume from 30000
    data:
      dataset: null  # Dummy data
      seed: 42

checkpoints:
  checkpoints_path: ./test_checkpoints
  resume_checkpoint_path: null
  save_initial_state: false

logging:
  iteration_step_info_interval: 1
  log_level: info'''
    
    with open("test_config.yaml", 'w') as f:
        f.write(config_content)
    return "test_config.yaml"

def main():
    try:
        import torch.distributed as dist
        from nanotron.trainer import DistributedTrainer
        from run_train import get_dataloader
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        if rank == 0:
            print("Testing dataloader fix...")
            config_file = create_test_config()
        else:
            config_file = "test_config.yaml"
        
        # Wait for config file
        if dist.is_initialized():
            dist.barrier()
        
        # Create trainer (this simulates resuming from step 30000)
        trainer = DistributedTrainer(config_file)
        dataloader = get_dataloader(trainer)
        
        if rank == 0:
            print(f"Initial state: iteration_step={trainer.iteration_step}")
            print(f"Data stage starts at step 1")
            print(f"Testing next iteration (step {trainer.iteration_step + 1})...")
        
        # This is where the original error occurred
        trainer.iteration_step += 1  # Now at step 30001
        trainer._update_dataloader_based_on_training_stages(dataloader)
        
        if trainer.current_dataloader is None:
            if rank == 0:
                print("✗ FAILED: current_dataloader is None")
            return False
        else:
            if rank == 0:
                print("✓ SUCCESS: current_dataloader is properly set")
        
        # Test a few more steps
        for i in range(2):
            trainer.iteration_step += 1
            trainer._update_dataloader_based_on_training_stages(dataloader)
            if trainer.current_dataloader is None:
                if rank == 0:
                    print(f"✗ FAILED at step {trainer.iteration_step}")
                return False
        
        if rank == 0:
            print("✓ All tests passed - fix is working!")
            print("\nNow test with your real script:")
            print("./run_passkey_finetune_300m.sh ./checkpoints/fineweb_4gpu_300m_infini/30000")
        
        return True
        
    except Exception as e:
        if rank == 0:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if rank == 0 and os.path.exists("test_config.yaml"):
            os.remove("test_config.yaml")

if __name__ == "__main__":
    main()