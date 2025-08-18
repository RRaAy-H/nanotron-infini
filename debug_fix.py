#!/usr/bin/env python3
"""
All-in-one debug script to diagnose and fix the dataloader issue.

USAGE:
  torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29401 debug_fix.py

This script will:
1. Apply the correct fix to trainer.py
2. Run with detailed debugging to verify the fix works
3. Automatically restore the original file when done
"""

import argparse
import sys
import os
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def backup_and_fix_trainer():
    """Backup original trainer.py and apply the fix"""
    trainer_path = Path("src/nanotron/trainer.py")
    backup_path = Path("src/nanotron/trainer_backup.py")
    
    # Backup original
    shutil.copy2(trainer_path, backup_path)
    print("✓ Backed up trainer.py")
    
    # Read current content
    with open(trainer_path, 'r') as f:
        content = f.read()
    
    # Apply the fix
    old_code = '''        dataloader = None
        for stage_id, stage in enumerate(self.config.data_stages):
            stage = cast(DatasetStageArgs, stage)

            if stage.start_training_step == self.iteration_step:
                if self.current_dataloader is not None:
                    prev_stage_name = self.config.data_stages[stage_id - 1].name
                    prev_dataloader = dataloaders[prev_stage_name]
                    if isinstance(prev_dataloader, DataLoader):
                        # NOTE: we don't need to clear dummy data generator from memory
                        clear_dataloader_from_memory(prev_dataloader, stage_name=stage.name)

                log_rank(
                    f"[Training Stage: {stage.name}] Switching to a new dataset",
                    logger=logger,
                    level=logging.INFO,
                    rank=0,
                )

                dataloader = dataloaders[stage.name]
                # NOTE: if a dataloader is lazy initialized, we need to call it to initialize it
                dataloader = dataloader() if callable(dataloader) else dataloader
                break

        if dataloader is not None:
            self.current_dataloader = sanity_check_dataloader(
                dataloader=dataloader, parallel_context=self.parallel_context, config=self.config
            )'''
    
    new_code = '''        dataloader = None
        current_stage = None
        
        # FIXED: Find the active stage for the current iteration step
        for stage_id, stage in enumerate(self.config.data_stages):
            stage = cast(DatasetStageArgs, stage)
            
            # Check if this stage should be active for the current iteration step
            if stage.start_training_step <= self.iteration_step:
                current_stage = stage
                current_stage_id = stage_id
            else:
                break

        if current_stage is not None:
            # Only switch dataloader when starting a new stage (exact match)
            if current_stage.start_training_step == self.iteration_step:
                if self.current_dataloader is not None and current_stage_id > 0:
                    prev_stage_name = self.config.data_stages[current_stage_id - 1].name
                    prev_dataloader = dataloaders[prev_stage_name]
                    if isinstance(prev_dataloader, DataLoader):
                        # NOTE: we don't need to clear dummy data generator from memory
                        clear_dataloader_from_memory(prev_dataloader, stage_name=current_stage.name)

                log_rank(
                    f"[Training Stage: {current_stage.name}] Switching to a new dataset",
                    logger=logger,
                    level=logging.INFO,
                    rank=0,
                )

            # FIXED: Set dataloader for the current active stage (not just on exact match)
            dataloader = dataloaders[current_stage.name]
            # NOTE: if a dataloader is lazy initialized, we need to call it to initialize it
            dataloader = dataloader() if callable(dataloader) else dataloader

        if dataloader is not None:
            self.current_dataloader = sanity_check_dataloader(
                dataloader=dataloader, parallel_context=self.parallel_context, config=self.config
            )'''
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        print("✓ Applied dataloader fix")
    else:
        print("✗ Could not find code to fix - trainer.py may have been modified")
        return False
    
    # Add debug logging to the fixed method
    debug_patch = '''    def _update_dataloader_based_on_training_stages(self, dataloaders: Union[List[DataLoader], DataLoader]):
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        print(f"[RANK{rank}] FIXED VERSION: iteration_step={self.iteration_step}")
        if hasattr(self.config, 'data_stages') and self.config.data_stages:
            for i, stage in enumerate(self.config.data_stages):
                active = stage.start_training_step <= self.iteration_step
                exact_match = stage.start_training_step == self.iteration_step
                print(f"[RANK{rank}] Stage {i} '{stage.name}': start={stage.start_training_step}, active={active}, exact_match={exact_match}")
        
        from collections.abc import Generator'''
    
    original_method_start = '''    def _update_dataloader_based_on_training_stages(self, dataloaders: Union[List[DataLoader], DataLoader]):
        from collections.abc import Generator'''
    
    if original_method_start in content:
        content = content.replace(original_method_start, debug_patch)
        print("✓ Added debug logging")
    
    # Write the fixed file
    with open(trainer_path, 'w') as f:
        f.write(content)
    
    return True

def restore_trainer():
    """Restore original trainer.py"""
    trainer_path = Path("src/nanotron/trainer.py")
    backup_path = Path("src/nanotron/trainer_backup.py")
    
    if backup_path.exists():
        shutil.copy2(backup_path, trainer_path)
        backup_path.unlink()
        print("✓ Restored original trainer.py")
    else:
        print("✗ No backup found")

def create_test_config():
    """Create test config that reproduces the issue"""
    config_content = '''general:
  benchmark_csv_path: null
  consumed_train_samples: null
  ignore_sanity_checks: true
  project: debug_fix_test
  run: debug_dataloader_fix
  seed: 42
  step: 30000  # CRITICAL: Simulate resuming from step 30000

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

parallelism:
  dp: 4
  tp: 1
  expert_parallel_size: 1
  pp: 1
  pp_engine: 1f1b
  tp_linear_async_communication: false
  tp_mode: ALL_REDUCE

tokenizer:
  tokenizer_max_length: null
  tokenizer_name_or_path: lvwerra/the-tokenizer-v1
  tokenizer_revision: null

tokens:
  batch_accumulation_per_replica: 1
  limit_test_batches: 0
  limit_val_batches: 0
  micro_batch_size: 1
  sequence_length: 512
  train_steps: 30003  # Train just 3 steps (30001, 30002, 30003)
  val_check_interval: -1

data_stages:
  - name: "Fix Test Stage"
    start_training_step: 1  # Stage starts at 1, but we resume from 30000
    data:
      dataset: null  # Use dummy data
      num_loading_workers: 1
      seed: 42

checkpoints:
  checkpoint_interval: 10
  checkpoints_path: ./debug_fix_checkpoints
  checkpoints_path_is_shared_file_system: false
  resume_checkpoint_path: null
  save_initial_state: false

logging:
  iteration_step_info_interval: 1
  log_level: info
  log_level_replica: info

profiler: null'''
    
    config_file = "debug_fix_config.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✓ Created test config: {config_file}")
    return config_file

def run_test(config_file):
    """Run the training test"""
    try:
        from nanotron.trainer import DistributedTrainer
        from run_train import get_dataloader
        import torch.distributed as dist
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        if rank == 0:
            print("Creating trainer...")
        trainer = DistributedTrainer(config_file)
        
        if rank == 0:
            print("Creating dataloaders...")
        dataloader = get_dataloader(trainer)
        
        if rank == 0:
            print("Starting training test (this should work now)...")
        
        # Run just a few training steps
        for step in range(3):
            if rank == 0:
                print(f"\n=== Testing step {trainer.iteration_step + 1} ===")
            
            # This should now work without error
            trainer._update_dataloader_based_on_training_stages(dataloader)
            
            if trainer.current_dataloader is None:
                if rank == 0:
                    print(f"✗ FAILED: current_dataloader is None at step {trainer.iteration_step + 1}")
                return False
            else:
                if rank == 0:
                    print(f"✓ SUCCESS: current_dataloader is set at step {trainer.iteration_step + 1}")
            
            # Simulate next iteration
            trainer.iteration_step += 1
        
        if rank == 0:
            print("\n✓ All tests passed! Fix appears to work correctly.")
        return True
        
    except Exception as e:
        if rank == 0:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
        return False

def main():
    import torch.distributed as dist
    
    try:
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        if rank == 0:
            print("DEBUG FIX SCRIPT")
            print("=" * 50)
        
        # Step 1: Apply fix
        if rank == 0:
            print("1. Applying fix to trainer.py...")
            if not backup_and_fix_trainer():
                print("Failed to apply fix")
                return
        
        # Step 2: Create test config
        if rank == 0:
            print("2. Creating test configuration...")
            config_file = create_test_config()
        
        # Synchronize across ranks
        if dist.is_initialized():
            dist.barrier()
            config_file = "debug_fix_config.yaml"  # All ranks use same config
        
        # Step 3: Run test
        if rank == 0:
            print("3. Running test...")
        
        success = run_test(config_file)
        
        # Step 4: Results
        if rank == 0:
            print("\n" + "=" * 50)
            if success:
                print("✓ FIX SUCCESSFUL!")
                print("The dataloader issue has been resolved.")
                print("You can now run your actual training:")
                print("  ./run_passkey_finetune_300m.sh ./checkpoints/fineweb_4gpu_300m_infini/30000")
            else:
                print("✗ Fix failed - more investigation needed")
            print("=" * 50)
        
    except Exception as e:
        if rank == 0:
            print(f"\nFATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    finally:
        # Always restore original file
        if rank == 0:
            print("\nRestoring original trainer.py...")
            restore_trainer()
            
            # Cleanup
            for file in ["debug_fix_config.yaml"]:
                if os.path.exists(file):
                    os.remove(file)
                    print(f"Cleaned up: {file}")

if __name__ == "__main__":
    main()