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
    """Use existing working config but modify for testing"""
    import yaml
    
    # Copy existing working config
    with open("passkey_finetune_300m_simple_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for testing
    config['general']['project'] = 'test_fix'
    config['general']['run'] = 'test_dataloader_fix'
    config['tokens']['train_steps'] = 30003  # Just 3 steps
    config['model']['model_config']['hidden_size'] = 32  # Smaller for speed
    config['model']['model_config']['intermediate_size'] = 128
    config['model']['model_config']['num_hidden_layers'] = 1
    config['tokens']['sequence_length'] = 512
    config['checkpoints']['checkpoints_path'] = './test_checkpoints'
    config['checkpoints']['resume_checkpoint_path'] = None
    
    config_file = "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_file

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
        
        # Create trainer
        trainer = DistributedTrainer(config_file)
        dataloader = get_dataloader(trainer)
        
        # Simulate resuming from step 30000 (like loading from checkpoint)
        trainer.iteration_step = 30000
        
        if rank == 0:
            print(f"Initial state: iteration_step={trainer.iteration_step}")
            print(f"Data stage starts at step 1")
            print(f"Testing next iteration (step {trainer.iteration_step + 1})...")
        
        # This is where the original error occurred
        trainer.iteration_step += 1  # Now at step 30001
        
        if rank == 0:
            print(f"Before update: current_dataloader = {trainer.current_dataloader}")
            print(f"Available dataloaders: {list(dataloader.keys())}")
            print(f"Data stages: {[(stage.name, stage.start_training_step) for stage in trainer.config.data_stages]}")
        
        trainer._update_dataloader_based_on_training_stages(dataloader)
        
        if rank == 0:
            print(f"After update: current_dataloader = {trainer.current_dataloader}")
        
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