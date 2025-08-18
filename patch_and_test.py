#!/usr/bin/env python3
"""
This script patches the trainer.py file to add debugging at the exact failure point,
then runs a minimal test to capture the state when the error occurs.

Usage: python patch_and_test.py --config-file passkey_finetune_300m_simple_config.yaml
"""

import argparse
import sys
import os
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def backup_trainer_file():
    """Backup the original trainer.py file"""
    trainer_path = Path("src/nanotron/trainer.py")
    backup_path = Path("src/nanotron/trainer_original_backup.py")
    
    if not backup_path.exists():
        shutil.copy2(trainer_path, backup_path)
        print(f"✓ Backed up trainer.py to {backup_path}")
    else:
        print(f"✓ Backup already exists at {backup_path}")

def restore_trainer_file():
    """Restore the original trainer.py file"""
    trainer_path = Path("src/nanotron/trainer.py")
    backup_path = Path("src/nanotron/trainer_original_backup.py")
    
    if backup_path.exists():
        shutil.copy2(backup_path, trainer_path)
        print(f"✓ Restored trainer.py from backup")
    else:
        print(f"✗ No backup found at {backup_path}")

def add_debug_patches():
    """Add debug patches to trainer.py"""
    print("Adding debug patches to trainer.py...")
    
    trainer_path = Path("src/nanotron/trainer.py")
    
    # Read the current file
    with open(trainer_path, 'r') as f:
        content = f.read()
    
    # Patch 1: Add debug logging to _update_dataloader_based_on_training_stages
    patch1_old = """        dataloader = None
        for stage_id, stage in enumerate(self.config.data_stages):
            stage = cast(DatasetStageArgs, stage)

            if stage.start_training_step == self.iteration_step:"""
    
    patch1_new = """        dataloader = None
        print(f"DEBUG: _update_dataloader_based_on_training_stages called at iteration_step={self.iteration_step}")
        print(f"DEBUG: data_stages = {[(s.name, s.start_training_step) for s in self.config.data_stages]}")
        print(f"DEBUG: current_dataloader before update = {type(self.current_dataloader) if self.current_dataloader else None}")
        
        for stage_id, stage in enumerate(self.config.data_stages):
            stage = cast(DatasetStageArgs, stage)
            print(f"DEBUG: Checking stage {stage_id} '{stage.name}' start_step={stage.start_training_step} vs iteration_step={self.iteration_step}")

            if stage.start_training_step == self.iteration_step:
                print(f"DEBUG: MATCH FOUND - stage {stage.name} matches iteration_step {self.iteration_step}")"""
    
    # Patch 2: Add debug logging after the loop
    patch2_old = """                break

        if dataloader is not None:"""
    
    patch2_new = """                break
        
        print(f"DEBUG: After stage loop - dataloader = {type(dataloader) if dataloader else None}")
        if dataloader is None:
            print(f"DEBUG: CRITICAL - No dataloader found for iteration_step {self.iteration_step}")
            print(f"DEBUG: This will cause current_dataloader to remain None and trigger the TypeError")

        if dataloader is not None:"""
    
    # Patch 3: Add debug logging in training_step
    patch3_old = """        train_batches = (next(dataloader) for _ in range(self.n_micro_batches_per_batch))"""
    
    patch3_new = """        print(f"DEBUG: training_step called with dataloader={type(dataloader) if dataloader else None}")
        print(f"DEBUG: iteration_step={self.iteration_step}, n_micro_batches_per_batch={self.n_micro_batches_per_batch}")
        
        if dataloader is None:
            print(f"DEBUG: FATAL - dataloader is None, this will cause TypeError!")
            print(f"DEBUG: About to execute: train_batches = (next(dataloader) for _ in range({self.n_micro_batches_per_batch}))")
            
        train_batches = (next(dataloader) for _ in range(self.n_micro_batches_per_batch))"""
    
    # Apply patches
    if patch1_old in content:
        content = content.replace(patch1_old, patch1_new)
        print("✓ Applied patch 1: dataloader update debug")
    else:
        print("✗ Could not apply patch 1")
    
    if patch2_old in content:
        content = content.replace(patch2_old, patch2_new)
        print("✓ Applied patch 2: post-loop debug")
    else:
        print("✗ Could not apply patch 2")
    
    if patch3_old in content:
        content = content.replace(patch3_old, patch3_new)
        print("✓ Applied patch 3: training_step debug")
    else:
        print("✗ Could not apply patch 3")
    
    # Write the patched file
    with open(trainer_path, 'w') as f:
        f.write(content)
    
    print("✓ Debug patches applied to trainer.py")

def run_debug_test(config_file):
    """Run a minimal test with the debug patches"""
    print("\nRunning debug test...")
    
    try:
        from nanotron.trainer import DistributedTrainer
        from run_train import get_dataloader
        
        print("1. Creating trainer...")
        trainer = DistributedTrainer(config_file)
        
        print("2. Creating dataloaders...")
        dataloaders = get_dataloader(trainer)
        
        print("3. Simulating resume scenario...")
        # This simulates the exact scenario from the error
        trainer.iteration_step = 30001
        
        print("4. Calling _update_dataloader_based_on_training_stages...")
        trainer._update_dataloader_based_on_training_stages(dataloaders)
        
        print("5. Simulating training_step call...")
        # This should trigger the error or show us exactly what's wrong
        try:
            outputs, loss_avg = trainer.training_step(dataloader=trainer.current_dataloader)
            print("✓ Unexpected success - no error occurred")
            return True
        except TypeError as e:
            print(f"✓ Reproduced TypeError: {e}")
            return True
        except Exception as e:
            print(f"✗ Different error: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML config file")
    parser.add_argument("--restore", action="store_true", help="Restore original trainer.py file")
    args = parser.parse_args()
    
    if args.restore:
        restore_trainer_file()
        return
    
    print("PATCH AND TEST DEBUGGING")
    print("=" * 60)
    print(f"Config: {args.config_file}")
    print("=" * 60)
    
    try:
        # Step 1: Backup original file
        backup_trainer_file()
        
        # Step 2: Add debug patches
        add_debug_patches()
        
        # Step 3: Run debug test
        success = run_debug_test(args.config_file)
        
        print("\n" + "=" * 60)
        if success:
            print("✓ Debug test completed - check output above for details")
        else:
            print("✗ Debug test failed")
        print("=" * 60)
        
        print("\nTo restore original file, run:")
        print(f"  python {sys.argv[0]} --restore")
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nRestoring original file due to error...")
        restore_trainer_file()

if __name__ == "__main__":
    main()