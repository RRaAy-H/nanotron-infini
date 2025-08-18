#!/usr/bin/env python3
"""
Comprehensive dataloader state inspection tool.
This script examines the dataloader state at various points to understand the issue.

Usage: python inspect_dataloader_state.py --config-file passkey_finetune_300m_simple_config.yaml
"""

import argparse
import sys
import os
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def inspect_config(config_file):
    """Inspect the configuration file"""
    print("=" * 80)
    print("CONFIG FILE INSPECTION")
    print("=" * 80)
    
    import yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Config file: {config_file}")
    print(f"General step: {config.get('general', {}).get('step', 'None')}")
    print(f"Train steps: {config.get('tokens', {}).get('train_steps', 'None')}")
    
    if 'data_stages' in config:
        print(f"Data stages count: {len(config['data_stages'])}")
        for i, stage in enumerate(config['data_stages']):
            print(f"  Stage {i}: {stage.get('name', 'unnamed')} starts at step {stage.get('start_training_step', 'unknown')}")
    else:
        print("No data_stages found in config")
    
    print()

def inspect_trainer_initialization(config_file):
    """Inspect trainer during initialization"""
    print("=" * 80)
    print("TRAINER INITIALIZATION INSPECTION")
    print("=" * 80)
    
    try:
        from nanotron.trainer import DistributedTrainer
        
        print("Creating DistributedTrainer...")
        trainer = DistributedTrainer(config_file)
        
        print(f"✓ Trainer created successfully")
        print(f"Start iteration step: {trainer.start_iteration_step}")
        print(f"Current iteration step: {trainer.iteration_step}")
        print(f"Current dataloader: {type(trainer.current_dataloader) if trainer.current_dataloader else None}")
        print(f"Has data_stages: {hasattr(trainer.config, 'data_stages')}")
        
        if hasattr(trainer.config, 'data_stages') and trainer.config.data_stages:
            print(f"Data stages:")
            for i, stage in enumerate(trainer.config.data_stages):
                print(f"  {i}: {stage.name} starts at {stage.start_training_step}")
        
        return trainer
        
    except Exception as e:
        print(f"✗ Trainer creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def inspect_dataloader_creation(trainer):
    """Inspect dataloader creation process"""
    print("=" * 80)
    print("DATALOADER CREATION INSPECTION")
    print("=" * 80)
    
    if trainer is None:
        print("Skipping dataloader inspection - trainer is None")
        return None
    
    try:
        from run_train import get_dataloader
        
        print("Creating dataloaders...")
        dataloaders = get_dataloader(trainer)
        
        print(f"✓ Dataloaders created successfully")
        print(f"Dataloaders type: {type(dataloaders)}")
        
        if isinstance(dataloaders, dict):
            print(f"Dataloader keys: {list(dataloaders.keys())}")
            for name, dl in dataloaders.items():
                print(f"  {name}: {type(dl)} (callable: {callable(dl)})")
        else:
            print(f"Single dataloader: {type(dataloaders)}")
        
        return dataloaders
        
    except Exception as e:
        print(f"✗ Dataloader creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def inspect_dataloader_update_process(trainer, dataloaders):
    """Inspect the dataloader update process that's causing the issue"""
    print("=" * 80)
    print("DATALOADER UPDATE PROCESS INSPECTION")
    print("=" * 80)
    
    if trainer is None or dataloaders is None:
        print("Skipping update inspection - trainer or dataloaders is None")
        return
    
    print(f"Before update:")
    print(f"  Trainer iteration step: {trainer.iteration_step}")
    print(f"  Current dataloader: {type(trainer.current_dataloader) if trainer.current_dataloader else None}")
    
    # Simulate different iteration steps to see what happens
    test_steps = [1, 30000, 30001, 30500]
    
    for step in test_steps:
        print(f"\n--- Testing iteration step {step} ---")
        
        # Temporarily set iteration step
        original_step = trainer.iteration_step
        trainer.iteration_step = step
        
        try:
            # Call the update method
            trainer._update_dataloader_based_on_training_stages(dataloaders)
            
            print(f"  After update at step {step}:")
            print(f"    Current dataloader: {type(trainer.current_dataloader) if trainer.current_dataloader else None}")
            print(f"    Current dataloader is None: {trainer.current_dataloader is None}")
            
            if trainer.current_dataloader is None:
                print(f"    ✗ PROBLEM: dataloader is None at step {step}")
                
                # Debug the stage matching logic
                print(f"    Debug stage matching:")
                for stage_id, stage in enumerate(trainer.config.data_stages):
                    matches = stage.start_training_step == step
                    print(f"      Stage {stage_id} '{stage.name}' starts at {stage.start_training_step}, matches step {step}: {matches}")
            else:
                print(f"    ✓ dataloader properly set at step {step}")
                
        except Exception as e:
            print(f"  ✗ Error during update at step {step}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Restore original step
            trainer.iteration_step = original_step

def inspect_training_step_simulation(trainer):
    """Simulate the training_step call to see where it fails"""
    print("=" * 80)
    print("TRAINING STEP SIMULATION")
    print("=" * 80)
    
    if trainer is None:
        print("Skipping training step simulation - trainer is None")
        return
    
    print(f"Current dataloader: {type(trainer.current_dataloader) if trainer.current_dataloader else None}")
    print(f"N micro batches per batch: {trainer.n_micro_batches_per_batch}")
    
    if trainer.current_dataloader is None:
        print("✗ CRITICAL: current_dataloader is None - this will cause the TypeError")
        print("This is the exact condition that causes: TypeError: 'NoneType' object is not an iterator")
        return False
    
    try:
        # Try to create the generator that's failing
        print("Attempting to create train_batches generator...")
        train_batches = (next(trainer.current_dataloader) for _ in range(trainer.n_micro_batches_per_batch))
        print("✓ train_batches generator created successfully")
        
        # Try to get first batch
        print("Attempting to get first batch...")
        first_batch = next(train_batches)
        print(f"✓ First batch retrieved: {type(first_batch)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in training step simulation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    args = parser.parse_args()
    
    print("COMPREHENSIVE DATALOADER STATE INSPECTION")
    print("=" * 80)
    print(f"Config file: {args.config_file}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Step 1: Inspect config
    inspect_config(args.config_file)
    
    # Step 2: Inspect trainer initialization
    trainer = inspect_trainer_initialization(args.config_file)
    
    # Step 3: Inspect dataloader creation
    dataloaders = inspect_dataloader_creation(trainer)
    
    # Step 4: Inspect dataloader update process
    inspect_dataloader_update_process(trainer, dataloaders)
    
    # Step 5: Simulate training step
    success = inspect_training_step_simulation(trainer)
    
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    if success:
        print("✓ All inspections passed - no obvious issues detected")
    else:
        print("✗ Issues detected - this explains the TypeError")
        
    print("\nTo reproduce the exact error, run the original training script:")
    print(f"  torchrun --nproc_per_node=4 run_train.py --config-file {args.config_file}")

if __name__ == "__main__":
    main()