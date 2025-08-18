#!/usr/bin/env python3
"""
Minimal script to reproduce the exact TypeError: 'NoneType' object is not an iterator error.
This script focuses on the specific scenario that's failing.

Usage: python reproduce_error.py --config-file passkey_finetune_300m_simple_config.yaml
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def reproduce_error_scenario(config_file):
    """Reproduce the exact error scenario"""
    print("REPRODUCING ERROR SCENARIO")
    print("=" * 60)
    
    try:
        from nanotron.trainer import DistributedTrainer
        from run_train import get_dataloader
        
        print("1. Creating trainer...")
        trainer = DistributedTrainer(config_file)
        print(f"   ✓ Created, iteration_step: {trainer.iteration_step}")
        
        print("2. Creating dataloaders...")
        dataloaders = get_dataloader(trainer)
        print(f"   ✓ Created, type: {type(dataloaders)}")
        
        print("3. Simulating resume from checkpoint (step 30000)...")
        # This simulates resuming from a checkpoint
        trainer.iteration_step = 30001  # This is what happens when resuming
        print(f"   Set iteration_step to: {trainer.iteration_step}")
        
        print("4. Calling _update_dataloader_based_on_training_stages...")
        print(f"   Before: current_dataloader = {trainer.current_dataloader}")
        
        trainer._update_dataloader_based_on_training_stages(dataloaders)
        
        print(f"   After: current_dataloader = {trainer.current_dataloader}")
        
        if trainer.current_dataloader is None:
            print("   ✗ REPRODUCED: current_dataloader is None!")
            print("   This will cause: TypeError: 'NoneType' object is not an iterator")
            print("")
            print("   Root cause analysis:")
            print("   - Data stage starts at step 1")
            print("   - Resume happens at step 30001") 
            print("   - Loop looks for: stage.start_training_step == 30001")
            print("   - No stage starts at 30001, so dataloader stays None")
            print("")
            
            # Now simulate the exact line that fails
            print("5. Simulating the exact failing line...")
            try:
                train_batches = (next(trainer.current_dataloader) for _ in range(trainer.n_micro_batches_per_batch))
                print("   This line should never be reached")
            except TypeError as e:
                print(f"   ✗ REPRODUCED ERROR: {e}")
                return True
                
        else:
            print("   ✓ current_dataloader is set - error not reproduced")
            return False
            
    except Exception as e:
        print(f"Error during reproduction: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_stage_matching_logic(config_file):
    """Analyze why the stage matching logic fails"""
    print("STAGE MATCHING LOGIC ANALYSIS")
    print("=" * 60)
    
    try:
        from nanotron.trainer import DistributedTrainer
        
        trainer = DistributedTrainer(config_file)
        
        print("Data stages configuration:")
        for i, stage in enumerate(trainer.config.data_stages):
            print(f"  Stage {i}: '{stage.name}' starts at step {stage.start_training_step}")
        
        print("\nTesting stage matching for different iteration steps:")
        test_steps = [1, 100, 1000, 10000, 30000, 30001, 30500]
        
        for step in test_steps:
            matches = []
            for stage_id, stage in enumerate(trainer.config.data_stages):
                if stage.start_training_step == step:
                    matches.append(f"Stage {stage_id} '{stage.name}'")
            
            if matches:
                print(f"  Step {step:5d}: Matches {', '.join(matches)}")
            else:
                print(f"  Step {step:5d}: No matches (dataloader would be None)")
        
        print("\nPROBLEM IDENTIFIED:")
        print("- The logic only matches when stage.start_training_step == iteration_step")
        print("- When resuming from checkpoint, iteration_step is much larger than start_training_step")
        print("- No stage matches, so dataloader stays None")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def test_proposed_fixes():
    """Test different approaches to fix the issue"""
    print("PROPOSED FIX TESTING")
    print("=" * 60)
    
    # Test data
    data_stages = [
        {'name': 'Passkey Finetune', 'start_training_step': 1}
    ]
    
    def original_logic(iteration_step):
        for stage_id, stage in enumerate(data_stages):
            if stage['start_training_step'] == iteration_step:
                return stage['name']
        return None
    
    def fix_approach_1(iteration_step):
        """Find the latest stage that has started"""
        current_stage = None
        for stage in data_stages:
            if stage['start_training_step'] <= iteration_step:
                current_stage = stage
        return current_stage['name'] if current_stage else None
    
    def fix_approach_2(iteration_step):
        """Always use first stage if no exact match (for single stage configs)"""
        for stage_id, stage in enumerate(data_stages):
            if stage['start_training_step'] == iteration_step:
                return stage['name']
        # If no exact match and only one stage, use it
        if len(data_stages) == 1:
            return data_stages[0]['name']
        return None
    
    test_steps = [1, 30001]
    
    print("Testing different fix approaches:")
    for step in test_steps:
        print(f"\nIteration step {step}:")
        print(f"  Original logic: {original_logic(step)}")
        print(f"  Fix approach 1: {fix_approach_1(step)}")
        print(f"  Fix approach 2: {fix_approach_2(step)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    args = parser.parse_args()
    
    print("COMPREHENSIVE DATALOADER ERROR ANALYSIS")
    print("=" * 80)
    print(f"Analyzing config: {args.config_file}")
    print()
    
    # Step 1: Inspect config
    inspect_config(args.config_file)
    
    # Step 2: Inspect trainer initialization
    trainer = inspect_trainer_initialization(args.config_file)
    
    # Step 3: Analyze stage matching logic
    analyze_stage_matching_logic(args.config_file)
    
    # Step 4: Reproduce the error
    print("")
    error_reproduced = reproduce_error_scenario(args.config_file)
    
    # Step 5: Test proposed fixes
    print("")
    test_proposed_fixes()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    if error_reproduced:
        print("✓ Successfully reproduced the original error")
        print("✓ Identified the root cause")
        print("✓ Tested potential fix approaches")
        print("\nRecommended fix: Use approach 1 (find latest started stage)")
    else:
        print("✗ Could not reproduce the error - may need more investigation")

if __name__ == "__main__":
    main()