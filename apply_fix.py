#!/usr/bin/env python3
"""
Simple script to apply the dataloader fix and test it.

USAGE:
  python3 apply_fix.py

This will:
1. Show the current problematic code
2. Apply the fix manually
3. Test with: torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29401 test_actual_error.py
"""

def show_current_code():
    """Show the current problematic code"""
    print("CURRENT PROBLEMATIC CODE:")
    print("=" * 60)
    
    with open("src/nanotron/trainer.py", 'r') as f:
        lines = f.readlines()
    
    # Find the problematic section
    for i, line in enumerate(lines):
        if "for stage_id, stage in enumerate(self.config.data_stages):" in line:
            start_line = i - 2
            # Show context around the problematic code
            for j in range(start_line, min(start_line + 25, len(lines))):
                print(f"{j+1:3d}: {lines[j].rstrip()}")
            break
    
    print("=" * 60)

def apply_fix():
    """Apply the fix to trainer.py"""
    print("\nAPPLYING FIX:")
    print("=" * 60)
    
    with open("src/nanotron/trainer.py", 'r') as f:
        content = f.read()
    
    # Find and replace the problematic section
    old_pattern = """        dataloader = None
        for stage_id, stage in enumerate(self.config.data_stages):
            stage = cast(DatasetStageArgs, stage)

            if stage.start_training_step == self.iteration_step:"""
    
    new_pattern = """        dataloader = None
        current_stage = None
        current_stage_id = 0
        
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
            if current_stage.start_training_step == self.iteration_step:"""
    
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        print("✓ Applied pattern 1")
    else:
        print("✗ Pattern 1 not found")
    
    # Fix the second part
    old_pattern2 = """                dataloader = dataloaders[stage.name]
                # NOTE: if a dataloader is lazy initialized, we need to call it to initialize it
                dataloader = dataloader() if callable(dataloader) else dataloader
                break"""
    
    new_pattern2 = """                dataloader = dataloaders[current_stage.name]
                # NOTE: if a dataloader is lazy initialized, we need to call it to initialize it
                dataloader = dataloader() if callable(dataloader) else dataloader
            else:
                # FIXED: Even if not exact match, use the active stage's dataloader
                dataloader = dataloaders[current_stage.name]
                dataloader = dataloader() if callable(dataloader) else dataloader"""
    
    if old_pattern2 in content:
        content = content.replace(old_pattern2, new_pattern2)
        print("✓ Applied pattern 2")
    else:
        print("✗ Pattern 2 not found")
    
    # Fix variable references
    old_pattern3 = """                if self.current_dataloader is not None:
                    prev_stage_name = self.config.data_stages[stage_id - 1].name
                    prev_dataloader = dataloaders[prev_stage_name]
                    if isinstance(prev_dataloader, DataLoader):
                        # NOTE: we don't need to clear dummy data generator from memory
                        clear_dataloader_from_memory(prev_dataloader, stage_name=stage.name)

                log_rank(
                    f"[Training Stage: {stage.name}] Switching to a new dataset","""
    
    new_pattern3 = """                if self.current_dataloader is not None and current_stage_id > 0:
                    prev_stage_name = self.config.data_stages[current_stage_id - 1].name
                    prev_dataloader = dataloaders[prev_stage_name]
                    if isinstance(prev_dataloader, DataLoader):
                        # NOTE: we don't need to clear dummy data generator from memory
                        clear_dataloader_from_memory(prev_dataloader, stage_name=current_stage.name)

                log_rank(
                    f"[Training Stage: {current_stage.name}] Switching to a new dataset","""
    
    if old_pattern3 in content:
        content = content.replace(old_pattern3, new_pattern3)
        print("✓ Applied pattern 3")
    else:
        print("✗ Pattern 3 not found")
    
    # Backup and write
    import shutil
    shutil.copy2("src/nanotron/trainer.py", "src/nanotron/trainer_backup.py")
    
    with open("src/nanotron/trainer.py", 'w') as f:
        f.write(content)
    
    print("✓ Fix applied and backup created")
    print("=" * 60)

def show_fixed_code():
    """Show the fixed code"""
    print("\nFIXED CODE:")
    print("=" * 60)
    
    with open("src/nanotron/trainer.py", 'r') as f:
        lines = f.readlines()
    
    # Find the fixed section
    for i, line in enumerate(lines):
        if "# FIXED: Find the active stage for the current iteration step" in line:
            start_line = i - 3
            # Show context around the fixed code
            for j in range(start_line, min(start_line + 35, len(lines))):
                print(f"{j+1:3d}: {lines[j].rstrip()}")
            break
    
    print("=" * 60)

def main():
    print("DATALOADER FIX APPLICATION")
    print("=" * 80)
    
    # Step 1: Show current problematic code
    show_current_code()
    
    # Step 2: Apply fix
    apply_fix()
    
    # Step 3: Show fixed code
    show_fixed_code()
    
    # Step 4: Instructions
    print("\nNEXT STEPS:")
    print("=" * 60)
    print("1. Test the fix:")
    print("   torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29401 test_actual_error.py")
    print()
    print("2. If it works, test with real passkey script:")
    print("   ./run_passkey_finetune_300m.sh ./checkpoints/fineweb_4gpu_300m_infini/30000")
    print()
    print("3. To restore original file if needed:")
    print("   cp src/nanotron/trainer_backup.py src/nanotron/trainer.py")
    print("=" * 60)

if __name__ == "__main__":
    main()