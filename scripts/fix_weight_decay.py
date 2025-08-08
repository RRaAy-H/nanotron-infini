#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/fix_weight_decay.py

"""
Script to fix weight decay handling in Adam optimizer to prevent None errors.

This script checks and updates specific code patterns in the codebase that might
be causing NoneType errors with weight_decay in the Adam optimizer.
"""

import os
import re
import sys
from pathlib import Path

def fix_file(file_path):
    """Fix weight decay handling in the given file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace patterns where weight_decay might be None
        modified = re.sub(
            r'weight_decay=([^,\)]+)',
            r'weight_decay=\1 if \1 is not None else 0.0',
            content
        )
        
        # Look for Adam parameter updates without checks
        modified = re.sub(
            r'param\.mul\(1 - lr \* weight_decay\)',
            r'param.mul(1 - lr * (weight_decay if weight_decay is not None else 0.0))',
            modified
        )
        
        if content != modified:
            with open(file_path, 'w') as f:
                f.write(modified)
            print(f"Fixed weight decay handling in: {file_path}")
            return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return False

def scan_directory(directory):
    """Scan directory for Python files and fix them."""
    fixed_files = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if fix_file(file_path):
                    fixed_files += 1
    
    return fixed_files

def main():
    """Main function to fix weight decay issues in the codebase."""
    # Get the project root directory
    project_root = Path(__file__).resolve().parents[1]
    
    # Focus on the source directories most likely to contain optimizer code
    target_dirs = [
        project_root / "src" / "nanotron" / "optim",
        project_root / "src" / "nanotron" / "helpers.py",
    ]
    
    total_fixed = 0
    for target_dir in target_dirs:
        if target_dir.is_file():
            if fix_file(target_dir):
                total_fixed += 1
        else:
            total_fixed += scan_directory(target_dir)
    
    print(f"Fixed weight decay handling in {total_fixed} files")
    print("This should prevent 'unsupported operand type(s) for *: 'float' and 'NoneType'' errors")

if __name__ == "__main__":
    main()
