#!/usr/bin/env python
# Script to fix Flash Attention deprecated warnings by updating torch.cuda.amp to torch.amp
# This fixes the FutureWarning errors seen during training

import os
import re
import sys
from pathlib import Path

def find_flash_attention_path():
    """Find the Flash Attention installation path"""
    try:
        import flash_attn
        return Path(flash_attn.__file__).parent
    except ImportError:
        print("Flash Attention is not installed. Please install it first.")
        sys.exit(1)

def fix_file(file_path):
    """Replace torch.cuda.amp.custom_fwd/bwd with torch.amp.custom_fwd/bwd"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace torch.cuda.amp.custom_fwd with torch.amp.custom_fwd
    modified = re.sub(
        r'torch\.cuda\.amp\.custom_fwd\((.*?)\)', 
        r'torch.amp.custom_fwd(\1, device_type="cuda")', 
        content
    )
    
    # Replace torch.cuda.amp.custom_bwd with torch.amp.custom_bwd
    modified = re.sub(
        r'torch\.cuda\.amp\.custom_bwd\((.*?)\)', 
        r'torch.amp.custom_bwd(\1, device_type="cuda")', 
        modified
    )
    
    if content != modified:
        print(f"Fixing {file_path}")
        with open(file_path, 'w') as f:
            f.write(modified)
        return True
    return False

def main():
    flash_attn_path = find_flash_attention_path()
    print(f"Flash Attention path: {flash_attn_path}")
    
    # Files to patch
    files_to_check = list(flash_attn_path.glob("**/*.py"))
    fixed_count = 0
    
    for file_path in files_to_check:
        if fix_file(file_path):
            fixed_count += 1
    
    print(f"Fixed {fixed_count} files.")
    if fixed_count > 0:
        print("Flash Attention warnings have been fixed!")
    else:
        print("No fixes were needed.")

if __name__ == "__main__":
    main()
