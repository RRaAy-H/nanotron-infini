#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/fix_flash_attention_glibc.py

"""
Script to patch the Infini-Llama code to handle Flash Attention GLIBC errors gracefully.
This script directly patches the llama.py file to catch and handle Flash Attention import errors.
"""

import os
import sys
import inspect
from pathlib import Path
import importlib.util
import re

# Get project root
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def find_llama_module():
    """Find the llama.py module path in the project."""
    try:
        # First try to import it properly
        from nanotron.models import llama
        return os.path.abspath(inspect.getfile(llama))
    except ImportError:
        # If import fails, search for it
        llama_paths = list(project_root.glob("**/llama.py"))
        if llama_paths:
            return str(llama_paths[0])
        return None

def create_standard_rotary_embedding():
    """Create a standard rotary embedding implementation that doesn't rely on Flash Attention."""
    import torch
    
    class StandardRotaryEmbedding:
        """Standard (non-flash) rotary embedding implementation"""
        def __init__(self, dim, base=10000, precision=torch.float32, learnable=False, device=None):
            super().__init__()
            self.dim = dim
            self.base = base
            self.precision = precision
            self.learnable = learnable
            self.device = device
            
            # Create standard rotary embeddings manually
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv_freq)
            
            if learnable:
                self.inv_freq = torch.nn.Parameter(self.inv_freq)
            
        def _set_cos_sin_cache(self, seq_len, device, dtype):
            self.max_seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device)
            
            # Create freqs
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(dtype))
            
            # Create rotation matrices
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        
        def forward(self, x, seq_len=None):
            if seq_len is None:
                seq_len = x.shape[1]
            
            if not hasattr(self, "max_seq_len_cached") or self.max_seq_len_cached < seq_len:
                self._set_cos_sin_cache(seq_len, x.device, x.dtype)
            
            return (
                self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
                self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            )
    
    return StandardRotaryEmbedding

def patch_llama_module(llama_path):
    """Patch the llama.py file to handle Flash Attention import errors gracefully."""
    if not llama_path or not os.path.exists(llama_path):
        print(f"Error: Llama module not found at {llama_path}")
        return False
    
    print(f"Patching llama module at {llama_path}")
    
    # Read the original file
    with open(llama_path, 'r') as f:
        content = f.read()
    
    # Create a backup
    backup_path = llama_path + '.backup'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"Created backup at {backup_path}")
    
    # Find the Flash Attention import in CausalSelfAttention.__init__
    pattern = r"([ \t]+)from flash_attn\.layers\.rotary import RotaryEmbedding as FlashRotaryEmbedding"
    match = re.search(pattern, content)
    
    if not match:
        print("Warning: Could not find Flash Attention import in llama.py")
        return False
    
    indent = match.group(1)
    
    # Create replacement code with try-except
    replacement = f"{indent}# Try to import Flash Attention, but handle GLIBC errors gracefully\n"
    replacement += f"{indent}try:\n"
    replacement += f"{indent}    from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding\n"
    replacement += f"{indent}    use_flash_rotary = True\n"
    replacement += f"{indent}except (ImportError, OSError) as e:\n"
    replacement += f"{indent}    import warnings\n"
    replacement += f"{indent}    warnings.warn(f\"Flash Attention import failed: {{e}}. Using standard attention implementation.\")\n"
    replacement += f"{indent}    use_flash_rotary = False\n"
    replacement += f"{indent}    # Define a standard rotary embedding as fallback\n"
    replacement += f"{indent}    class FlashRotaryEmbedding:\n"
    replacement += f"{indent}        def __init__(self, dim, base=10000, precision=None, learnable=False, device=None):\n"
    replacement += f"{indent}            import torch\n"
    replacement += f"{indent}            super().__init__()\n"
    replacement += f"{indent}            self.dim = dim\n"
    replacement += f"{indent}            self.base = base\n"
    replacement += f"{indent}            self.precision = precision\n"
    replacement += f"{indent}            self.learnable = learnable\n"
    replacement += f"{indent}            \n"
    replacement += f"{indent}            # Create standard rotary embeddings manually\n"
    replacement += f"{indent}            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))\n"
    replacement += f"{indent}            self.register_buffer(\"inv_freq\", inv_freq)\n"
    replacement += f"{indent}            \n"
    replacement += f"{indent}            if learnable:\n"
    replacement += f"{indent}                self.inv_freq = torch.nn.Parameter(self.inv_freq)\n"
    replacement += f"{indent}            \n"
    replacement += f"{indent}        def _set_cos_sin_cache(self, seq_len, device, dtype):\n"
    replacement += f"{indent}            self.max_seq_len_cached = seq_len\n"
    replacement += f"{indent}            t = torch.arange(seq_len, device=device)\n"
    replacement += f"{indent}            \n"
    replacement += f"{indent}            # Create freqs\n"
    replacement += f"{indent}            freqs = torch.einsum(\"i,j->ij\", t, self.inv_freq.to(dtype))\n"
    replacement += f"{indent}            \n"
    replacement += f"{indent}            # Create rotation matrices\n"
    replacement += f"{indent}            emb = torch.cat((freqs, freqs), dim=-1)\n"
    replacement += f"{indent}            self.cos_cached = emb.cos()[None, None, :, :]\n"
    replacement += f"{indent}            self.sin_cached = emb.sin()[None, None, :, :]\n"
    replacement += f"{indent}        \n"
    replacement += f"{indent}        def forward(self, x, seq_len=None):\n"
    replacement += f"{indent}            if seq_len is None:\n"
    replacement += f"{indent}                seq_len = x.shape[1]\n"
    replacement += f"{indent}            \n"
    replacement += f"{indent}            if not hasattr(self, \"max_seq_len_cached\") or self.max_seq_len_cached < seq_len:\n"
    replacement += f"{indent}                self._set_cos_sin_cache(seq_len, x.device, x.dtype)\n"
    replacement += f"{indent}            \n"
    replacement += f"{indent}            return (\n"
    replacement += f"{indent}                self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),\n"
    replacement += f"{indent}                self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),\n"
    replacement += f"{indent}            )\n"
    
    # Replace the import line
    patched_content = re.sub(pattern, replacement, content)
    
    # Also patch the flash attention usage
    flash_attn_pattern = r"([ \t]+)if use_flash_attention:"
    flash_attn_match = re.search(flash_attn_pattern, patched_content)
    
    if flash_attn_match:
        flash_indent = flash_attn_match.group(1)
        flash_replacement = f"{flash_indent}if use_flash_attention and use_flash_rotary:"
        patched_content = re.sub(flash_attn_pattern, flash_replacement, patched_content)
    else:
        print("Warning: Could not find flash attention usage pattern")
    
    # Write the patched content
    with open(llama_path, 'w') as f:
        f.write(patched_content)
    
    print(f"Successfully patched {llama_path} to handle Flash Attention import errors")
    return True

def main():
    print("Running Flash Attention GLIBC fix for Infini-Llama...")
    llama_path = find_llama_module()
    
    if not llama_path:
        print("Error: Could not find llama module")
        sys.exit(1)
    
    success = patch_llama_module(llama_path)
    
    if success:
        print("Patching successful!")
        print("Flash Attention GLIBC errors will now be caught and handled gracefully.")
        print("The model will automatically fall back to standard attention implementation.")
    else:
        print("Patching failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
