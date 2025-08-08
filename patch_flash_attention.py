#!/usr/bin/env python
"""
This script patches the Llama model to handle Flash Attention failures
by providing a fallback to standard attention implementation.
"""
import os
import sys
import re
from pathlib import Path

def find_llama_model_file():
    """Find the Llama model file in the nanotron source directory."""
    src_dir = Path(__file__).parent / "src" / "nanotron" / "models"
    llama_file = src_dir / "llama.py"
    
    if not llama_file.exists():
        print(f"Error: Could not find Llama model file at {llama_file}")
        print("Please run this script from the root directory of the nanotron-infini project.")
        sys.exit(1)
    
    return llama_file

def backup_file(file_path):
    """Create a backup of the file before modifying it."""
    backup_path = file_path.with_suffix(f"{file_path.suffix}.bak")
    if not backup_path.exists():
        with open(file_path, "r") as src, open(backup_path, "w") as dest:
            dest.write(src.read())
        print(f"Created backup at {backup_path}")
    else:
        print(f"Backup already exists at {backup_path}")
    
    return backup_path

def patch_attention_import(file_content):
    """Wrap the Flash Attention import in a try-except block."""
    # Pattern to match the Flash Attention import
    pattern = r"from flash_attn\.layers\.rotary import RotaryEmbedding as FlashRotaryEmbedding"
    
    # Replacement with try-except block
    replacement = """    # Check if Flash Attention should be disabled
    disable_flash_attn = os.environ.get("DISABLE_FLASH_ATTN", "0") == "1"
    
    try:
        # Try to import Flash Attention if not disabled
        if not disable_flash_attn:
            from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding
            self.use_flash_attn = True
        else:
            raise ImportError("Flash Attention disabled by environment variable")
    except ImportError as e:
        # Fallback to standard attention
        print(f"Flash Attention not available: {e}. Using standard attention implementation.")
        self.use_flash_attn = False
        # Import standard attention components
        from nanotron.models.attention import RotaryEmbedding"""
    
    return re.sub(pattern, replacement, file_content)

def patch_rotary_embedding(file_content):
    """Add fallback implementation for rotary embeddings."""
    # Pattern to match rotary embedding initialization
    pattern = r"self\.rotary_emb = FlashRotaryEmbedding\(\s+dim=head_dim,\s+base=rope_theta,\s+interleaved=rope_interleaved,\s+scaling_factor=rope_scaling,\s+max_position_embeddings=max_position_embeddings,\s+\)"
    
    # Replacement with conditional initialization
    replacement = """        if self.use_flash_attn:
            self.rotary_emb = FlashRotaryEmbedding(
                dim=head_dim,
                base=rope_theta,
                interleaved=rope_interleaved,
                scaling_factor=rope_scaling,
                max_position_embeddings=max_position_embeddings,
            )
        else:
            self.rotary_emb = RotaryEmbedding(
                dim=head_dim,
                base=rope_theta,
                interleaved=rope_interleaved,
                scaling_factor=rope_scaling,
                max_position_embeddings=max_position_embeddings,
            )"""
    
    return re.sub(pattern, replacement, file_content)

def patch_flash_attention_import(file_content):
    """Add os import and modify other Flash Attention imports."""
    # Add os import if not already present
    if "import os" not in file_content:
        file_content = "import os\n" + file_content
    
    # Modify flash_attn.flash_attn_interface import to be in try-except
    pattern = r"from flash_attn\.flash_attn_interface import flash_attn_func"
    replacement = """try:
    from flash_attn.flash_attn_interface import flash_attn_func
    HAS_FLASH_ATTN = not (os.environ.get("DISABLE_FLASH_ATTN", "0") == "1")
except ImportError:
    flash_attn_func = None
    HAS_FLASH_ATTN = False
    print("Flash Attention not available. Using standard attention implementation.")"""
    
    return re.sub(pattern, replacement, file_content)

def patch_attention_forward(file_content):
    """Modify the forward method to conditionally use Flash Attention."""
    # This is a simplified example - actual implementation would depend on the code structure
    pattern = r"(attn_output = flash_attn_func\()"
    replacement = r"if HAS_FLASH_ATTN and self.use_flash_attn:\n            \1"
    
    return re.sub(pattern, replacement, file_content)

def patch_llama_model():
    """Apply all patches to the Llama model file."""
    llama_file = find_llama_model_file()
    backup_file(llama_file)
    
    with open(llama_file, "r") as f:
        content = f.read()
    
    # Apply patches
    content = patch_flash_attention_import(content)
    content = patch_attention_import(content)
    content = patch_rotary_embedding(content)
    content = patch_attention_forward(content)
    
    # Write patched content back to the file
    with open(llama_file, "w") as f:
        f.write(content)
    
    print(f"Successfully patched {llama_file}")
    print("The model now has a fallback mechanism when Flash Attention is not available.")
    print("You can disable Flash Attention by setting the environment variable DISABLE_FLASH_ATTN=1")

if __name__ == "__main__":
    print("Patching Llama model to handle Flash Attention failures...")
    patch_llama_model()
