#!/usr/bin/env python3
"""
Debug script to see all files in checkpoint directories
"""

from pathlib import Path
import os

def inspect_files(checkpoint_path: str):
    checkpoint_path = Path(checkpoint_path)
    model_path = checkpoint_path / "model" / "model"
    
    print("=== File Structure ===")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Model path: {model_path}")
    
    # Check all directories
    for dir_name in ["decoder", "final_layer_norm", "lm_head", "token_position_embeddings"]:
        dir_path = model_path / dir_name
        print(f"\n=== {dir_name.upper()} ===")
        if dir_path.exists():
            print(f"Directory exists: {dir_path}")
            try:
                files = list(dir_path.iterdir())
                if files:
                    for file in sorted(files):
                        print(f"  {file.name} ({file.stat().st_size} bytes)")
                else:
                    print("  Directory is empty")
            except Exception as e:
                print(f"  Error reading directory: {e}")
        else:
            print(f"Directory does not exist: {dir_path}")
    
    # Also check if there are any .safetensors files anywhere in the checkpoint
    print(f"\n=== ALL .safetensors FILES ===")
    for safetensor_file in checkpoint_path.rglob("*.safetensors"):
        print(f"Found: {safetensor_file}")

if __name__ == "__main__":
    checkpoint_path = "/data1/infini-attn/infini-llama/nanotron-infini/checkpoints/fineweb_4gpu_300m_infini/30000"
    inspect_files(checkpoint_path)