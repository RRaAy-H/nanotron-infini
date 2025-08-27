#!/usr/bin/env python3
"""
Debug script to inspect Nanotron checkpoint structure
"""

from safetensors.torch import load_file
from pathlib import Path

def inspect_checkpoint(checkpoint_path: str):
    checkpoint_path = Path(checkpoint_path)
    model_path = checkpoint_path / "model" / "model"
    
    print("=== Nanotron Checkpoint Structure ===")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Model path: {model_path}")
    
    # Check all directories
    for dir_name in ["decoder", "final_layer_norm", "lm_head", "token_position_embeddings"]:
        dir_path = model_path / dir_name
        if dir_path.exists():
            print(f"\n=== {dir_name.upper()} ===")
            for safetensor_file in sorted(dir_path.glob("*.safetensors")):
                print(f"\nFile: {safetensor_file.name}")
                try:
                    weights = load_file(safetensor_file)
                    for key in sorted(weights.keys()):
                        print(f"  {key}: {weights[key].shape}")
                except Exception as e:
                    print(f"  Error loading {safetensor_file}: {e}")
        else:
            print(f"\n=== {dir_name.upper()} === (NOT FOUND)")

if __name__ == "__main__":
    checkpoint_path = "/data1/infini-attn/infini-llama/nanotron-infini/checkpoints/fineweb_4gpu_300m_infini/30000"
    inspect_checkpoint(checkpoint_path)