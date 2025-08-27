#!/usr/bin/env python3
"""
Debug the actual weight keys in the checkpoint
"""

from safetensors.torch import load_file
from pathlib import Path

checkpoint_path = Path("/data1/infini-attn/infini-llama/nanotron-infini/checkpoints/fineweb_4gpu_300m_infini/30000")
model_path = checkpoint_path / "model" / "model"

# Check a few sample files to see actual key names
sample_files = [
    model_path / "token_position_embeddings" / "pp_block" / "token_embedding" / "model_weight_pp-rank-0-of-1_tp-rank-0-of-1.safetensors",
    model_path / "decoder" / "0" / "pp_block" / "attn" / "qkv_proj" / "model_weight_pp-rank-0-of-1_tp-rank-0-of-1.safetensors",
    model_path / "decoder" / "0" / "pp_block" / "input_layernorm" / "model_weight.safetensors",
    model_path / "final_layer_norm" / "pp_block" / "model_weight.safetensors",
    model_path / "lm_head" / "pp_block" / "model_weight_pp-rank-0-of-1_tp-rank-0-of-1.safetensors"
]

for file_path in sample_files:
    if file_path.exists():
        print(f"\n=== {file_path.name} ===")
        weights = load_file(file_path)
        for key in weights.keys():
            print(f"  {key}: {weights[key].shape}")
    else:
        print(f"\n=== {file_path.name} === (NOT FOUND)")