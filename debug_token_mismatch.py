#!/usr/bin/env python3
"""
Debug token embedding mismatch
"""

import torch
from safetensors.torch import load_file
from pathlib import Path

def debug_token_mismatch():
    nanotron_path = Path("/data1/infini-attn/infini-llama/nanotron-infini/checkpoints/fineweb_4gpu_300m_infini/30000")
    hf_path = Path("./hf_converted_model")
    
    # Load Nanotron token embeddings
    nanotron_token_path = nanotron_path / "model" / "model" / "token_position_embeddings" / "pp_block" / "token_embedding" / "model_weight_pp-rank-0-of-1_tp-rank-0-of-1.safetensors"
    nanotron_tokens = load_file(nanotron_token_path)["data"]
    
    # Load HF token embeddings
    hf_tokens = load_file(hf_path / "model.safetensors")["model.embed_tokens.weight"]
    
    print(f"Nanotron shape: {nanotron_tokens.shape}")
    print(f"HF shape: {hf_tokens.shape}")
    print(f"Nanotron dtype: {nanotron_tokens.dtype}")
    print(f"HF dtype: {hf_tokens.dtype}")
    
    # Check if shapes match
    if nanotron_tokens.shape != hf_tokens.shape:
        print("✗ Shapes don't match!")
        return
    
    # Check exact equality
    exact_match = torch.equal(nanotron_tokens, hf_tokens)
    print(f"Exact match: {exact_match}")
    
    # Check approximate equality with different tolerances
    rtol_tests = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    for rtol in rtol_tests:
        close_match = torch.allclose(nanotron_tokens, hf_tokens, rtol=rtol, atol=1e-8)
        print(f"Close match (rtol={rtol}): {close_match}")
        if close_match:
            break
    
    # Check max difference
    diff = torch.abs(nanotron_tokens - hf_tokens)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    print(f"Max absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    
    # Check relative difference
    rel_diff = diff / (torch.abs(nanotron_tokens) + 1e-8)
    max_rel_diff = torch.max(rel_diff).item()
    mean_rel_diff = torch.mean(rel_diff).item()
    print(f"Max relative difference: {max_rel_diff}")
    print(f"Mean relative difference: {mean_rel_diff}")
    
    # Check a few specific values
    print(f"\nFirst 5 values comparison:")
    print(f"Nanotron: {nanotron_tokens[0, :5]}")
    print(f"HF:       {hf_tokens[0, :5]}")
    print(f"Diff:     {(nanotron_tokens - hf_tokens)[0, :5]}")
    
    # Check data types and potential conversion issues
    print(f"\nData type info:")
    print(f"Nanotron min/max: {nanotron_tokens.min().item():.6f} / {nanotron_tokens.max().item():.6f}")
    print(f"HF min/max: {hf_tokens.min().item():.6f} / {hf_tokens.max().item():.6f}")

if __name__ == "__main__":
    debug_token_mismatch()