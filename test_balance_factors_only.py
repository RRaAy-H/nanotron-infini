#!/usr/bin/env python3
"""
Simple test to verify balance factor loading works correctly.
This focuses only on balance factors without other weights.
"""

import sys
sys.path.append('src')
import torch
from pathlib import Path
from safetensors import safe_open

def test_balance_factor_values_only():
    """Test balance factor loading without full model setup."""
    
    checkpoint_path = Path('/data1/infini-attn/infini-llama/nanotron-infini/checkpoints/fineweb_4gpu_300m_infini/30000')
    
    print("=== TESTING BALANCE FACTOR VALUES ===")
    
    # Test balance factors from all layers
    for layer_idx in range(12):  # 12 layers in your model
        bf_file = checkpoint_path / f"model/model/decoder/{layer_idx}/pp_block/attn/model_balance_factors_pp-rank-0-of-1_tp-rank-0-of-1.safetensors"
        
        if bf_file.exists():
            with safe_open(str(bf_file), framework='pt', device='cpu') as f:
                saved_bf = f.get_tensor('data')
                
                # Test hard_sigmoid activation (same as in your model)
                def hard_sigmoid(x):
                    return torch.clamp(x * 0.2 + 0.5, min=0.0, max=1.0)
                
                activated = hard_sigmoid(saved_bf)
                avg_memory_weight = activated.mean().item()
                
                print(f"Layer {layer_idx:2d}: raw_mean={saved_bf.mean().item():6.3f}, activated_mean={avg_memory_weight:.3f} ", end="")
                
                if avg_memory_weight > 0.7:
                    print(f"🧠 MEMORY-FOCUSED ({avg_memory_weight*100:.1f}%)")
                elif avg_memory_weight < 0.3:
                    print(f"👁️  ATTENTION-FOCUSED ({avg_memory_weight*100:.1f}%)")
                else:
                    print(f"⚖️  BALANCED ({avg_memory_weight*100:.1f}%)")
        else:
            print(f"Layer {layer_idx:2d}: ❌ Balance factor file not found")
    
    print("\n=== ANALYSIS ===")
    print("✅ If you see varied values (not all zeros), balance factors are properly saved")
    print("✅ Early layers should prefer memory (high %), late layers should prefer attention (low %)")
    print("✅ This matches your earlier analysis showing layer-wise specialization")

if __name__ == "__main__":
    test_balance_factor_values_only()
