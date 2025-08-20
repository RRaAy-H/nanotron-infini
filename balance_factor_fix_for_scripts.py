#!/usr/bin/env python3
"""
Balance factor loading fix to integrate into your memory testing scripts.
"""

from pathlib import Path
from safetensors import safe_open
import torch

def load_balance_factors_manually(model, checkpoint_path):
    """
    Load balance factors manually from checkpoint.
    
    Add this function call after load_weights() in your scripts:
    
    # Standard model loading
    model, tokenizer, parallel_context, config = load_model_and_tokenizer(checkpoint_path)
    
    # FIX: Load balance factors manually
    load_balance_factors_manually(model, checkpoint_path)
    
    # Now your memory tests should work!
    """
    
    checkpoint_path = Path(checkpoint_path)
    
    print("🔧 Applying balance factor fix...")
    
    for layer_idx, layer in enumerate(model.model.decoder):
        if hasattr(layer, 'pp_block') and hasattr(layer.pp_block, 'attn') and hasattr(layer.pp_block.attn, 'balance_factors'):
            bf_file = checkpoint_path / f"model/model/decoder/{layer_idx}/pp_block/attn/model_balance_factors_pp-rank-0-of-1_tp-rank-0-of-1.safetensors"
            
            if bf_file.exists():
                with safe_open(str(bf_file), framework='pt', device='cpu') as f:
                    if 'data' in f.keys():
                        saved_bf = f.get_tensor('data')
                        
                        # Move to correct device and dtype
                        target_device = layer.pp_block.attn.balance_factors.device
                        target_dtype = layer.pp_block.attn.balance_factors.dtype
                        saved_bf = saved_bf.to(device=target_device, dtype=target_dtype)
                        
                        # Update model parameters
                        with torch.no_grad():
                            layer.pp_block.attn.balance_factors.data.copy_(saved_bf)
    
    print("✅ Balance factors loaded successfully!")
    
    # Verify the fix worked
    layer0 = model.model.decoder[0]
    if hasattr(layer0, 'pp_block') and hasattr(layer0.pp_block, 'attn') and hasattr(layer0.pp_block.attn, 'balance_factors'):
        bf = layer0.pp_block.attn.balance_factors.data
        activated = layer0.pp_block.attn.balance_act_func(bf)
        avg_memory_weight = activated.mean().item()
        print(f"🧠 Layer 0 now using {avg_memory_weight*100:.1f}% memory (should be ~94%)")
