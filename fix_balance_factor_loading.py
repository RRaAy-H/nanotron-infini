#!/usr/bin/env python3
"""
Manual fix for balance factor loading issue.
This script demonstrates how to properly load balance factors from checkpoint.
"""

import sys
sys.path.append('src')
import torch
from pathlib import Path
from safetensors import safe_open
from nanotron import constants
from nanotron.config import get_config_from_file

def load_balance_factors_manually(model, checkpoint_path):
    """Manually load balance factors from checkpoint after model loading."""
    
    checkpoint_path = Path(checkpoint_path)
    
    print("=== MANUALLY LOADING BALANCE FACTORS ===")
    
    for layer_idx, layer in enumerate(model.model.decoder):
        if hasattr(layer, 'attn') and hasattr(layer.attn, 'balance_factors'):
            # Construct the checkpoint file path
            bf_file = checkpoint_path / f"model/model/decoder/{layer_idx}/pp_block/attn/model_balance_factors_pp-rank-0-of-1_tp-rank-0-of-1.safetensors"
            
            if bf_file.exists():
                print(f"Loading balance factors for layer {layer_idx}...")
                
                # Load from checkpoint
                with safe_open(str(bf_file), framework='pt', device='cpu') as f:
                    keys = list(f.keys())
                    if 'data' in keys:
                        saved_bf = f.get_tensor('data')
                        
                        # Move to correct device and dtype
                        target_device = layer.attn.balance_factors.device
                        target_dtype = layer.attn.balance_factors.dtype
                        saved_bf = saved_bf.to(device=target_device, dtype=target_dtype)
                        
                        # Update model parameters
                        with torch.no_grad():
                            layer.attn.balance_factors.data.copy_(saved_bf)
                        
                        print(f"  ✅ Layer {layer_idx}: Loaded balance factors with mean={saved_bf.mean().item():.6f}, std={saved_bf.std().item():.6f}")
                    else:
                        print(f"  ❌ Layer {layer_idx}: No 'data' key found in {keys}")
            else:
                print(f"  ❌ Layer {layer_idx}: Balance factor file not found: {bf_file}")
        else:
            print(f"  ❌ Layer {layer_idx}: No balance_factors attribute found")
    
    print("=== BALANCE FACTOR LOADING COMPLETE ===")

def test_balance_factor_fix():
    """Test the balance factor loading fix."""
    
    # Load your model normally (this will initialize balance factors to zero)
    from nanotron.models import build_model
    from nanotron.parallel import ParallelContext
    from nanotron.serialize import load_weights
    from nanotron.trainer import CONFIG_TO_MODEL_CLASS
    from nanotron.random import RandomStates, get_current_random_state, get_synced_random_state, set_random_seed
    
    checkpoint_path = '/data1/infini-attn/infini-llama/nanotron-infini/checkpoints/fineweb_4gpu_300m_infini/30000'
    
    # Load configuration  
    config = get_config_from_file(f'{checkpoint_path}/config.yaml')
    constants.CONFIG = config
    
    # Setup model
    parallel_context = ParallelContext(
        data_parallel_size=1,
        pipeline_parallel_size=1, 
        tensor_parallel_size=1,
    )
    
    set_random_seed(42)
    
    model_config = config.model.model_config
    model_config_cls = model_config.__class__.__name__
    
    random_states = RandomStates({"tp_synced": get_synced_random_state(
        random_state=get_current_random_state(), 
        pg=parallel_context.tp_pg
    )})
    
    model = build_model(
        model_builder=lambda: CONFIG_TO_MODEL_CLASS[model_config_cls](
            config=model_config,
            parallel_context=parallel_context,
            parallel_config=None,
            random_states=random_states,
        ),
        dtype=torch.bfloat16,
        parallel_context=parallel_context,
    )
    
    print("\n=== BEFORE STANDARD WEIGHT LOADING ===")
    for layer_idx in range(min(3, len(model.model.decoder))):
        layer = model.model.decoder[layer_idx]
        if hasattr(layer, 'attn') and hasattr(layer.attn, 'balance_factors'):
            bf = layer.attn.balance_factors.data
            print(f"Layer {layer_idx}: mean={bf.mean().item():.6f}, std={bf.std().item():.6f}")
    
    # Load standard weights (this won't fix balance factors)
    load_weights(model=model, parallel_context=parallel_context, root_folder=Path(checkpoint_path))
    
    print("\n=== AFTER STANDARD WEIGHT LOADING (BROKEN) ===")
    for layer_idx in range(min(3, len(model.model.decoder))):
        layer = model.model.decoder[layer_idx]
        if hasattr(layer, 'attn') and hasattr(layer.attn, 'balance_factors'):
            bf = layer.attn.balance_factors.data
            print(f"Layer {layer_idx}: mean={bf.mean().item():.6f}, std={bf.std().item():.6f}")
    
    # NOW APPLY THE FIX
    load_balance_factors_manually(model, checkpoint_path)
    
    print("\n=== AFTER MANUAL BALANCE FACTOR LOADING (FIXED!) ===")
    for layer_idx in range(min(3, len(model.model.decoder))):
        layer = model.model.decoder[layer_idx]
        if hasattr(layer, 'attn') and hasattr(layer.attn, 'balance_factors'):
            bf = layer.attn.balance_factors.data
            print(f"Layer {layer_idx}: mean={bf.mean().item():.6f}, std={bf.std().item():.6f}")
            
            # Test activation function
            activated = layer.attn.balance_act_func(bf)
            print(f"  After activation: mean={activated.mean().item():.6f}, range=[{activated.min().item():.6f}, {activated.max().item():.6f}]")

if __name__ == "__main__":
    test_balance_factor_fix()
