#!/usr/bin/env python3
"""
Debug balance factor loading to see if they're properly restored from checkpoint
"""

import sys
sys.path.append('src')
import torch
from nanotron import constants
from nanotron.config import get_config_from_file
from nanotron.models import build_model
from nanotron.parallel import ParallelContext
from nanotron.serialize import load_weights
from nanotron.trainer import CONFIG_TO_MODEL_CLASS
from nanotron.random import RandomStates, get_current_random_state, get_synced_random_state, set_random_seed

def debug_balance_factors():
    print("=== BALANCE FACTOR LOADING DEBUG ===")
    
    # Load configuration  
    config = get_config_from_file('/data1/infini-attn/infini-llama/nanotron-infini/checkpoints/fineweb_4gpu_300m_infini/30000/config.yaml')
    constants.CONFIG = config
    
    # Setup minimal model
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
    
    # Build model BEFORE loading weights
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
    
    print("\n=== BEFORE WEIGHT LOADING ===")
    for layer_idx in range(min(3, len(model.model.decoder))):  # Check first 3 layers
        layer = model.model.decoder[layer_idx]
        if hasattr(layer, 'attn') and hasattr(layer.attn, 'balance_factors'):
            bf = layer.attn.balance_factors.data
            print(f"Layer {layer_idx}: balance_factors shape={bf.shape}, mean={bf.mean().item():.6f}, std={bf.std().item():.6f}")
            print(f"  First few values: {bf.flatten()[:5].tolist()}")
        else:
            print(f"Layer {layer_idx}: No balance_factors found")
    
    # Load weights
    print("\n=== LOADING WEIGHTS ===")
    load_weights(model=model, parallel_context=parallel_context, root_folder='/data1/infini-attn/infini-llama/nanotron-infini/checkpoints/fineweb_4gpu_300m_infini/30000')
    model.eval()
    
    print("\n=== AFTER WEIGHT LOADING ===")
    for layer_idx in range(min(3, len(model.model.decoder))):  # Check first 3 layers
        layer = model.model.decoder[layer_idx]
        if hasattr(layer, 'attn') and hasattr(layer.attn, 'balance_factors'):
            bf = layer.attn.balance_factors.data
            print(f"Layer {layer_idx}: balance_factors shape={bf.shape}, mean={bf.mean().item():.6f}, std={bf.std().item():.6f}")
            print(f"  First few values: {bf.flatten()[:5].tolist()}")
            
            # Check if these match your known good values
            if bf.std().item() > 0.1:  # Should have variation if loaded correctly
                print(f"  ✅ GOOD: Balance factors loaded with variation")
            else:
                print(f"  ❌ BAD: Balance factors appear not loaded (low variation)")
        else:
            print(f"Layer {layer_idx}: No balance_factors found")
    
    # Test the balance activation function
    print("\n=== TESTING BALANCE ACTIVATION ===")
    layer = model.model.decoder[0]
    if hasattr(layer, 'attn') and hasattr(layer.attn, 'balance_factors'):
        bf = layer.attn.balance_factors.data
        activated = layer.attn.balance_act_func(bf)
        print(f"Raw balance factors: {bf.flatten()[:5].tolist()}")
        print(f"After activation: {activated.flatten()[:5].tolist()}")
        print(f"Activation function: {layer.attn.balance_act_func}")

if __name__ == "__main__":
    debug_balance_factors()
