#!/usr/bin/env python3
"""
Debug the model structure to find where balance factors are actually located.
"""

import sys
sys.path.append('src')
import torch
from nanotron import constants
from nanotron.config import get_config_from_file
from nanotron.models import build_model
from nanotron.parallel import ParallelContext
from nanotron.trainer import CONFIG_TO_MODEL_CLASS
from nanotron.random import RandomStates, get_current_random_state, get_synced_random_state, set_random_seed

def debug_model_structure():
    """Debug where balance factors are actually located in the model."""
    
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
    
    print("=== MODEL STRUCTURE ANALYSIS ===")
    
    # Check overall model structure
    print(f"Model type: {type(model)}")
    print(f"Model.model type: {type(model.model)}")
    print(f"Number of decoder layers: {len(model.model.decoder)}")
    
    # Deep dive into first layer
    print(f"\n=== LAYER 0 DETAILED STRUCTURE ===")
    layer0 = model.model.decoder[0]
    print(f"Layer 0 type: {type(layer0)}")
    print(f"Layer 0 attributes: {[attr for attr in dir(layer0) if not attr.startswith('_')]}")
    
    if hasattr(layer0, 'attn'):
        print(f"\nLayer 0 attn type: {type(layer0.attn)}")
        print(f"Layer 0 attn attributes: {[attr for attr in dir(layer0.attn) if not attr.startswith('_')]}")
        
        # Look for balance-related attributes
        balance_attrs = [attr for attr in dir(layer0.attn) if 'balance' in attr.lower()]
        print(f"Balance-related attributes: {balance_attrs}")
        
        # Check all parameters
        print(f"\nAll parameters in layer 0 attention:")
        for name, param in layer0.attn.named_parameters():
            print(f"  {name}: {param.shape}")
    
    # Check the entire model for balance factor parameters
    print(f"\n=== ALL BALANCE FACTOR PARAMETERS IN MODEL ===")
    balance_params = []
    for name, param in model.named_parameters():
        if 'balance' in name.lower():
            balance_params.append((name, param.shape))
            print(f"  {name}: {param.shape}")
    
    if not balance_params:
        print("  ❌ NO BALANCE FACTOR PARAMETERS FOUND!")
        print("  This explains why loading fails - they don't exist in the model")
    else:
        print(f"  ✅ Found {len(balance_params)} balance factor parameters")
    
    # Look for all parameters to understand the naming
    print(f"\n=== SAMPLE PARAMETER NAMES ===")
    all_param_names = list(model.named_parameters())
    print(f"Total parameters: {len(all_param_names)}")
    print("First 10 parameter names:")
    for i, (name, param) in enumerate(all_param_names[:10]):
        print(f"  {i+1:2d}. {name}: {param.shape}")
    
    print("\nAttention-related parameter names:")
    attn_params = [(name, param.shape) for name, param in all_param_names if 'attn' in name]
    for name, shape in attn_params[:10]:  # Show first 10
        print(f"  {name}: {shape}")

if __name__ == "__main__":
    debug_model_structure()
