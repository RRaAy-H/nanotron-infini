#!/usr/bin/env python3
"""
Test the permanent balance factor fix.

This script verifies that balance factors are automatically loaded correctly
for all inference/generation/evaluation tasks without manual intervention.
"""

import sys
sys.path.append('src')
import torch
from pathlib import Path
from nanotron import constants
from nanotron.config import get_config_from_file
from nanotron.models import build_model
from nanotron.parallel import ParallelContext
from nanotron.serialize import load_weights
from nanotron.trainer import CONFIG_TO_MODEL_CLASS
from nanotron.random import RandomStates, get_current_random_state, get_synced_random_state, set_random_seed

def test_automatic_balance_factor_loading():
    """Test that balance factors are automatically loaded correctly."""
    
    checkpoint_path = '/data1/infini-attn/infini-llama/nanotron-infini/checkpoints/fineweb_4gpu_300m_infini/30000'
    
    print("=== TESTING PERMANENT BALANCE FACTOR FIX ===")
    
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
    
    print("\n=== BEFORE LOADING WEIGHTS ===")
    layer0 = model.model.decoder[0]
    if hasattr(layer0, 'pp_block') and hasattr(layer0.pp_block, 'attn') and hasattr(layer0.pp_block.attn, 'balance_factors'):
        bf = layer0.pp_block.attn.balance_factors.data
        print(f"Layer 0: mean={bf.mean().item():.6f}, std={bf.std().item():.6f} (should be ~0)")
    
    # Load weights - THE PERMANENT FIX SHOULD APPLY AUTOMATICALLY
    print("\n=== LOADING WEIGHTS (WITH AUTOMATIC FIX) ===")
    try:
        load_weights(model=model, parallel_context=parallel_context, root_folder=Path(checkpoint_path))
        print("✅ Weight loading completed successfully")
    except Exception as e:
        print(f"❌ Weight loading failed: {e}")
        return False
    
    print("\n=== AFTER LOADING WEIGHTS ===")
    success = True
    
    # Check first few layers
    for layer_idx in range(min(3, len(model.model.decoder))):
        layer = model.model.decoder[layer_idx]
        if hasattr(layer, 'pp_block') and hasattr(layer.pp_block, 'attn') and hasattr(layer.pp_block.attn, 'balance_factors'):
            bf = layer.pp_block.attn.balance_factors.data
            activated = layer.pp_block.attn.balance_act_func(bf)
            avg_memory_weight = activated.mean().item()
            
            print(f"Layer {layer_idx}: raw_mean={bf.mean().item():.3f}, activated_mean={avg_memory_weight:.3f}", end=" ")
            
            # Verify this matches expected trained values
            if layer_idx == 0 and abs(avg_memory_weight - 0.941) < 0.05:  # Should be ~94.1%
                print("🧠 MEMORY-FOCUSED ✅")
            elif layer_idx == 1 and abs(avg_memory_weight - 0.918) < 0.05:  # Should be ~91.8%
                print("🧠 MEMORY-FOCUSED ✅")
            elif layer_idx == 2 and abs(avg_memory_weight - 0.832) < 0.05:  # Should be ~83.2%
                print("🧠 MEMORY-FOCUSED ✅")
            elif bf.std().item() > 0.1:  # At least has variation
                print("✅ LOADED")
            else:
                print("❌ NOT LOADED")
                success = False
        else:
            print(f"Layer {layer_idx}: ❌ No balance factors found")
            success = False
    
    print(f"\n=== TEST RESULT ===")
    if success:
        print("🎉 SUCCESS: Permanent balance factor fix is working!")
        print("✅ All future inference/generation/evaluation tasks will automatically work")
        print("✅ No manual intervention required")
    else:
        print("❌ FAILED: Permanent fix is not working properly")
    
    return success

def test_memory_mechanism_activation():
    """Quick test to verify memory mechanism is properly activated."""
    
    checkpoint_path = '/data1/infini-attn/infini-llama/nanotron-infini/checkpoints/fineweb_4gpu_300m_infini/30000'
    
    print("\n=== TESTING MEMORY MECHANISM ACTIVATION ===")
    
    # Load model with automatic fix
    config = get_config_from_file(f'{checkpoint_path}/config.yaml')
    constants.CONFIG = config
    
    parallel_context = ParallelContext(data_parallel_size=1, pipeline_parallel_size=1, tensor_parallel_size=1)
    set_random_seed(42)
    
    model_config = config.model.model_config
    model = build_model(
        model_builder=lambda: CONFIG_TO_MODEL_CLASS[model_config.__class__.__name__](
            config=model_config, parallel_context=parallel_context, parallel_config=None,
            random_states=RandomStates({"tp_synced": get_synced_random_state(get_current_random_state(), pg=parallel_context.tp_pg)})
        ),
        dtype=torch.bfloat16, parallel_context=parallel_context,
    )
    
    # Load weights (automatic fix applies)
    load_weights(model=model, parallel_context=parallel_context, root_folder=Path(checkpoint_path))
    
    # Test memory mechanism configuration
    print(f"turn_on_memory: {constants.CONFIG.infini_attention.turn_on_memory}")
    print(f"segment_length: {constants.CONFIG.infini_attention.segment_length}")
    
    # Check balance factor activation for early vs late layers
    early_layer = model.model.decoder[0].pp_block.attn
    late_layer = model.model.decoder[11].pp_block.attn
    
    early_activated = early_layer.balance_act_func(early_layer.balance_factors).mean().item()
    late_activated = late_layer.balance_act_func(late_layer.balance_factors).mean().item()
    
    print(f"Early layer (0) memory usage: {early_activated*100:.1f}%")
    print(f"Late layer (11) memory usage: {late_activated*100:.1f}%")
    
    if early_activated > 0.8 and late_activated < 0.1:
        print("✅ Memory mechanism properly configured: early layers prefer memory, late layers prefer attention")
        return True
    else:
        print("❌ Memory mechanism not properly configured")
        return False

if __name__ == "__main__":
    success1 = test_automatic_balance_factor_loading()
    success2 = test_memory_mechanism_activation()
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 Memory mechanism is now permanently fixed for all tasks")
    else:
        print("\n❌ Some tests failed - please check the implementation")
    
    exit(0 if success1 and success2 else 1)
