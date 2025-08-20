#!/usr/bin/env python3
"""
Standalone balance factor fix that works independently of load_weights.
Apply this after model creation but before inference/generation.
"""

import sys
sys.path.append('src')
from pathlib import Path
from safetensors import safe_open
import torch

def apply_balance_factor_fix_standalone(model, checkpoint_path, verbose=True):
    """
    Apply balance factor fix independently of load_weights.
    
    This works around the parameter type issues in load_weights by applying
    only the balance factor fix directly.
    """
    
    checkpoint_path = Path(checkpoint_path)
    
    if verbose:
        print("🔧 Applying standalone balance factor fix...")
    
    loaded_count = 0
    total_count = 0
    
    for layer_idx, layer in enumerate(model.model.decoder):
        total_count += 1
        
        if hasattr(layer, 'pp_block') and hasattr(layer.pp_block, 'attn') and hasattr(layer.pp_block.attn, 'balance_factors'):
            bf_file = checkpoint_path / f"model/model/decoder/{layer_idx}/pp_block/attn/model_balance_factors_pp-rank-0-of-1_tp-rank-0-of-1.safetensors"
            
            if bf_file.exists():
                try:
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
                            
                            loaded_count += 1
                            
                            if verbose:
                                activated = layer.pp_block.attn.balance_act_func(saved_bf)
                                avg_memory_weight = activated.mean().item()
                                print(f"  ✅ Layer {layer_idx}: {avg_memory_weight*100:.1f}% memory")
                                
                except Exception as e:
                    if verbose:
                        print(f"  ❌ Layer {layer_idx}: Failed to load: {e}")
            else:
                if verbose:
                    print(f"  ❌ Layer {layer_idx}: File not found")
    
    if loaded_count > 0:
        if verbose:
            print(f"✅ Successfully loaded balance factors for {loaded_count}/{total_count} layers")
        return True
    else:
        if verbose:
            print(f"❌ Failed to load any balance factors")
        return False

# Test it directly
if __name__ == "__main__":
    from nanotron import constants
    from nanotron.config import get_config_from_file
    from nanotron.models import build_model
    from nanotron.parallel import ParallelContext
    from nanotron.trainer import CONFIG_TO_MODEL_CLASS
    from nanotron.random import RandomStates, get_current_random_state, get_synced_random_state, set_random_seed
    
    checkpoint_path = '/data1/infini-attn/infini-llama/nanotron-infini/checkpoints/fineweb_4gpu_300m_infini/30000'
    
    print("=== TESTING STANDALONE BALANCE FACTOR FIX ===")
    
    # Load configuration  
    config = get_config_from_file(f'{checkpoint_path}/config.yaml')
    constants.CONFIG = config
    
    # Setup model (minimal, no weight loading)
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
    
    print("\n=== BEFORE FIX ===")
    layer0 = model.model.decoder[0]
    bf = layer0.pp_block.attn.balance_factors.data
    print(f"Layer 0: mean={bf.mean().item():.6f}, std={bf.std().item():.6f}")
    
    # Apply standalone fix
    print("\n=== APPLYING STANDALONE FIX ===")
    success = apply_balance_factor_fix_standalone(model, checkpoint_path, verbose=True)
    
    print(f"\n=== VERIFICATION ===")
    if success:
        layer0 = model.model.decoder[0]
        bf = layer0.pp_block.attn.balance_factors.data
        activated = layer0.pp_block.attn.balance_act_func(bf)
        print(f"Layer 0: raw_mean={bf.mean().item():.3f}, activated_mean={activated.mean().item():.3f}")
        print("🎉 Standalone fix working!")
    else:
        print("❌ Standalone fix failed")
