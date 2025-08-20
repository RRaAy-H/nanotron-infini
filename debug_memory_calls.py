#!/usr/bin/env python3
"""
Simple debug script to trace if memory functions are being called at all.
"""

import sys
sys.path.append('src')

import torch
from nanotron import constants
from nanotron.config import get_config_from_file, ParallelismArgs
from nanotron.models import build_model
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.parallel.pipeline_parallel.engine import OneForwardOneBackwardPipelineEngine
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.random import RandomStates, get_current_random_state, get_synced_random_state, set_random_seed
from nanotron.serialize import load_weights
from nanotron.trainer import CONFIG_TO_MODEL_CLASS, mark_tied_parameters
from pathlib import Path

def debug_memory_calls(checkpoint_path: str):
    """Debug if memory functions are being called."""
    
    print("🔍 DEBUGGING MEMORY FUNCTION CALLS")
    print("=" * 50)
    
    checkpoint_path = Path(checkpoint_path)
    
    # Load configuration
    config = get_config_from_file((checkpoint_path / "config.yaml").as_posix())
    constants.CONFIG = config
    
    print(f"✅ Config loaded: turn_on_memory = {config.infini_attention.turn_on_memory}")
    print(f"✅ Segment length: {config.infini_attention.segment_length}")
    
    model_config = config.model.model_config
    
    # Setup parallelism
    parallel_config = ParallelismArgs(
        dp=1, pp=1, tp=1,
        pp_engine=OneForwardOneBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        tp_linear_async_communication=False,
    )
    
    parallel_context = ParallelContext(
        data_parallel_size=1, pipeline_parallel_size=1, tensor_parallel_size=1,
    )
    
    set_random_seed(42)
    
    # Build model
    print("🏗️  Building model...")
    model_config_cls = model_config.__class__.__name__
    
    if parallel_config.tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        random_states = RandomStates(
            {"tp_synced": get_synced_random_state(random_state=get_current_random_state(), pg=parallel_context.tp_pg)}
        )
    else:
        random_states = RandomStates({})
    
    model = build_model(
        model_builder=lambda: CONFIG_TO_MODEL_CLASS[model_config_cls](
            config=model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=random_states,
        ),
        dtype=torch.bfloat16,
        parallel_context=parallel_context,
    )
    
    mark_tied_parameters(model=model, parallel_context=parallel_context, parallel_config=parallel_config)
    sanity_check(root_module=model)
    
    # Load weights
    print("📥 Loading weights...")
    load_weights(model=model, parallel_context=parallel_context, root_folder=checkpoint_path)
    
    # Apply balance factor fix
    print("🔧 Applying balance factor fix...")
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    try:
        from apply_balance_fix_standalone import apply_balance_factor_fix_standalone
        fix_success = apply_balance_factor_fix_standalone(model, checkpoint_path, verbose=False)
        if fix_success:
            print("✅ Balance factors loaded successfully")
        else:
            print("⚠️  Balance factor fix failed")
    except Exception as e:
        print(f"⚠️  Balance factor fix failed: {e}")
    
    model.eval()
    
    # Add simple call counters to first layer
    first_layer = model.model.decoder[0].pp_block.attn
    
    retrieve_call_count = 0
    update_call_count = 0
    
    original_retrieve = first_layer._retrieve_from_memory
    original_update = first_layer._update_memory
    
    def counting_retrieve(query_states, prev_memory, prev_normalization):
        nonlocal retrieve_call_count
        retrieve_call_count += 1
        print(f"🧠 _retrieve_from_memory called #{retrieve_call_count}")
        print(f"   prev_memory is None: {prev_memory is None}")
        print(f"   prev_normalization is None: {prev_normalization is None}")
        if prev_memory is not None:
            print(f"   prev_memory shape: {prev_memory.shape}, norm: {prev_memory.norm().item():.6f}")
        result = original_retrieve(query_states, prev_memory, prev_normalization)
        print(f"   result shape: {result.shape}, norm: {result.norm().item():.6f}")
        return result
    
    def counting_update(prev_memory, prev_normalization, key_states, value_states):
        nonlocal update_call_count
        update_call_count += 1
        print(f"💾 _update_memory called #{update_call_count}")
        print(f"   prev_memory is None: {prev_memory is None}")
        print(f"   key_states shape: {key_states.shape}")
        print(f"   value_states shape: {value_states.shape}")
        memory, normalization = original_update(prev_memory, prev_normalization, key_states, value_states)
        print(f"   returned memory shape: {memory.shape}, norm: {memory.norm().item():.6f}")
        print(f"   returned normalization shape: {normalization.shape}")
        return memory, normalization
    
    first_layer._retrieve_from_memory = counting_retrieve
    first_layer._update_memory = counting_update
    
    # Test with multi-segment input
    batch_size = 1
    seq_len = 2048  # 2 segments
    hidden_size = model_config.hidden_size
    
    print(f"\n🧪 Testing with {seq_len} tokens (should create {seq_len//1024} segments)")
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device=torch.device("cuda"))
    sequence_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=torch.device("cuda"))
    
    print("🔄 Running forward pass...")
    with torch.no_grad():
        output = first_layer(hidden_states, sequence_mask)
    
    print(f"\n📊 RESULTS:")
    print(f"   _retrieve_from_memory called: {retrieve_call_count} times")
    print(f"   _update_memory called: {update_call_count} times")
    print(f"   Expected calls for {seq_len//1024} segments: {seq_len//1024} retrieve, {seq_len//1024} update")
    
    if retrieve_call_count == 0 and update_call_count == 0:
        print("❌ NO MEMORY FUNCTIONS CALLED - Infini-Attention not activating!")
        print("🔍 Checking constants.CONFIG.infini_attention.turn_on_memory...")
        print(f"   Value: {constants.CONFIG.infini_attention.turn_on_memory}")
        print(f"   Type: {type(constants.CONFIG.infini_attention.turn_on_memory)}")
        return False
    elif retrieve_call_count > 0 and update_call_count > 0:
        print("✅ Memory functions are being called - monitoring issue was the problem!")
        return True
    else:
        print("⚠️  Partial calls - something is partially working")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    args = parser.parse_args()
    
    try:
        success = debug_memory_calls(args.checkpoint)
        if success:
            print("\n🎉 Memory functions are working - the issue was monitoring!")
        else:
            print("\n💥 Memory functions are NOT being called - deeper issue exists!")
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
