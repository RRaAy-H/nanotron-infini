#!/usr/bin/env python3
"""
Test hooking pipeline blocks directly instead of model.forward.
"""

import sys
sys.path.append('src')

import torch
from nanotron import constants
from nanotron.config import get_config_from_file, GenerationArgs, ParallelismArgs
from nanotron.generation.decode import GenerationInput, TokenizerConfig, decode_text
from nanotron.models import build_model
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.parallel.pipeline_parallel.engine import OneForwardOneBackwardPipelineEngine
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.random import RandomStates, get_current_random_state, get_synced_random_state, set_random_seed
from nanotron.serialize import load_weights
from nanotron.trainer import CONFIG_TO_MODEL_CLASS, mark_tied_parameters
from transformers import AutoTokenizer
from pathlib import Path

def test_pipeline_hooks():
    """Test hooking pipeline blocks directly."""
    
    checkpoint_path = "/data1/infini-attn/infini-llama/nanotron-infini/checkpoints/fineweb_4gpu_300m_infini/30000"
    checkpoint_path = Path(checkpoint_path)
    
    # Setup model (same as before)
    config = get_config_from_file((checkpoint_path / "config.yaml").as_posix())
    constants.CONFIG = config
    
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
    
    model_config = config.model.model_config
    random_states = RandomStates({
        "tp_synced": get_synced_random_state(random_state=get_current_random_state(), pg=parallel_context.tp_pg)
    })
    
    model = build_model(
        model_builder=lambda: CONFIG_TO_MODEL_CLASS[model_config.__class__.__name__](
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
    load_weights(model=model, parallel_context=parallel_context, root_folder=checkpoint_path)
    
    # Apply balance factor fix
    from apply_balance_fix_standalone import apply_balance_factor_fix_standalone
    apply_balance_factor_fix_standalone(model, checkpoint_path, verbose=False)
    
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    print("🔧 Hooking PIPELINE BLOCKS instead of model.forward...")
    
    # Hook the pipeline blocks directly
    call_counts = {'block_calls': 0, 'retrieve': 0, 'update': 0}
    
    # Hook every decoder pipeline block
    for layer_idx, pipeline_block in enumerate(model.model.decoder):
        print(f"   Hooking pipeline block {layer_idx}: {type(pipeline_block)}")
        
        # Hook the pipeline block's forward method
        original_forward = pipeline_block.forward
        
        def create_block_forward_hook(layer_idx):
            def hooked_forward(*args, **kwargs):
                call_counts['block_calls'] += 1
                print(f"🔧 Pipeline block {layer_idx} forward call #{call_counts['block_calls']}")
                
                # Also hook the attention layer within this block
                if hasattr(pipeline_block, 'pp_block') and hasattr(pipeline_block.pp_block, 'attn'):
                    attn_layer = pipeline_block.pp_block.attn
                    
                    # Hook memory functions for this specific call
                    original_retrieve = attn_layer._retrieve_from_memory
                    original_update = attn_layer._update_memory
                    
                    def counting_retrieve(query_states, prev_memory, prev_normalization):
                        call_counts['retrieve'] += 1
                        print(f"    🧠 Layer {layer_idx}: Memory retrieve #{call_counts['retrieve']} (prev_memory exists: {prev_memory is not None})")
                        return original_retrieve(query_states, prev_memory, prev_normalization)
                    
                    def counting_update(prev_memory, prev_normalization, key_states, value_states):
                        call_counts['update'] += 1
                        print(f"    💾 Layer {layer_idx}: Memory update #{call_counts['update']}")
                        return original_update(prev_memory, prev_normalization, key_states, value_states)
                    
                    # Temporarily hook the memory functions
                    attn_layer._retrieve_from_memory = counting_retrieve
                    attn_layer._update_memory = counting_update
                    
                    try:
                        result = original_forward(*args, **kwargs)
                        return result
                    finally:
                        # Restore original functions
                        attn_layer._retrieve_from_memory = original_retrieve
                        attn_layer._update_memory = original_update
                else:
                    return original_forward(*args, **kwargs)
            
            return hooked_forward
        
        # Replace the pipeline block's forward method
        pipeline_block.forward = create_block_forward_hook(layer_idx)
    
    print(f"✅ Hooked {len(model.model.decoder)} pipeline blocks")
    
    # Test with decode_text
    print("\n🚀 Testing decode_text with pipeline block hooks...")
    prompt_text = "The quick brown fox jumps over the lazy dog. " * 200  # ~1800 tokens
    
    try:
        outputs = decode_text(
            input_iter=[GenerationInput(text=prompt_text)],
            tokenizer=tokenizer,
            model=model,  # Note: Pass the full model, not model.model
            parallel_context=parallel_context,
            max_new_tokens=3,
            max_micro_batch_size=1,
            generation_config=GenerationArgs(sampler="greedy", use_cache=False),
            tokenizer_config=TokenizerConfig(max_input_length=4096),
        )
        
        print(f"\n✅ decode_text completed!")
        print(f"📊 Results:")
        print(f"   Pipeline block calls: {call_counts['block_calls']}")
        print(f"   Memory retrievals: {call_counts['retrieve']}")
        print(f"   Memory updates: {call_counts['update']}")
        
        if call_counts['block_calls'] > 0:
            print("✅ Pipeline blocks are being called!")
            if call_counts['retrieve'] > 0 or call_counts['update'] > 0:
                print("🎉 MEMORY FUNCTIONS ARE WORKING IN decode_text!")
            else:
                print("❌ Pipeline blocks called but no memory activity")
        else:
            print("❌ No pipeline block calls detected")
            
    except Exception as e:
        print(f"❌ decode_text failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline_hooks()

