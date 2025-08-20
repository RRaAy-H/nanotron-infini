#!/usr/bin/env python3
"""
Compare direct model call vs decode_text to understand execution differences.
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

def test_execution_paths():
    """Test direct model call vs decode_text execution paths."""
    
    checkpoint_path = "/data1/infini-attn/infini-llama/nanotron-infini/checkpoints/fineweb_4gpu_300m_infini/30000"
    checkpoint_path = Path(checkpoint_path)
    
    # Setup model (same as monitoring script)
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
    
    # Hook first layer
    first_layer = model.model.decoder[0].pp_block.attn
    call_counts = {'retrieve': 0, 'update': 0}
    
    original_retrieve = first_layer._retrieve_from_memory
    original_update = first_layer._update_memory
    
    def counting_retrieve(query_states, prev_memory, prev_normalization):
        call_counts['retrieve'] += 1
        print(f"🧠 Memory retrieve #{call_counts['retrieve']} (prev_memory exists: {prev_memory is not None})")
        return original_retrieve(query_states, prev_memory, prev_normalization)
    
    def counting_update(prev_memory, prev_normalization, key_states, value_states):
        call_counts['update'] += 1
        print(f"💾 Memory update #{call_counts['update']}")
        return original_update(prev_memory, prev_normalization, key_states, value_states)
    
    first_layer._retrieve_from_memory = counting_retrieve
    first_layer._update_memory = counting_update
    
    # Test 1: Direct model call with long sequence
    print("=" * 60)
    print("TEST 1: Direct model call with 2048 tokens")
    print("=" * 60)
    
    call_counts = {'retrieve': 0, 'update': 0}
    
    # Create a 2048 token input
    prompt_text = "The quick brown fox jumps over the lazy dog. " * 200
    input_ids = torch.tensor([tokenizer.encode(prompt_text)[:2048]], dtype=torch.long, device=torch.device("cuda"))
    input_mask = torch.ones_like(input_ids, dtype=torch.bool)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Sequence length: {input_ids.shape[1]}")
    print(f"Expected segments: {input_ids.shape[1] // 1024}")
    
    with torch.no_grad():
        output = model.model(input_ids=input_ids, input_mask=input_mask)
    
    print(f"Direct call results: {call_counts['retrieve']} retrieve, {call_counts['update']} update")
    
    # Test 2: decode_text with same prompt
    print("\n" + "=" * 60)
    print("TEST 2: decode_text with same prompt")
    print("=" * 60)
    
    call_counts = {'retrieve': 0, 'update': 0}
    
    prompt_text = "The quick brown fox jumps over the lazy dog. " * 200
    
    try:
        outputs = decode_text(
            input_iter=[GenerationInput(text=prompt_text)],
            tokenizer=tokenizer,
            model=model.model,
            parallel_context=parallel_context,
            max_new_tokens=5,
            max_micro_batch_size=1,
            generation_config=GenerationArgs(sampler="greedy", use_cache=False),
            tokenizer_config=TokenizerConfig(max_input_length=4096),
        )
        
        print(f"decode_text results: {call_counts['retrieve']} retrieve, {call_counts['update']} update")
        
    except Exception as e:
        print(f"decode_text failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Analyze decode_text behavior by hooking model.forward
    print("\n" + "=" * 60)
    print("TEST 3: Analyzing decode_text forward calls")
    print("=" * 60)
    
    call_counts = {'retrieve': 0, 'update': 0}
    forward_calls = []
    
    original_forward = model.model.forward
    
    def debug_forward(input_ids, input_mask, *args, **kwargs):
        forward_calls.append({
            'input_shape': input_ids.shape if isinstance(input_ids, torch.Tensor) else str(type(input_ids)),
            'sequence_length': input_ids.shape[1] if isinstance(input_ids, torch.Tensor) else 'Unknown'
        })
        print(f"🔍 Forward call #{len(forward_calls)}: input_ids shape = {input_ids.shape if isinstance(input_ids, torch.Tensor) else type(input_ids)}")
        return original_forward(input_ids, input_mask, *args, **kwargs)
    
    model.model.forward = debug_forward
    
    try:
        outputs = decode_text(
            input_iter=[GenerationInput(text=prompt_text)],
            tokenizer=tokenizer,
            model=model.model,
            parallel_context=parallel_context,
            max_new_tokens=3,
            max_micro_batch_size=1,
            generation_config=GenerationArgs(sampler="greedy", use_cache=False),
            tokenizer_config=TokenizerConfig(max_input_length=4096),
        )
        
        print(f"\nSummary of forward calls:")
        for i, call in enumerate(forward_calls):
            print(f"  Call {i+1}: {call['input_shape']}, seq_len = {call['sequence_length']}")
        
        print(f"\nMemory function calls during decode_text: {call_counts['retrieve']} retrieve, {call_counts['update']} update")
        
        if not forward_calls:
            print("❌ NO FORWARD CALLS DETECTED!")
        elif all(call['sequence_length'] < 1024 for call in forward_calls if isinstance(call['sequence_length'], int)):
            print("❌ ALL FORWARD CALLS HAVE SEQUENCE LENGTH < 1024 - Memory never triggered!")
        
    except Exception as e:
        print(f"decode_text failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original forward
        model.model.forward = original_forward

if __name__ == "__main__":
    test_execution_paths()
