#!/usr/bin/env python3
"""
Quick test to verify basic generation is working with balance factor fix applied.
This will help debug why evaluation scripts are failing.
"""

import sys
sys.path.append('src')

import torch
from nanotron import constants
from nanotron.config import get_config_from_file, GenerationArgs, ParallelismArgs
from nanotron.generation.decode import GenerationInput, TokenizerConfig, decode_text
from nanotron.models import build_model
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.engine import OneForwardOneBackwardPipelineEngine
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.random import RandomStates, get_current_random_state, get_synced_random_state, set_random_seed
from nanotron.serialize import load_weights
from nanotron.trainer import CONFIG_TO_MODEL_CLASS, mark_tied_parameters
from pathlib import Path

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

def test_basic_generation(checkpoint_path: str):
    """Test basic text generation to debug evaluation failures."""
    
    print("=== QUICK GENERATION TEST ===")
    print(f"Checkpoint: {checkpoint_path}")
    
    checkpoint_path = Path(checkpoint_path)
    
    # Load configuration
    config = get_config_from_file((checkpoint_path / "config.yaml").as_posix())
    constants.CONFIG = config
    
    model_config = config.model.model_config
    tokenizer_path = config.tokenizer.tokenizer_name_or_path
    
    # Setup parallelism
    parallel_config = ParallelismArgs(
        dp=1,
        pp=1,
        tp=1,
        pp_engine=OneForwardOneBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        tp_linear_async_communication=False,
    )
    
    # Initialize parallel context
    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )
    
    # Set random seed
    set_random_seed(42)
    
    # Build model
    print("Building model...")
    model = build_model(
        model_builder=lambda: CONFIG_TO_MODEL_CLASS[config.model_config](
            config=model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=RandomStates(),
        ),
        parallel_context=parallel_context,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    
    # Mark tied parameters
    mark_tied_parameters(model=model, parallel_context=parallel_context, parallel_config=parallel_config)
    
    # Load weights
    print("Loading weights...")
    load_weights(model=model, parallel_context=parallel_context, root_folder=checkpoint_path)
    
    # Apply balance factor fix
    print("🔧 Applying balance factor fix...")
    try:
        from apply_balance_fix_standalone import apply_balance_factor_fix_standalone
        fix_success = apply_balance_factor_fix_standalone(model, checkpoint_path, verbose=False)
        if fix_success:
            print("✅ Balance factors loaded successfully")
        else:
            print("⚠️ Balance factor fix may have failed")
    except Exception as e:
        print(f"⚠️ Balance factor fix failed: {e}")
    
    model.eval()
    
    # Load tokenizer
    print("Loading tokenizer...")
    if AutoTokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
    else:
        raise ImportError("transformers library is required for tokenizer")
    
    # Test simple generation
    print("\n=== TESTING BASIC GENERATION ===")
    test_prompt = "The quick brown fox"
    print(f"Input: '{test_prompt}'")
    
    try:
        print("Attempting text generation...")
        outputs = decode_text(
            input_iter=[GenerationInput(text=test_prompt)],
            tokenizer=tokenizer,
            model=model.model,
            parallel_context=parallel_context,
            max_new_tokens=10,
            max_micro_batch_size=1,
            generation_config=GenerationArgs(sampler="greedy", use_cache=False),
            tokenizer_config=TokenizerConfig(max_input_length=100),
        )
        
        output_list = list(outputs)
        if output_list and len(output_list) > 0:
            generated_text = output_list[0]
            print(f"✅ Generation successful!")
            print(f"Output: '{generated_text}'")
            return True
        else:
            print("❌ Generation returned empty result")
            return False
            
    except Exception as e:
        import traceback
        print(f"❌ Generation failed: {type(e).__name__}: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    args = parser.parse_args()
    
    success = test_basic_generation(args.checkpoint)
    if success:
        print("\n🎉 Basic generation test PASSED!")
        print("The model can generate text correctly with balance factors loaded.")
    else:
        print("\n💥 Basic generation test FAILED!")
        print("There's an issue with the generation pipeline itself.")
