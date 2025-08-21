#!/usr/bin/env python3
"""
Memory-Aware Passkey Retrieval Test for Infini-Attention

This script demonstrates how the memory mechanism works with real passkey retrieval tasks:
1. Tests retrieval at different depths (early, middle, late in context)
2. Shows detailed memory usage during generation
3. Compares memory activity for successful vs failed retrievals
4. Provides insights into how memory foundation helps with long-context understanding
"""

import sys
import os
import json
import time
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Any

# Ensure we're loading from the correct path
correct_path = "/data1/infini-attn/infini-llama/nanotron-infini/src"
if correct_path not in sys.path:
    sys.path.insert(0, correct_path)

# Remove any conflicting paths
sys.path = [p for p in sys.path if 'fiery/infini-nanotron' not in p]

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

class MemoryAwarePasskeyMonitor:
    """Monitor memory usage during passkey retrieval tasks."""
    
    def __init__(self):
        self.memory_calls = {'retrieve': 0, 'update': 0}
        self.memory_stats = []
        self.generation_stats = []
        self.hooked_blocks = []
        self.current_test = None
        
    def start_test(self, test_info):
        """Start monitoring a new test."""
        self.current_test = test_info
        self.memory_calls = {'retrieve': 0, 'update': 0}
        self.memory_stats = []
        self.generation_stats = []
        
    def hook_memory_functions(self, model):
        """Hook memory functions using the CORRECT pipeline block execution path."""
        print("Setting up memory monitoring for passkey tests...")
        
        # Get the actual model (unwrap LlamaForTraining if needed)
        actual_model = model.model if hasattr(model, 'model') else model
        
        if not hasattr(actual_model, 'decoder'):
            print(f"   ERROR: Model has no decoder attribute")
            return
            
        print(f"   Monitoring {len(actual_model.decoder)} layers...")
        
        # Hook each pipeline block's forward method
        for layer_idx, pipeline_block in enumerate(actual_model.decoder):
            if hasattr(pipeline_block, 'pp_block') and hasattr(pipeline_block.pp_block, 'attn'):
                attn_layer = pipeline_block.pp_block.attn
                
                if hasattr(attn_layer, '_retrieve_from_memory') and hasattr(attn_layer, '_update_memory'):
                    # Save original methods
                    original_forward = pipeline_block.forward
                    original_retrieve = attn_layer._retrieve_from_memory
                    original_update = attn_layer._update_memory
                    
                    def create_monitored_forward(layer_idx):
                        def monitored_forward(*args, **kwargs):
                            # Temporarily hook memory functions for this forward call
                            def counting_retrieve(query_states, prev_memory, prev_normalization):
                                self.memory_calls['retrieve'] += 1
                                has_memory = prev_memory is not None
                                memory_norm = prev_memory.norm().item() if has_memory else 0.0
                                
                                # Only print for a few layers to avoid spam
                                if layer_idx < 3:
                                    print(f"    Layer {layer_idx}: Retrieving memory #{self.memory_calls['retrieve']} "
                                          f"(has_prev: {'Yes' if has_memory else 'No'}, norm: {memory_norm:.3f})")
                                
                                self.memory_stats.append({
                                    'type': 'retrieve',
                                    'layer': layer_idx,
                                    'timestamp': time.time(),
                                    'has_prev_memory': has_memory,
                                    'memory_norm': memory_norm,
                                    'test_info': self.current_test
                                })
                                
                                return original_retrieve(query_states, prev_memory, prev_normalization)
                            
                            def counting_update(prev_memory, prev_normalization, key_states, value_states):
                                self.memory_calls['update'] += 1
                                prev_norm = prev_memory.norm().item() if prev_memory is not None else 0.0
                                
                                # Call original to get result
                                result = original_update(prev_memory, prev_normalization, key_states, value_states)
                                new_memory, new_normalization = result
                                new_norm = new_memory.norm().item()
                                
                                # Only print for a few layers
                                if layer_idx < 3:
                                    print(f"    Layer {layer_idx}: Updating memory #{self.memory_calls['update']} "
                                          f"(prev: {prev_norm:.3f} -> new: {new_norm:.3f})")
                                
                                self.memory_stats.append({
                                    'type': 'update',
                                    'layer': layer_idx,
                                    'timestamp': time.time(),
                                    'prev_memory_norm': prev_norm,
                                    'new_memory_norm': new_norm,
                                    'test_info': self.current_test
                                })
                                
                                return result
                            
                            # Hook memory functions temporarily
                            attn_layer._retrieve_from_memory = counting_retrieve
                            attn_layer._update_memory = counting_update
                            
                            try:
                                # Call original forward
                                result = original_forward(*args, **kwargs)
                                return result
                            finally:
                                # Restore original functions
                                attn_layer._retrieve_from_memory = original_retrieve
                                attn_layer._update_memory = original_update
                        
                        return monitored_forward
                    
                    # Replace pipeline block forward method
                    pipeline_block.forward = create_monitored_forward(layer_idx)
                    self.hooked_blocks.append((layer_idx, pipeline_block, original_forward))
        
        print(f"SUCCESS: Hooked {len(self.hooked_blocks)} pipeline blocks for memory monitoring")
        
    def get_test_summary(self):
        """Get summary for current test."""
        return {
            'total_retrievals': self.memory_calls['retrieve'],
            'total_updates': self.memory_calls['update'],
            'memory_active': self.memory_calls['retrieve'] > 0 or self.memory_calls['update'] > 0,
            'layers_with_memory': len(set(stat['layer'] for stat in self.memory_stats)),
            'avg_memory_norm': sum(s.get('memory_norm', 0) for s in self.memory_stats if s['type'] == 'retrieve') / max(1, sum(1 for s in self.memory_stats if s['type'] == 'retrieve')),
            'stats': self.memory_stats
        }

def load_model_and_tokenizer(checkpoint_path: Path):
    """Load model and tokenizer with balance factor fix."""
    
    # Load config
    config = get_config_from_file((checkpoint_path / "config.yaml").as_posix())
    constants.CONFIG = config
    
    # Setup parallel context
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
    base_path = "/data1/infini-attn/infini-llama/nanotron-infini"
    if base_path not in sys.path:
        sys.path.insert(0, base_path)
    
    from apply_balance_fix_standalone import apply_balance_factor_fix_standalone
    fix_success = apply_balance_factor_fix_standalone(model, checkpoint_path, verbose=False)
    if fix_success:
        print("SUCCESS: Balance factors loaded successfully")
    else:
        print("WARNING: Balance factor fix may not have worked properly")
    
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    return model, tokenizer, parallel_context

def create_passkey_example(tokenizer, context_length: int, depth_percent: int, passkey: str = None):
    """Create a passkey retrieval example at specified depth."""
    
    if passkey is None:
        # Generate random 6-digit passkey
        passkey = f"{random.randint(100000, 999999)}"
    
    # Create base context using repeating pattern
    base_text = """The history of science is a tapestry of human curiosity and discovery. From ancient civilizations studying the stars to modern quantum physicists probing the nature of reality, each generation builds upon the knowledge of those who came before. The scientific method, with its emphasis on observation, hypothesis, and experimentation, has revolutionized our understanding of the natural world. Researchers across disciplines continue to push the boundaries of knowledge, uncovering new phenomena and developing innovative technologies that shape our daily lives. The collaborative nature of modern science, enabled by global communication networks, allows for unprecedented cooperation in tackling humanity's greatest challenges. """
    
    # Create the needle (passkey insertion)
    needle = f" The secret passkey for this document is {passkey}. Remember this number carefully. "
    
    # Create the question
    question = f"What is the secret passkey mentioned in the document? Answer with just the number:"
    
    # Calculate approximate tokens needed
    approx_tokens_per_char = 0.25  # Rough estimate
    target_chars = int(context_length / approx_tokens_per_char)
    
    # Repeat base text to reach approximate target length
    repeat_count = max(1, target_chars // len(base_text))
    context = base_text * repeat_count
    
    # Insert needle at specified depth
    if depth_percent == 0:
        # At the beginning
        context = needle + context
    elif depth_percent == 100:
        # At the end (before question)
        context = context + needle
    else:
        # At specified percentage
        insertion_point = int(len(context) * (depth_percent / 100))
        # Find nearest sentence end
        while insertion_point < len(context) and context[insertion_point] not in '.!?':
            insertion_point += 1
        if insertion_point < len(context):
            insertion_point += 1  # Include the punctuation
        
        context = context[:insertion_point] + needle + context[insertion_point:]
    
    # Create full prompt
    full_prompt = f"Please read the following document carefully:\n\n{context}\n\n{question}"
    
    # Truncate to target length if needed
    tokens = tokenizer.encode(full_prompt)
    if len(tokens) > context_length:
        # Truncate context, keeping needle and question
        target_context_tokens = context_length - len(tokenizer.encode(question)) - 50  # Buffer
        context_tokens = tokenizer.encode(context)
        
        if len(context_tokens) > target_context_tokens:
            # Maintain needle position proportion
            needle_tokens = tokenizer.encode(needle)
            
            if depth_percent == 0:
                # Keep from beginning
                truncated_context_tokens = context_tokens[:target_context_tokens]
            elif depth_percent == 100:
                # Keep to end
                truncated_context_tokens = context_tokens[-target_context_tokens:]
            else:
                # Keep around needle position
                needle_pos_tokens = int(target_context_tokens * (depth_percent / 100))
                start_pos = max(0, needle_pos_tokens - target_context_tokens // 2)
                end_pos = min(len(context_tokens), start_pos + target_context_tokens)
                truncated_context_tokens = context_tokens[start_pos:end_pos]
            
            context = tokenizer.decode(truncated_context_tokens, skip_special_tokens=True)
            full_prompt = f"Please read the following document carefully:\n\n{context}\n\n{question}"
    
    return {
        'prompt': full_prompt,
        'passkey': passkey,
        'depth_percent': depth_percent,
        'context_length': len(tokenizer.encode(full_prompt)),
        'needle': needle,
        'question': question
    }

def run_passkey_test(model, tokenizer, parallel_context, monitor, test_case):
    """Run a single passkey test with memory monitoring."""
    
    prompt = test_case['prompt']
    expected_passkey = test_case['passkey']
    depth = test_case['depth_percent']
    
    print(f"\n{'='*60}")
    print(f"PASSKEY TEST: Depth {depth}%, Context Length {test_case['context_length']} tokens")
    print(f"Expected Passkey: {expected_passkey}")
    print(f"{'='*60}")
    
    # Start monitoring
    monitor.start_test({
        'depth_percent': depth,
        'context_length': test_case['context_length'],
        'expected_passkey': expected_passkey
    })
    
    try:
        # Run generation with memory monitoring
        print("Starting generation with memory monitoring...")
        start_time = time.time()
        
        outputs = list(decode_text(
            input_iter=[GenerationInput(text=prompt)],
            tokenizer=tokenizer,
            model=model.model,
            parallel_context=parallel_context,
            max_new_tokens=20,
            max_micro_batch_size=1,
            generation_config=GenerationArgs(sampler="greedy", use_cache=False),
            tokenizer_config=TokenizerConfig(max_input_length=test_case['context_length'] + 100),
        ))
        
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f}s")
        
        # Extract response
        if outputs:
            output = outputs[0]
            if hasattr(output, 'generation_ids') and hasattr(output, 'input_ids'):
                generated_ids = output.generation_ids
                input_ids = output.input_ids
                answer_ids = generated_ids[len(input_ids):]
                response = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            else:
                response = str(output)
        else:
            response = "No output generated"
        
        print(f"Generated Response: '{response}'")
        
        # Check if passkey was retrieved correctly
        success = expected_passkey in response or response.strip() == expected_passkey.strip()
        
        # Get memory usage summary
        memory_summary = monitor.get_test_summary()
        
        print(f"\nMemory Usage Summary:")
        print(f"   Memory Retrievals: {memory_summary['total_retrievals']}")
        print(f"   Memory Updates: {memory_summary['total_updates']}")
        print(f"   Active Layers: {memory_summary['layers_with_memory']}")
        print(f"   Avg Memory Norm: {memory_summary['avg_memory_norm']:.4f}")
        
        if success:
            print(f"SUCCESS: Correctly retrieved passkey!")
        else:
            print(f"FAILED: Expected '{expected_passkey}', got '{response}'")
        
        return {
            'depth_percent': depth,
            'context_length': test_case['context_length'],
            'expected_passkey': expected_passkey,
            'generated_response': response,
            'success': success,
            'generation_time': generation_time,
            'memory_usage': memory_summary,
            'prompt_preview': prompt[:200] + "..." if len(prompt) > 200 else prompt
        }
        
    except Exception as e:
        print(f"ERROR during generation: {e}")
        return {
            'depth_percent': depth,
            'context_length': test_case['context_length'],
            'expected_passkey': expected_passkey,
            'error': str(e),
            'success': False,
            'memory_usage': monitor.get_test_summary()
        }

def main():
    parser = argparse.ArgumentParser(description="Memory-Aware Passkey Retrieval Test")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[2048, 4096, 8192], 
                        help="Context lengths to test")
    parser.add_argument("--depth-percents", nargs="+", type=int, default=[0, 25, 50, 75, 100], 
                        help="Depths to test (percentage of context)")
    parser.add_argument("--output", type=str, default="./passkey_memory_results", 
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("Memory-Aware Passkey Retrieval Test for Infini-Attention")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Context lengths: {args.context_lengths}")
    print(f"Depth percentages: {args.depth_percents}")
    print(f"Output directory: {args.output}")
    print()
    
    random.seed(args.seed)
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint path does not exist: {checkpoint_path}")
        return
    
    print("Loading model and tokenizer...")
    model, tokenizer, parallel_context = load_model_and_tokenizer(checkpoint_path)
    
    # Setup monitoring
    monitor = MemoryAwarePasskeyMonitor()
    monitor.hook_memory_functions(model)
    
    # Run tests
    all_results = []
    
    for context_length in args.context_lengths:
        for depth_percent in args.depth_percents:
            
            # Create test case
            test_case = create_passkey_example(tokenizer, context_length, depth_percent)
            
            # Run test
            result = run_passkey_test(model, tokenizer, parallel_context, monitor, test_case)
            all_results.append(result)
    
    # Analyze results
    print(f"\n{'='*70}")
    print(f"FINAL ANALYSIS")
    print(f"{'='*70}")
    
    successful_tests = [r for r in all_results if r.get('success', False)]
    failed_tests = [r for r in all_results if not r.get('success', False)]
    
    print(f"Total tests: {len(all_results)}")
    print(f"Successful: {len(successful_tests)} ({len(successful_tests)/len(all_results)*100:.1f}%)")
    print(f"Failed: {len(failed_tests)} ({len(failed_tests)/len(all_results)*100:.1f}%)")
    
    # Memory analysis
    if successful_tests:
        avg_memory_successful = sum(r['memory_usage']['total_retrievals'] for r in successful_tests) / len(successful_tests)
        print(f"\nMemory usage in successful tests:")
        print(f"  Average retrievals: {avg_memory_successful:.1f}")
        
    if failed_tests:
        avg_memory_failed = sum(r.get('memory_usage', {}).get('total_retrievals', 0) for r in failed_tests) / len(failed_tests)
        print(f"\nMemory usage in failed tests:")
        print(f"  Average retrievals: {avg_memory_failed:.1f}")
    
    # Depth analysis
    print(f"\nSuccess rate by depth:")
    for depth in sorted(set(r['depth_percent'] for r in all_results)):
        depth_results = [r for r in all_results if r['depth_percent'] == depth]
        depth_success = [r for r in depth_results if r.get('success', False)]
        success_rate = len(depth_success) / len(depth_results) * 100
        avg_memory = sum(r['memory_usage']['total_retrievals'] for r in depth_results) / len(depth_results)
        print(f"  {depth:3d}%: {success_rate:5.1f}% success, {avg_memory:5.1f} avg memory retrievals")
    
    # Save detailed results
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / f"passkey_memory_analysis_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'test_configuration': {
                'checkpoint': args.checkpoint,
                'context_lengths': args.context_lengths,
                'depth_percents': args.depth_percents,
                'seed': args.seed
            },
            'results': all_results,
            'summary': {
                'total_tests': len(all_results),
                'successful_tests': len(successful_tests),
                'failed_tests': len(failed_tests),
                'success_rate': len(successful_tests) / len(all_results) * 100 if all_results else 0
            }
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    print(f"\nMemory-aware passkey testing completed!")
    print(f"The memory mechanism actively participated in {len([r for r in all_results if r.get('memory_usage', {}).get('total_retrievals', 0) > 0])} out of {len(all_results)} tests.")

if __name__ == "__main__":
    main()
