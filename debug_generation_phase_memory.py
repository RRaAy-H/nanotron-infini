#!/usr/bin/env python3
"""
Debug memory usage specifically during the generation phase vs context encoding phase.
This will help identify if memory works during encoding but fails during generation.
"""

import sys
import os
sys.path.append('src')
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import time
from pathlib import Path
from nanotron.config import get_config_from_file
from nanotron.models.llama import LlamaForTraining
from nanotron.generation.decode import decode_text
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.parallel.tensor_parallel.nn import TensorParallelLinearMode
from nanotron.random import RandomStates, get_current_random_state, get_synced_random_state
from nanotron.serialize.weights import load_weights
from transformers import AutoTokenizer
from torch.nn import functional as F
import torch.distributed as dist
import argparse
import json
from collections import defaultdict
from nanotron.serialize.weights import save_weights

# Import balance factor fix
try:
    from apply_balance_fix_standalone import apply_balance_factor_fix_standalone
except ImportError:
    sys.path.append('.')
    from apply_balance_fix_standalone import apply_balance_factor_fix_standalone

def safe_norm_extract(norm_tensor):
    """Safely extract scalar from norm tensor"""
    if norm_tensor is None:
        return 0.0
    try:
        return norm_tensor.item() if norm_tensor.numel() == 1 else norm_tensor.mean().item()
    except:
        return 0.0

class GenerationPhaseMemoryMonitor:
    def __init__(self):
        self.encoding_stats = defaultdict(list)  # Memory during context encoding
        self.generation_stats = defaultdict(list)  # Memory during answer generation
        self.current_phase = "encoding"  # Track which phase we're in
        self.generation_started = False
        
    def set_phase(self, phase):
        """Set current phase: 'encoding' or 'generation'"""
        self.current_phase = phase
        if phase == "generation":
            self.generation_started = True
            
    def hook_memory_update(self, layer_idx):
        def hook_fn(module, args, kwargs, result):
            try:
                if len(result) >= 2:
                    memory, norm = result[0], result[1]
                    
                    # Record based on current phase
                    stats = self.generation_stats if self.generation_started else self.encoding_stats
                    
                    stats[f'layer_{layer_idx}_updates'].append({
                        'timestamp': time.time(),
                        'phase': self.current_phase,
                        'memory_norm': safe_norm_extract(norm),
                        'memory_shape': list(memory.shape) if memory is not None else None
                    })
                    
                    print(f"[{self.current_phase}] Layer {layer_idx}: Memory UPDATE (norm: {safe_norm_extract(norm):.1f})")
                    
            except Exception as e:
                print(f"Error in update hook: {e}")
            return result
        return hook_fn
        
    def hook_memory_retrieve(self, layer_idx):
        def hook_fn(module, args, kwargs, result):
            try:
                if len(result) >= 2:
                    retrieved, norm = result[0], result[1]
                    
                    # Record based on current phase  
                    stats = self.generation_stats if self.generation_started else self.encoding_stats
                    
                    stats[f'layer_{layer_idx}_retrievals'].append({
                        'timestamp': time.time(),
                        'phase': self.current_phase,
                        'memory_norm': safe_norm_extract(norm),
                        'retrieved_shape': list(retrieved.shape) if retrieved is not None else None
                    })
                    
                    print(f"[{self.current_phase}] Layer {layer_idx}: Memory RETRIEVE (norm: {safe_norm_extract(norm):.1f})")
                    
            except Exception as e:
                print(f"Error in retrieve hook: {e}")
            return result
        return hook_fn

def load_model_and_tokenizer(checkpoint_path):
    """Load model with memory monitoring"""
    print("Loading model and tokenizer...")
    
    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    
    # Load config
    config_path = Path(checkpoint_path) / "config.yaml"
    config = get_config_from_file(str(config_path))
    
    # Setup parallelism for single GPU
    parallel_config = config.parallelism
    parallel_config.dp = 1
    parallel_config.pp = 1  
    parallel_config.tp = 1
    
    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )
    
    # Model config
    model_config = config.model.model_config
    
    # Setup random states
    random_states = RandomStates({
        "tp_synced": get_synced_random_state(random_state=get_current_random_state(), pg=parallel_context.tp_pg)
    }) if parallel_config.tp_mode is TensorParallelLinearMode.ALL_REDUCE else RandomStates({})
    
    # Build model
    from nanotron.models import build_model
    from nanotron import constants
    from nanotron.models.llama import LlamaConfig
    
    model = build_model(
        model_builder=lambda: LlamaForTraining(
            config=model_config,
            parallel_context=parallel_context, 
            parallel_config=parallel_config,
            random_states=random_states,
        ),
        dtype=torch.bfloat16,
        parallel_context=parallel_context,
    )
    
    # Handle tied groups
    tied_groups = getattr(model_config, 'tied_groups', None)
    if tied_groups is not None:
        from nanotron.models.base import mark_tied_parameters
        mark_tied_parameters(model=model, parallel_context=parallel_context, tied_groups=tied_groups)
    
    # Load weights
    try:
        load_weights(model=model, parallel_context=parallel_context, root_folder=checkpoint_path)
    except NotImplementedError as e:
        if "should be a NanotronParameter" in str(e):
            print("Expected balance factor loading error - will fix with standalone loader")
        else:
            raise e
    
    # Apply balance factor fix
    print("Applying balance factor fix...")
    try:
        apply_balance_factor_fix_standalone(model, checkpoint_path)
        print("SUCCESS: Balance factors loaded successfully")
    except Exception as e:
        print(f"WARNING: Balance factor fix failed: {e}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, parallel_context

def setup_phase_monitoring(model, monitor):
    """Setup hooks to monitor memory during different phases"""
    hooks = []
    
    for layer_idx, layer in enumerate(model.model.decoder):
        attn_layer = layer.pp_block.attn
        
        # Hook update function
        update_hook = attn_layer._update_memory.register_forward_hook(
            monitor.hook_memory_update(layer_idx), 
            with_kwargs=True
        )
        hooks.append(update_hook)
        
        # Hook retrieve function  
        retrieve_hook = attn_layer._retrieve_from_memory.register_forward_hook(
            monitor.hook_memory_retrieve(layer_idx),
            with_kwargs=True
        )
        hooks.append(retrieve_hook)
        
    print(f"SUCCESS: Phase monitoring active on {len(model.model.decoder)} layers")
    return hooks

def test_passkey_phases(model, tokenizer, parallel_context, checkpoint_path):
    """Test passkey with phase-specific memory monitoring"""
    
    # Create a longer context passkey test
    passkey = "789012"
    context_parts = [
        "This is a very long document that contains important information. "
        "It has many paragraphs and discusses various topics including science, "
        "technology, history, and literature. The content is designed to be "
        "comprehensive and detailed, spanning multiple segments of text. "
    ] * 20  # Repeat to make it long
    
    # Insert passkey in the middle
    context_parts.insert(10, f"The secret verification code is {passkey}. ")
    context_parts.append("What is the secret verification code mentioned in this document?")
    
    full_text = "".join(context_parts)
    
    print(f"\nTesting passkey '{passkey}' in {len(full_text)} character context")
    print(f"Estimated tokens: ~{len(full_text.split())}")
    
    # Setup monitoring
    monitor = GenerationPhaseMemoryMonitor()
    hooks = setup_phase_monitoring(model, monitor)
    
    try:
        # Tokenize input
        inputs = tokenizer(full_text, return_tensors="pt", padding=True, truncation=False)
        input_ids = inputs["input_ids"]
        
        print(f"Actual input tokens: {input_ids.shape[-1]}")
        
        # Phase 1: Context Encoding
        print("\n" + "="*60)
        print("PHASE 1: CONTEXT ENCODING")
        print("="*60)
        monitor.set_phase("encoding")
        
        # Generate with memory monitoring
        print("Starting generation...")
        generation_config = {
            "max_new_tokens": 50,
            "temperature": 0.1,
            "top_p": 0.9,
            "do_sample": False
        }
        
        # This will trigger both encoding and generation phases
        outputs = list(decode_text(
            model=model.model,
            input_ids=input_ids,
            input_mask=torch.ones_like(input_ids),
            **generation_config
        ))
        
        # Phase 2: Answer Generation (automatic during decode_text)
        print("\n" + "="*60) 
        print("PHASE 2: ANSWER GENERATION")
        print("="*60)
        monitor.set_phase("generation")
        
        # Extract answer
        if outputs:
            output = outputs[0]
            if hasattr(output, 'generation_ids') and hasattr(output, 'input_ids'):
                try:
                    generation_ids = output.generation_ids
                    input_ids_out = output.input_ids
                    
                    if generation_ids.dim() <= 1:
                        generation_ids = generation_ids.unsqueeze(0)
                    if input_ids_out.dim() <= 1:
                        input_ids_out = input_ids_out.unsqueeze(0)
                        
                    input_len = input_ids_out.shape[-1]
                    if generation_ids.shape[-1] > input_len:
                        answer_ids = generation_ids[0][input_len:]
                        answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
                    else:
                        answer = tokenizer.decode(generation_ids[0], skip_special_tokens=True).strip()
                        
                except Exception as e:
                    print(f"Error extracting answer: {e}")
                    answer = str(output)
            else:
                answer = str(output)
        else:
            answer = "No output generated"
            
        print(f"\nGenerated Answer: '{answer}'")
        print(f"Expected Passkey: '{passkey}'")
        print(f"Correct: {'YES' if passkey in answer else 'NO'}")
        
        # Analyze phase differences
        print("\n" + "="*60)
        print("PHASE ANALYSIS")
        print("="*60)
        
        # Count operations by phase
        encoding_ops = sum(len(ops) for key, ops in monitor.encoding_stats.items())
        generation_ops = sum(len(ops) for key, ops in monitor.generation_stats.items())
        
        print(f"Context Encoding Phase: {encoding_ops} memory operations")
        print(f"Answer Generation Phase: {generation_ops} memory operations")
        
        # Layer-wise breakdown
        for layer_idx in range(12):
            enc_updates = len(monitor.encoding_stats.get(f'layer_{layer_idx}_updates', []))
            enc_retrievals = len(monitor.encoding_stats.get(f'layer_{layer_idx}_retrievals', []))
            gen_updates = len(monitor.generation_stats.get(f'layer_{layer_idx}_updates', []))
            gen_retrievals = len(monitor.generation_stats.get(f'layer_{layer_idx}_retrievals', []))
            
            print(f"Layer {layer_idx:2d}: Encoding({enc_updates}U/{enc_retrievals}R) Generation({gen_updates}U/{gen_retrievals}R)")
        
        # Save detailed results
        results = {
            'passkey': passkey,
            'answer': answer,
            'correct': passkey in answer,
            'context_length': len(full_text),
            'token_count': input_ids.shape[-1],
            'encoding_stats': dict(monitor.encoding_stats),
            'generation_stats': dict(monitor.generation_stats),
            'encoding_operations': encoding_ops,
            'generation_operations': generation_ops
        }
        
        # Save to file
        output_dir = Path("generation_phase_analysis")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        output_file = output_dir / f"phase_analysis_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"\nDetailed analysis saved to: {output_file}")
        
        return results
        
    finally:
        # Clean up hooks
        for hook in hooks:
            hook.remove()

def main():
    parser = argparse.ArgumentParser(description="Debug memory usage during generation phases")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    
    args = parser.parse_args()
    
    print("Generation Phase Memory Analysis")
    print("="*50)
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load model
    model, tokenizer, parallel_context = load_model_and_tokenizer(args.checkpoint)
    
    # Test passkey with phase monitoring
    results = test_passkey_phases(model, tokenizer, parallel_context, args.checkpoint)
    
    print("\n" + "="*50)
    print("CONCLUSION:")
    if results['correct']:
        print("SUCCESS: Passkey retrieved correctly!")
    else:
        print("FAILED: Passkey not retrieved")
        print("\nPossible issues:")
        if results['generation_operations'] == 0:
            print("- No memory operations during generation phase")
        elif results['generation_operations'] < results['encoding_operations']:
            print("- Reduced memory activity during generation")
        else:
            print("- Memory active but not producing correct answers")

if __name__ == "__main__":
    main()
