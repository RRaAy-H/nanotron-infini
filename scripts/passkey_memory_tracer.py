#!/usr/bin/env python3
"""
Passkey Memory Tracer for Infini-Attention

This script provides detailed tracing of memory content during passkey retrieval tasks,
showing exactly what information is stored and retrieved from beginning to end.
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

# Ensure we're loading from the correct path
correct_path = "/data1/infini-attn/infini-llama/nanotron-infini/src"
if correct_path not in sys.path:
    sys.path.insert(0, correct_path)

# Remove any conflicting paths
sys.path = [p for p in sys.path if 'fiery/infini-nanotron' not in p]

import torch
import torch.nn.functional as F
from nanotron import constants, distributed as dist
from nanotron.config import get_config_from_file, GenerationArgs, ParallelismArgs
from nanotron.generation.decode import GenerationInput, TokenizerConfig, decode_text
from nanotron.models import build_model
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.engine import OneForwardOneBackwardPipelineEngine
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.random import RandomStates, get_current_random_state, get_synced_random_state, set_random_seed
from nanotron.serialize import load_weights
from nanotron.trainer import CONFIG_TO_MODEL_CLASS, mark_tied_parameters
from transformers import AutoTokenizer

class PasskeyMemoryTracer:
    """Traces memory content specifically during passkey retrieval tasks."""
    
    def __init__(self, tokenizer, passkey):
        self.tokenizer = tokenizer
        self.passkey = str(passkey)
        self.passkey_tokens = tokenizer.encode(self.passkey, add_special_tokens=False)
        
        # Memory tracing data
        self.memory_operations = []  # Complete timeline
        self.passkey_encounters = []  # When passkey info is processed
        self.memory_retrievals = []  # When memory is accessed
        self.memory_snapshots = {}  # layer -> memory states
        self.hooked_blocks = []
        
        # Analysis state
        self.current_token_position = 0
        self.passkey_position = None  # Where in sequence the passkey appears
        self.generation_phase = False  # Whether we're in generation vs encoding
        
        print(f"Tracking passkey: '{self.passkey}' (tokens: {self.passkey_tokens})")
        
    def hook_passkey_memory_tracing(self, model):
        """Hook memory functions to trace passkey-related content."""
        print("Setting up passkey memory tracing...")
        
        # Get the actual model (unwrap if needed)
        actual_model = model.model if hasattr(model, 'model') else model
        
        if not hasattr(actual_model, 'decoder'):
            print(f"   ERROR: Model has no decoder attribute")
            return
            
        print(f"   Tracing passkey memory across {len(actual_model.decoder)} layers...")
        
        # Hook each pipeline block's forward method
        for layer_idx, pipeline_block in enumerate(actual_model.decoder):
            if hasattr(pipeline_block, 'pp_block') and hasattr(pipeline_block.pp_block, 'attn'):
                attn_layer = pipeline_block.pp_block.attn
                
                if hasattr(attn_layer, '_retrieve_from_memory') and hasattr(attn_layer, '_update_memory'):
                    # Save original methods
                    original_forward = pipeline_block.forward
                    original_retrieve = attn_layer._retrieve_from_memory
                    original_update = attn_layer._update_memory
                    
                    def create_passkey_tracing_forward(layer_idx):
                        def tracing_forward(*args, **kwargs):
                            # Hook memory functions for passkey tracing
                            def passkey_tracing_retrieve(query_states, prev_memory, prev_normalization):
                                result = original_retrieve(query_states, prev_memory, prev_normalization)
                                
                                # Trace retrieval with passkey context
                                self._trace_memory_retrieval(
                                    layer_idx=layer_idx,
                                    query_states=query_states,
                                    retrieved_memory=prev_memory,
                                    memory_norm=prev_normalization,
                                    result=result
                                )
                                
                                return result
                            
                            def passkey_tracing_update(prev_memory, prev_normalization, key_states, value_states):
                                # Call original to get updated memory
                                result = original_update(prev_memory, prev_normalization, key_states, value_states)
                                new_memory, new_normalization = result
                                
                                # Trace update with passkey context
                                self._trace_memory_update(
                                    layer_idx=layer_idx,
                                    prev_memory=prev_memory,
                                    new_memory=new_memory,
                                    key_states=key_states,
                                    value_states=value_states,
                                    prev_norm=prev_normalization,
                                    new_norm=new_normalization
                                )
                                
                                return result
                            
                            # Temporarily replace memory functions
                            attn_layer._retrieve_from_memory = passkey_tracing_retrieve
                            attn_layer._update_memory = passkey_tracing_update
                            
                            try:
                                result = original_forward(*args, **kwargs)
                                return result
                            finally:
                                # Restore original functions
                                attn_layer._retrieve_from_memory = original_retrieve
                                attn_layer._update_memory = original_update
                        
                        return tracing_forward
                    
                    # Replace pipeline block forward method
                    pipeline_block.forward = create_passkey_tracing_forward(layer_idx)
                    self.hooked_blocks.append((layer_idx, pipeline_block, original_forward))
        
        print(f"SUCCESS: Passkey memory tracing active on {len(self.hooked_blocks)} layers")
        
    def _trace_memory_retrieval(self, layer_idx, query_states, retrieved_memory, memory_norm, result):
        """Trace what is retrieved from memory with passkey context."""
        if retrieved_memory is None:
            return
            
        # Analyze retrieved memory content
        memory_stats = self._analyze_tensor_content(retrieved_memory, "retrieved_memory")
        query_stats = self._analyze_tensor_content(query_states, "query")
        
        # Check if this might be passkey-related retrieval
        passkey_relevance = self._assess_passkey_relevance(retrieved_memory, query_states)
        
        retrieval_info = {
            'timestamp': time.time(),
            'operation': 'retrieve',
            'layer': layer_idx,
            'token_position': self.current_token_position,
            'generation_phase': self.generation_phase,
            'memory_norm': memory_norm.item() if memory_norm is not None else 0.0,
            'memory_stats': memory_stats,
            'query_stats': query_stats,
            'passkey_relevance': passkey_relevance,
            'phase': self._get_current_phase()
        }
        
        self.memory_operations.append(retrieval_info)
        self.memory_retrievals.append(retrieval_info)
        
        # Store memory snapshot for this layer
        if layer_idx not in self.memory_snapshots:
            self.memory_snapshots[layer_idx] = []
        self.memory_snapshots[layer_idx].append({
            'type': 'retrieval',
            'memory_tensor': retrieved_memory.detach().cpu().clone(),
            'info': retrieval_info
        })
        
        # Print important retrievals
        if passkey_relevance['potential_passkey_info'] > 0.3:
            print(f"    Layer {layer_idx}: POTENTIAL PASSKEY RETRIEVAL (relevance: {passkey_relevance['potential_passkey_info']:.3f}) [{self._get_current_phase()}]")
        elif retrieval_info['memory_norm'] > 100000:  # High-value memory
            print(f"    Layer {layer_idx}: High-value memory retrieval (norm: {retrieval_info['memory_norm']:.0f}) [{self._get_current_phase()}]")
            
    def _trace_memory_update(self, layer_idx, prev_memory, new_memory, key_states, value_states, prev_norm, new_norm):
        """Trace what is stored in memory with passkey context."""
        
        # Analyze memory content change
        memory_change_stats = self._analyze_memory_change(prev_memory, new_memory)
        new_memory_stats = self._analyze_tensor_content(new_memory, "new_memory")
        
        # Check if passkey information might be being stored
        passkey_storage = self._assess_passkey_storage(key_states, value_states, new_memory)
        
        update_info = {
            'timestamp': time.time(),
            'operation': 'update',
            'layer': layer_idx,
            'token_position': self.current_token_position,
            'generation_phase': self.generation_phase,
            'prev_memory_norm': prev_norm.item() if prev_norm is not None else 0.0,
            'new_memory_norm': new_norm.item() if new_norm is not None else 0.0,
            'memory_change': memory_change_stats,
            'new_memory_stats': new_memory_stats,
            'passkey_storage': passkey_storage,
            'phase': self._get_current_phase()
        }
        
        self.memory_operations.append(update_info)
        
        # Store memory snapshot
        if layer_idx not in self.memory_snapshots:
            self.memory_snapshots[layer_idx] = []
        self.memory_snapshots[layer_idx].append({
            'type': 'update',
            'memory_tensor': new_memory.detach().cpu().clone(),
            'info': update_info
        })
        
        # Print important updates
        if passkey_storage['potential_passkey_storage'] > 0.3:
            print(f"    Layer {layer_idx}: POTENTIAL PASSKEY STORAGE (confidence: {passkey_storage['potential_passkey_storage']:.3f}) [{self._get_current_phase()}]")
        elif memory_change_stats['change_magnitude'] > 50000:  # Significant change
            print(f"    Layer {layer_idx}: Major memory update (change: {memory_change_stats['change_magnitude']:.0f}) [{self._get_current_phase()}]")
            
    def _analyze_tensor_content(self, tensor, tensor_name):
        """Analyze statistical properties of a tensor."""
        if tensor is None:
            return None
            
        return {
            'shape': list(tensor.shape),
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'max': tensor.max().item(),
            'min': tensor.min().item(),
            'norm': tensor.norm().item(),
            'sparsity': (tensor.abs() < 1e-6).float().mean().item()  # Fraction of near-zero values
        }
        
    def _analyze_memory_change(self, prev_memory, new_memory):
        """Analyze how memory content changes."""
        if prev_memory is None:
            return {
                'change_magnitude': new_memory.norm().item(),
                'content_similarity': 0.0,
                'is_first_storage': True
            }
        
        memory_change = new_memory - prev_memory
        change_magnitude = memory_change.norm().item()
        
        # Cosine similarity between old and new memory
        content_similarity = F.cosine_similarity(
            prev_memory.flatten(), 
            new_memory.flatten(), 
            dim=0
        ).item()
        
        return {
            'change_magnitude': change_magnitude,
            'content_similarity': content_similarity,
            'is_first_storage': False,
            'relative_change': change_magnitude / (prev_memory.norm().item() + 1e-8)
        }
        
    def _assess_passkey_relevance(self, retrieved_memory, query_states):
        """Assess if retrieved memory might contain passkey information."""
        # This is a heuristic assessment based on memory patterns
        # In practice, you'd need more sophisticated analysis
        
        memory_complexity = retrieved_memory.std().item() / (retrieved_memory.mean().abs().item() + 1e-8)
        query_complexity = query_states.std().item() / (query_states.mean().abs().item() + 1e-8)
        
        # High similarity between query and memory might indicate relevant retrieval
        similarity = F.cosine_similarity(
            retrieved_memory.flatten(),
            query_states.flatten(),
            dim=0
        ).item()
        
        # Heuristic: relevant passkey info often has specific complexity patterns
        potential_passkey_info = min(1.0, (abs(similarity) + memory_complexity / 10) / 2)
        
        return {
            'query_memory_similarity': similarity,
            'memory_complexity': memory_complexity,
            'query_complexity': query_complexity,
            'potential_passkey_info': potential_passkey_info
        }
        
    def _assess_passkey_storage(self, key_states, value_states, new_memory):
        """Assess if passkey information might be being stored."""
        if key_states is None or value_states is None:
            return {'potential_passkey_storage': 0.0}
        
        # Check if we're in the vicinity of where passkey should be
        near_passkey_position = (
            self.passkey_position is not None and 
            abs(self.current_token_position - self.passkey_position) < 50
        )
        
        # Analyze key-value patterns that might indicate important information
        key_complexity = key_states.std().item() / (key_states.mean().abs().item() + 1e-8)
        value_complexity = value_states.std().item() / (value_states.mean().abs().item() + 1e-8)
        
        # High key-memory similarity might indicate specific information storage
        key_memory_sim = F.cosine_similarity(
            key_states.flatten(),
            new_memory.flatten(),
            dim=0
        ).item()
        
        # Heuristic assessment
        base_score = (key_complexity + value_complexity) / 20
        position_bonus = 0.5 if near_passkey_position else 0.0
        similarity_bonus = abs(key_memory_sim) * 0.3
        
        potential_storage = min(1.0, base_score + position_bonus + similarity_bonus)
        
        return {
            'potential_passkey_storage': potential_storage,
            'key_complexity': key_complexity,
            'value_complexity': value_complexity,
            'key_memory_similarity': key_memory_sim,
            'near_passkey_position': near_passkey_position
        }
        
    def _get_current_phase(self):
        """Determine current processing phase."""
        if self.generation_phase:
            return "generation"
        elif self.passkey_position is not None and self.current_token_position >= self.passkey_position:
            return "post_passkey_encoding"
        elif self.passkey_position is not None and abs(self.current_token_position - self.passkey_position) < 10:
            return "passkey_encoding"
        else:
            return "pre_passkey_encoding"
            
    def set_passkey_position(self, text):
        """Find where the passkey appears in the input text."""
        passkey_match = re.search(re.escape(self.passkey), text)
        if passkey_match:
            # Estimate token position (rough approximation)
            char_position = passkey_match.start()
            estimated_token_pos = len(self.tokenizer.encode(text[:char_position], add_special_tokens=False))
            self.passkey_position = estimated_token_pos
            print(f"Passkey found at estimated token position {estimated_token_pos}")
        else:
            print("WARNING: Passkey not found in input text!")
            
    def analyze_passkey_memory_journey(self):
        """Analyze the complete memory journey for passkey retrieval."""
        print("\n" + "="*70)
        print("PASSKEY MEMORY JOURNEY ANALYSIS")
        print("="*70)
        
        # Phase-based analysis
        phases = {
            'pre_passkey_encoding': [],
            'passkey_encoding': [],
            'post_passkey_encoding': [], 
            'generation': []
        }
        
        for op in self.memory_operations:
            phases[op['phase']].append(op)
        
        analysis = {
            'phases': {},
            'passkey_encounters': [],
            'retrieval_analysis': {},
            'storage_analysis': {}
        }
        
        for phase_name, ops in phases.items():
            if not ops:
                continue
                
            retrievals = [op for op in ops if op['operation'] == 'retrieve']
            updates = [op for op in ops if op['operation'] == 'update']
            
            # Find potential passkey operations
            passkey_retrievals = [r for r in retrievals if r['passkey_relevance']['potential_passkey_info'] > 0.3]
            passkey_storages = [u for u in updates if u['passkey_storage']['potential_passkey_storage'] > 0.3]
            
            phase_analysis = {
                'total_operations': len(ops),
                'retrievals': len(retrievals),
                'updates': len(updates),
                'potential_passkey_retrievals': len(passkey_retrievals),
                'potential_passkey_storages': len(passkey_storages),
                'memory_activity_summary': {
                    'avg_memory_norm': sum(op.get('memory_norm', op.get('new_memory_norm', 0)) for op in ops) / len(ops),
                    'peak_memory_norm': max(op.get('memory_norm', op.get('new_memory_norm', 0)) for op in ops)
                }
            }
            
            analysis['phases'][phase_name] = phase_analysis
            
            print(f"\n{phase_name.replace('_', ' ').title()}:")
            print(f"  Operations: {len(ops)} ({len(retrievals)} retrievals, {len(updates)} updates)")
            print(f"  Potential passkey operations: {len(passkey_retrievals)} retrievals, {len(passkey_storages)} storages")
            print(f"  Avg memory norm: {phase_analysis['memory_activity_summary']['avg_memory_norm']:.0f}")
            
            if passkey_retrievals:
                print(f"  HIGH-CONFIDENCE passkey retrievals detected!")
            if passkey_storages:
                print(f"  HIGH-CONFIDENCE passkey storage detected!")
        
        # Overall assessment
        total_passkey_ops = sum(
            phase['potential_passkey_retrievals'] + phase['potential_passkey_storages'] 
            for phase in analysis['phases'].values()
        )
        
        print(f"\n" + "="*50)
        print(f"PASSKEY MEMORY ASSESSMENT:")
        print(f"  Total operations: {len(self.memory_operations)}")
        print(f"  Potential passkey-related operations: {total_passkey_ops}")
        print(f"  Memory utilization: {'ACTIVE' if len(self.memory_operations) > 0 else 'INACTIVE'}")
        
        if total_passkey_ops > 0:
            print(f"  Passkey handling: DETECTED - Memory appears to process passkey information")
        else:
            print(f"  Passkey handling: UNCLEAR - No clear passkey-specific patterns detected")
            
        return analysis
        
    def save_detailed_trace(self, output_path):
        """Save complete memory trace to file."""
        trace_data = {
            'passkey_info': {
                'passkey': self.passkey,
                'passkey_tokens': self.passkey_tokens,
                'passkey_position': self.passkey_position
            },
            'memory_operations': self.memory_operations,
            'analysis': self.analyze_passkey_memory_journey(),
            'trace_summary': {
                'total_operations': len(self.memory_operations),
                'layers_active': len(self.memory_snapshots),
                'retrieval_count': len(self.memory_retrievals)
            }
        }
        
        output_file = Path(output_path) / f"passkey_memory_trace_{int(time.time())}.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(trace_data, f, indent=2)
        
        print(f"\nDetailed passkey memory trace saved to: {output_file}")
        return output_file

def load_model_and_tokenizer(checkpoint_path):
    """Load model and tokenizer with balance factor fixes."""
    config = get_config_from_file((checkpoint_path / "config.yaml").as_posix())
    constants.CONFIG = config
    
    # Initialize distributed
    dist.initialize_torch_distributed()
    
    # Setup parallelism for single process
    parallel_config = ParallelismArgs(
        dp=1, pp=1, tp=1,
        pp_engine=OneForwardOneBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        tp_linear_async_communication=False,
    )
    
    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )
    
    set_random_seed(42)
    
    # Build model
    model_config = config.model.model_config
    model_config.turn_on_memory = True  # Ensure memory is enabled
    
    # Get the correct model class from config
    model_type = None
    
    # Try different ways to determine model type
    if hasattr(config.model.model_config, 'model_type'):
        model_type = config.model.model_config.model_type
    elif hasattr(config.model.model_config, '__class__'):
        # Get model type from class name
        class_name = config.model.model_config.__class__.__name__
        if "llama" in class_name.lower():
            model_type = "llama"
        else:
            model_type = class_name.replace("Config", "").lower()
    
    # Check available model classes and pick the best match
    available_classes = list(CONFIG_TO_MODEL_CLASS.keys())
    print(f"Available model classes: {available_classes}")
    print(f"Detected model type: {model_type}")
    
    # Ensure model_type exists in CONFIG_TO_MODEL_CLASS
    if model_type is None or model_type not in CONFIG_TO_MODEL_CLASS:
        # Try to find llama-related class
        llama_classes = [cls for cls in available_classes if "llama" in cls.lower()]
        if llama_classes:
            model_type = llama_classes[0]
            print(f"Using llama-related model class: {model_type}")
        else:
            # Default fallback - use the first available model class
            model_type = available_classes[0]
            print(f"Warning: Using default model class: {model_type}")
    
    print(f"Final model type: {model_type}")
    
    # Setup random states for model building
    if parallel_config.tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        random_states = RandomStates({
            "tp_synced": get_synced_random_state(random_state=get_current_random_state(), pg=parallel_context.tp_pg)
        })
    else:
        random_states = RandomStates({})
    
    model = build_model(
        model_builder=lambda: CONFIG_TO_MODEL_CLASS[model_type](
            config=model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=random_states,
        ),
        dtype=torch.bfloat16,
        parallel_context=parallel_context,
    )
    
    # Mark tied parameters if they exist in the config
    tied_groups = getattr(model_config, 'tied_groups', None)
    if tied_groups is not None:
        mark_tied_parameters(model=model, parallel_context=parallel_context, tied_groups=tied_groups)
    
    # Load checkpoint (may fail due to balance factor parameters)
    print(f"Loading checkpoint from {checkpoint_path}")
    try:
        load_weights(model=model, parallel_context=parallel_context, root_folder=checkpoint_path)
        print("Standard weight loading completed")
    except NotImplementedError as e:
        if "should be a NanotronParameter" in str(e):
            print("Expected balance factor loading error - will fix with standalone loader")
        else:
            raise e
    
    # Apply balance factor fix (this handles both regular params and balance factors)
    print("Applying balance factor fix...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(current_dir, '..'))
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, parallel_context

def create_passkey_prompt(passkey, context_length=2048, depth_percent=50):
    """Create a passkey retrieval prompt."""
    base_text = "The following is a story about a brave knight on a quest. "
    filler_text = "The knight traveled through many lands, meeting various people and having adventures. He crossed rivers, climbed mountains, and explored dark forests. Along the way, he learned valuable lessons about courage, friendship, and perseverance. "
    
    # Calculate insertion point
    passkey_text = f"The secret password is {passkey}. Remember this important information. "
    question = f"What is the secret password mentioned in the text above?"
    
    # Build the prompt
    words_before = int((context_length * depth_percent / 100) / 4)  # Rough word count estimation
    words_after = int((context_length * (100 - depth_percent) / 100) / 4)
    
    before_text = (filler_text * (words_before // len(filler_text.split()) + 1))[:words_before * 4]
    after_text = (filler_text * (words_after // len(filler_text.split()) + 1))[:words_after * 4]
    
    prompt = f"{base_text}{before_text}{passkey_text}{after_text}\n\nQuestion: {question}\nAnswer:"
    
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Trace memory content during passkey retrieval")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--passkey", type=str, default="987654", help="Passkey to trace")
    parser.add_argument("--context-length", type=int, default=2048, help="Context length for test")
    parser.add_argument("--depth-percent", type=int, default=50, help="Depth percentage for passkey placement")
    parser.add_argument("--output", type=str, default="./passkey_memory_trace", 
                        help="Output directory for trace results")
    
    args = parser.parse_args()
    
    print("Passkey Memory Tracer for Infini-Attention")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Passkey: {args.passkey}")
    print(f"Context length: {args.context_length}")
    print(f"Depth: {args.depth_percent}%")
    print(f"Output directory: {args.output}")
    print()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint path does not exist: {checkpoint_path}")
        return
    
    print("Loading model and tokenizer...")
    model, tokenizer, parallel_context = load_model_and_tokenizer(checkpoint_path)
    
    # Create passkey prompt
    prompt = create_passkey_prompt(args.passkey, args.context_length, args.depth_percent)
    print(f"\nGenerated passkey prompt ({len(prompt)} chars):")
    print(f"'{prompt[:200]}...{prompt[-100:]}'")
    print()
    
    # Setup passkey memory tracer
    tracer = PasskeyMemoryTracer(tokenizer, args.passkey)
    tracer.set_passkey_position(prompt)
    tracer.hook_passkey_memory_tracing(model)
    
    print(f"Starting passkey memory trace...")
    print(f"Tracking: '{args.passkey}' at {args.depth_percent}% depth")
    print()
    
    # Run generation with passkey memory tracing
    start_time = time.time()
    
    # Mark that we're starting generation phase
    tracer.generation_phase = True
    
    outputs = list(decode_text(
        input_iter=[GenerationInput(text=prompt)],
        tokenizer=tokenizer,
        model=model.model,
        parallel_context=parallel_context,
        max_new_tokens=15,
        max_micro_batch_size=1,
        generation_config=GenerationArgs(sampler="greedy", use_cache=False),
        tokenizer_config=TokenizerConfig(max_input_length=len(prompt) + 100),
    ))
    
    generation_time = time.time() - start_time
    print(f"\nGeneration completed in {generation_time:.2f}s")
    
    # Extract and display the answer
    if outputs:
        output = outputs[0]
        if hasattr(output, 'generation_ids') and hasattr(output, 'input_ids'):
            answer_ids = output.generation_ids[0][len(output.input_ids[0]):]
            answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        else:
            answer = str(output)
    else:
        answer = "No output generated"
    
    print(f"Generated Answer: '{answer}'")
    print(f"Expected Passkey: '{args.passkey}'")
    print(f"Correct: {'YES' if args.passkey in answer else 'NO'}")
    
    # Analyze the complete memory journey
    analysis = tracer.analyze_passkey_memory_journey()
    
    # Save detailed trace
    output_file = tracer.save_detailed_trace(args.output)
    
    print(f"\nPasskey memory tracing completed!")
    print(f"Total memory operations: {len(tracer.memory_operations)}")
    print(f"Layers with activity: {len(tracer.memory_snapshots)}")

if __name__ == "__main__":
    main()
