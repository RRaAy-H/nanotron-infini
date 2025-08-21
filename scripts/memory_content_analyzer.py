#!/usr/bin/env python3
"""
Memory Content Analyzer for Infini-Attention

This script analyzes what information is actually stored in the Infini-Attention memory
during inference, providing insights into memory content evolution and effectiveness.
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

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

class MemoryContentAnalyzer:
    """Analyzes what information is stored in Infini-Attention memory."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.memory_snapshots = defaultdict(list)  # layer -> list of memory states
        self.memory_timeline = []  # chronological memory evolution
        self.hooked_blocks = []
        self.input_embeddings = None
        self.token_sequence = None
        
    def hook_memory_content_analysis(self, model):
        """Hook memory functions to capture and analyze content."""
        print("Setting up memory content analysis...")
        
        # Get the actual model (unwrap if needed)
        actual_model = model.model if hasattr(model, 'model') else model
        
        if not hasattr(actual_model, 'decoder'):
            print(f"   ERROR: Model has no decoder attribute")
            return
            
        print(f"   Analyzing memory content across {len(actual_model.decoder)} layers...")
        
        # Hook each pipeline block's forward method
        for layer_idx, pipeline_block in enumerate(actual_model.decoder):
            if hasattr(pipeline_block, 'pp_block') and hasattr(pipeline_block.pp_block, 'attn'):
                attn_layer = pipeline_block.pp_block.attn
                
                if hasattr(attn_layer, '_retrieve_from_memory') and hasattr(attn_layer, '_update_memory'):
                    # Save original methods
                    original_forward = pipeline_block.forward
                    original_retrieve = attn_layer._retrieve_from_memory
                    original_update = attn_layer._update_memory
                    
                    def create_content_analyzing_forward(layer_idx):
                        def analyzing_forward(*args, **kwargs):
                            # Hook memory functions for content analysis
                            def content_analyzing_retrieve(query_states, prev_memory, prev_normalization):
                                result = original_retrieve(query_states, prev_memory, prev_normalization)
                                
                                # Analyze retrieved memory content
                                if prev_memory is not None:
                                    self._analyze_memory_retrieval(
                                        layer_idx=layer_idx,
                                        query_states=query_states,
                                        retrieved_memory=prev_memory,
                                        memory_norm=prev_normalization
                                    )
                                
                                return result
                            
                            def content_analyzing_update(prev_memory, prev_normalization, key_states, value_states):
                                # Call original to get updated memory
                                result = original_update(prev_memory, prev_normalization, key_states, value_states)
                                new_memory, new_normalization = result
                                
                                # Analyze memory content change
                                self._analyze_memory_update(
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
                            attn_layer._retrieve_from_memory = content_analyzing_retrieve
                            attn_layer._update_memory = content_analyzing_update
                            
                            try:
                                result = original_forward(*args, **kwargs)
                                return result
                            finally:
                                # Restore original functions
                                attn_layer._retrieve_from_memory = original_retrieve
                                attn_layer._update_memory = original_update
                        
                        return analyzing_forward
                    
                    # Replace pipeline block forward method
                    pipeline_block.forward = create_content_analyzing_forward(layer_idx)
                    self.hooked_blocks.append((layer_idx, pipeline_block, original_forward))
        
        print(f"SUCCESS: Content analysis active on {len(self.hooked_blocks)} layers")
        
    def _analyze_memory_retrieval(self, layer_idx, query_states, retrieved_memory, memory_norm):
        """Analyze what information is being retrieved from memory."""
        if retrieved_memory is None:
            return
            
        # Compute similarity between query and retrieved memory
        query_flat = query_states.flatten(-2)  # [batch, seq, hidden]
        memory_flat = retrieved_memory.flatten(-2)
        
        # Cosine similarity
        query_norm = F.normalize(query_flat, dim=-1)
        memory_norm_vec = F.normalize(memory_flat, dim=-1)
        similarities = torch.sum(query_norm * memory_norm_vec, dim=-1)
        
        # Handle normalization values that might be tensors or scalars
        def safe_norm_extract(norm_tensor):
            if norm_tensor is None:
                return 0.0
            try:
                if norm_tensor.numel() == 1:
                    return norm_tensor.item()
                else:
                    # If it's a tensor with multiple elements, take the mean
                    return norm_tensor.mean().item()
            except:
                return 0.0
        
        # Analyze memory content statistics
        memory_stats = {
            'layer': layer_idx,
            'operation': 'retrieve',
            'timestamp': time.time(),
            'memory_shape': list(retrieved_memory.shape),
            'memory_norm': safe_norm_extract(memory_norm),
            'memory_mean': retrieved_memory.mean().item(),
            'memory_std': retrieved_memory.std().item(),
            'memory_max': retrieved_memory.max().item(),
            'memory_min': retrieved_memory.min().item(),
            'query_memory_similarity': {
                'mean': similarities.mean().item(),
                'max': similarities.max().item(),
                'min': similarities.min().item(),
                'std': similarities.std().item()
            }
        }
        
        self.memory_snapshots[layer_idx].append({
            'type': 'retrieval',
            'memory_tensor': retrieved_memory.detach().cpu().clone(),
            'stats': memory_stats
        })
        
        self.memory_timeline.append(memory_stats)
        
        # Print interesting findings
        if similarities.mean().item() > 0.5:  # High similarity
            print(f"    Layer {layer_idx}: HIGH query-memory similarity ({similarities.mean().item():.3f}) - relevant retrieval!")
        elif similarities.mean().item() < 0.1:  # Low similarity
            print(f"    Layer {layer_idx}: LOW query-memory similarity ({similarities.mean().item():.3f}) - diverse content")
            
    def _analyze_memory_update(self, layer_idx, prev_memory, new_memory, key_states, value_states, prev_norm, new_norm):
        """Analyze how memory content changes during updates."""
        
        # Analyze the change in memory content
        if prev_memory is not None:
            memory_change = new_memory - prev_memory
            change_magnitude = memory_change.norm().item()
            content_similarity = F.cosine_similarity(
                prev_memory.flatten(), 
                new_memory.flatten(), 
                dim=0
            ).item()
        else:
            change_magnitude = new_memory.norm().item()
            content_similarity = 0.0  # No previous memory to compare
            
        # Handle normalization values that might be tensors or scalars
        def safe_norm_extract(norm_tensor):
            if norm_tensor is None:
                return 0.0
            try:
                if norm_tensor.numel() == 1:
                    return norm_tensor.item()
                else:
                    # If it's a tensor with multiple elements, take the mean
                    return norm_tensor.mean().item()
            except:
                return 0.0
        
        # Analyze what new information is being stored
        new_info_stats = {
            'layer': layer_idx,
            'operation': 'update',
            'timestamp': time.time(),
            'prev_memory_norm': safe_norm_extract(prev_norm),
            'new_memory_norm': safe_norm_extract(new_norm),
            'memory_change_magnitude': change_magnitude,
            'content_similarity': content_similarity,
            'new_memory_stats': {
                'mean': new_memory.mean().item(),
                'std': new_memory.std().item(),
                'max': new_memory.max().item(),
                'min': new_memory.min().item()
            }
        }
        
        # Analyze relationship between keys/values and stored memory
        if key_states is not None and value_states is not None:
            try:
                # Handle key-memory similarity with shape checking
                key_flat = key_states.flatten()
                memory_flat = new_memory.flatten()
                
                if key_flat.shape == memory_flat.shape:
                    key_memory_sim = F.cosine_similarity(key_flat, memory_flat, dim=0).item()
                else:
                    min_size = min(key_flat.shape[0], memory_flat.shape[0])
                    key_memory_sim = F.cosine_similarity(
                        key_flat[:min_size], 
                        memory_flat[:min_size], 
                        dim=0
                    ).item()
                
                # Handle value-memory similarity with shape checking
                value_flat = value_states.flatten()
                
                if value_flat.shape == memory_flat.shape:
                    value_memory_sim = F.cosine_similarity(value_flat, memory_flat, dim=0).item()
                else:
                    min_size = min(value_flat.shape[0], memory_flat.shape[0])
                    value_memory_sim = F.cosine_similarity(
                        value_flat[:min_size], 
                        memory_flat[:min_size], 
                        dim=0
                    ).item()
                
                new_info_stats['key_memory_similarity'] = key_memory_sim
                new_info_stats['value_memory_similarity'] = value_memory_sim
            except Exception as e:
                # Fallback if similarity computation fails
                new_info_stats['key_memory_similarity'] = 0.0
                new_info_stats['value_memory_similarity'] = 0.0
        
        self.memory_snapshots[layer_idx].append({
            'type': 'update',
            'memory_tensor': new_memory.detach().cpu().clone(),
            'stats': new_info_stats
        })
        
        self.memory_timeline.append(new_info_stats)
        
        # Print significant changes
        if change_magnitude > 1000:  # Significant change
            print(f"    Layer {layer_idx}: MAJOR memory update (change: {change_magnitude:.1f}, similarity: {content_similarity:.3f})")
        elif content_similarity > 0.9 and prev_memory is not None:  # Minor refinement
            print(f"    Layer {layer_idx}: Minor memory refinement (similarity: {content_similarity:.3f})")
            
    def analyze_memory_patterns(self):
        """Analyze patterns in memory content across layers and time."""
        print("\n" + "="*60)
        print("MEMORY CONTENT ANALYSIS")
        print("="*60)
        
        analysis = {
            'layer_analysis': {},
            'temporal_patterns': {},
            'content_insights': {}
        }
        
        # Analyze each layer's memory patterns
        for layer_idx, snapshots in self.memory_snapshots.items():
            if not snapshots:
                continue
                
            retrievals = [s for s in snapshots if s['type'] == 'retrieval']
            updates = [s for s in snapshots if s['type'] == 'update']
            
            layer_analysis = {
                'total_operations': len(snapshots),
                'retrievals': len(retrievals),
                'updates': len(updates),
                'memory_evolution': []
            }
            
            # Analyze memory evolution in this layer
            memory_norms = [s['stats']['new_memory_norm'] if 'new_memory_norm' in s['stats'] 
                          else s['stats']['memory_norm'] for s in snapshots if 'new_memory_norm' in s['stats'] or 'memory_norm' in s['stats']]
            
            if memory_norms:
                layer_analysis['memory_norm_progression'] = {
                    'start': memory_norms[0],
                    'end': memory_norms[-1],
                    'max': max(memory_norms),
                    'trend': 'increasing' if memory_norms[-1] > memory_norms[0] else 'decreasing'
                }
            
            # Analyze content similarity patterns
            if retrievals:
                similarities = [r['stats']['query_memory_similarity']['mean'] for r in retrievals]
                layer_analysis['retrieval_relevance'] = {
                    'avg_similarity': sum(similarities) / len(similarities),
                    'max_similarity': max(similarities),
                    'highly_relevant_retrievals': sum(1 for s in similarities if s > 0.5)
                }
            
            analysis['layer_analysis'][layer_idx] = layer_analysis
            
            print(f"\nLayer {layer_idx}:")
            print(f"  Operations: {len(snapshots)} ({len(retrievals)} retrievals, {len(updates)} updates)")
            if memory_norms:
                print(f"  Memory norm: {memory_norms[0]:.1f} -> {memory_norms[-1]:.1f} ({layer_analysis['memory_norm_progression']['trend']})")
            if retrievals:
                print(f"  Avg retrieval relevance: {layer_analysis['retrieval_relevance']['avg_similarity']:.3f}")
                print(f"  Highly relevant retrievals: {layer_analysis['retrieval_relevance']['highly_relevant_retrievals']}/{len(retrievals)}")
        
        # Analyze temporal patterns
        if self.memory_timeline:
            print(f"\nTemporal Patterns:")
            print(f"  Total memory operations: {len(self.memory_timeline)}")
            
            # Memory growth pattern
            memory_ops = [op for op in self.memory_timeline if 'new_memory_norm' in op]
            if memory_ops:
                norms = [op['new_memory_norm'] for op in memory_ops]
                print(f"  Memory growth: {norms[0]:.1f} -> {norms[-1]:.1f}")
                print(f"  Peak memory norm: {max(norms):.1f}")
        
        return analysis
    
    def save_detailed_analysis(self, output_path):
        """Save detailed memory content analysis to file."""
        analysis_data = {
            'memory_snapshots_summary': {
                layer_idx: {
                    'operation_count': len(snapshots),
                    'retrieval_count': sum(1 for s in snapshots if s['type'] == 'retrieval'),
                    'update_count': sum(1 for s in snapshots if s['type'] == 'update'),
                    'final_memory_stats': snapshots[-1]['stats'] if snapshots else None
                }
                for layer_idx, snapshots in self.memory_snapshots.items()
            },
            'timeline_analysis': self.memory_timeline,
            'full_analysis': self.analyze_memory_patterns()
        }
        
        output_file = Path(output_path) / f"memory_content_analysis_{int(time.time())}.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"\nDetailed memory content analysis saved to: {output_file}")
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

def main():
    parser = argparse.ArgumentParser(description="Analyze Infini-Attention memory content")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--text", type=str, 
                        default="The quick brown fox jumps over the lazy dog. This sentence contains important information that should be remembered. The fox was very clever and managed to escape from the hunter. Later, the fox met a wise owl who taught it about survival in the forest.",
                        help="Text to analyze memory content for")
    parser.add_argument("--output", type=str, default="./memory_content_analysis", 
                        help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    print("Infini-Attention Memory Content Analyzer")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output}")
    print()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint path does not exist: {checkpoint_path}")
        return
    
    print("Loading model and tokenizer...")
    model, tokenizer, parallel_context = load_model_and_tokenizer(checkpoint_path)
    
    # Setup memory content analyzer
    analyzer = MemoryContentAnalyzer(tokenizer)
    analyzer.hook_memory_content_analysis(model)
    
    print(f"\nAnalyzing memory content for text:")
    print(f"'{args.text[:100]}{'...' if len(args.text) > 100 else ''}'")
    print()
    
    # Run generation with memory content analysis
    start_time = time.time()
    outputs = list(decode_text(
        input_iter=[GenerationInput(text=args.text)],
        tokenizer=tokenizer,
        model=model.model,
        parallel_context=parallel_context,
        max_new_tokens=10,
        max_micro_batch_size=1,
        generation_config=GenerationArgs(sampler="greedy", use_cache=False),
        tokenizer_config=TokenizerConfig(max_input_length=len(args.text) + 100),
    ))
    
    generation_time = time.time() - start_time
    print(f"\nGeneration completed in {generation_time:.2f}s")
    
    # Analyze memory patterns
    analysis = analyzer.analyze_memory_patterns()
    
    # Save detailed analysis
    output_file = analyzer.save_detailed_analysis(args.output)
    
    print(f"\nMemory content analysis completed!")
    print(f"Total memory operations captured: {len(analyzer.memory_timeline)}")
    print(f"Layers with memory activity: {len(analyzer.memory_snapshots)}")

if __name__ == "__main__":
    main()
