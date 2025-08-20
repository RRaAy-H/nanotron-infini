#!/usr/bin/env python3
"""
Infini-Attention Memory Usage Debugger

This script provides comprehensive monitoring of memory usage during inference,
including real-time tracking of memory retrieval, storage, and cross-segment
information flow.

Usage:
    python scripts/debug_memory_usage.py --checkpoint ./checkpoints/model/30000
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# Import nanotron components
import sys
sys.path.append('src')
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


class MemoryUsageMonitor:
    """Monitor infini-attention memory usage during inference."""
    
    def __init__(self, output_dir: str = "./memory_debug"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_states = []
        self.retrieval_patterns = []
        self.segment_stats = []
        self.hooks = []
        
    def hook_memory_functions(self, model):
        """Hook into memory retrieval and update functions."""
        
        def create_retrieve_hook(layer_idx):
            def retrieve_hook(query_states, prev_memory, prev_normalization):
                # Record memory retrieval
                if prev_memory is not None:
                    memory_norm = prev_memory.norm().item()
                    memory_shape = prev_memory.shape
                    
                    retrieval_info = {
                        'layer_idx': layer_idx,
                        'timestamp': time.time(),
                        'memory_norm': memory_norm,
                        'memory_shape': list(memory_shape),
                        'has_memory': True
                    }
                else:
                    retrieval_info = {
                        'layer_idx': layer_idx,
                        'timestamp': time.time(),
                        'memory_norm': 0.0,
                        'memory_shape': None,
                        'has_memory': False
                    }
                
                self.retrieval_patterns.append(retrieval_info)
                return retrieval_info
            return retrieve_hook
        
        def create_update_hook(layer_idx):
            def update_hook(prev_memory, prev_normalization, key_states, value_states):
                # Record memory update
                key_norm = key_states.norm().item()
                value_norm = value_states.norm().item()
                
                if prev_memory is not None:
                    prev_memory_norm = prev_memory.norm().item()
                else:
                    prev_memory_norm = 0.0
                
                update_info = {
                    'layer_idx': layer_idx,
                    'timestamp': time.time(),
                    'prev_memory_norm': prev_memory_norm,
                    'key_norm': key_norm,
                    'value_norm': value_norm
                }
                
                self.memory_states.append(update_info)
                return update_info
            return update_hook
        
        # Hook all decoder layers
        for layer_idx, layer in enumerate(model.model.decoder):
            if hasattr(layer, 'attn'):
                # Store original functions
                original_retrieve = layer.attn._retrieve_from_memory
                original_update = layer.attn._update_memory
                
                # Create monitoring wrapper
                def monitored_retrieve(query_states, prev_memory, prev_normalization, layer_idx=layer_idx):
                    # Record monitoring data
                    create_retrieve_hook(layer_idx)(query_states, prev_memory, prev_normalization)
                    # Call original function
                    return original_retrieve(query_states, prev_memory, prev_normalization)
                
                def monitored_update(prev_memory, prev_normalization, key_states, value_states, layer_idx=layer_idx):
                    # Record monitoring data
                    create_update_hook(layer_idx)(prev_memory, prev_normalization, key_states, value_states)
                    # Call original function
                    return original_update(prev_memory, prev_normalization, key_states, value_states)
                
                # Replace functions with monitored versions
                layer.attn._retrieve_from_memory = monitored_retrieve
                layer.attn._update_memory = monitored_update
    
    def analyze_segment_patterns(self, segment_length: int = 1024):
        """Analyze memory usage patterns by segment."""
        
        # Group retrieval patterns by segment
        segment_data = {}
        
        for retrieval in self.retrieval_patterns:
            # Estimate segment based on layer progression
            # This is approximate - in practice you'd track token position
            segment_idx = len(self.segment_stats)  # Simplified
            
            if segment_idx not in segment_data:
                segment_data[segment_idx] = {
                    'retrievals': [],
                    'memory_active_layers': 0,
                    'total_memory_norm': 0.0
                }
            
            segment_data[segment_idx]['retrievals'].append(retrieval)
            if retrieval['has_memory']:
                segment_data[segment_idx]['memory_active_layers'] += 1
                segment_data[segment_idx]['total_memory_norm'] += retrieval['memory_norm']
        
        # Analyze patterns
        analysis = {
            'total_segments': len(segment_data),
            'segments_with_memory': sum(1 for s in segment_data.values() if s['total_memory_norm'] > 0),
            'memory_activation_rate': 0.0,
            'average_memory_norm': 0.0,
            'segment_details': []
        }
        
        if segment_data:
            total_retrievals = sum(len(s['retrievals']) for s in segment_data.values())
            memory_retrievals = sum(s['memory_active_layers'] for s in segment_data.values())
            
            analysis['memory_activation_rate'] = memory_retrievals / max(total_retrievals, 1)
            analysis['average_memory_norm'] = np.mean([s['total_memory_norm'] for s in segment_data.values()])
            
            for seg_idx, seg_data in segment_data.items():
                analysis['segment_details'].append({
                    'segment': seg_idx,
                    'active_layers': seg_data['memory_active_layers'],
                    'total_norm': seg_data['total_memory_norm'],
                    'avg_norm_per_layer': seg_data['total_memory_norm'] / max(seg_data['memory_active_layers'], 1)
                })
        
        return analysis
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory usage report."""
        
        segment_analysis = self.analyze_segment_patterns()
        
        # Overall statistics
        total_retrievals = len(self.retrieval_patterns)
        memory_retrievals = sum(1 for r in self.retrieval_patterns if r['has_memory'])
        
        report = {
            'summary': {
                'total_retrievals': total_retrievals,
                'memory_retrievals': memory_retrievals,
                'memory_usage_rate': memory_retrievals / max(total_retrievals, 1),
                'segments_analyzed': segment_analysis['total_segments'],
                'segments_with_memory': segment_analysis['segments_with_memory']
            },
            'memory_patterns': {
                'activation_rate': segment_analysis['memory_activation_rate'],
                'average_norm': segment_analysis['average_memory_norm'],
                'segment_details': segment_analysis['segment_details']
            },
            'retrieval_timeline': self.retrieval_patterns,
            'memory_updates': self.memory_states,
            'analysis': {
                'memory_mechanism_active': memory_retrievals > 0,
                'cross_segment_flow': segment_analysis['segments_with_memory'] > 1,
                'memory_effectiveness': self._assess_effectiveness(segment_analysis)
            }
        }
        
        return report
    
    def _assess_effectiveness(self, segment_analysis: Dict) -> str:
        """Assess the effectiveness of memory mechanism."""
        
        if segment_analysis['segments_with_memory'] == 0:
            return "INACTIVE: No memory usage detected"
        elif segment_analysis['segments_with_memory'] == 1:
            return "LIMITED: Memory only active in single segment"
        elif segment_analysis['memory_activation_rate'] > 0.5:
            return "HIGHLY_ACTIVE: Strong memory usage across segments"
        elif segment_analysis['memory_activation_rate'] > 0.2:
            return "MODERATELY_ACTIVE: Decent memory usage"
        else:
            return "WEAKLY_ACTIVE: Minimal memory usage"
    
    def save_results(self, report: Dict[str, Any], prefix: str = "memory_debug"):
        """Save analysis results."""
        
        # Save JSON report
        json_path = self.output_dir / f"{prefix}_report.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save detailed logs
        logs_path = self.output_dir / f"{prefix}_detailed_logs.json"
        detailed_logs = {
            'retrieval_patterns': self.retrieval_patterns,
            'memory_states': self.memory_states,
            'segment_stats': self.segment_stats
        }
        with open(logs_path, 'w') as f:
            json.dump(detailed_logs, f, indent=2)
        
        print(f"Results saved to: {self.output_dir}")
        return json_path, logs_path


def load_model_and_tokenizer(checkpoint_path: str):
    """Load model and tokenizer from checkpoint."""
    
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists(), f"Checkpoint path {checkpoint_path} does not exist"
    
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
        data_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
    )
    
    # Set random seed
    set_random_seed(42)
    
    # Build model
    model_config_cls = model_config.__class__.__name__
    if model_config_cls not in CONFIG_TO_MODEL_CLASS:
        raise ValueError(f"Unsupported model config {model_config_cls}")
    
    random_states = RandomStates({"tp_synced": get_synced_random_state(
        random_state=get_current_random_state(), 
        pg=parallel_context.tp_pg
    )})
    
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
    
    # Mark tied parameters
    mark_tied_parameters(model=model, parallel_context=parallel_context, parallel_config=parallel_config)
    
    # Load weights
    load_weights(model=model, parallel_context=parallel_context, root_folder=checkpoint_path)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    return model, tokenizer, parallel_context, config


def test_memory_usage(
    model, 
    tokenizer, 
    parallel_context, 
    config,
    context_lengths: List[int],
    monitor: MemoryUsageMonitor,
    num_samples: int = 5
):
    """Test memory usage across different context lengths."""
    
    results = {}
    
    for context_length in context_lengths:
        print(f"\nTesting context length: {context_length} tokens")
        
        # Create test prompts of specified length
        test_prompts = []
        for i in range(num_samples):
            # Generate a prompt that will be close to context_length tokens
            base_text = "The quick brown fox jumps over the lazy dog. " * (context_length // 10)
            test_prompts.append(base_text)
        
        context_results = {
            'context_length': context_length,
            'expected_segments': max(1, context_length // config.model.model_config.max_position_embeddings),  # Simplified
            'samples': []
        }
        
        for i, prompt in enumerate(test_prompts):
            print(f"  Sample {i+1}/{num_samples}")
            
            # Clear previous monitoring data
            monitor.memory_states.clear()
            monitor.retrieval_patterns.clear()
            
            # Generate response
            try:
                outputs = decode_text(
                    input_iter=[GenerationInput(text=prompt)],
                    tokenizer=tokenizer,
                    model=model.model,
                    parallel_context=parallel_context,
                    max_new_tokens=20,
                    max_micro_batch_size=1,
                    generation_config=GenerationArgs(sampler="greedy", use_cache=False),
                    tokenizer_config=TokenizerConfig(max_input_length=context_length),
                )
                
                # Analyze this sample's memory usage
                sample_report = monitor.generate_report()
                context_results['samples'].append({
                    'sample_id': i,
                    'prompt_length': len(tokenizer.encode(prompt)),
                    'memory_usage': sample_report['summary'],
                    'effectiveness': sample_report['analysis']['memory_effectiveness']
                })
                
            except Exception as e:
                print(f"    Error processing sample {i}: {e}")
                context_results['samples'].append({
                    'sample_id': i,
                    'error': str(e)
                })
        
        results[context_length] = context_results
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Debug Infini-Attention Memory Usage")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--context-lengths", type=str, default="1024,2048,4096",
                       help="Comma-separated list of context lengths to test")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of samples per context length")
    parser.add_argument("--output-dir", type=str, default="./memory_debug_results",
                       help="Output directory for results")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Parse context lengths
    context_lengths = [int(x.strip()) for x in args.context_lengths.split(',')]
    
    print("Infini-Attention Memory Usage Debugger")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Context lengths: {context_lengths}")
    print(f"Samples per length: {args.num_samples}")
    print(f"Output directory: {args.output_dir}")
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer, parallel_context, config = load_model_and_tokenizer(args.checkpoint)
    
    # Initialize monitor
    monitor = MemoryUsageMonitor(args.output_dir)
    
    # Hook memory functions
    print("Setting up memory monitoring hooks...")
    monitor.hook_memory_functions(model)
    
    # Test memory usage
    print("\nTesting memory usage...")
    results = test_memory_usage(
        model, tokenizer, parallel_context, config,
        context_lengths, monitor, args.num_samples
    )
    
    # Generate final report
    print("\nGenerating comprehensive report...")
    final_report = {
        'test_configuration': {
            'checkpoint': args.checkpoint,
            'context_lengths': context_lengths,
            'num_samples': args.num_samples,
            'model_config': {
                'segment_length': getattr(config.infini_attention, 'segment_length', 'unknown'),
                'turn_on_memory': getattr(config.infini_attention, 'turn_on_memory', 'unknown'),
                'balance_factor_lr': getattr(config.infini_attention, 'balance_factor_lr', 'unknown')
            }
        },
        'results_by_context': results,
        'overall_analysis': analyze_overall_patterns(results)
    }
    
    # Save results
    json_path, logs_path = monitor.save_results(final_report, "comprehensive_memory_debug")
    
    # Print summary
    print("\n" + "=" * 50)
    print("MEMORY USAGE ANALYSIS SUMMARY")
    print("=" * 50)
    
    overall = final_report['overall_analysis']
    print(f"Memory Mechanism Status: {overall['status']}")
    print(f"Cross-Segment Capability: {overall['cross_segment_capable']}")
    print(f"Effectiveness Rating: {overall['effectiveness_rating']}")
    
    if overall['recommendations']:
        print("\nRecommendations:")
        for rec in overall['recommendations']:
            print(f"  - {rec}")
    
    print(f"\nDetailed results saved to: {json_path}")
    
    return final_report


def analyze_overall_patterns(results: Dict) -> Dict[str, Any]:
    """Analyze overall patterns across all context lengths."""
    
    total_samples = 0
    memory_active_samples = 0
    context_with_memory = 0
    
    effectiveness_ratings = []
    
    for context_length, context_data in results.items():
        has_memory_in_context = False
        
        for sample in context_data['samples']:
            if 'memory_usage' in sample:
                total_samples += 1
                if sample['memory_usage']['memory_retrievals'] > 0:
                    memory_active_samples += 1
                    has_memory_in_context = True
                
                # Extract effectiveness
                if 'effectiveness' in sample:
                    effectiveness_ratings.append(sample['effectiveness'])
        
        if has_memory_in_context:
            context_with_memory += 1
    
    # Overall analysis
    memory_activation_rate = memory_active_samples / max(total_samples, 1)
    context_coverage = context_with_memory / max(len(results), 1)
    
    # Determine status
    if memory_activation_rate > 0.8 and context_coverage > 0.8:
        status = "EXCELLENT: Memory highly active across contexts"
    elif memory_activation_rate > 0.5 and context_coverage > 0.5:
        status = "GOOD: Memory moderately active"
    elif memory_activation_rate > 0.1:
        status = "WEAK: Limited memory usage"
    else:
        status = "BROKEN: No memory activity detected"
    
    # Generate recommendations
    recommendations = []
    if memory_activation_rate < 0.1:
        recommendations.append("Check if model was trained with turn_on_memory=true")
        recommendations.append("Verify balance_factor_lr was > 0 during training")
    if context_coverage < 0.5:
        recommendations.append("Test with longer contexts to trigger memory usage")
    if memory_activation_rate < 0.3:
        recommendations.append("Examine balance factors - they may be stuck at extremes")
    
    return {
        'status': status,
        'memory_activation_rate': memory_activation_rate,
        'context_coverage': context_coverage,
        'cross_segment_capable': context_coverage > 0.5,
        'effectiveness_rating': max(effectiveness_ratings, default="UNKNOWN"),
        'recommendations': recommendations,
        'statistics': {
            'total_samples': total_samples,
            'memory_active_samples': memory_active_samples,
            'contexts_tested': len(results),
            'contexts_with_memory': context_with_memory
        }
    }


if __name__ == "__main__":
    main()