#!/usr/bin/env python3
"""
Memory vs No-Memory Performance Comparison

This script performs A/B testing to compare performance with memory enabled
vs disabled, providing statistical evidence of memory mechanism effectiveness.

Usage:
    python scripts/compare_memory_vs_no_memory.py --checkpoint ./checkpoints/model/30000
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import copy

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datasets import Dataset
from transformers import AutoTokenizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


@dataclass
class ComparisonResult:
    """Results from memory vs no-memory comparison."""
    context_length: int
    with_memory_scores: List[float]
    without_memory_scores: List[float]
    statistical_test: Dict[str, float]
    effect_size: float
    interpretation: str


class MemoryComparison:
    """Compare performance with and without memory mechanism."""
    
    def __init__(self, checkpoint_path: str, output_dir: str = "./memory_comparison"):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.parallel_context = None
        self.config = None
        self.original_config = None
        
    def load_model_components(self):
        """Load model, tokenizer, and configuration."""
        
        print("Loading model components...")
        
        # Load configuration
        config_path = self.checkpoint_path / "config.yaml"
        self.original_config = get_config_from_file(config_path.as_posix())
        constants.CONFIG = self.original_config
        
        model_config = self.original_config.model.model_config
        tokenizer_path = self.original_config.tokenizer.tokenizer_name_or_path
        
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
        self.parallel_context = ParallelContext(
            data_parallel_size=1,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
        )
        
        # Set random seed for reproducibility
        set_random_seed(42)
        
        # Build model
        model_config_cls = model_config.__class__.__name__
        if model_config_cls not in CONFIG_TO_MODEL_CLASS:
            raise ValueError(f"Unsupported model config {model_config_cls}")
        
        random_states = RandomStates({"tp_synced": get_synced_random_state(
            random_state=get_current_random_state(), 
            pg=self.parallel_context.tp_pg
        )})
        
        self.model = build_model(
            model_builder=lambda: CONFIG_TO_MODEL_CLASS[model_config_cls](
                config=model_config,
                parallel_context=self.parallel_context,
                parallel_config=parallel_config,
                random_states=random_states,
            ),
            dtype=torch.bfloat16,
            parallel_context=self.parallel_context,
        )
        
        # Mark tied parameters
        mark_tied_parameters(model=self.model, parallel_context=self.parallel_context, parallel_config=parallel_config)
        
        # Load weights
        load_weights(model=self.model, parallel_context=self.parallel_context, root_folder=self.checkpoint_path)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        
        print("Model components loaded successfully")
    
    def create_test_dataset(self, context_lengths: List[int], samples_per_length: int = 20) -> Dict[int, List[str]]:
        """Create test dataset with passkey retrieval tasks."""
        
        print("Creating test dataset...")
        
        dataset = {}
        
        for context_length in context_lengths:
            print(f"  Creating {samples_per_length} samples for {context_length} tokens")
            
            samples = []
            for i in range(samples_per_length):
                sample = self._create_passkey_sample(context_length, seed=i)
                samples.append(sample)
            
            dataset[context_length] = samples
        
        return dataset
    
    def _create_passkey_sample(self, context_length: int, seed: int = 42) -> Dict[str, Any]:
        """Create a single passkey retrieval sample."""
        
        np.random.seed(seed)
        
        # Generate random passkey
        passkey = np.random.randint(10000, 99999)
        
        # Choose random position for passkey (avoid very beginning and end)
        passkey_position = np.random.randint(100, context_length - 100)
        
        # Create filler text
        filler_sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "The weather today is sunny with a chance of rain later.",
            "Machine learning is transforming many industries worldwide.",
            "Scientific research continues to advance our understanding.",
            "Technology plays an increasingly important role in society.",
            "Education is the foundation of personal and societal growth.",
            "Environmental conservation is crucial for future generations.",
            "Art and culture enrich our lives in countless ways.",
            "Communication skills are essential in the modern workplace."
        ]
        
        # Generate text to reach approximately the target length
        text_parts = []
        current_length = 0
        
        while current_length < passkey_position:
            sentence = np.random.choice(filler_sentences)
            text_parts.append(sentence)
            current_length += len(self.tokenizer.encode(sentence))
        
        # Insert passkey
        passkey_text = f" The passkey is {passkey}. Remember this number."
        text_parts.append(passkey_text)
        current_length += len(self.tokenizer.encode(passkey_text))
        
        # Add more filler to reach target length
        while current_length < context_length:
            sentence = np.random.choice(filler_sentences)
            text_parts.append(sentence)
            current_length += len(self.tokenizer.encode(sentence))
            if current_length > context_length:
                break
        
        # Create final text and question
        context = " ".join(text_parts)
        question = " What is the passkey?"
        full_prompt = context + question
        
        # Truncate to exact length if needed
        encoded = self.tokenizer.encode(full_prompt)
        if len(encoded) > context_length:
            encoded = encoded[:context_length]
            full_prompt = self.tokenizer.decode(encoded)
        
        return {
            "prompt": full_prompt,
            "target": str(passkey),
            "context_length": len(self.tokenizer.encode(full_prompt)),
            "passkey_position": passkey_position
        }
    
    def evaluate_with_memory(self, dataset: Dict[int, List[str]]) -> Dict[int, List[float]]:
        """Evaluate performance with memory enabled."""
        
        print("Evaluating with memory enabled...")
        
        # Ensure memory is enabled
        original_memory_setting = constants.CONFIG.infini_attention.turn_on_memory
        constants.CONFIG.infini_attention.turn_on_memory = True
        
        results = {}
        
        try:
            for context_length, samples in dataset.items():
                print(f"  Testing {context_length} tokens...")
                
                scores = []
                for i, sample in enumerate(samples):
                    score = self._evaluate_single_sample(sample)
                    scores.append(score)
                    
                    if (i + 1) % 5 == 0:
                        print(f"    Completed {i + 1}/{len(samples)} samples")
                
                results[context_length] = scores
                avg_score = np.mean(scores)
                print(f"    Average accuracy: {avg_score:.3f}")
        
        finally:
            # Restore original setting
            constants.CONFIG.infini_attention.turn_on_memory = original_memory_setting
        
        return results
    
    def evaluate_without_memory(self, dataset: Dict[int, List[str]]) -> Dict[int, List[float]]:
        """Evaluate performance with memory disabled."""
        
        print("Evaluating with memory disabled...")
        
        # Disable memory
        original_memory_setting = constants.CONFIG.infini_attention.turn_on_memory
        constants.CONFIG.infini_attention.turn_on_memory = False
        
        results = {}
        
        try:
            for context_length, samples in dataset.items():
                print(f"  Testing {context_length} tokens...")
                
                scores = []
                for i, sample in enumerate(samples):
                    score = self._evaluate_single_sample(sample)
                    scores.append(score)
                    
                    if (i + 1) % 5 == 0:
                        print(f"    Completed {i + 1}/{len(samples)} samples")
                
                results[context_length] = scores
                avg_score = np.mean(scores)
                print(f"    Average accuracy: {avg_score:.3f}")
        
        finally:
            # Restore original setting
            constants.CONFIG.infini_attention.turn_on_memory = original_memory_setting
        
        return results
    
    def _evaluate_single_sample(self, sample: Dict[str, Any]) -> float:
        """Evaluate a single passkey sample."""
        
        try:
            # Generate response
            outputs = decode_text(
                input_iter=[GenerationInput(text=sample["prompt"])],
                tokenizer=self.tokenizer,
                model=self.model.model,
                parallel_context=self.parallel_context,
                max_new_tokens=10,  # Only need a few tokens for the number
                max_micro_batch_size=1,
                generation_config=GenerationArgs(sampler="greedy", use_cache=False),
                tokenizer_config=TokenizerConfig(max_input_length=sample["context_length"] + 10),
            )
            
            # Extract generated text
            output_list = list(outputs)  # Convert generator to list
            if output_list and len(output_list) > 0:
                generated_text = output_list[0]
                # Extract the answer from the generated text
                # Look for the passkey number in the response
                import re
                numbers = re.findall(r'\b\d{5}\b', generated_text)
                
                if numbers and str(sample["target"]) in numbers:
                    return 1.0  # Correct
                else:
                    return 0.0  # Incorrect
            else:
                return 0.0
                
        except Exception as e:
            print(f"    Error evaluating sample: {e}")
            return 0.0
    
    def perform_statistical_analysis(self, with_memory: Dict[int, List[float]], 
                                   without_memory: Dict[int, List[float]]) -> List[ComparisonResult]:
        """Perform statistical analysis of the comparison."""
        
        print("Performing statistical analysis...")
        
        results = []
        
        for context_length in with_memory.keys():
            mem_scores = with_memory[context_length]
            no_mem_scores = without_memory[context_length]
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(mem_scores, no_mem_scores)
            
            # Calculate effect size (Cohen's d)
            mem_mean = np.mean(mem_scores)
            no_mem_mean = np.mean(no_mem_scores)
            pooled_std = np.sqrt(((len(mem_scores) - 1) * np.var(mem_scores, ddof=1) + 
                                 (len(no_mem_scores) - 1) * np.var(no_mem_scores, ddof=1)) / 
                                (len(mem_scores) + len(no_mem_scores) - 2))
            
            effect_size = (mem_mean - no_mem_mean) / pooled_std if pooled_std > 0 else 0
            
            # Interpret results
            interpretation = self._interpret_comparison(mem_mean, no_mem_mean, p_value, effect_size)
            
            result = ComparisonResult(
                context_length=context_length,
                with_memory_scores=mem_scores,
                without_memory_scores=no_mem_scores,
                statistical_test={
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'with_memory_mean': float(mem_mean),
                    'without_memory_mean': float(no_mem_mean),
                    'with_memory_std': float(np.std(mem_scores)),
                    'without_memory_std': float(np.std(no_mem_scores))
                },
                effect_size=float(effect_size),
                interpretation=interpretation
            )
            
            results.append(result)
            
            print(f"  {context_length} tokens:")
            print(f"    With memory: {mem_mean:.3f} ± {np.std(mem_scores):.3f}")
            print(f"    Without memory: {no_mem_mean:.3f} ± {np.std(no_mem_scores):.3f}")
            print(f"    p-value: {p_value:.6f}")
            print(f"    Effect size: {effect_size:.3f}")
            print(f"    Interpretation: {interpretation}")
        
        return results
    
    def _interpret_comparison(self, mem_mean: float, no_mem_mean: float, 
                            p_value: float, effect_size: float) -> str:
        """Interpret the comparison results."""
        
        # Check statistical significance
        is_significant = p_value < 0.05
        
        # Check practical significance (effect size)
        if abs(effect_size) >= 0.8:
            effect_magnitude = "large"
        elif abs(effect_size) >= 0.5:
            effect_magnitude = "medium"
        elif abs(effect_size) >= 0.2:
            effect_magnitude = "small"
        else:
            effect_magnitude = "negligible"
        
        # Determine direction
        if mem_mean > no_mem_mean:
            direction = "Memory improves performance"
        elif mem_mean < no_mem_mean:
            direction = "Memory hurts performance"
        else:
            direction = "No performance difference"
        
        # Combine interpretations
        if is_significant and effect_magnitude in ["large", "medium"]:
            if mem_mean > no_mem_mean:
                return f"SIGNIFICANT_IMPROVEMENT: {direction} with {effect_magnitude} effect"
            else:
                return f"SIGNIFICANT_DEGRADATION: {direction} with {effect_magnitude} effect"
        elif is_significant:
            return f"STATISTICALLY_SIGNIFICANT: {direction} but {effect_magnitude} effect"
        elif effect_magnitude in ["large", "medium"]:
            return f"PRACTICALLY_SIGNIFICANT: {direction} with {effect_magnitude} effect (not statistically significant)"
        else:
            return f"NO_SIGNIFICANT_DIFFERENCE: {direction} with {effect_magnitude} effect"
    
    def create_visualizations(self, results: List[ComparisonResult]) -> List[str]:
        """Create comprehensive visualizations."""
        
        viz_files = []
        
        # 1. Performance comparison plot
        fig = self._create_performance_plot(results)
        perf_path = self.output_dir / "performance_comparison.html"
        fig.write_html(str(perf_path))
        viz_files.append(str(perf_path))
        
        # 2. Statistical significance plot
        fig = self._create_significance_plot(results)
        sig_path = self.output_dir / "statistical_significance.html"
        fig.write_html(str(sig_path))
        viz_files.append(str(sig_path))
        
        # 3. Effect size analysis
        fig = self._create_effect_size_plot(results)
        effect_path = self.output_dir / "effect_size_analysis.html"
        fig.write_html(str(effect_path))
        viz_files.append(str(effect_path))
        
        # 4. Static plots for publications
        self._create_publication_plots(results)
        viz_files.extend([
            str(self.output_dir / "comparison_static.png"),
            str(self.output_dir / "statistical_analysis_static.png")
        ])
        
        return viz_files
    
    def _create_performance_plot(self, results: List[ComparisonResult]):
        """Create performance comparison plot."""
        
        context_lengths = [r.context_length for r in results]
        with_memory_means = [r.statistical_test['with_memory_mean'] for r in results]
        without_memory_means = [r.statistical_test['without_memory_mean'] for r in results]
        with_memory_stds = [r.statistical_test['with_memory_std'] for r in results]
        without_memory_stds = [r.statistical_test['without_memory_std'] for r in results]
        
        fig = go.Figure()
        
        # With memory
        fig.add_trace(go.Scatter(
            x=context_lengths,
            y=with_memory_means,
            error_y=dict(type='data', array=with_memory_stds, visible=True),
            mode='lines+markers',
            name='With Memory',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        # Without memory
        fig.add_trace(go.Scatter(
            x=context_lengths,
            y=without_memory_means,
            error_y=dict(type='data', array=without_memory_stds, visible=True),
            mode='lines+markers',
            name='Without Memory',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Performance Comparison: Memory vs No Memory',
            xaxis_title='Context Length (tokens)',
            yaxis_title='Accuracy',
            yaxis=dict(range=[0, 1]),
            legend=dict(x=0.7, y=0.95),
            width=800,
            height=500
        )
        
        return fig
    
    def _create_significance_plot(self, results: List[ComparisonResult]):
        """Create statistical significance visualization."""
        
        context_lengths = [r.context_length for r in results]
        p_values = [r.statistical_test['p_value'] for r in results]
        
        # Create significance categories
        sig_colors = []
        sig_labels = []
        for p in p_values:
            if p < 0.001:
                sig_colors.append('darkgreen')
                sig_labels.append('p < 0.001 (Highly Significant)')
            elif p < 0.01:
                sig_colors.append('green')
                sig_labels.append('p < 0.01 (Very Significant)')
            elif p < 0.05:
                sig_colors.append('orange')
                sig_labels.append('p < 0.05 (Significant)')
            else:
                sig_colors.append('red')
                sig_labels.append('p ≥ 0.05 (Not Significant)')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=context_lengths,
            y=[-np.log10(p) for p in p_values],
            marker_color=sig_colors,
            text=sig_labels,
            textposition='outside',
            name='Statistical Significance'
        ))
        
        # Add significance threshold lines
        fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="orange", 
                     annotation_text="p = 0.05 threshold")
        fig.add_hline(y=-np.log10(0.01), line_dash="dash", line_color="green",
                     annotation_text="p = 0.01 threshold")
        
        fig.update_layout(
            title='Statistical Significance of Memory Effect',
            xaxis_title='Context Length (tokens)',
            yaxis_title='-log10(p-value)',
            showlegend=False,
            width=800,
            height=500
        )
        
        return fig
    
    def _create_effect_size_plot(self, results: List[ComparisonResult]):
        """Create effect size visualization."""
        
        context_lengths = [r.context_length for r in results]
        effect_sizes = [r.effect_size for r in results]
        
        # Color code by effect size magnitude
        colors = []
        for es in effect_sizes:
            if abs(es) >= 0.8:
                colors.append('darkgreen')
            elif abs(es) >= 0.5:
                colors.append('green')
            elif abs(es) >= 0.2:
                colors.append('orange')
            else:
                colors.append('red')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=context_lengths,
            y=effect_sizes,
            marker_color=colors,
            name='Effect Size (Cohen\'s d)'
        ))
        
        # Add effect size interpretation lines
        fig.add_hline(y=0.2, line_dash="dash", line_color="orange",
                     annotation_text="Small effect (0.2)")
        fig.add_hline(y=0.5, line_dash="dash", line_color="green",
                     annotation_text="Medium effect (0.5)")
        fig.add_hline(y=0.8, line_dash="dash", line_color="darkgreen",
                     annotation_text="Large effect (0.8)")
        fig.add_hline(y=-0.2, line_dash="dash", line_color="orange")
        fig.add_hline(y=-0.5, line_dash="dash", line_color="green")
        fig.add_hline(y=-0.8, line_dash="dash", line_color="darkgreen")
        
        fig.update_layout(
            title='Effect Size of Memory Mechanism',
            xaxis_title='Context Length (tokens)',
            yaxis_title='Effect Size (Cohen\'s d)',
            showlegend=False,
            width=800,
            height=500
        )
        
        return fig
    
    def _create_publication_plots(self, results: List[ComparisonResult]):
        """Create static plots for publications."""
        
        plt.style.use('seaborn-v0_8')
        
        # Performance comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        context_lengths = [r.context_length for r in results]
        with_memory_means = [r.statistical_test['with_memory_mean'] for r in results]
        without_memory_means = [r.statistical_test['without_memory_mean'] for r in results]
        with_memory_stds = [r.statistical_test['with_memory_std'] for r in results]
        without_memory_stds = [r.statistical_test['without_memory_std'] for r in results]
        
        ax1.errorbar(context_lengths, with_memory_means, yerr=with_memory_stds,
                    marker='o', linewidth=2, label='With Memory', color='blue')
        ax1.errorbar(context_lengths, without_memory_means, yerr=without_memory_stds,
                    marker='s', linewidth=2, label='Without Memory', color='red')
        ax1.set_xlabel('Context Length (tokens)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Performance Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Effect sizes
        effect_sizes = [r.effect_size for r in results]
        colors = ['darkgreen' if abs(es) >= 0.8 else 'green' if abs(es) >= 0.5 
                 else 'orange' if abs(es) >= 0.2 else 'red' for es in effect_sizes]
        
        bars = ax2.bar(context_lengths, effect_sizes, color=colors, alpha=0.7)
        ax2.axhline(y=0.2, linestyle='--', color='orange', alpha=0.7, label='Small effect')
        ax2.axhline(y=0.5, linestyle='--', color='green', alpha=0.7, label='Medium effect')
        ax2.axhline(y=0.8, linestyle='--', color='darkgreen', alpha=0.7, label='Large effect')
        ax2.axhline(y=0, linestyle='-', color='black', alpha=0.5)
        ax2.set_xlabel('Context Length (tokens)')
        ax2.set_ylabel('Effect Size (Cohen\'s d)')
        ax2.set_title('Effect Size Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "comparison_static.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Statistical significance plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        p_values = [r.statistical_test['p_value'] for r in results]
        log_p_values = [-np.log10(p) for p in p_values]
        
        sig_colors = ['darkgreen' if p < 0.001 else 'green' if p < 0.01 
                     else 'orange' if p < 0.05 else 'red' for p in p_values]
        
        bars = ax.bar(context_lengths, log_p_values, color=sig_colors, alpha=0.7)
        ax.axhline(y=-np.log10(0.05), linestyle='--', color='orange', label='p = 0.05')
        ax.axhline(y=-np.log10(0.01), linestyle='--', color='green', label='p = 0.01')
        ax.axhline(y=-np.log10(0.001), linestyle='--', color='darkgreen', label='p = 0.001')
        
        ax.set_xlabel('Context Length (tokens)')
        ax.set_ylabel('-log10(p-value)')
        ax.set_title('Statistical Significance of Memory Effect')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "statistical_analysis_static.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self, results: List[ComparisonResult]) -> Dict[str, Any]:
        """Generate comprehensive comparison report."""
        
        # Overall statistics
        overall_with_memory = np.concatenate([r.with_memory_scores for r in results])
        overall_without_memory = np.concatenate([r.without_memory_scores for r in results])
        
        overall_t_stat, overall_p_value = stats.ttest_ind(overall_with_memory, overall_without_memory)
        
        # Effect size
        overall_effect_size = (np.mean(overall_with_memory) - np.mean(overall_without_memory)) / \
                             np.sqrt((np.var(overall_with_memory) + np.var(overall_without_memory)) / 2)
        
        # Create visualizations
        viz_files = self.create_visualizations(results)
        
        report = {
            'experiment_info': {
                'checkpoint_path': str(self.checkpoint_path),
                'context_lengths_tested': [r.context_length for r in results],
                'samples_per_length': len(results[0].with_memory_scores) if results else 0,
                'total_comparisons': len(results)
            },
            'overall_results': {
                'with_memory_mean': float(np.mean(overall_with_memory)),
                'without_memory_mean': float(np.mean(overall_without_memory)),
                'overall_p_value': float(overall_p_value),
                'overall_effect_size': float(overall_effect_size),
                'memory_advantage': float(np.mean(overall_with_memory) - np.mean(overall_without_memory))
            },
            'detailed_results': [
                {
                    'context_length': r.context_length,
                    'statistical_test': r.statistical_test,
                    'effect_size': r.effect_size,
                    'interpretation': r.interpretation
                }
                for r in results
            ],
            'visualizations': viz_files,
            'conclusion': self._draw_overall_conclusion(results, overall_effect_size, overall_p_value),
            'recommendations': self._generate_recommendations(results)
        }
        
        return report
    
    def _draw_overall_conclusion(self, results: List[ComparisonResult], 
                               overall_effect_size: float, overall_p_value: float) -> Dict[str, Any]:
        """Draw overall conclusion from comparison results."""
        
        # Count significant improvements
        significant_improvements = sum(1 for r in results 
                                     if 'IMPROVEMENT' in r.interpretation and 
                                        r.statistical_test['p_value'] < 0.05)
        
        total_contexts = len(results)
        improvement_rate = significant_improvements / total_contexts if total_contexts > 0 else 0
        
        # Determine overall memory effectiveness
        if improvement_rate >= 0.8 and overall_effect_size > 0.5:
            effectiveness = "HIGHLY_EFFECTIVE"
        elif improvement_rate >= 0.5 and overall_effect_size > 0.3:
            effectiveness = "MODERATELY_EFFECTIVE"
        elif improvement_rate >= 0.3 or overall_effect_size > 0.2:
            effectiveness = "SOMEWHAT_EFFECTIVE"
        else:
            effectiveness = "INEFFECTIVE"
        
        return {
            'memory_effectiveness': effectiveness,
            'improvement_rate': improvement_rate,
            'significant_improvements': significant_improvements,
            'total_contexts_tested': total_contexts,
            'overall_statistical_significance': overall_p_value < 0.05,
            'overall_effect_magnitude': 'large' if abs(overall_effect_size) >= 0.8 
                                      else 'medium' if abs(overall_effect_size) >= 0.5
                                      else 'small' if abs(overall_effect_size) >= 0.2
                                      else 'negligible',
            'summary': f"Memory mechanism shows {effectiveness.lower().replace('_', ' ')} "
                      f"performance with {improvement_rate:.1%} of contexts showing significant improvement"
        }
    
    def _generate_recommendations(self, results: List[ComparisonResult]) -> List[str]:
        """Generate recommendations based on comparison results."""
        
        recommendations = []
        
        # Check for consistent patterns
        improvements = [r for r in results if 'IMPROVEMENT' in r.interpretation]
        degradations = [r for r in results if 'DEGRADATION' in r.interpretation]
        
        if len(improvements) > len(degradations):
            if len(improvements) == len(results):
                recommendations.append("Memory mechanism is working well across all context lengths")
            else:
                recommendations.append("Memory shows benefits but may need optimization for some context lengths")
        elif len(degradations) > len(improvements):
            recommendations.append("Memory mechanism may be interfering with performance - check implementation")
        else:
            recommendations.append("Memory effect is inconsistent - investigate training or configuration issues")
        
        # Context length specific recommendations
        long_context_results = [r for r in results if r.context_length >= 4096]
        if long_context_results:
            long_context_improvements = [r for r in long_context_results if 'IMPROVEMENT' in r.interpretation]
            if len(long_context_improvements) / len(long_context_results) < 0.5:
                recommendations.append("Memory mechanism underperforming at longer contexts - may need training adjustments")
        
        # Effect size recommendations
        small_effects = [r for r in results if abs(r.effect_size) < 0.2]
        if len(small_effects) > len(results) / 2:
            recommendations.append("Effect sizes are small - consider increasing balance_factor_lr or training longer")
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(description="Compare Memory vs No-Memory Performance")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--context-lengths", type=str, default="1024,2048,4096",
                       help="Comma-separated list of context lengths to test")
    parser.add_argument("--samples-per-length", type=int, default=20,
                       help="Number of samples per context length")
    parser.add_argument("--output-dir", type=str, default="./memory_comparison_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Parse context lengths
    context_lengths = [int(x.strip()) for x in args.context_lengths.split(',')]
    
    print("Memory vs No-Memory Performance Comparison")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Context lengths: {context_lengths}")
    print(f"Samples per length: {args.samples_per_length}")
    print(f"Output directory: {args.output_dir}")
    
    # Initialize comparison
    comparison = MemoryComparison(args.checkpoint, args.output_dir)
    
    # Load model components
    comparison.load_model_components()
    
    # Create test dataset
    dataset = comparison.create_test_dataset(context_lengths, args.samples_per_length)
    
    # Run evaluations
    with_memory_results = comparison.evaluate_with_memory(dataset)
    without_memory_results = comparison.evaluate_without_memory(dataset)
    
    # Perform statistical analysis
    statistical_results = comparison.perform_statistical_analysis(
        with_memory_results, without_memory_results
    )
    
    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    report = comparison.generate_comprehensive_report(statistical_results)
    
    # Save report
    report_path = Path(args.output_dir) / "comparison_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    
    # Print summary
    print("\n" + "=" * 50)
    print("MEMORY VS NO-MEMORY COMPARISON SUMMARY")
    print("=" * 50)
    
    conclusion = report['conclusion']
    print(f"Memory Effectiveness: {conclusion['memory_effectiveness']}")
    print(f"Improvement Rate: {conclusion['improvement_rate']:.1%}")
    print(f"Overall Statistical Significance: {conclusion['overall_statistical_significance']}")
    print(f"Overall Effect Size: {conclusion['overall_effect_magnitude']}")
    
    print(f"\nSummary: {conclusion['summary']}")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
    
    print(f"\nDetailed report saved to: {report_path}")
    
    return report


if __name__ == "__main__":
    main()