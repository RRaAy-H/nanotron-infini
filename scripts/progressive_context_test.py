#!/usr/bin/env python3
"""
Progressive Context Length Tester for Infini-Attention

This script systematically tests memory mechanism effectiveness across
progressively increasing context lengths to understand scaling behavior
and identify optimal operating ranges.

Usage:
    python scripts/progressive_context_test.py --checkpoint ./checkpoints/model/30000
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import copy

import numpy as np
import torch
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Import formal plotting style configuration
try:
    from formal_plot_style import (
        ACADEMIC_COLORS, save_plotly_figure, save_matplotlib_figure,
        create_comparison_colors
    )
except ImportError:
    print("⚠️  Warning: formal_plot_style module not found. Using default styling.")
    # Fallback colors if formal_plot_style is not available
    ACADEMIC_COLORS = {
        'memory_enabled': '#1f77b4',
        'memory_disabled': '#d62728',
        'primary_blue': '#1f77b4',
        'primary_red': '#d62728',
        'primary_green': '#2ca02c',
        'primary_orange': '#ff7f0e',
        'improvement': '#2ca02c',
        'degradation': '#d62728',
        'neutral': '#7f7f7f'
    }
    
    def save_plotly_figure(fig, output_path, html_filename, vector_filename, width=1200, height=800,
                          vector_format='pdf', include_png=False):
        """Fallback function if formal_plot_style is not available."""
        output_path.mkdir(parents=True, exist_ok=True)
        saved_files = []
        
        # Save HTML
        html_path = output_path / f"{html_filename}.html"
        fig.write_html(str(html_path))
        saved_files.append(str(html_path))
        
        # Try vector format first
        try:
            vector_path = output_path / f"{vector_filename}.{vector_format}"
            fig.write_image(str(vector_path), format=vector_format, width=width, height=height)
            saved_files.append(str(vector_path))
        except Exception as e:
            print(f"Warning: Could not save {vector_format} image: {e}")
            # Fallback to PNG
            try:
                png_path = output_path / f"{vector_filename}.png"
                fig.write_image(str(png_path), width=width, height=height, scale=2)
                saved_files.append(str(png_path))
            except Exception as png_e:
                print(f"Warning: PNG fallback also failed: {png_e}")
        
        return saved_files
    
    def save_matplotlib_figure(fig, output_path, filename, figsize=(12, 8), vector_format='pdf',
                              include_png=False, dpi=300):
        """Fallback function if formal_plot_style is not available."""
        output_path.mkdir(parents=True, exist_ok=True)
        fig.set_size_inches(figsize)
        
        # Try vector format first
        try:
            vector_path = output_path / f"{filename}.{vector_format}"
            fig.savefig(str(vector_path), format=vector_format, bbox_inches='tight', facecolor='white')
            return str(vector_path)
        except Exception as e:
            print(f"Warning: Could not save {vector_format} format: {e}")
            # Fallback to PNG
            try:
                png_path = output_path / f"{filename}.png"
                fig.savefig(str(png_path), dpi=dpi, bbox_inches='tight', facecolor='white')
                return str(png_path)
            except Exception as png_e:
                print(f"Warning: PNG fallback also failed: {png_e}")
                return None
    
    def create_comparison_colors(values, threshold=0.0):
        """Fallback function if formal_plot_style is not available."""
        return ['#2ca02c' if v > threshold else '#d62728' if v < threshold else '#7f7f7f' for v in values]

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


class ProgressiveContextTester:
    """Test memory mechanism across progressively increasing context lengths."""
    
    def __init__(self, checkpoint_path: str, output_dir: str = "./progressive_context_analysis"):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.parallel_context = None
        self.config = None
        
        # Test results
        self.test_results = {}
        self.performance_trends = {}
        self.memory_scaling_data = {}
        
    def load_model_components(self):
        """Load model, tokenizer, and configuration."""
        
        print("Loading model components...")
        
        # Load configuration
        config_path = self.checkpoint_path / "config.yaml"
        self.config = get_config_from_file(config_path.as_posix())
        constants.CONFIG = self.config
        
        model_config = self.config.model.model_config
        tokenizer_path = self.config.tokenizer.tokenizer_name_or_path
        
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
        
        mark_tied_parameters(model=self.model, parallel_context=self.parallel_context, parallel_config=parallel_config)
        load_weights(model=self.model, parallel_context=self.parallel_context, root_folder=self.checkpoint_path)
        
        # Apply balance factor fix for Infini-Attention
        print("🔧 Applying balance factor fix...")
        
        # Add root directory to Python path (more robust)
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(current_dir, '..')
        root_dir = os.path.abspath(root_dir)
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        
        try:
            from apply_balance_fix_standalone import apply_balance_factor_fix_standalone
            fix_success = apply_balance_factor_fix_standalone(self.model, self.checkpoint_path, verbose=False)
            if fix_success:
                print("✅ Balance factors loaded successfully")
            else:
                print("⚠️  Balance factor fix may not have worked properly")
        except Exception as e:
            print(f"⚠️  Balance factor fix failed: {e}")
        
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
    
    def generate_progressive_contexts(self, min_length: int, max_length: int, 
                                    step_size: int, samples_per_length: int = 10) -> Dict[int, List[Dict]]:
        """Generate test contexts of progressively increasing lengths."""
        
        print(f"Generating contexts from {min_length} to {max_length} tokens (step: {step_size})...")
        
        contexts = {}
        context_lengths = list(range(min_length, max_length + 1, step_size))
        
        # Base content templates
        templates = {
            'narrative': self._create_narrative_template(),
            'factual': self._create_factual_template(), 
            'technical': self._create_technical_template(),
            'conversational': self._create_conversational_template()
        }
        
        for context_length in context_lengths:
            print(f"  Generating {samples_per_length} samples for {context_length} tokens...")
            
            length_contexts = []
            
            for sample_idx in range(samples_per_length):
                # Rotate through different templates
                template_name = list(templates.keys())[sample_idx % len(templates)]
                template = templates[template_name]
                
                # Generate context with embedded test information
                context_data = self._generate_single_context(
                    template, context_length, sample_idx
                )
                context_data['template_type'] = template_name
                length_contexts.append(context_data)
            
            contexts[context_length] = length_contexts
        
        return contexts
    
    def _create_narrative_template(self) -> Dict[str, Any]:
        """Create narrative text template."""
        
        return {
            'type': 'narrative',
            'base_sentences': [
                "The old lighthouse stood majestically on the rocky cliff, its beacon cutting through the foggy night.",
                "Sarah walked through the ancient forest, her footsteps echoing among the towering oak trees.",
                "The mysterious letter arrived on a rainy Tuesday morning, changing everything forever.",
                "In the small village, time seemed to move more slowly than in the bustling city.",
                "The grandfather clock in the hallway chimed midnight as the story began to unfold.",
                "Through the library windows, moonlight illuminated the dusty volumes on the shelves.",
                "The train journey across the countryside revealed landscapes both familiar and strange.",
                "At the edge of the lake, the old fisherman cast his line into the still water."
            ],
            'question_templates': [
                "What happened at {time}?",
                "Who was {character}?",
                "Where did {character} go?",
                "What was found in the {location}?"
            ]
        }
    
    def _create_factual_template(self) -> Dict[str, Any]:
        """Create factual/educational text template."""
        
        return {
            'type': 'factual',
            'base_sentences': [
                "The scientific method involves systematic observation, hypothesis formation, and experimental testing.",
                "Climate change affects global weather patterns through complex atmospheric interactions.",
                "Photosynthesis in plants converts carbon dioxide and water into glucose using solar energy.",
                "The human brain contains approximately 86 billion neurons connected by trillions of synapses.",
                "Ocean currents play a crucial role in regulating Earth's temperature and weather systems.",
                "Artificial intelligence algorithms learn patterns from large datasets to make predictions.",
                "Renewable energy sources include solar, wind, hydroelectric, and geothermal power.",
                "DNA contains the genetic instructions for all living organisms' growth and development."
            ],
            'question_templates': [
                "What is the definition of {concept}?",
                "How does {process} work?",
                "What are the components of {system}?",
                "What is the significance of {discovery}?"
            ]
        }
    
    def _create_technical_template(self) -> Dict[str, Any]:
        """Create technical documentation template."""
        
        return {
            'type': 'technical',
            'base_sentences': [
                "The API endpoint accepts POST requests with JSON payload containing user credentials.",
                "Database indexing improves query performance by creating sorted data structures.",
                "Network protocols ensure reliable data transmission across distributed systems.",
                "Memory allocation algorithms manage computer resources efficiently during program execution.",
                "Encryption standards protect sensitive information through mathematical transformations.",
                "Software testing frameworks automate quality assurance processes in development cycles.",
                "Version control systems track changes in source code throughout project development.",
                "Load balancing distributes incoming requests across multiple server instances."
            ],
            'question_templates': [
                "What does the {component} do?",
                "How is {process} implemented?",
                "What are the requirements for {system}?",
                "What is the purpose of {feature}?"
            ]
        }
    
    def _create_conversational_template(self) -> Dict[str, Any]:
        """Create conversational text template."""
        
        return {
            'type': 'conversational',
            'base_sentences': [
                "Alice: I think we should consider the environmental impact of our decision.",
                "Bob: That's a good point. What specific concerns do you have about the project?",
                "Carol: From my experience, these initiatives usually take longer than expected.",
                "David: We need to balance efficiency with sustainability in our approach.",
                "Emma: The budget constraints might limit our available options significantly.",
                "Frank: Perhaps we could explore alternative solutions that address both issues.",
                "Grace: I've seen similar projects succeed with proper planning and community support.",
                "Henry: The timeline seems ambitious, but it's certainly achievable with dedication."
            ],
            'question_templates': [
                "What did {speaker} say about {topic}?",
                "Who mentioned {concern}?",
                "What was discussed regarding {subject}?",
                "What opinion did {person} express?"
            ]
        }
    
    def _generate_single_context(self, template: Dict, target_length: int, seed: int) -> Dict[str, Any]:
        """Generate a single context of specified length with embedded test information."""
        
        np.random.seed(seed)
        
        # Choose test information to embed
        test_info_types = ['number', 'name', 'fact', 'location']
        test_type = np.random.choice(test_info_types)
        
        if test_type == 'number':
            test_value = str(np.random.randint(10000, 99999))
            test_sentence = f"The important code is {test_value}. Remember this number."
            question = "What is the important code?"
            expected_answer = test_value
        
        elif test_type == 'name':
            names = ['Alexander', 'Victoria', 'Benjamin', 'Isabella', 'Christopher']
            test_value = np.random.choice(names)
            test_sentence = f"The key person in this story is {test_value}. This name is crucial."
            question = f"Who is the key person mentioned?"
            expected_answer = test_value
        
        elif test_type == 'fact':
            facts = [
                ('capital of Mars colony', 'New Terra'),
                ('speed of quantum processor', '15.7 THz'),
                ('inventor of teleportation', 'Dr. Sarah Chen'),
                ('distance to Alpha Station', '47.3 light years'),
                ('formula for time dilation', 'T = T0 * sqrt(1 - v²/c²)')
            ]
            fact_name, fact_value = facts[seed % len(facts)]
            test_sentence = f"It's important to know that the {fact_name} is {fact_value}."
            question = f"What is the {fact_name}?"
            expected_answer = fact_value
        
        else:  # location
            locations = ['Neo Tokyo', 'Crystal Falls', 'Meridian City', 'Azure Heights', 'Stellar Bay']
            test_value = np.random.choice(locations)
            test_sentence = f"The secret meeting was held in {test_value}. This location is significant."
            question = "Where was the secret meeting held?"
            expected_answer = test_value
        
        # Choose position for test information (not too early or late)
        info_position = np.random.randint(target_length // 4, 3 * target_length // 4)
        
        # Build context
        sentences = template['base_sentences']
        context_parts = []
        current_length = 0
        
        # Add sentences until we reach the info position
        while current_length < info_position:
            sentence = np.random.choice(sentences)
            context_parts.append(sentence)
            current_length += len(self.tokenizer.encode(sentence))
        
        # Insert test information
        context_parts.append(test_sentence)
        current_length += len(self.tokenizer.encode(test_sentence))
        
        # Fill remaining space
        while current_length < target_length:
            sentence = np.random.choice(sentences)
            context_parts.append(sentence)
            current_length += len(self.tokenizer.encode(sentence))
            if current_length > target_length:
                break
        
        # Create final context
        context = " ".join(context_parts)
        
        # Truncate to exact length if needed
        tokens = self.tokenizer.encode(context)
        if len(tokens) > target_length:
            tokens = tokens[:target_length]
            context = self.tokenizer.decode(tokens)
        
        return {
            'context': context,
            'question': question,
            'expected_answer': expected_answer,
            'test_type': test_type,
            'info_position': info_position,
            'actual_length': len(self.tokenizer.encode(context)),
            'segments_spanned': max(1, len(self.tokenizer.encode(context)) // 1024)  # Assuming 1024 segment length
        }
    
    def run_progressive_testing(self, contexts: Dict[int, List[Dict]], 
                              test_memory_enabled: bool = True,
                              test_memory_disabled: bool = True) -> Dict[str, Any]:
        """Run progressive context length testing."""
        
        print("Running progressive context length testing...")
        
        results = {
            'test_configuration': {
                'context_lengths': list(contexts.keys()),
                'samples_per_length': len(next(iter(contexts.values()))) if contexts else 0,
                'memory_enabled_tested': test_memory_enabled,
                'memory_disabled_tested': test_memory_disabled
            },
            'results_with_memory': {} if test_memory_enabled else None,
            'results_without_memory': {} if test_memory_disabled else None,
            'performance_trends': {},
            'scaling_analysis': {}
        }
        
        # Test with memory enabled
        if test_memory_enabled:
            print("  Testing with memory enabled...")
            constants.CONFIG.infini_attention.turn_on_memory = True
            results['results_with_memory'] = self._test_all_contexts(contexts, "with_memory")
        
        # Test with memory disabled
        if test_memory_disabled:
            print("  Testing with memory disabled...")
            constants.CONFIG.infini_attention.turn_on_memory = False
            results['results_without_memory'] = self._test_all_contexts(contexts, "without_memory")
        
        # Analyze performance trends
        results['performance_trends'] = self._analyze_performance_trends(results)
        
        # Analyze scaling behavior
        results['scaling_analysis'] = self._analyze_scaling_behavior(results)
        
        return results
    
    def _test_all_contexts(self, contexts: Dict[int, List[Dict]], test_mode: str) -> Dict[int, Dict]:
        """Test all context lengths in the specified mode."""
        
        mode_results = {}
        
        for context_length, context_samples in contexts.items():
            print(f"    Testing {context_length} tokens...")
            
            length_results = {
                'context_length': context_length,
                'sample_results': [],
                'statistics': {}
            }
            
            # Test each sample
            for i, context_data in enumerate(context_samples):
                if (i + 1) % 5 == 0:
                    print(f"      Sample {i + 1}/{len(context_samples)}")
                
                sample_result = self._test_single_sample(context_data)
                length_results['sample_results'].append(sample_result)
            
            # Calculate statistics for this length
            accuracies = [r['accuracy'] for r in length_results['sample_results']]
            response_times = [r['response_time'] for r in length_results['sample_results'] if 'response_time' in r]
            
            length_results['statistics'] = {
                'accuracy_mean': float(np.mean(accuracies)),
                'accuracy_std': float(np.std(accuracies)),
                'accuracy_median': float(np.median(accuracies)),
                'total_samples': len(accuracies),
                'successful_samples': sum(accuracies),
                'success_rate': float(np.mean(accuracies))
            }
            
            if response_times:
                length_results['statistics']['response_time_mean'] = float(np.mean(response_times))
                length_results['statistics']['response_time_std'] = float(np.std(response_times))
            
            mode_results[context_length] = length_results
            print(f"      Accuracy: {length_results['statistics']['success_rate']:.1%}")
        
        return mode_results
    
    def _test_single_sample(self, context_data: Dict) -> Dict[str, Any]:
        """Test a single context sample."""
        
        start_time = time.time()
        
        # Create full prompt
        full_prompt = context_data['context'] + " " + context_data['question'] + " Answer:"
        
        try:
            # Generate response
            outputs = decode_text(
                input_iter=[GenerationInput(text=full_prompt)],
                tokenizer=self.tokenizer,
                model=self.model.model,
                parallel_context=self.parallel_context,
                max_new_tokens=50,  # Increased for longer answers
                max_micro_batch_size=1,
                generation_config=GenerationArgs(sampler="greedy", use_cache=False),
                tokenizer_config=TokenizerConfig(max_input_length=context_data['actual_length'] + 100),  # Use actual context length
            )
            
            response_time = time.time() - start_time
            
            # Analyze response
            output_list = list(outputs)  # Convert generator to list
            if output_list and len(output_list) > 0:
                generated_text = output_list[0].strip()
                
                # Check accuracy
                accuracy = self._check_accuracy(
                    generated_text, 
                    context_data['expected_answer'], 
                    context_data['test_type']
                )
                
                result = {
                    'accuracy': accuracy,
                    'generated_response': generated_text,
                    'expected_answer': context_data['expected_answer'],
                    'response_time': response_time,
                    'test_type': context_data['test_type'],
                    'info_position': context_data['info_position'],
                    'segments_spanned': context_data['segments_spanned'],
                    'success': True
                }
            else:
                result = {
                    'accuracy': 0.0,
                    'generated_response': "",
                    'expected_answer': context_data['expected_answer'],
                    'response_time': response_time,
                    'test_type': context_data['test_type'],
                    'info_position': context_data['info_position'],
                    'segments_spanned': context_data['segments_spanned'],
                    'success': False,
                    'error': "No output generated"
                }
        
        except Exception as e:
            response_time = time.time() - start_time
            result = {
                'accuracy': 0.0,
                'generated_response': "",
                'expected_answer': context_data['expected_answer'],
                'response_time': response_time,
                'test_type': context_data['test_type'],
                'info_position': context_data['info_position'],
                'segments_spanned': context_data['segments_spanned'],
                'success': False,
                'error': str(e)
            }
        
        return result
    
    def _check_accuracy(self, generated: str, expected: str, test_type: str) -> float:
        """Check accuracy of generated response."""
        
        generated_lower = generated.lower().strip()
        expected_lower = expected.lower().strip()
        
        if test_type == 'number':
            # Extract numbers from response
            import re
            numbers = re.findall(r'\b\d+\b', generated)
            return 1.0 if expected in numbers else 0.0
        
        elif test_type in ['name', 'location']:
            # Check if expected name/location appears
            return 1.0 if expected_lower in generated_lower else 0.0
        
        elif test_type == 'fact':
            # Check if expected fact value appears
            return 1.0 if expected_lower in generated_lower else 0.0
        
        else:
            # Default exact match
            return 1.0 if expected_lower == generated_lower else 0.0
    
    def _analyze_performance_trends(self, results: Dict) -> Dict[str, Any]:
        """Analyze performance trends across context lengths."""
        
        trends = {
            'with_memory': {},
            'without_memory': {},
            'comparison': {}
        }
        
        # Analyze trends with memory
        if results['results_with_memory']:
            trends['with_memory'] = self._extract_performance_trend(results['results_with_memory'])
        
        # Analyze trends without memory
        if results['results_without_memory']:
            trends['without_memory'] = self._extract_performance_trend(results['results_without_memory'])
        
        # Compare trends
        if results['results_with_memory'] and results['results_without_memory']:
            trends['comparison'] = self._compare_performance_trends(
                trends['with_memory'], trends['without_memory']
            )
        
        return trends
    
    def _extract_performance_trend(self, results: Dict) -> Dict[str, Any]:
        """Extract performance trend from results."""
        
        context_lengths = sorted(results.keys())
        accuracies = [results[length]['statistics']['success_rate'] for length in context_lengths]
        
        # Calculate trend statistics
        if len(context_lengths) > 1:
            # Linear regression to find trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(context_lengths, accuracies)
            
            trend_analysis = {
                'context_lengths': context_lengths,
                'accuracies': accuracies,
                'trend_slope': float(slope),
                'trend_intercept': float(intercept),
                'correlation_coefficient': float(r_value),
                'p_value': float(p_value),
                'trend_direction': 'improving' if slope > 0 else 'degrading' if slope < 0 else 'stable',
                'trend_strength': abs(float(r_value)),
                'performance_range': {
                    'min_accuracy': float(min(accuracies)),
                    'max_accuracy': float(max(accuracies)),
                    'accuracy_span': float(max(accuracies) - min(accuracies))
                }
            }
        else:
            trend_analysis = {
                'context_lengths': context_lengths,
                'accuracies': accuracies,
                'insufficient_data': True
            }
        
        return trend_analysis
    
    def _compare_performance_trends(self, with_memory: Dict, without_memory: Dict) -> Dict[str, Any]:
        """Compare performance trends between memory modes."""
        
        comparison = {}
        
        if 'accuracies' in with_memory and 'accuracies' in without_memory:
            mem_accuracies = with_memory['accuracies']
            no_mem_accuracies = without_memory['accuracies']
            
            # Calculate differences
            differences = [m - nm for m, nm in zip(mem_accuracies, no_mem_accuracies)]
            
            comparison = {
                'accuracy_differences': differences,
                'average_difference': float(np.mean(differences)),
                'difference_trend': float(np.polyfit(range(len(differences)), differences, 1)[0]),
                'memory_advantage': float(np.mean(differences)) > 0,
                'advantage_magnitude': abs(float(np.mean(differences))),
                'contexts_where_memory_helps': sum(1 for d in differences if d > 0),
                'total_contexts': len(differences),
                'consistent_advantage': all(d > 0 for d in differences)
            }
        
        return comparison
    
    def _analyze_scaling_behavior(self, results: Dict) -> Dict[str, Any]:
        """Analyze how memory mechanism scales with context length."""
        
        scaling = {
            'memory_scaling': {},
            'performance_plateaus': {},
            'optimal_ranges': {},
            'scaling_efficiency': {}
        }
        
        # Analyze memory scaling
        if results['results_with_memory']:
            mem_results = results['results_with_memory']
            context_lengths = sorted(mem_results.keys())
            
            # Find performance plateaus and drops
            accuracies = [mem_results[length]['statistics']['success_rate'] for length in context_lengths]
            
            # Detect significant performance drops
            drop_threshold = 0.1  # 10% drop
            drops = []
            for i in range(1, len(accuracies)):
                if accuracies[i-1] - accuracies[i] > drop_threshold:
                    drops.append({
                        'context_length': context_lengths[i],
                        'drop_amount': accuracies[i-1] - accuracies[i],
                        'previous_accuracy': accuracies[i-1],
                        'new_accuracy': accuracies[i]
                    })
            
            # Find optimal context range (highest sustained performance)
            optimal_start = 0
            optimal_end = len(accuracies) - 1
            best_avg_accuracy = 0
            
            for start in range(len(accuracies)):
                for end in range(start + 2, len(accuracies) + 1):  # At least 3 points
                    avg_accuracy = np.mean(accuracies[start:end])
                    if avg_accuracy > best_avg_accuracy:
                        best_avg_accuracy = avg_accuracy
                        optimal_start = start
                        optimal_end = end - 1
            
            scaling['memory_scaling'] = {
                'context_range': (min(context_lengths), max(context_lengths)),
                'accuracy_range': (min(accuracies), max(accuracies)),
                'performance_drops': drops,
                'scaling_trend': 'degrading' if accuracies[-1] < accuracies[0] else 'stable'
            }
            
            scaling['optimal_ranges'] = {
                'optimal_context_range': (context_lengths[optimal_start], context_lengths[optimal_end]),
                'optimal_accuracy': best_avg_accuracy,
                'recommended_max_context': context_lengths[optimal_end]
            }
        
        return scaling
    
    def create_visualizations(self, results: Dict) -> List[str]:
        """Create comprehensive visualizations of progressive testing results with formal academic styling."""
        
        viz_files = []
        
        print("  Creating formal publication-ready visualizations...")
        
        # 1. Performance vs context length
        if results['results_with_memory'] or results['results_without_memory']:
            fig = self._create_performance_vs_context_plot(results)
            files = save_plotly_figure(
                fig, self.output_dir,
                "performance_vs_context_length", "performance_vs_context_plot",
                width=1200, height=800, vector_format='pdf'
            )
            viz_files.extend(files)
        
        # 2. Scaling analysis plot
        if 'scaling_analysis' in results:
            fig = self._create_scaling_analysis_plot(results)
            files = save_plotly_figure(
                fig, self.output_dir,
                "scaling_analysis", "scaling_analysis_plot",
                width=1200, height=900, vector_format='pdf'
            )
            viz_files.extend(files)
        
        # 3. Performance trends plot
        if 'performance_trends' in results:
            fig = self._create_trends_plot(results['performance_trends'])
            files = save_plotly_figure(
                fig, self.output_dir,
                "performance_trends", "performance_trends_plot",
                width=1200, height=800, vector_format='pdf'
            )
            viz_files.extend(files)
        
        # 4. Static plots for publications
        static_files = self._create_publication_plots(results)
        viz_files.extend(static_files)
        
        print(f"  Generated {len(viz_files)} visualization files")
        
        return viz_files
    
    def _create_performance_vs_context_plot(self, results: Dict):
        """Create performance vs context length plot with formal academic styling."""
        
        fig = go.Figure()
        
        # Plot with memory if available
        if results['results_with_memory']:
            mem_results = results['results_with_memory']
            context_lengths = sorted(mem_results.keys())
            accuracies = [mem_results[length]['statistics']['success_rate'] for length in context_lengths]
            stds = [mem_results[length]['statistics']['accuracy_std'] for length in context_lengths]
            
            fig.add_trace(go.Scatter(
                x=context_lengths,
                y=accuracies,
                error_y=dict(
                    type='data', 
                    array=stds, 
                    visible=True,
                    color=ACADEMIC_COLORS['memory_enabled'],
                    thickness=2,
                    width=4
                ),
                mode='lines+markers',
                name='With Infini-Attention Memory',
                line=dict(color=ACADEMIC_COLORS['memory_enabled'], width=3),
                marker=dict(size=10, symbol='circle', line=dict(width=2, color='white'))
            ))
        
        # Plot without memory if available
        if results['results_without_memory']:
            no_mem_results = results['results_without_memory']
            context_lengths = sorted(no_mem_results.keys())
            accuracies = [no_mem_results[length]['statistics']['success_rate'] for length in context_lengths]
            stds = [no_mem_results[length]['statistics']['accuracy_std'] for length in context_lengths]
            
            fig.add_trace(go.Scatter(
                x=context_lengths,
                y=accuracies,
                error_y=dict(
                    type='data', 
                    array=stds, 
                    visible=True,
                    color=ACADEMIC_COLORS['memory_disabled'],
                    thickness=2,
                    width=4
                ),
                mode='lines+markers',
                name='Without Memory (Baseline)',
                line=dict(color=ACADEMIC_COLORS['memory_disabled'], width=3),
                marker=dict(size=10, symbol='square', line=dict(width=2, color='white'))
            ))
        
        fig.update_layout(
            title=dict(
                text='Progressive Context Length Performance Analysis',
                font=dict(size=18, family='Times New Roman'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title='Context Length (tokens)',
                title_font=dict(size=14, family='Times New Roman'),
                tickfont=dict(size=12, family='Times New Roman'),
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                showline=True,
                linewidth=2,
                linecolor='black'
            ),
            yaxis=dict(
                title='Task Accuracy',
                title_font=dict(size=14, family='Times New Roman'),
                tickfont=dict(size=12, family='Times New Roman'),
                range=[0, 1.05],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                showline=True,
                linewidth=2,
                linecolor='black'
            ),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=12, family='Times New Roman')
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=80, r=50, t=80, b=60)
        )
        
        return fig
    
    def _create_scaling_analysis_plot(self, results: Dict):
        """Create scaling analysis visualization."""
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Memory Advantage by Context Length', 'Performance Drop Detection']
        )
        
        if ('performance_trends' in results and 
            'comparison' in results['performance_trends'] and
            'accuracy_differences' in results['performance_trends']['comparison']):
            
            comparison = results['performance_trends']['comparison']
            
            if results['results_with_memory']:
                context_lengths = sorted(results['results_with_memory'].keys())
                differences = comparison['accuracy_differences']
                
                # Memory advantage plot
                colors = ['green' if d > 0 else 'red' for d in differences]
                fig.add_trace(
                    go.Bar(
                        x=context_lengths,
                        y=differences,
                        marker_color=colors,
                        name='Memory Advantage',
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                # Performance drops (if available in scaling analysis)
                if ('scaling_analysis' in results and 
                    'memory_scaling' in results['scaling_analysis'] and
                    'performance_drops' in results['scaling_analysis']['memory_scaling']):
                    
                    drops = results['scaling_analysis']['memory_scaling']['performance_drops']
                    if drops:
                        drop_contexts = [d['context_length'] for d in drops]
                        drop_amounts = [d['drop_amount'] for d in drops]
                        
                        fig.add_trace(
                            go.Bar(
                                x=drop_contexts,
                                y=drop_amounts,
                                marker_color='orange',
                                name='Performance Drops',
                                showlegend=False
                            ),
                            row=2, col=1
                        )
        
        fig.update_layout(
            title='Memory Mechanism Scaling Analysis',
            height=800,
            showlegend=False
        )
        
        fig.update_xaxes(title_text='Context Length (tokens)', row=2, col=1)
        fig.update_yaxes(title_text='Accuracy Difference', row=1, col=1)
        fig.update_yaxes(title_text='Performance Drop', row=2, col=1)
        
        return fig
    
    def _create_trends_plot(self, trends: Dict):
        """Create performance trends visualization."""
        
        fig = go.Figure()
        
        if 'with_memory' in trends and 'context_lengths' in trends['with_memory']:
            mem_trend = trends['with_memory']
            
            # Actual data points
            fig.add_trace(go.Scatter(
                x=mem_trend['context_lengths'],
                y=mem_trend['accuracies'],
                mode='markers',
                name='With Memory (Actual)',
                marker=dict(color='blue', size=10)
            ))
            
            # Trend line
            if 'trend_slope' in mem_trend:
                trend_y = [mem_trend['trend_slope'] * x + mem_trend['trend_intercept'] 
                          for x in mem_trend['context_lengths']]
                
                fig.add_trace(go.Scatter(
                    x=mem_trend['context_lengths'],
                    y=trend_y,
                    mode='lines',
                    name=f'Memory Trend (slope: {mem_trend["trend_slope"]:.4f})',
                    line=dict(color='blue', dash='dash')
                ))
        
        if 'without_memory' in trends and 'context_lengths' in trends['without_memory']:
            no_mem_trend = trends['without_memory']
            
            # Actual data points
            fig.add_trace(go.Scatter(
                x=no_mem_trend['context_lengths'],
                y=no_mem_trend['accuracies'],
                mode='markers',
                name='Without Memory (Actual)',
                marker=dict(color='red', size=10)
            ))
            
            # Trend line
            if 'trend_slope' in no_mem_trend:
                trend_y = [no_mem_trend['trend_slope'] * x + no_mem_trend['trend_intercept'] 
                          for x in no_mem_trend['context_lengths']]
                
                fig.add_trace(go.Scatter(
                    x=no_mem_trend['context_lengths'],
                    y=trend_y,
                    mode='lines',
                    name=f'No Memory Trend (slope: {no_mem_trend["trend_slope"]:.4f})',
                    line=dict(color='red', dash='dash')
                ))
        
        fig.update_layout(
            title='Performance Trends Analysis',
            xaxis_title='Context Length (tokens)',
            yaxis_title='Accuracy',
            width=900,
            height=600
        )
        
        return fig
    
    def _create_publication_plots(self, results: Dict):
        """Create static plots for publications with formal academic styling."""
        
        created_files = []
        
        # Performance vs context length
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if results['results_with_memory']:
            mem_results = results['results_with_memory']
            context_lengths = sorted(mem_results.keys())
            accuracies = [mem_results[length]['statistics']['success_rate'] for length in context_lengths]
            stds = [mem_results[length]['statistics']['accuracy_std'] for length in context_lengths]
            
            ax.errorbar(context_lengths, accuracies, yerr=stds, 
                       marker='o', linewidth=3, markersize=10, capsize=5, capthick=2,
                       label='With Infini-Attention Memory', 
                       color=ACADEMIC_COLORS['memory_enabled'])
        
        if results['results_without_memory']:
            no_mem_results = results['results_without_memory']
            context_lengths = sorted(no_mem_results.keys())
            accuracies = [no_mem_results[length]['statistics']['success_rate'] for length in context_lengths]
            stds = [no_mem_results[length]['statistics']['accuracy_std'] for length in context_lengths]
            
            ax.errorbar(context_lengths, accuracies, yerr=stds,
                       marker='s', linewidth=3, markersize=10, capsize=5, capthick=2,
                       label='Without Memory (Baseline)', 
                       color=ACADEMIC_COLORS['memory_disabled'])
        
        ax.set_xlabel('Context Length (tokens)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Task Accuracy', fontsize=14, fontweight='bold')
        ax.set_title('Progressive Context Length Performance Analysis', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linewidth=0.8)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        file_path = save_matplotlib_figure(fig, self.output_dir, "progressive_performance_analysis", 
                                         figsize=(12, 8), vector_format='pdf')
        if file_path:
            created_files.append(file_path)
        plt.close()
        
        # Memory scaling analysis
        if ('performance_trends' in results and 
            'comparison' in results['performance_trends'] and
            'accuracy_differences' in results['performance_trends']['comparison']):
            
            comparison = results['performance_trends']['comparison']
            
            if results['results_with_memory']:
                context_lengths = sorted(results['results_with_memory'].keys())
                differences = comparison['accuracy_differences']
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                colors = create_comparison_colors(differences, threshold=0.0)
                bars = ax.bar(context_lengths, differences, color=colors, alpha=0.8,
                             edgecolor='black', linewidth=1)
                
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=2)
                ax.set_xlabel('Context Length (tokens)', fontsize=14, fontweight='bold')
                ax.set_ylabel('Memory Advantage (Accuracy Difference)', fontsize=14, fontweight='bold')
                ax.set_title('Memory Mechanism Scaling Analysis', fontsize=16, fontweight='bold')
                ax.grid(True, alpha=0.3, linewidth=0.8)
                ax.tick_params(axis='both', which='major', labelsize=12)
                
                # Add value labels on bars
                for bar, diff in zip(bars, differences):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., 
                           height + (0.01 if height >= 0 else -0.01),
                           f'{diff:.2f}', ha='center', 
                           va='bottom' if height >= 0 else 'top', 
                           fontsize=11, fontweight='bold')
                
                plt.tight_layout()
                file_path = save_matplotlib_figure(fig, self.output_dir, "memory_scaling_analysis", 
                                                 figsize=(12, 6), vector_format='pdf')
                if file_path:
                    created_files.append(file_path)
                plt.close()
        
        return created_files
    
    def generate_comprehensive_report(self, results: Dict) -> Dict[str, Any]:
        """Generate comprehensive progressive testing report."""
        
        # Extract key metrics
        summary_metrics = self._extract_summary_metrics(results)
        
        # Create visualizations
        viz_files = self.create_visualizations(results)
        
        # Generate insights
        insights = self._generate_insights(results, summary_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results, insights)
        
        report = {
            'experiment_configuration': results['test_configuration'],
            'summary_metrics': summary_metrics,
            'detailed_results': results,
            'visualizations': viz_files,
            'insights': insights,
            'recommendations': recommendations,
            'conclusion': self._draw_conclusion(summary_metrics, insights)
        }
        
        return report
    
    def _extract_summary_metrics(self, results: Dict) -> Dict[str, Any]:
        """Extract key summary metrics from results."""
        
        metrics = {}
        
        if results['results_with_memory']:
            mem_results = results['results_with_memory']
            context_lengths = sorted(mem_results.keys())
            accuracies = [mem_results[length]['statistics']['success_rate'] for length in context_lengths]
            
            metrics['with_memory'] = {
                'context_range': (min(context_lengths), max(context_lengths)),
                'accuracy_range': (min(accuracies), max(accuracies)),
                'average_accuracy': np.mean(accuracies),
                'accuracy_decline': accuracies[0] - accuracies[-1],
                'performance_stability': 1.0 - (np.std(accuracies) / max(np.mean(accuracies), 0.001))
            }
        
        if results['results_without_memory']:
            no_mem_results = results['results_without_memory']
            context_lengths = sorted(no_mem_results.keys())
            accuracies = [no_mem_results[length]['statistics']['success_rate'] for length in context_lengths]
            
            metrics['without_memory'] = {
                'context_range': (min(context_lengths), max(context_lengths)),
                'accuracy_range': (min(accuracies), max(accuracies)),
                'average_accuracy': np.mean(accuracies),
                'accuracy_decline': accuracies[0] - accuracies[-1],
                'performance_stability': 1.0 - (np.std(accuracies) / max(np.mean(accuracies), 0.001))
            }
        
        # Comparative metrics
        if 'with_memory' in metrics and 'without_memory' in metrics:
            metrics['comparison'] = {
                'average_advantage': metrics['with_memory']['average_accuracy'] - metrics['without_memory']['average_accuracy'],
                'stability_advantage': metrics['with_memory']['performance_stability'] - metrics['without_memory']['performance_stability'],
                'decline_difference': metrics['without_memory']['accuracy_decline'] - metrics['with_memory']['accuracy_decline']
            }
        
        return metrics
    
    def _generate_insights(self, results: Dict, metrics: Dict) -> List[str]:
        """Generate key insights from the analysis."""
        
        insights = []
        
        # Memory effectiveness insights
        if 'comparison' in metrics:
            avg_advantage = metrics['comparison']['average_advantage']
            if avg_advantage > 0.2:
                insights.append("Memory mechanism provides substantial performance benefits across context lengths")
            elif avg_advantage > 0.1:
                insights.append("Memory mechanism provides moderate performance benefits")
            elif avg_advantage > 0.05:
                insights.append("Memory mechanism provides slight performance benefits")
            else:
                insights.append("Memory mechanism shows minimal or no performance benefit")
        
        # Scaling behavior insights
        if 'with_memory' in metrics:
            decline = metrics['with_memory']['accuracy_decline']
            if decline < 0.1:
                insights.append("Memory mechanism maintains performance well across increasing context lengths")
            elif decline < 0.2:
                insights.append("Memory mechanism shows moderate performance decline with context length")
            else:
                insights.append("Memory mechanism shows significant performance decline at longer contexts")
        
        # Stability insights
        if 'comparison' in metrics:
            stability_adv = metrics['comparison']['stability_advantage']
            if stability_adv > 0.1:
                insights.append("Memory provides more stable performance across different context lengths")
            elif stability_adv < -0.1:
                insights.append("Memory actually reduces performance stability")
        
        # Trend insights
        if ('performance_trends' in results and 
            'comparison' in results['performance_trends'] and
            'consistent_advantage' in results['performance_trends']['comparison']):
            
            consistent = results['performance_trends']['comparison']['consistent_advantage']
            if consistent:
                insights.append("Memory advantage is consistent across all tested context lengths")
            else:
                insights.append("Memory advantage varies across different context lengths")
        
        return insights
    
    def _generate_recommendations(self, results: Dict, insights: List[str]) -> List[str]:
        """Generate recommendations based on analysis."""
        
        recommendations = []
        
        # Based on memory effectiveness
        if any('substantial' in insight for insight in insights):
            recommendations.append("Memory mechanism is working well - consider testing even longer contexts")
        elif any('minimal or no' in insight for insight in insights):
            recommendations.append("Memory mechanism needs improvement - check training configuration")
            recommendations.append("Consider increasing balance_factor_lr or retraining with memory-focused objectives")
        
        # Based on scaling behavior
        if any('significant performance decline' in insight for insight in insights):
            recommendations.append("Memory mechanism struggles at longer contexts - investigate memory capacity limits")
            recommendations.append("Consider architectural modifications to improve long-context scaling")
        elif any('maintains performance well' in insight for insight in insights):
            recommendations.append("Memory scaling is good - can confidently use at tested context lengths")
        
        # Based on consistency
        if any('varies across different' in insight for insight in insights):
            recommendations.append("Investigate why memory effectiveness varies - may indicate training instability")
        
        # Based on optimal ranges (if available)
        if ('scaling_analysis' in results and 
            'optimal_ranges' in results['scaling_analysis'] and
            'recommended_max_context' in results['scaling_analysis']['optimal_ranges']):
            
            max_context = results['scaling_analysis']['optimal_ranges']['recommended_max_context']
            recommendations.append(f"Recommended maximum context length: {max_context} tokens for optimal performance")
        
        return recommendations
    
    def _draw_conclusion(self, metrics: Dict, insights: List[str]) -> Dict[str, Any]:
        """Draw overall conclusion from progressive testing."""
        
        conclusion = {
            'memory_effectiveness': 'UNKNOWN',
            'scaling_behavior': 'UNKNOWN',
            'recommended_action': 'Further investigation needed',
            'confidence_level': 'LOW'
        }
        
        # Determine memory effectiveness
        if 'comparison' in metrics:
            avg_advantage = metrics['comparison']['average_advantage']
            if avg_advantage > 0.15:
                conclusion['memory_effectiveness'] = 'HIGHLY_EFFECTIVE'
                conclusion['confidence_level'] = 'HIGH'
            elif avg_advantage > 0.05:
                conclusion['memory_effectiveness'] = 'MODERATELY_EFFECTIVE'
                conclusion['confidence_level'] = 'MEDIUM'
            else:
                conclusion['memory_effectiveness'] = 'INEFFECTIVE'
                conclusion['confidence_level'] = 'MEDIUM'
        
        # Determine scaling behavior
        if 'with_memory' in metrics:
            decline = metrics['with_memory']['accuracy_decline']
            stability = metrics['with_memory']['performance_stability']
            
            if decline < 0.1 and stability > 0.8:
                conclusion['scaling_behavior'] = 'EXCELLENT'
            elif decline < 0.2 and stability > 0.6:
                conclusion['scaling_behavior'] = 'GOOD'
            else:
                conclusion['scaling_behavior'] = 'POOR'
        
        # Recommended action
        if conclusion['memory_effectiveness'] == 'HIGHLY_EFFECTIVE':
            conclusion['recommended_action'] = 'Memory mechanism is working well - proceed with deployment'
        elif conclusion['memory_effectiveness'] == 'MODERATELY_EFFECTIVE':
            conclusion['recommended_action'] = 'Memory shows promise - consider optimization or further training'
        else:
            conclusion['recommended_action'] = 'Memory mechanism needs significant improvement or debugging'
        
        return conclusion


def main():
    parser = argparse.ArgumentParser(description="Progressive Context Length Testing for Infini-Attention")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--min-context", type=int, default=1024,
                       help="Minimum context length to test")
    parser.add_argument("--max-context", type=int, default=8192,
                       help="Maximum context length to test")
    parser.add_argument("--step-size", type=int, default=1024,
                       help="Step size for context length progression")
    parser.add_argument("--samples-per-length", type=int, default=10,
                       help="Number of samples per context length")
    parser.add_argument("--test-memory-enabled", action="store_true", default=True,
                       help="Test with memory enabled")
    parser.add_argument("--test-memory-disabled", action="store_true", default=True,
                       help="Test with memory disabled")
    parser.add_argument("--output-dir", type=str, default="./progressive_context_analysis",
                       help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    print("Progressive Context Length Tester for Infini-Attention")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Context range: {args.min_context} - {args.max_context} tokens (step: {args.step_size})")
    print(f"Samples per length: {args.samples_per_length}")
    print(f"Output directory: {args.output_dir}")
    
    # Initialize tester
    tester = ProgressiveContextTester(args.checkpoint, args.output_dir)
    
    # Load model components
    tester.load_model_components()
    
    # Generate progressive contexts
    contexts = tester.generate_progressive_contexts(
        args.min_context, args.max_context, args.step_size, args.samples_per_length
    )
    
    # Run progressive testing
    results = tester.run_progressive_testing(
        contexts, 
        test_memory_enabled=args.test_memory_enabled,
        test_memory_disabled=args.test_memory_disabled
    )
    
    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    report = tester.generate_comprehensive_report(results)
    
    # Save report
    report_path = Path(args.output_dir) / "progressive_context_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROGRESSIVE CONTEXT TESTING SUMMARY")
    print("=" * 60)
    
    conclusion = report['conclusion']
    print(f"Memory Effectiveness: {conclusion['memory_effectiveness']}")
    print(f"Scaling Behavior: {conclusion['scaling_behavior']}")
    print(f"Confidence Level: {conclusion['confidence_level']}")
    print(f"Recommended Action: {conclusion['recommended_action']}")
    
    if report['insights']:
        print("\nKey Insights:")
        for insight in report['insights']:
            print(f"  • {insight}")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
    
    print(f"\nDetailed report saved to: {report_path}")
    
    return report


if __name__ == "__main__":
    main()