#!/usr/bin/env python3
"""
Memory Content Analyzer for Infini-Attention

This script analyzes what information is actually stored and retrieved from
the infini-attention memory mechanism, providing insights into the semantic
quality and utility of the compressed memory.

Usage:
    python scripts/memory_content_analysis.py --checkpoint ./checkpoints/model/30000
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import copy
import re

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        'primary_blue': '#1f77b4',
        'primary_red': '#d62728',
        'primary_green': '#2ca02c',
        'primary_orange': '#ff7f0e',
        'primary_purple': '#9467bd',
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


class MemoryStateCapture:
    """Capture and analyze memory states during inference."""
    
    def __init__(self):
        self.memory_states = []  # List of memory tensors
        self.normalization_states = []  # List of normalization tensors
        self.key_value_pairs = []  # List of (key, value) pairs that created memory
        self.timestamps = []  # When each memory was created
        self.layer_indices = []  # Which layer created each memory
        self.token_positions = []  # Token position when memory was created
        
    def capture_memory_update(self, layer_idx: int, token_pos: int, 
                            memory_tensor: torch.Tensor, 
                            normalization_tensor: torch.Tensor,
                            key_states: torch.Tensor, 
                            value_states: torch.Tensor):
        """Capture memory state during update."""
        
        self.memory_states.append(memory_tensor.detach().cpu().clone())
        self.normalization_states.append(normalization_tensor.detach().cpu().clone())
        self.key_value_pairs.append((key_states.detach().cpu().clone(), 
                                   value_states.detach().cpu().clone()))
        self.timestamps.append(time.time())
        self.layer_indices.append(layer_idx)
        self.token_positions.append(token_pos)


class MemoryContentAnalyzer:
    """Analyze the content and quality of infini-attention memory."""
    
    def __init__(self, checkpoint_path: str, output_dir: str = "./memory_content_analysis"):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.parallel_context = None
        self.config = None
        
        # Analysis results
        self.memory_captures = {}  # Dict[experiment_name] -> MemoryStateCapture
        self.content_analysis = {}
        self.semantic_analysis = {}
        
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
    
    def setup_memory_hooks(self, capture: MemoryStateCapture):
        """Setup hooks to capture memory states."""
        
        token_counter = [0]  # Mutable counter for token position
        
        def create_memory_hook(layer_idx):
            original_update = self.model.model.decoder[layer_idx].attn._update_memory
            
            def hooked_update(prev_memory, prev_normalization, key_states, value_states):
                # Call original function
                result = original_update(prev_memory, prev_normalization, key_states, value_states)
                
                # Capture the result
                if isinstance(result, tuple) and len(result) == 2:
                    memory, normalization = result
                    capture.capture_memory_update(
                        layer_idx=layer_idx,
                        token_pos=token_counter[0],
                        memory_tensor=memory,
                        normalization_tensor=normalization,
                        key_states=key_states,
                        value_states=value_states
                    )
                    token_counter[0] += key_states.shape[-2]  # Update token counter
                
                return result
            
            return hooked_update
        
        # Hook all layers
        for layer_idx, layer in enumerate(self.model.model.decoder):
            if hasattr(layer, 'attn'):
                layer.attn._update_memory = create_memory_hook(layer_idx)
    
    def create_information_retention_test(self, context_length: int = 4096) -> List[Dict[str, Any]]:
        """Create test cases for information retention analysis."""
        
        test_cases = []
        
        # Test case 1: Named entity recall
        entities = ["Alice", "Bob", "Charlie", "Diana", "Edward"]
        professions = ["doctor", "engineer", "teacher", "artist", "scientist"]
        locations = ["New York", "London", "Tokyo", "Paris", "Sydney"]
        
        for i, (name, profession, location) in enumerate(zip(entities, professions, locations)):
            # Place information at different depths
            info_position = i * (context_length // 5)
            
            test_case = {
                'type': 'named_entity_recall',
                'target_info': f"{name} is a {profession} living in {location}",
                'question': f"What is {name}'s profession?",
                'expected_answer': profession,
                'info_position': info_position,
                'entity_name': name
            }
            test_cases.append(test_case)
        
        # Test case 2: Numerical recall
        numbers = [1423, 5769, 8234, 9876, 3456]
        for i, number in enumerate(numbers):
            info_position = i * (context_length // 5) + 50
            
            test_case = {
                'type': 'numerical_recall',
                'target_info': f"The secret code is {number}.",
                'question': f"What is the secret code mentioned earlier?",
                'expected_answer': str(number),
                'info_position': info_position,
                'number': number
            }
            test_cases.append(test_case)
        
        # Test case 3: Fact recall
        facts = [
            ("The capital of Australia", "Canberra"),
            ("The largest planet", "Jupiter"), 
            ("The inventor of the telephone", "Alexander Graham Bell"),
            ("The chemical symbol for gold", "Au"),
            ("The speed of light", "299,792,458 m/s")
        ]
        
        for i, (fact_setup, answer) in enumerate(facts):
            info_position = i * (context_length // 5) + 100
            
            test_case = {
                'type': 'fact_recall',
                'target_info': f"{fact_setup} is {answer}.",
                'question': f"{fact_setup}?",
                'expected_answer': answer,
                'info_position': info_position,
                'fact_category': fact_setup
            }
            test_cases.append(test_case)
        
        return test_cases
    
    def create_test_document(self, test_cases: List[Dict], context_length: int) -> Tuple[str, Dict]:
        """Create a test document with embedded information."""
        
        # Base filler text
        filler_sentences = [
            "The weather today is quite pleasant with clear skies.",
            "Technology continues to advance at a rapid pace.",
            "Scientists are making remarkable discoveries every day.",
            "Education plays a crucial role in societal development.",
            "Art and culture enrich our understanding of humanity.",
            "Environmental conservation is essential for future generations.",
            "Communication skills are vital in the modern workplace.",
            "Innovation drives progress across all industries.",
            "Collaboration between teams leads to better outcomes.",
            "Research and development fuel technological advancement."
        ]
        
        # Create document with embedded test information
        document_parts = []
        info_map = {}  # Maps position to test case info
        
        current_length = 0
        
        for test_case in test_cases:
            target_pos = test_case['info_position']
            
            # Add filler text until target position
            while current_length < target_pos and current_length < context_length:
                sentence = np.random.choice(filler_sentences)
                document_parts.append(sentence)
                current_length += len(self.tokenizer.encode(sentence))
            
            # Insert target information
            if current_length < context_length:
                document_parts.append(test_case['target_info'])
                info_map[current_length] = test_case
                current_length += len(self.tokenizer.encode(test_case['target_info']))
        
        # Fill remaining space
        while current_length < context_length:
            sentence = np.random.choice(filler_sentences)
            document_parts.append(sentence)
            current_length += len(self.tokenizer.encode(sentence))
            if current_length > context_length:
                break
        
        document = " ".join(document_parts)
        
        # Truncate to exact length if needed
        tokens = self.tokenizer.encode(document)
        if len(tokens) > context_length:
            tokens = tokens[:context_length]
            document = self.tokenizer.decode(tokens)
        
        return document, info_map
    
    def run_information_retention_experiment(self, context_length: int = 4096) -> Dict[str, Any]:
        """Run information retention experiment."""
        
        print(f"Running information retention experiment (context length: {context_length})...")
        
        # Create test cases
        test_cases = self.create_information_retention_test(context_length)
        
        # Create test document
        document, info_map = self.create_test_document(test_cases, context_length)
        
        # Setup memory capture
        capture = MemoryStateCapture()
        self.setup_memory_hooks(capture)
        
        results = {
            'experiment_config': {
                'context_length': context_length,
                'num_test_cases': len(test_cases),
                'test_types': list(set(tc['type'] for tc in test_cases))
            },
            'test_results': [],
            'memory_analysis': {},
            'information_map': info_map
        }
        
        # Process document through model to capture memory states
        print("  Processing document to capture memory states...")
        try:
            outputs = decode_text(
                input_iter=[GenerationInput(text=document)],
                tokenizer=self.tokenizer,
                model=self.model.model,
                parallel_context=self.parallel_context,
                max_new_tokens=1,  # Just need to process the context
                max_micro_batch_size=1,
                generation_config=GenerationArgs(sampler="greedy", use_cache=False),
                tokenizer_config=TokenizerConfig(max_input_length=context_length + 100),
            )
        except Exception as e:
            print(f"    Error processing document: {e}")
            return results
        
        # Store memory captures for this experiment
        experiment_name = f"retention_test_{context_length}"
        self.memory_captures[experiment_name] = capture
        
        # Test information retrieval for each test case
        print("  Testing information retrieval...")
        for i, test_case in enumerate(test_cases):
            print(f"    Testing case {i+1}/{len(test_cases)}: {test_case['type']}")
            
            # Create retrieval prompt
            retrieval_prompt = document + " " + test_case['question'] + " Answer:"
            
            try:
                # Generate answer
                outputs = decode_text(
                    input_iter=[GenerationInput(text=retrieval_prompt)],
                    tokenizer=self.tokenizer,
                    model=self.model.model,
                    parallel_context=self.parallel_context,
                    max_new_tokens=50,  # Increased for longer answers
                    max_micro_batch_size=1,
                    generation_config=GenerationArgs(sampler="greedy", use_cache=False),
                    tokenizer_config=TokenizerConfig(max_input_length=context_length + 100),  # Use context length limit
                )
                
                # Analyze answer
                output_list = list(outputs)  # Convert generator to list
                if output_list and len(output_list) > 0:
                    generated_answer = output_list[0].strip()
                    
                    # Check if answer is correct
                    is_correct = self._check_answer_correctness(
                        generated_answer, test_case['expected_answer'], test_case['type']
                    )
                    
                    test_result = {
                        'test_case_index': i,
                        'type': test_case['type'],
                        'question': test_case['question'],
                        'expected_answer': test_case['expected_answer'],
                        'generated_answer': generated_answer,
                        'is_correct': is_correct,
                        'info_position': test_case['info_position'],
                        'segments_crossed': test_case['info_position'] // 1024,  # Assuming 1024 segment length
                    }
                else:
                    test_result = {
                        'test_case_index': i,
                        'type': test_case['type'],
                        'question': test_case['question'],
                        'expected_answer': test_case['expected_answer'],
                        'generated_answer': "",
                        'is_correct': False,
                        'info_position': test_case['info_position'],
                        'segments_crossed': test_case['info_position'] // 1024,
                        'error': "No output generated"
                    }
                
            except Exception as e:
                test_result = {
                    'test_case_index': i,
                    'type': test_case['type'],
                    'question': test_case['question'],
                    'expected_answer': test_case['expected_answer'],
                    'generated_answer': "",
                    'is_correct': False,
                    'info_position': test_case['info_position'],
                    'segments_crossed': test_case['info_position'] // 1024,
                    'error': str(e)
                }
            
            results['test_results'].append(test_result)
        
        # Analyze memory content
        results['memory_analysis'] = self._analyze_memory_content(capture, info_map)
        
        return results
    
    def _check_answer_correctness(self, generated: str, expected: str, test_type: str) -> bool:
        """Check if generated answer is correct."""
        
        generated_lower = generated.lower().strip()
        expected_lower = expected.lower().strip()
        
        if test_type == 'numerical_recall':
            # Extract numbers from generated text
            numbers = re.findall(r'\b\d+\b', generated)
            return expected in numbers
        
        elif test_type in ['named_entity_recall', 'fact_recall']:
            # Check if expected answer appears in generated text
            return expected_lower in generated_lower
        
        else:
            # Default: exact match
            return generated_lower == expected_lower
    
    def _analyze_memory_content(self, capture: MemoryStateCapture, info_map: Dict) -> Dict[str, Any]:
        """Analyze the content stored in memory."""
        
        if not capture.memory_states:
            return {'error': 'No memory states captured'}
        
        analysis = {
            'memory_evolution': [],
            'information_compression': {},
            'semantic_similarity': {},
            'memory_statistics': {}
        }
        
        # Analyze memory evolution over time
        for i, (memory_state, token_pos, layer_idx) in enumerate(
            zip(capture.memory_states, capture.token_positions, capture.layer_indices)
        ):
            memory_norm = memory_state.norm().item()
            memory_entropy = self._calculate_tensor_entropy(memory_state)
            
            evolution_point = {
                'step': i,
                'token_position': token_pos,
                'layer_index': layer_idx,
                'memory_norm': memory_norm,
                'memory_entropy': memory_entropy,
                'memory_shape': list(memory_state.shape),
                'contains_target_info': token_pos in info_map
            }
            
            analysis['memory_evolution'].append(evolution_point)
        
        # Analyze information compression
        if len(capture.key_value_pairs) > 1:
            analysis['information_compression'] = self._analyze_information_compression(capture)
        
        # Analyze semantic similarity between memory states
        if len(capture.memory_states) > 1:
            analysis['semantic_similarity'] = self._analyze_semantic_similarity(capture)
        
        # Overall statistics
        memory_norms = [m.norm().item() for m in capture.memory_states]
        analysis['memory_statistics'] = {
            'total_memory_updates': len(capture.memory_states),
            'average_memory_norm': float(np.mean(memory_norms)),
            'memory_norm_std': float(np.std(memory_norms)),
            'memory_growth_rate': float((memory_norms[-1] - memory_norms[0]) / len(memory_norms)) if len(memory_norms) > 1 else 0,
            'layers_with_memory': list(set(capture.layer_indices)),
            'token_span': (min(capture.token_positions), max(capture.token_positions)) if capture.token_positions else (0, 0)
        }
        
        return analysis
    
    def _calculate_tensor_entropy(self, tensor: torch.Tensor) -> float:
        """Calculate entropy of tensor values."""
        
        # Flatten tensor and convert to numpy
        values = tensor.flatten().numpy()
        
        # Create histogram
        hist, _ = np.histogram(values, bins=50)
        
        # Calculate probabilities
        probs = hist / hist.sum()
        probs = probs[probs > 0]  # Remove zero probabilities
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs))
        
        return float(entropy)
    
    def _analyze_information_compression(self, capture: MemoryStateCapture) -> Dict[str, Any]:
        """Analyze how information is compressed in memory."""
        
        compression_analysis = {
            'compression_ratios': [],
            'key_value_evolution': [],
            'compression_efficiency': {}
        }
        
        # Calculate compression ratios
        for i, (keys, values) in enumerate(capture.key_value_pairs):
            key_size = keys.numel()
            value_size = values.numel()
            memory_size = capture.memory_states[i].numel()
            
            compression_ratio = (key_size + value_size) / memory_size if memory_size > 0 else 0
            
            compression_analysis['compression_ratios'].append({
                'step': i,
                'original_size': key_size + value_size,
                'compressed_size': memory_size,
                'compression_ratio': compression_ratio
            })
        
        # Overall compression efficiency
        if compression_analysis['compression_ratios']:
            ratios = [cr['compression_ratio'] for cr in compression_analysis['compression_ratios']]
            compression_analysis['compression_efficiency'] = {
                'average_ratio': float(np.mean(ratios)),
                'compression_trend': 'improving' if ratios[-1] > ratios[0] else 'degrading',
                'best_compression': float(max(ratios)),
                'worst_compression': float(min(ratios))
            }
        
        return compression_analysis
    
    def _analyze_semantic_similarity(self, capture: MemoryStateCapture) -> Dict[str, Any]:
        """Analyze semantic similarity between memory states."""
        
        # Convert memory states to vectors for similarity analysis
        memory_vectors = []
        for memory_state in capture.memory_states:
            # Flatten and normalize memory tensor
            vector = memory_state.flatten().numpy()
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            memory_vectors.append(vector)
        
        similarity_analysis = {
            'pairwise_similarities': [],
            'temporal_similarity': {},
            'clustering_analysis': {}
        }
        
        # Calculate pairwise similarities
        n_states = len(memory_vectors)
        similarity_matrix = np.zeros((n_states, n_states))
        
        for i in range(n_states):
            for j in range(i+1, n_states):
                similarity = 1 - cosine(memory_vectors[i], memory_vectors[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
                
                similarity_analysis['pairwise_similarities'].append({
                    'state_i': i,
                    'state_j': j,
                    'similarity': float(similarity),
                    'token_distance': abs(capture.token_positions[i] - capture.token_positions[j])
                })
        
        # Analyze temporal similarity patterns
        if n_states > 2:
            consecutive_similarities = []
            for i in range(n_states - 1):
                sim = similarity_matrix[i, i+1]
                consecutive_similarities.append(sim)
            
            similarity_analysis['temporal_similarity'] = {
                'average_consecutive_similarity': float(np.mean(consecutive_similarities)),
                'similarity_trend': 'increasing' if consecutive_similarities[-1] > consecutive_similarities[0] else 'decreasing',
                'similarity_stability': float(np.std(consecutive_similarities))
            }
        
        return similarity_analysis
    
    def create_visualizations(self, results: Dict[str, Any]) -> List[str]:
        """Create comprehensive visualizations of memory content analysis with formal academic styling."""
        
        viz_files = []
        
        print("  Creating formal publication-ready visualizations...")
        
        # 1. Information retention performance
        if 'test_results' in results and results['test_results']:
            fig = self._create_retention_performance_plot(results['test_results'])
            files = save_plotly_figure(
                fig, self.output_dir,
                "retention_performance", "retention_performance_plot",
                width=1200, height=700, vector_format='pdf'
            )
            viz_files.extend(files)
        
        # 2. Memory evolution plot
        if 'memory_analysis' in results and 'memory_evolution' in results['memory_analysis']:
            fig = self._create_memory_evolution_plot(results['memory_analysis']['memory_evolution'])
            files = save_plotly_figure(
                fig, self.output_dir,
                "memory_evolution", "memory_evolution_plot",
                width=1200, height=800, vector_format='pdf'
            )
            viz_files.extend(files)
        
        # 3. Compression analysis
        if ('memory_analysis' in results and 
            'information_compression' in results['memory_analysis'] and
            'compression_ratios' in results['memory_analysis']['information_compression']):
            fig = self._create_compression_plot(results['memory_analysis']['information_compression'])
            files = save_plotly_figure(
                fig, self.output_dir,
                "compression_analysis", "compression_analysis_plot",
                width=1200, height=700, vector_format='pdf'
            )
            viz_files.extend(files)
        
        # 4. Static plots for publications
        static_files = self._create_publication_plots(results)
        viz_files.extend(static_files)
        
        print(f"  Generated {len(viz_files)} visualization files")
        
        return viz_files
    
    def _create_retention_performance_plot(self, test_results: List[Dict]):
        """Create information retention performance plot with formal academic styling."""
        
        # Group by test type
        types = {}
        for result in test_results:
            test_type = result['type']
            if test_type not in types:
                types[test_type] = {'positions': [], 'accuracies': [], 'segments': []}
            
            types[test_type]['positions'].append(result['info_position'])
            types[test_type]['accuracies'].append(1.0 if result['is_correct'] else 0.0)
            types[test_type]['segments'].append(result['segments_crossed'])
        
        # Create subplot with formal styling
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                '(a) Accuracy vs Information Position',
                '(b) Accuracy vs Memory Segments'
            ],
            horizontal_spacing=0.12
        )
        
        # Use academic color palette
        academic_colors = [
            ACADEMIC_COLORS['primary_blue'],
            ACADEMIC_COLORS['primary_red'],
            ACADEMIC_COLORS['primary_green'],
            ACADEMIC_COLORS['primary_orange'],
            ACADEMIC_COLORS['primary_purple']
        ]
        
        # Define marker symbols for different test types
        marker_symbols = ['circle', 'square', 'diamond', 'triangle-up', 'star']
        
        for i, (test_type, data) in enumerate(types.items()):
            color = academic_colors[i % len(academic_colors)]
            symbol = marker_symbols[i % len(marker_symbols)]
            formatted_name = test_type.replace('_', ' ').title()
            
            # Position plot
            fig.add_trace(
                go.Scatter(
                    x=data['positions'],
                    y=data['accuracies'],
                    mode='markers',
                    name=formatted_name,
                    marker=dict(
                        color=color, 
                        size=12, 
                        symbol=symbol,
                        line=dict(width=2, color='white')
                    ),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Segments plot
            fig.add_trace(
                go.Scatter(
                    x=data['segments'],
                    y=data['accuracies'],
                    mode='markers',
                    name=formatted_name,
                    marker=dict(
                        color=color, 
                        size=12, 
                        symbol=symbol,
                        line=dict(width=2, color='white')
                    ),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Update layout with formal academic styling
        fig.update_layout(
            title=dict(
                text='Information Retention Performance Analysis',
                font=dict(size=18, family='Times New Roman'),
                x=0.5,
                xanchor='center'
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
        
        # Update axes with formal styling
        fig.update_xaxes(
            title_text='Token Position in Context',
            title_font=dict(size=14, family='Times New Roman'),
            tickfont=dict(size=12, family='Times New Roman'),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=2,
            linecolor='black',
            row=1, col=1
        )
        fig.update_xaxes(
            title_text='Memory Segments Traversed',
            title_font=dict(size=14, family='Times New Roman'),
            tickfont=dict(size=12, family='Times New Roman'),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=2,
            linecolor='black',
            row=1, col=2
        )
        fig.update_yaxes(
            title_text='Retrieval Accuracy',
            title_font=dict(size=14, family='Times New Roman'),
            tickfont=dict(size=12, family='Times New Roman'),
            range=[-0.05, 1.05],
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=2,
            linecolor='black',
            row=1, col=1
        )
        fig.update_yaxes(
            title_text='Retrieval Accuracy',
            title_font=dict(size=14, family='Times New Roman'),
            tickfont=dict(size=12, family='Times New Roman'),
            range=[-0.05, 1.05],
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=2,
            linecolor='black',
            row=1, col=2
        )
        
        return fig
    
    def _create_memory_evolution_plot(self, evolution_data: List[Dict]):
        """Create memory evolution plot with formal academic styling."""
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[
                '(a) Memory Norm Evolution',
                '(b) Memory Entropy Evolution'
            ],
            vertical_spacing=0.12
        )
        
        steps = [e['step'] for e in evolution_data]
        norms = [e['memory_norm'] for e in evolution_data]
        entropies = [e['memory_entropy'] for e in evolution_data]
        positions = [e['token_position'] for e in evolution_data]
        
        # Memory norm with formal styling
        fig.add_trace(
            go.Scatter(
                x=positions,
                y=norms,
                mode='lines+markers',
                name='Memory Norm',
                line=dict(color=ACADEMIC_COLORS['primary_blue'], width=3),
                marker=dict(size=8, symbol='circle', line=dict(width=2, color='white'))
            ),
            row=1, col=1
        )
        
        # Memory entropy with formal styling
        fig.add_trace(
            go.Scatter(
                x=positions,
                y=entropies,
                mode='lines+markers',
                name='Memory Entropy',
                line=dict(color=ACADEMIC_COLORS['primary_red'], width=3),
                marker=dict(size=8, symbol='square', line=dict(width=2, color='white')),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Update layout with formal academic styling
        fig.update_layout(
            title=dict(
                text='Memory Content Evolution Over Processing',
                font=dict(size=18, family='Times New Roman'),
                x=0.5,
                xanchor='center'
            ),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=80, r=50, t=80, b=60)
        )
        
        # Update axes with formal styling
        fig.update_xaxes(
            title_text='Token Position in Context',
            title_font=dict(size=14, family='Times New Roman'),
            tickfont=dict(size=12, family='Times New Roman'),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=2,
            linecolor='black',
            row=2, col=1
        )
        fig.update_yaxes(
            title_text='Memory Norm (L2)',
            title_font=dict(size=14, family='Times New Roman'),
            tickfont=dict(size=12, family='Times New Roman'),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=2,
            linecolor='black',
            row=1, col=1
        )
        fig.update_yaxes(
            title_text='Memory Entropy (bits)',
            title_font=dict(size=14, family='Times New Roman'),
            tickfont=dict(size=12, family='Times New Roman'),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=2,
            linecolor='black',
            row=2, col=1
        )
        
        return fig
    
    def _create_compression_plot(self, compression_data: Dict):
        """Create compression analysis plot with formal academic styling."""
        
        ratios_data = compression_data['compression_ratios']
        steps = [r['step'] for r in ratios_data]
        ratios = [r['compression_ratio'] for r in ratios_data]
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=ratios,
                mode='lines+markers',
                name='Compression Ratio',
                line=dict(color=ACADEMIC_COLORS['primary_green'], width=3),
                marker=dict(
                    size=10, 
                    symbol='circle',
                    line=dict(width=2, color='white')
                )
            )
        )
        
        # Add optimal compression reference line if needed
        if ratios:
            optimal_ratio = max(ratios)
            fig.add_hline(
                y=optimal_ratio,
                line_dash="dash",
                line_color=ACADEMIC_COLORS['neutral'],
                line_width=2,
                annotation_text=f"Peak Efficiency: {optimal_ratio:.2f}",
                annotation_position="top right",
                annotation_font=dict(size=12, family='Times New Roman')
            )
        
        fig.update_layout(
            title=dict(
                text='Information Compression Efficiency Over Time',
                font=dict(size=18, family='Times New Roman'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title='Memory Update Step',
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
                title='Compression Ratio (Original Size / Compressed Size)',
                title_font=dict(size=14, family='Times New Roman'),
                tickfont=dict(size=12, family='Times New Roman'),
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                showline=True,
                linewidth=2,
                linecolor='black'
            ),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=80, r=50, t=80, b=60)
        )
        
        return fig
    
    def _create_publication_plots(self, results: Dict):
        """Create static plots for publications with formal academic styling."""
        
        created_files = []
        
        if 'test_results' in results and results['test_results']:
            # Retention performance plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            test_results = results['test_results']
            
            # Group by type
            types = {}
            for result in test_results:
                test_type = result['type']
                if test_type not in types:
                    types[test_type] = {'positions': [], 'accuracies': []}
                
                types[test_type]['positions'].append(result['info_position'])
                types[test_type]['accuracies'].append(1.0 if result['is_correct'] else 0.0)
            
            # Use academic colors
            academic_colors = [
                ACADEMIC_COLORS['primary_blue'],
                ACADEMIC_COLORS['primary_red'],
                ACADEMIC_COLORS['primary_green'],
                ACADEMIC_COLORS['primary_orange'],
                ACADEMIC_COLORS['primary_purple']
            ]
            
            # Marker symbols for different types
            marker_symbols = ['o', 's', '^', 'D', '*']
            
            for i, (test_type, data) in enumerate(types.items()):
                color = academic_colors[i % len(academic_colors)]
                marker = marker_symbols[i % len(marker_symbols)]
                formatted_name = test_type.replace('_', ' ').title()
                
                ax1.scatter(data['positions'], data['accuracies'], 
                           label=formatted_name, 
                           color=color, s=80, alpha=0.8, marker=marker,
                           edgecolors='white', linewidth=1.5)
            
            ax1.set_xlabel('Information Position (tokens)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Retrieval Accuracy', fontsize=14, fontweight='bold')
            ax1.set_title('(a) Information Retention by Position', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
            ax1.grid(True, alpha=0.3, linewidth=0.8)
            ax1.set_ylim(-0.05, 1.05)
            ax1.tick_params(axis='both', which='major', labelsize=12)
            
            # Overall accuracy by test type
            type_accuracies = []
            type_names = []
            for test_type, data in types.items():
                type_accuracies.append(np.mean(data['accuracies']))
                type_names.append(test_type.replace('_', ' ').title())
            
            bars = ax2.bar(type_names, type_accuracies, 
                          color=academic_colors[:len(type_names)], alpha=0.8,
                          edgecolor='black', linewidth=1)
            ax2.set_ylabel('Average Accuracy', fontsize=14, fontweight='bold')
            ax2.set_title('(b) Accuracy by Information Type', fontsize=14, fontweight='bold')
            ax2.set_ylim(0, 1.05)
            ax2.tick_params(axis='both', which='major', labelsize=12)
            ax2.grid(True, alpha=0.3, linewidth=0.8)
            
            # Add value labels on bars
            for bar, acc in zip(bars, type_accuracies):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{acc:.2f}', ha='center', va='bottom', 
                        fontsize=11, fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            file_path = save_matplotlib_figure(fig, self.output_dir, "retention_performance_analysis", 
                                             figsize=(16, 6), vector_format='pdf')
            if file_path:
                created_files.append(file_path)
            plt.close()
        
        # Memory content evolution analysis plot
        if ('memory_analysis' in results and 
            'memory_evolution' in results['memory_analysis']):
            
            evolution_data = results['memory_analysis']['memory_evolution']
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            positions = [e['token_position'] for e in evolution_data]
            norms = [e['memory_norm'] for e in evolution_data]
            entropies = [e['memory_entropy'] for e in evolution_data]
            
            # Memory norm evolution
            ax1.plot(positions, norms, 
                    color=ACADEMIC_COLORS['primary_blue'], linewidth=3, 
                    marker='o', markersize=6, markerfacecolor='white',
                    markeredgecolor=ACADEMIC_COLORS['primary_blue'], markeredgewidth=2)
            ax1.set_ylabel('Memory Norm (L2)', fontsize=14, fontweight='bold')
            ax1.set_title('(a) Memory Norm Evolution', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, linewidth=0.8)
            ax1.tick_params(axis='both', which='major', labelsize=12)
            
            # Memory entropy evolution
            ax2.plot(positions, entropies, 
                    color=ACADEMIC_COLORS['primary_red'], linewidth=3,
                    marker='s', markersize=6, markerfacecolor='white',
                    markeredgecolor=ACADEMIC_COLORS['primary_red'], markeredgewidth=2)
            ax2.set_xlabel('Token Position in Context', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Memory Entropy (bits)', fontsize=14, fontweight='bold')
            ax2.set_title('(b) Memory Entropy Evolution', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, linewidth=0.8)
            ax2.tick_params(axis='both', which='major', labelsize=12)
            
            plt.tight_layout()
            file_path = save_matplotlib_figure(fig, self.output_dir, "memory_content_evolution_analysis", 
                                             figsize=(12, 10), vector_format='pdf')
            if file_path:
                created_files.append(file_path)
            plt.close()
        
        return created_files
    
    def generate_comprehensive_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive memory content analysis report."""
        
        # Calculate overall statistics
        if 'test_results' in results:
            test_results = results['test_results']
            overall_accuracy = np.mean([1.0 if r['is_correct'] else 0.0 for r in test_results])
            
            # Accuracy by information type
            type_accuracies = {}
            for result in test_results:
                test_type = result['type']
                if test_type not in type_accuracies:
                    type_accuracies[test_type] = []
                type_accuracies[test_type].append(1.0 if result['is_correct'] else 0.0)
            
            for test_type in type_accuracies:
                type_accuracies[test_type] = np.mean(type_accuracies[test_type])
            
            # Distance-based analysis
            distance_analysis = self._analyze_by_distance(test_results)
        else:
            overall_accuracy = 0.0
            type_accuracies = {}
            distance_analysis = {}
        
        # Create visualizations
        viz_files = self.create_visualizations(results)
        
        # Generate insights
        insights = self._generate_insights(results, overall_accuracy, type_accuracies)
        
        report = {
            'experiment_summary': {
                'overall_accuracy': float(overall_accuracy),
                'type_accuracies': {k: float(v) for k, v in type_accuracies.items()},
                'distance_analysis': distance_analysis,
                'memory_mechanism_assessment': insights['memory_assessment']
            },
            'detailed_results': results,
            'visualizations': viz_files,
            'insights_and_conclusions': insights,
            'recommendations': self._generate_recommendations(results, insights)
        }
        
        return report
    
    def _analyze_by_distance(self, test_results: List[Dict]) -> Dict[str, Any]:
        """Analyze accuracy by distance/depth of information."""
        
        # Group by segments crossed
        segment_groups = {}
        for result in test_results:
            segments = result['segments_crossed']
            if segments not in segment_groups:
                segment_groups[segments] = []
            segment_groups[segments].append(1.0 if result['is_correct'] else 0.0)
        
        segment_analysis = {}
        for segments, accuracies in segment_groups.items():
            segment_analysis[f"{segments}_segments"] = {
                'accuracy': np.mean(accuracies),
                'count': len(accuracies),
                'std': np.std(accuracies)
            }
        
        return segment_analysis
    
    def _generate_insights(self, results: Dict, overall_accuracy: float, 
                          type_accuracies: Dict) -> Dict[str, Any]:
        """Generate insights from the analysis."""
        
        insights = {
            'memory_assessment': 'UNKNOWN',
            'key_findings': [],
            'performance_patterns': [],
            'memory_characteristics': []
        }
        
        # Assess overall memory mechanism
        if overall_accuracy >= 0.8:
            insights['memory_assessment'] = 'EXCELLENT'
        elif overall_accuracy >= 0.6:
            insights['memory_assessment'] = 'GOOD'
        elif overall_accuracy >= 0.4:
            insights['memory_assessment'] = 'MODERATE'
        else:
            insights['memory_assessment'] = 'POOR'
        
        # Key findings
        insights['key_findings'].append(f"Overall information retention: {overall_accuracy:.1%}")
        
        if type_accuracies:
            best_type = max(type_accuracies.items(), key=lambda x: x[1])
            worst_type = min(type_accuracies.items(), key=lambda x: x[1])
            
            insights['key_findings'].append(f"Best performance on {best_type[0]}: {best_type[1]:.1%}")
            insights['key_findings'].append(f"Worst performance on {worst_type[0]}: {worst_type[1]:.1%}")
        
        # Memory characteristics
        if 'memory_analysis' in results and 'memory_statistics' in results['memory_analysis']:
            stats = results['memory_analysis']['memory_statistics']
            insights['memory_characteristics'].append(f"Average memory norm: {stats['average_memory_norm']:.3f}")
            insights['memory_characteristics'].append(f"Memory growth rate: {stats['memory_growth_rate']:.3f}")
            insights['memory_characteristics'].append(f"Layers using memory: {len(stats['layers_with_memory'])}")
        
        return insights
    
    def _generate_recommendations(self, results: Dict, insights: Dict) -> List[str]:
        """Generate recommendations for improving memory mechanism."""
        
        recommendations = []
        
        assessment = insights['memory_assessment']
        
        if assessment == 'POOR':
            recommendations.append("Memory mechanism is underperforming - check training configuration")
            recommendations.append("Consider increasing balance_factor_lr or training longer")
        elif assessment == 'MODERATE':
            recommendations.append("Memory shows some effectiveness but has room for improvement")
            recommendations.append("Consider fine-tuning on long-context tasks")
        elif assessment in ['GOOD', 'EXCELLENT']:
            recommendations.append("Memory mechanism is performing well")
            recommendations.append("Consider testing on even longer contexts or more complex tasks")
        
        # Type-specific recommendations
        if 'test_results' in results:
            test_results = results['test_results']
            numerical_results = [r for r in test_results if r['type'] == 'numerical_recall']
            if numerical_results:
                numerical_accuracy = np.mean([1.0 if r['is_correct'] else 0.0 for r in numerical_results])
                if numerical_accuracy < 0.5:
                    recommendations.append("Poor numerical recall - memory may not preserve precise information")
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(description="Analyze Infini-Attention Memory Content")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--context-lengths", type=str, default="4096",
                       help="Comma-separated list of context lengths to test")
    parser.add_argument("--output-dir", type=str, default="./memory_content_analysis",
                       help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    # Parse context lengths
    context_lengths = [int(x.strip()) for x in args.context_lengths.split(',')]
    
    print("Infini-Attention Memory Content Analyzer")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Context lengths: {context_lengths}")
    print(f"Output directory: {args.output_dir}")
    
    # Initialize analyzer
    analyzer = MemoryContentAnalyzer(args.checkpoint, args.output_dir)
    
    # Load model components
    analyzer.load_model_components()
    
    # Run experiments for each context length
    all_results = {}
    
    for context_length in context_lengths:
        print(f"\nAnalyzing context length: {context_length}")
        results = analyzer.run_information_retention_experiment(context_length)
        all_results[context_length] = results
    
    # Generate comprehensive reports for each context length
    final_reports = {}
    
    for context_length, results in all_results.items():
        print(f"\nGenerating report for {context_length} tokens...")
        report = analyzer.generate_comprehensive_report(results)
        final_reports[context_length] = report
        
        # Save individual report
        report_path = Path(args.output_dir) / f"memory_content_report_{context_length}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        print(f"Report saved to: {report_path}")
    
    # Generate overall summary
    overall_summary = {
        'analysis_configuration': {
            'checkpoint_path': args.checkpoint,
            'context_lengths': context_lengths,
            'total_experiments': len(context_lengths)
        },
        'results_by_context_length': final_reports,
        'overall_conclusions': analyzer._generate_overall_conclusions(final_reports)
    }
    
    # Save overall summary
    summary_path = Path(args.output_dir) / "memory_content_overall_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(overall_summary, f, indent=2, cls=NumpyEncoder)
    
    # Print summary
    print("\n" + "=" * 50)
    print("MEMORY CONTENT ANALYSIS SUMMARY")
    print("=" * 50)
    
    for context_length, report in final_reports.items():
        summary = report['experiment_summary']
        print(f"\nContext Length: {context_length} tokens")
        print(f"  Overall Accuracy: {summary['overall_accuracy']:.1%}")
        print(f"  Memory Assessment: {summary['memory_mechanism_assessment']}")
        
        if 'type_accuracies' in summary:
            print("  Performance by Type:")
            for info_type, accuracy in summary['type_accuracies'].items():
                print(f"    {info_type.replace('_', ' ').title()}: {accuracy:.1%}")
    
    print(f"\nOverall summary saved to: {summary_path}")
    
    return overall_summary


if __name__ == "__main__":
    # Add method to analyzer class for overall conclusions
    def _generate_overall_conclusions(self, final_reports: Dict) -> Dict[str, Any]:
        """Generate overall conclusions across all context lengths."""
        
        overall_accuracies = []
        assessments = []
        
        for context_length, report in final_reports.items():
            summary = report['experiment_summary']
            overall_accuracies.append(summary['overall_accuracy'])
            assessments.append(summary['memory_mechanism_assessment'])
        
        # Overall statistics
        avg_accuracy = np.mean(overall_accuracies)
        accuracy_std = np.std(overall_accuracies)
        
        # Most common assessment
        from collections import Counter
        assessment_counts = Counter(assessments)
        most_common_assessment = assessment_counts.most_common(1)[0][0]
        
        conclusions = {
            'average_accuracy_across_contexts': float(avg_accuracy),
            'accuracy_variability': float(accuracy_std),
            'predominant_assessment': most_common_assessment,
            'consistency': 'high' if accuracy_std < 0.1 else 'medium' if accuracy_std < 0.2 else 'low',
            'context_length_effect': 'minimal' if accuracy_std < 0.1 else 'moderate' if accuracy_std < 0.2 else 'significant',
            'recommendation': 'Memory mechanism shows good information retention' if avg_accuracy > 0.6 else 'Memory mechanism needs improvement'
        }
        
        return conclusions
    
    # Add method to class
    MemoryContentAnalyzer._generate_overall_conclusions = _generate_overall_conclusions
    
    main()