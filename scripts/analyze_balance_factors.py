#!/usr/bin/env python3
"""
Infini-Attention Balance Factor Analyzer

This script analyzes balance factors from trained checkpoints to determine
whether the model learned to use memory vs local attention effectively.

Usage:
    python scripts/analyze_balance_factors.py --checkpoint ./checkpoints/model/30000
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import safetensors
from safetensors import safe_open


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


class BalanceFactorAnalyzer:
    """Analyze balance factors from Infini-Attention checkpoints."""
    
    def __init__(self, checkpoint_path: str, output_dir: str = "./balance_factor_analysis"):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.balance_factors = None
        self.global_weights = None
        self.analysis_results = {}
        
    def load_balance_factors(self) -> np.ndarray:
        """Load balance factors from checkpoint."""
        
        print("Loading balance factors from checkpoint...")
        
        # Look for balance factor files in the checkpoint
        decoder_path = self.checkpoint_path / "model" / "model" / "decoder"
        
        if not decoder_path.exists():
            raise FileNotFoundError(f"Decoder path not found: {decoder_path}")
        
        # Detect the structure and number of layers
        layer_dirs = sorted([d for d in decoder_path.iterdir() if d.is_dir() and d.name.isdigit()])
        num_layers = len(layer_dirs)
        
        print(f"Found {num_layers} layers")
        
        merged_tensors = []
        
        for layer_idx in range(num_layers):
            layer_path = decoder_path / str(layer_idx) / "pp_block" / "attn"
            
            # Find balance factor files
            balance_files = list(layer_path.glob("model_balance_factors_*.safetensors"))
            
            if not balance_files:
                print(f"Warning: No balance factor files found in layer {layer_idx}")
                continue
            
            # Load and merge tensor parallel files
            layer_tensors = []
            for file_path in sorted(balance_files):
                try:
                    tensor_file = safe_open(file_path, framework="pt", device="cpu")
                    tensor_data = tensor_file.get_tensor("data").to(torch.float32).numpy()
                    layer_tensors.append(tensor_data)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
            
            if layer_tensors:
                # Concatenate tensor parallel chunks
                merged_tensor = np.concatenate(layer_tensors)
                merged_tensors.append(merged_tensor)
                print(f"Layer {layer_idx}: loaded {len(merged_tensor)} balance factors")
        
        if not merged_tensors:
            raise ValueError("No balance factor data found in checkpoint")
        
        # Convert to numpy array: [num_layers, num_heads]
        self.balance_factors = np.array(merged_tensors)
        
        # Convert to global weights (sigmoid activation)
        self.global_weights = 1 / (1 + np.exp(-self.balance_factors))  # sigmoid
        
        print(f"Loaded balance factors: shape {self.balance_factors.shape}")
        print(f"Global weights range: [{self.global_weights.min():.3f}, {self.global_weights.max():.3f}]")
        
        return self.balance_factors
    
    def analyze_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of balance factors."""
        
        if self.global_weights is None:
            raise ValueError("Balance factors not loaded. Call load_balance_factors() first.")
        
        # Flatten for overall statistics
        flat_weights = self.global_weights.flatten()
        
        # Basic statistics
        stats = {
            'mean': float(np.mean(flat_weights)),
            'std': float(np.std(flat_weights)),
            'min': float(np.min(flat_weights)),
            'max': float(np.max(flat_weights)),
            'median': float(np.median(flat_weights)),
            'q25': float(np.percentile(flat_weights, 25)),
            'q75': float(np.percentile(flat_weights, 75))
        }
        
        # Memory vs attention preferences
        memory_preference = (flat_weights >= 0.5).sum()
        attention_preference = (flat_weights < 0.5).sum()
        
        distribution_analysis = {
            'total_factors': len(flat_weights),
            'memory_preference_count': int(memory_preference),
            'attention_preference_count': int(attention_preference),
            'memory_preference_rate': float(memory_preference / len(flat_weights)),
            'attention_preference_rate': float(attention_preference / len(flat_weights))
        }
        
        # Binned distribution
        bins = np.arange(0, 1.1, 0.1)
        hist, bin_edges = np.histogram(flat_weights, bins=bins)
        bin_labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)]
        
        binned_distribution = {
            'bins': bin_labels,
            'counts': hist.tolist(),
            'percentages': (hist / len(flat_weights) * 100).tolist()
        }
        
        # Layer-wise analysis
        layer_stats = []
        for layer_idx in range(self.global_weights.shape[0]):
            layer_weights = self.global_weights[layer_idx]
            layer_stats.append({
                'layer': layer_idx,
                'mean': float(np.mean(layer_weights)),
                'std': float(np.std(layer_weights)),
                'memory_preference_rate': float((layer_weights >= 0.5).sum() / len(layer_weights)),
                'min': float(np.min(layer_weights)),
                'max': float(np.max(layer_weights))
            })
        
        analysis = {
            'basic_statistics': stats,
            'distribution_analysis': distribution_analysis,
            'binned_distribution': binned_distribution,
            'layer_wise_stats': layer_stats,
            'interpretation': self._interpret_distribution(stats, distribution_analysis)
        }
        
        return analysis
    
    def _interpret_distribution(self, stats: Dict, distribution: Dict) -> Dict[str, str]:
        """Interpret the balance factor distribution."""
        
        interpretations = {}
        
        # Overall health
        if stats['std'] < 0.1:
            interpretations['variability'] = "LOW: Balance factors are too uniform - may not be learning properly"
        elif stats['std'] > 0.3:
            interpretations['variability'] = "HIGH: Good diversity in balance factor learning"
        else:
            interpretations['variability'] = "MODERATE: Decent balance factor diversity"
        
        # Memory vs attention balance
        memory_rate = distribution['memory_preference_rate']
        if memory_rate < 0.1:
            interpretations['preference'] = "ATTENTION_DOMINANT: Model strongly prefers local attention"
        elif memory_rate > 0.9:
            interpretations['preference'] = "MEMORY_DOMINANT: Model strongly prefers memory retrieval"
        elif 0.3 <= memory_rate <= 0.7:
            interpretations['preference'] = "BALANCED: Good mix of memory and attention preferences"
        else:
            interpretations['preference'] = "SKEWED: Moderate bias toward memory or attention"
        
        # Extreme values
        if stats['min'] < 0.05 and stats['max'] > 0.95:
            interpretations['range'] = "FULL_RANGE: Balance factors use full range (good)"
        elif stats['max'] - stats['min'] < 0.3:
            interpretations['range'] = "LIMITED_RANGE: Balance factors stuck in narrow range (concerning)"
        else:
            interpretations['range'] = "PARTIAL_RANGE: Balance factors use partial range"
        
        # Overall assessment
        good_indicators = 0
        if stats['std'] > 0.15:
            good_indicators += 1
        if 0.2 <= memory_rate <= 0.8:
            good_indicators += 1
        if stats['max'] - stats['min'] > 0.4:
            good_indicators += 1
        
        if good_indicators >= 2:
            interpretations['overall'] = "HEALTHY: Balance factors show good learning patterns"
        elif good_indicators == 1:
            interpretations['overall'] = "QUESTIONABLE: Some concerning patterns in balance factors"
        else:
            interpretations['overall'] = "PROBLEMATIC: Balance factors may not be learning properly"
        
        return interpretations
    
    def create_visualizations(self) -> List[str]:
        """Create comprehensive visualizations."""
        
        if self.global_weights is None:
            raise ValueError("Balance factors not loaded.")
        
        viz_files = []
        
        # 1. Distribution histogram and pie chart
        fig = self._create_distribution_plot()
        dist_path = self.output_dir / "balance_factor_distribution.html"
        fig.write_html(str(dist_path))
        viz_files.append(str(dist_path))
        
        # 2. Layer-head heatmap
        fig = self._create_heatmap()
        heatmap_path = self.output_dir / "balance_factor_heatmap.html"
        fig.write_html(str(heatmap_path))
        viz_files.append(str(heatmap_path))
        
        # 3. Layer-wise statistics
        fig = self._create_layer_analysis()
        layer_path = self.output_dir / "layer_wise_analysis.html"
        fig.write_html(str(layer_path))
        viz_files.append(str(layer_path))
        
        # 4. Static matplotlib plots for papers/presentations
        self._create_publication_plots()
        viz_files.extend([
            str(self.output_dir / "balance_factor_distribution.png"),
            str(self.output_dir / "memory_attention_preference.png"),
            str(self.output_dir / "heatmap_static.png"),
            str(self.output_dir / "layer_stats_static.png")
        ])
        
        return viz_files
    
    def _create_distribution_plot(self):
        """Create distribution analysis plot."""
        
        flat_weights = self.global_weights.flatten()
        
        # Create bins
        bins = np.arange(0, 1.1, 0.1)
        hist, bin_edges = np.histogram(flat_weights, bins=bins)
        labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)]
        
        # Create subplot
        fig = make_subplots(
            rows=1, cols=2, 
            specs=[[{'type':'xy'}, {'type':'domain'}]],
            subplot_titles=['Balance Factor Distribution', 'Memory vs Attention Preference']
        )
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=flat_weights, 
                xbins=dict(start=0, end=1, size=0.1),
                name='Distribution',
                marker_color='skyblue'
            ), 
            row=1, col=1
        )
        
        # Add pie chart
        memory_count = (flat_weights >= 0.5).sum()
        attention_count = (flat_weights < 0.5).sum()
        
        fig.add_trace(
            go.Pie(
                labels=['Memory Preference (≥0.5)', 'Attention Preference (<0.5)'], 
                values=[memory_count, attention_count],
                name='Preference Distribution',
                marker_colors=['lightcoral', 'lightblue']
            ), 
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Balance Factor Analysis",
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(title_text='Balance Factor Value', range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text='Frequency', row=1, col=1)
        
        return fig
    
    def _create_heatmap(self):
        """Create layer-head heatmap."""
        
        num_layers, num_heads = self.global_weights.shape
        
        fig = go.Figure(data=go.Heatmap(
            z=self.global_weights,
            x=[f'Head {i}' for i in range(num_heads)],
            y=[f'Layer {i}' for i in range(num_layers)],
            colorscale='RdYlBu_r',
            colorbar=dict(title='Balance Factor (0=Attention, 1=Memory)'),
            zmin=0,
            zmax=1,
            hoverongaps=False
        ))
        
        # Add text annotations
        for layer in range(num_layers):
            for head in range(num_heads):
                value = self.global_weights[layer, head]
                fig.add_annotation(
                    x=head,
                    y=layer,
                    text=f'{value:.2f}',
                    showarrow=False,
                    font=dict(
                        color='white' if 0.3 <= value <= 0.7 else 'black',
                        size=8
                    )
                )
        
        fig.update_layout(
            title='Balance Factors Across Layers and Heads',
            xaxis_title='Attention Head',
            yaxis_title='Layer',
            width=max(800, num_heads * 40),
            height=max(600, num_layers * 25),
            yaxis=dict(autorange='reversed')
        )
        
        return fig
    
    def _create_layer_analysis(self):
        """Create layer-wise analysis plots."""
        
        # Calculate layer statistics
        layer_means = np.mean(self.global_weights, axis=1)
        layer_stds = np.std(self.global_weights, axis=1)
        layer_memory_rates = np.mean(self.global_weights >= 0.5, axis=1)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Mean Balance Factor by Layer',
                'Standard Deviation by Layer', 
                'Memory Preference Rate by Layer',
                'Range (Max - Min) by Layer'
            ]
        )
        
        layers = list(range(len(layer_means)))
        
        # Mean by layer
        fig.add_trace(
            go.Scatter(x=layers, y=layer_means, mode='lines+markers', name='Mean'),
            row=1, col=1
        )
        
        # Std by layer
        fig.add_trace(
            go.Scatter(x=layers, y=layer_stds, mode='lines+markers', name='Std Dev'),
            row=1, col=2
        )
        
        # Memory preference rate by layer
        fig.add_trace(
            go.Scatter(x=layers, y=layer_memory_rates, mode='lines+markers', name='Memory Rate'),
            row=2, col=1
        )
        
        # Range by layer
        layer_ranges = np.max(self.global_weights, axis=1) - np.min(self.global_weights, axis=1)
        fig.add_trace(
            go.Scatter(x=layers, y=layer_ranges, mode='lines+markers', name='Range'),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Layer-wise Balance Factor Analysis',
            height=800,
            showlegend=False
        )
        
        # Update axis labels
        fig.update_xaxes(title_text='Layer Index')
        fig.update_yaxes(title_text='Mean Balance Factor', row=1, col=1)
        fig.update_yaxes(title_text='Standard Deviation', row=1, col=2)
        fig.update_yaxes(title_text='Memory Preference Rate', row=2, col=1)
        fig.update_yaxes(title_text='Range', row=2, col=2)
        
        return fig
    
    def _create_publication_plots(self):
        """Create static plots for publications."""
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        flat_weights = self.global_weights.flatten()
        
        # 1. Distribution histogram (separate PNG)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.hist(flat_weights, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Balance Factor Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Balance Factor Distribution')
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.7, label='Memory/Attention Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "balance_factor_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Memory vs Attention Preference pie chart (separate PNG)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        memory_count = (flat_weights >= 0.5).sum()
        attention_count = (flat_weights < 0.5).sum()
        ax.pie([memory_count, attention_count], 
               labels=['Memory Preference (≥0.5)', 'Attention Preference (<0.5)'],
               autopct='%1.1f%%', colors=['lightcoral', 'lightblue'],
               startangle=90, textprops={'fontsize': 12})
        ax.set_title('Memory vs Attention Preference', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "memory_attention_preference.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Heatmap
        fig, ax = plt.subplots(figsize=(max(8, self.global_weights.shape[1] * 0.3), 
                                       max(6, self.global_weights.shape[0] * 0.2)))
        
        im = ax.imshow(self.global_weights, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        ax.set_xlabel('Attention Head')
        ax.set_ylabel('Layer')
        ax.set_title('Balance Factors Heatmap')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Balance Factor (0=Attention, 1=Memory)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "heatmap_static.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Layer statistics
        layer_means = np.mean(self.global_weights, axis=1)
        layer_stds = np.std(self.global_weights, axis=1)
        layers = list(range(len(layer_means)))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(layers, layer_means, 'o-', label='Mean')
        ax1.fill_between(layers, layer_means - layer_stds, layer_means + layer_stds, 
                        alpha=0.3, label='±1 Std')
        ax1.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Memory Threshold')
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Balance Factor')
        ax1.set_title('Layer-wise Balance Factor Statistics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        memory_rates = np.mean(self.global_weights >= 0.5, axis=1)
        ax2.plot(layers, memory_rates, 's-', color='orange', label='Memory Preference Rate')
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Memory Preference Rate')
        ax2.set_title('Memory vs Attention Preference by Layer')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "layer_stats_static.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        
        if self.balance_factors is None:
            self.load_balance_factors()
        
        # Perform analysis
        distribution_analysis = self.analyze_distribution()
        
        # Create visualizations
        viz_files = self.create_visualizations()
        
        # Generate report
        report = {
            'checkpoint_info': {
                'path': str(self.checkpoint_path),
                'balance_factors_shape': list(self.balance_factors.shape),
                'total_parameters': int(self.balance_factors.size)
            },
            'analysis': distribution_analysis,
            'visualizations': viz_files,
            'recommendations': self._generate_recommendations(distribution_analysis),
            'conclusion': self._draw_conclusion(distribution_analysis)
        }
        
        return report
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate actionable recommendations."""
        
        recommendations = []
        interpretation = analysis['interpretation']
        stats = analysis['basic_statistics']
        distribution = analysis['distribution_analysis']
        
        if 'PROBLEMATIC' in interpretation['overall']:
            recommendations.append("Consider retraining with different balance_factor_lr")
            recommendations.append("Check if memory mechanism is properly implemented")
        
        if stats['std'] < 0.1:
            recommendations.append("Increase balance_factor_lr to encourage more diverse learning")
        
        if distribution['memory_preference_rate'] < 0.1:
            recommendations.append("Model may not be learning to use memory - check training configuration")
        elif distribution['memory_preference_rate'] > 0.9:
            recommendations.append("Model may be over-relying on memory - consider reducing balance_factor_lr")
        
        if stats['max'] - stats['min'] < 0.3:
            recommendations.append("Balance factors are in narrow range - may need longer training or higher learning rate")
        
        return recommendations
    
    def _draw_conclusion(self, analysis: Dict) -> Dict[str, Any]:
        """Draw overall conclusion about memory mechanism."""
        
        interpretation = analysis['interpretation']
        stats = analysis['basic_statistics']
        distribution = analysis['distribution_analysis']
        
        # Determine confidence level
        confidence_indicators = 0
        if stats['std'] > 0.15:
            confidence_indicators += 1
        if 0.2 <= distribution['memory_preference_rate'] <= 0.8:
            confidence_indicators += 1
        if stats['max'] - stats['min'] > 0.4:
            confidence_indicators += 1
        
        if confidence_indicators >= 2:
            confidence = "HIGH"
            memory_working = True
        elif confidence_indicators == 1:
            confidence = "MEDIUM"
            memory_working = True
        else:
            confidence = "LOW"
            memory_working = False
        
        return {
            'memory_mechanism_learned': memory_working,
            'confidence': confidence,
            'confidence_score': confidence_indicators / 3,
            'primary_finding': interpretation['overall'],
            'key_insights': [
                f"Balance factor variability: {interpretation['variability']}",
                f"Memory vs attention preference: {interpretation['preference']}",
                f"Parameter range usage: {interpretation['range']}"
            ]
        }


def main():
    parser = argparse.ArgumentParser(description="Analyze Infini-Attention Balance Factors")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./balance_factor_analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--save-raw", action="store_true",
                       help="Save raw balance factor tensors")
    
    args = parser.parse_args()
    
    print("Infini-Attention Balance Factor Analyzer")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    
    # Initialize analyzer
    analyzer = BalanceFactorAnalyzer(args.checkpoint, args.output_dir)
    
    try:
        # Generate comprehensive report
        print("\nGenerating comprehensive analysis...")
        report = analyzer.generate_report()
        
        # Save report
        report_path = Path(args.output_dir) / "balance_factor_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        # Save raw data if requested
        if args.save_raw:
            raw_path = Path(args.output_dir) / "raw_balance_factors.npz"
            np.savez(raw_path, 
                    balance_factors=analyzer.balance_factors,
                    global_weights=analyzer.global_weights)
            print(f"Raw data saved to: {raw_path}")
        
        # Print summary
        print("\n" + "=" * 50)
        print("BALANCE FACTOR ANALYSIS SUMMARY")
        print("=" * 50)
        
        conclusion = report['conclusion']
        print(f"Memory Mechanism Learned: {conclusion['memory_mechanism_learned']}")
        print(f"Confidence Level: {conclusion['confidence']} ({conclusion['confidence_score']:.2f})")
        print(f"Primary Finding: {conclusion['primary_finding']}")
        
        print("\nKey Insights:")
        for insight in conclusion['key_insights']:
            print(f"  • {insight}")
        
        if report['recommendations']:
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        print(f"\nDetailed report saved to: {report_path}")
        print(f"Visualizations created in: {args.output_dir}")
        
        return report
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()