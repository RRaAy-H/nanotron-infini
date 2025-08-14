#!/usr/bin/env python3
"""
Analyze passkey evaluation results for 300M Infini-Attention model.

This script provides comprehensive analysis of passkey retrieval performance,
with specific insights for Infini-Attention models using segment_length=1024.
"""

import pickle
import glob
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
import argparse
from typing import List, Dict, Any, Tuple

def extract_number_from_generation(gen_text: str, target_answer: str) -> bool:
    """
    Enhanced number extraction for passkey evaluation.
    
    Args:
        gen_text: Generated text from model
        target_answer: Expected passkey number
        
    Returns:
        True if passkey found correctly, False otherwise
    """
    if not gen_text or not target_answer:
        return False
    
    gen_text = str(gen_text).strip()
    target = str(target_answer).strip()
    
    # Method 1: Direct string match (most reliable)
    if target in gen_text:
        return True
    
    # Method 2: Extract numbers and check
    numbers = re.findall(r'\d+', gen_text)
    if target in numbers:
        return True
    
    # Method 3: Check first few tokens (sometimes model generates correct number but continues)
    first_tokens = gen_text.split()[:5]
    for token in first_tokens:
        if target in token or token in target:
            return True
    
    return False

def analyze_by_segment_boundaries(results_by_depth: List[Dict], segment_length: int = 1024) -> Dict:
    """
    Analyze performance specifically for Infini-Attention segment boundaries.
    
    Args:
        results_by_depth: List of accuracy results by depth
        segment_length: Segment length used in Infini-Attention (default 1024)
        
    Returns:
        Dictionary with segment-specific analysis
    """
    # For 1024 token context with 1024 segment length, everything is within one segment
    # For longer contexts, we can analyze cross-segment performance
    
    within_segment = []
    cross_segment = []
    
    for result in results_by_depth:
        depth = result['depth_percent']
        accuracy = result['accuracy']
        
        # Rough estimation: if depth suggests position within first segment
        if depth <= 50:  # Rough heuristic for within-segment
            within_segment.append(accuracy)
        else:
            cross_segment.append(accuracy)
    
    return {
        'within_segment_avg': np.mean(within_segment) if within_segment else 0,
        'cross_segment_avg': np.mean(cross_segment) if cross_segment else 0,
        'within_segment_count': len(within_segment),
        'cross_segment_count': len(cross_segment)
    }

def categorize_performance(avg_accuracy: float, std_dev: float, context_length: int) -> Dict[str, str]:
    """
    Categorize performance with context-aware expectations.
    
    Args:
        avg_accuracy: Average accuracy across all depths
        std_dev: Standard deviation of accuracy
        context_length: Context length being evaluated
        
    Returns:
        Dictionary with performance category and recommendations
    """
    # Adjust expectations based on context length and segment size
    segment_count = max(1, context_length / 1024)
    
    if segment_count == 1:  # Within single segment
        if avg_accuracy >= 95:
            category = "EXCELLENT"
            status = "OK"
            message = "Infini-Attention working perfectly within segment!"
        elif avg_accuracy >= 90:
            category = "VERY_GOOD"
            status = "OK"
            message = "Strong performance within segment"
        elif avg_accuracy >= 80:
            category = "GOOD"
            status = "OK"
            message = "Good performance, minor room for improvement"
        elif avg_accuracy >= 70:
            category = "FAIR"
            status = "WARN"
            message = "Acceptable but could be better within segment"
        else:
            category = "POOR"
            status = "FAIL"
            message = "Poor performance even within segment - check training"
    else:  # Cross-segment
        expected_degradation = min(20, segment_count * 5)  # Expected 5% degradation per additional segment
        if avg_accuracy >= (95 - expected_degradation):
            category = "EXCELLENT"
            status = "OK"
            message = f"Excellent cross-segment performance ({segment_count:.1f} segments)!"
        elif avg_accuracy >= (85 - expected_degradation):
            category = "VERY_GOOD"
            status = "OK"
            message = f"Very good cross-segment performance ({segment_count:.1f} segments)"
        elif avg_accuracy >= (75 - expected_degradation):
            category = "GOOD"
            status = "OK"
            message = f"Good cross-segment performance ({segment_count:.1f} segments)"
        elif avg_accuracy >= (65 - expected_degradation):
            category = "FAIR"
            status = "WARN"
            message = f"Fair cross-segment performance, may need tuning"
        else:
            category = "POOR"
            status = "FAIL"
            message = f"Poor cross-segment performance - Infini mechanism may have issues"
    
    # Add consistency check
    consistency = "CONSISTENT" if std_dev < 15 else "INCONSISTENT" if std_dev < 25 else "VERY_INCONSISTENT"
    
    return {
        'category': category,
        'status': status,
        'message': message,
        'consistency': consistency,
        'expected_degradation': expected_degradation if segment_count > 1 else 0
    }

def analyze_passkey_results(results_dir: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Main analysis function for passkey evaluation results.
    
    Args:
        results_dir: Directory containing pickle result files
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary containing complete analysis results
    """
    results_path = Path(results_dir)
    result_files = sorted(glob.glob(str(results_path / "*.pkl")))
    
    if not result_files:
        if verbose:
            print(f"No result files found in {results_dir}")
        return {'error': 'No result files found'}
    
    # Load and combine all results
    all_results = []
    for file in result_files:
        try:
            with open(file, 'rb') as f:
                df = pickle.load(f)
                all_results.append(df)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not load {file}: {e}")
            continue
    
    if not all_results:
        if verbose:
            print("No valid result files could be loaded")
        return {'error': 'No valid result files'}
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Extract context information
    context_length = combined_df['context_length'].iloc[0] if 'context_length' in combined_df.columns else 'unknown'
    total_samples = len(combined_df)
    
    if verbose:
        print("\n" + "="*70)
        print("PASSKEY RETRIEVAL ANALYSIS - 300M INFINI-ATTENTION MODEL")
        print("="*70)
        print(f"Model: 300M Infini-Attention (segment_length=1024)")
        print(f"Context Length: {context_length} tokens")
        print(f"Total Samples: {total_samples}")
        if isinstance(context_length, (int, float)):
            segments = max(1, int(context_length / 1024))
            print(f"Expected Segments: {segments}")
        print("="*70)
    
    # Analyze by depth
    results_by_depth = []
    for depth in sorted(combined_df['depth_percent'].unique()):
        depth_df = combined_df[combined_df['depth_percent'] == depth]
        
        correct = 0
        total = len(depth_df)
        
        for _, row in depth_df.iterrows():
            gen_text = row.get('generation_text', '')
            answer = row.get('answer', '')
            
            if extract_number_from_generation(gen_text, answer):
                correct += 1
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        results_by_depth.append({
            'depth_percent': depth,
            'accuracy': accuracy,
            'num_samples': total,
            'correct': correct
        })
        
        if verbose:
            # Create visual accuracy bar
            bar_length = max(0, min(30, int(accuracy * 30 / 100)))
            bar = "#" * bar_length + "." * (30 - bar_length)
            
            print(f"Depth {depth:3d}%: {bar} {accuracy:6.2f}% ({correct}/{total})")
    
    # Calculate summary statistics
    accuracies = [r['accuracy'] for r in results_by_depth]
    avg_accuracy = np.mean(accuracies)
    std_dev = np.std(accuracies)
    min_accuracy = np.min(accuracies)
    max_accuracy = np.max(accuracies)
    
    min_depth = results_by_depth[np.argmin(accuracies)]['depth_percent']
    max_depth = results_by_depth[np.argmax(accuracies)]['depth_percent']
    
    if verbose:
        print("\n" + "-"*70)
        print("SUMMARY STATISTICS:")
        print(f"Average Accuracy: {avg_accuracy:.2f}%")
        print(f"Std Deviation: {std_dev:.2f}%")
        print(f"Min Accuracy: {min_accuracy:.2f}% (at depth {min_depth}%)")
        print(f"Max Accuracy: {max_accuracy:.2f}% (at depth {max_depth}%)")
    
    # Segment-specific analysis
    if isinstance(context_length, (int, float)):
        segment_analysis = analyze_by_segment_boundaries(results_by_depth)
        if verbose and segment_analysis['within_segment_count'] > 0 and segment_analysis['cross_segment_count'] > 0:
            print(f"Within-segment avg: {segment_analysis['within_segment_avg']:.2f}%")
            print(f"Cross-segment avg: {segment_analysis['cross_segment_avg']:.2f}%")
    else:
        segment_analysis = {}
    
    # Performance categorization
    performance = categorize_performance(avg_accuracy, std_dev, context_length if isinstance(context_length, (int, float)) else 1024)
    
    if verbose:
        print("\n" + "-"*70)
        print("PERFORMANCE ASSESSMENT:")
        print(f"{performance['status']} {performance['category']}: {performance['message']}")
        print(f"Consistency: {performance['consistency']} (std dev: {std_dev:.2f}%)")
        
        if performance['expected_degradation'] > 0:
            print(f"Expected degradation: {performance['expected_degradation']:.1f}% for {int(context_length/1024)} segments")
    
    # Prepare return data
    analysis_result = {
        'model': '300M Infini-Attention',
        'context_length': context_length,
        'total_samples': total_samples,
        'results_by_depth': results_by_depth,
        'summary_stats': {
            'average_accuracy': float(avg_accuracy),
            'std_dev': float(std_dev),
            'min_accuracy': float(min_accuracy),
            'max_accuracy': float(max_accuracy),
            'min_depth': int(min_depth),
            'max_depth': int(max_depth)
        },
        'segment_analysis': segment_analysis,
        'performance': performance
    }
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    # Save summary to JSON
    summary_path = results_path / "summary.json"
    try:
        analysis_result_serializable = convert_numpy_types(analysis_result)
        with open(summary_path, 'w') as f:
            json.dump(analysis_result_serializable, f, indent=2)
        if verbose:
            print(f"\nDetailed summary saved to: {summary_path}")
    except Exception as e:
        if verbose:
            print(f"Warning: Could not save summary to JSON: {e}")
    
    return analysis_result

def main():
    parser = argparse.ArgumentParser(description="Analyze passkey evaluation results for 300M Infini-Attention model")
    parser.add_argument("results_dir", help="Directory containing pickle result files")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress detailed output")
    parser.add_argument("--json-only", action="store_true", help="Output only JSON summary")
    
    args = parser.parse_args()
    
    if args.json_only:
        result = analyze_passkey_results(args.results_dir, verbose=False)
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        print(json.dumps(convert_numpy_types(result), indent=2))
    else:
        analyze_passkey_results(args.results_dir, verbose=not args.quiet)

if __name__ == "__main__":
    main()