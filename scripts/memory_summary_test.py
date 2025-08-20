#!/usr/bin/env python3
"""
Condensed Memory Mechanism Summary Test

Runs essential memory tests and outputs a concise summary with key insights.
Perfect for remote testing and quick evaluation.

Usage:
    python scripts/memory_summary_test.py --checkpoint ./checkpoints/model/30000
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Any
import sys
import os


def run_comprehensive_tests(checkpoint_path: str, output_dir: str, context_lengths: str = "8192,16384,32768", content_length: int = 16384) -> Dict[str, Any]:
    """Run comprehensive tests and extract key insights."""
    
    print("Running Infini-Attention Memory Summary Test...")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 60)
    
    # Set up environment for distributed PyTorch
    env = os.environ.copy()
    env.update({
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '29500', 
        'WORLD_SIZE': '1',
        'RANK': '0',
        'LOCAL_RANK': '0'
    })
    
    script_dir = Path(__file__).parent
    results = {}
    
    # 1. Balance Factor Analysis (Always run - fast and critical)
    print("1. Analyzing Balance Factors...")
    try:
        cmd = [sys.executable, str(script_dir / 'analyze_balance_factors.py'),
               '--checkpoint', checkpoint_path, '--output-dir', f"{output_dir}/balance"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        
        if result.returncode == 0:
            # Load and extract key info
            balance_file = Path(f"{output_dir}/balance/balance_factor_report.json")
            if balance_file.exists():
                with open(balance_file, 'r') as f:
                    data = json.load(f)
                
                conclusion = data.get('conclusion', {})
                stats = data.get('analysis', {}).get('basic_statistics', {})
                
                results['balance_factors'] = {
                    'status': 'success',
                    'learned': conclusion.get('memory_mechanism_learned', False),
                    'confidence': conclusion.get('confidence', 'UNKNOWN'),
                    'mean_factor': round(stats.get('mean', 0), 3),
                    'std_factor': round(stats.get('std', 0), 3),
                    'range': [round(stats.get('min', 0), 3), round(stats.get('max', 0), 3)],
                    'finding': conclusion.get('primary_finding', 'Unknown')
                }
            else:
                results['balance_factors'] = {'status': 'success', 'data_missing': True}
        else:
            results['balance_factors'] = {'status': 'failed', 'error': result.stderr[:200]}
    except Exception as e:
        results['balance_factors'] = {'status': 'error', 'error': str(e)[:200]}
    
    # 2. Quick Memory Comparison (Most important for effectiveness)
    print("2. Testing Memory vs No-Memory Performance...")
    try:
        cmd = [sys.executable, str(script_dir / 'compare_memory_vs_no_memory.py'),
               '--checkpoint', checkpoint_path, '--context-lengths', context_lengths, 
               '--samples-per-length', '3', '--output-dir', f"{output_dir}/comparison"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
        
        if result.returncode == 0:
            comp_file = Path(f"{output_dir}/comparison/comparison_report.json")
            if comp_file.exists():
                with open(comp_file, 'r') as f:
                    data = json.load(f)
                
                conclusion = data.get('conclusion', {})
                overall = data.get('overall_results', {})
                
                results['memory_comparison'] = {
                    'status': 'success',
                    'effectiveness': conclusion.get('memory_effectiveness', 'UNKNOWN'),
                    'with_memory_mean': round(overall.get('with_memory_mean', 0), 4),
                    'without_memory_mean': round(overall.get('without_memory_mean', 0), 4),
                    'advantage': round(overall.get('memory_advantage', 0), 4),
                    'p_value': overall.get('overall_p_value', 'N/A'),
                    'effect_size': round(overall.get('overall_effect_size', 0), 3)
                }
            else:
                results['memory_comparison'] = {'status': 'success', 'data_missing': True}
        else:
            results['memory_comparison'] = {'status': 'failed', 'error': result.stderr[:200]}
    except Exception as e:
        results['memory_comparison'] = {'status': 'error', 'error': str(e)[:200]}
    
    # 3. Content Analysis (If time permits)
    print("3. Testing Information Retention...")
    try:
        cmd = [sys.executable, str(script_dir / 'memory_content_analysis.py'),
               '--checkpoint', checkpoint_path, '--context-lengths', str(content_length),
               '--output-dir', f"{output_dir}/content"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, env=env)
        
        if result.returncode == 0:
            content_file = Path(f"{output_dir}/content/memory_content_overall_summary.json")
            if content_file.exists():
                with open(content_file, 'r') as f:
                    data = json.load(f)
                
                conclusions = data.get('overall_conclusions', {})
                
                results['content_analysis'] = {
                    'status': 'success',
                    'average_accuracy': round(conclusions.get('average_accuracy_across_contexts', 0), 3),
                    'consistency': conclusions.get('consistency', 'unknown'),
                    'assessment': conclusions.get('memory_mechanism_assessment', 'UNKNOWN'),
                    'context_lengths_tested': data.get('context_lengths_analyzed', [])
                }
            else:
                results['content_analysis'] = {'status': 'success', 'data_missing': True}
        else:
            results['content_analysis'] = {'status': 'failed', 'error': result.stderr[:200]}
    except Exception as e:
        results['content_analysis'] = {'status': 'error', 'error': str(e)[:200]}
    
    return results


def generate_summary(results: Dict[str, Any], checkpoint_path: str) -> Dict[str, Any]:
    """Generate concise summary with key insights."""
    
    summary = {
        'test_metadata': {
            'checkpoint_path': checkpoint_path,
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tests_run': list(results.keys())
        },
        'key_findings': [],
        'overall_assessment': 'UNKNOWN',
        'memory_mechanism_working': False,
        'critical_metrics': {},
        'recommendations': [],
        'detailed_results': results
    }
    
    # Extract key findings
    findings = []
    working_indicators = 0
    total_indicators = 0
    
    # Balance factors analysis
    if results.get('balance_factors', {}).get('status') == 'success':
        bf = results['balance_factors']
        if bf.get('learned', False):
            findings.append(f"✅ Balance factors learned (confidence: {bf.get('confidence', 'unknown')})")
            working_indicators += 1
        else:
            findings.append(f"❌ Balance factors not learned properly")
        total_indicators += 1
        
        summary['critical_metrics']['balance_factor_range'] = bf.get('range', [0, 0])
        summary['critical_metrics']['balance_factor_mean'] = bf.get('mean_factor', 0)
    
    # Memory comparison analysis  
    if results.get('memory_comparison', {}).get('status') == 'success':
        mc = results['memory_comparison']
        effectiveness = mc.get('effectiveness', 'UNKNOWN')
        advantage = mc.get('advantage', 0)
        
        if 'EFFECTIVE' in effectiveness and effectiveness != 'INEFFECTIVE':
            findings.append(f"✅ Memory provides performance benefit ({effectiveness})")
            working_indicators += 1
        else:
            findings.append(f"❌ Memory shows no performance benefit ({effectiveness})")
        total_indicators += 1
        
        summary['critical_metrics']['memory_advantage'] = advantage
        summary['critical_metrics']['performance_with_memory'] = mc.get('with_memory_mean', 0)
        summary['critical_metrics']['performance_without_memory'] = mc.get('without_memory_mean', 0)
    
    # Content analysis
    if results.get('content_analysis', {}).get('status') == 'success':
        ca = results['content_analysis']
        accuracy = ca.get('average_accuracy', 0)
        
        if accuracy > 0.5:
            findings.append(f"✅ Good information retention ({accuracy:.1%})")
            working_indicators += 1
        elif accuracy > 0.1:
            findings.append(f"⚠️ Moderate information retention ({accuracy:.1%})")
            working_indicators += 0.5
        else:
            findings.append(f"❌ Poor information retention ({accuracy:.1%})")
        total_indicators += 1
        
        summary['critical_metrics']['information_retention'] = accuracy
    
    # Overall assessment
    if total_indicators > 0:
        success_rate = working_indicators / total_indicators
        summary['memory_mechanism_working'] = success_rate >= 0.5
        
        if success_rate >= 0.8:
            summary['overall_assessment'] = 'EXCELLENT'
        elif success_rate >= 0.6:
            summary['overall_assessment'] = 'GOOD' 
        elif success_rate >= 0.4:
            summary['overall_assessment'] = 'MODERATE'
        else:
            summary['overall_assessment'] = 'POOR'
    
    summary['key_findings'] = findings
    
    # Generate recommendations
    recommendations = []
    if summary['memory_mechanism_working']:
        recommendations.append("Memory mechanism shows positive signs - consider longer context testing")
        if summary['overall_assessment'] in ['EXCELLENT', 'GOOD']:
            recommendations.append("Good performance - ready for production use")
        else:
            recommendations.append("Moderate performance - consider fine-tuning")
    else:
        recommendations.append("Memory mechanism needs attention - check training configuration")
        recommendations.append("Consider increasing balance_factor_lr or training longer")
    
    summary['recommendations'] = recommendations
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Condensed Infini-Attention Memory Test")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./memory_summary_test",
                       help="Output directory")
    parser.add_argument("--context-lengths", type=str, default="8192,16384,32768",
                       help="Context lengths for memory comparison (comma-separated)")
    parser.add_argument("--content-length", type=int, default=16384,
                       help="Context length for content analysis")
    parser.add_argument("--extreme-test", action="store_true",
                       help="Test with very long contexts (32K, 64K, 128K)")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint {checkpoint_path} does not exist")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle extreme test mode
    if args.extreme_test:
        context_lengths = "32768,65536,131072"
        content_length = 65536
        print("🚀 EXTREME TEST MODE: Testing very long contexts (32K, 64K, 128K)")
        print("⚠️  This may take 10-20 minutes and require significant GPU memory")
    else:
        context_lengths = args.context_lengths
        content_length = args.content_length
    
    print(f"📏 Context lengths: {context_lengths}")
    print(f"📄 Content analysis length: {content_length}")
    print()
    
    start_time = time.time()
    
    # Run tests
    results = run_comprehensive_tests(str(checkpoint_path), str(output_dir), context_lengths, content_length)
    
    # Generate summary
    summary = generate_summary(results, str(checkpoint_path))
    
    # Save summary
    summary_file = output_dir / "memory_test_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print concise report
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("🧠 INFINI-ATTENTION MEMORY TEST SUMMARY")
    print("=" * 60)
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"📊 Overall Assessment: {summary['overall_assessment']}")
    print(f"🎯 Memory Working: {'YES' if summary['memory_mechanism_working'] else 'NO'}")
    
    print(f"\n📋 Key Findings:")
    for finding in summary['key_findings']:
        print(f"  {finding}")
    
    print(f"\n📏 Context Lengths Tested: {context_lengths}")
    print(f"\n📈 Critical Metrics:")
    for key, value in summary['critical_metrics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n💡 Recommendations:")
    for rec in summary['recommendations']:
        print(f"  • {rec}")
    
    print(f"\n📄 Detailed report: {summary_file}")
    print("=" * 60)
    
    # Exit code based on results
    if summary['overall_assessment'] in ['EXCELLENT', 'GOOD']:
        sys.exit(0)
    elif summary['overall_assessment'] == 'MODERATE':
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
