#!/usr/bin/env python3
"""
Comprehensive Infini-Attention Memory Testing Master Script

This script orchestrates all individual testing tools to provide a complete
analysis of memory mechanism effectiveness. It runs all tests, generates
unified reports, and provides clear pass/fail determinations.

Usage:
    python scripts/test_memory_comprehensive.py --checkpoint ./checkpoints/model/30000
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jinja2 import Template


class ComprehensiveMemoryTester:
    """Master orchestrator for comprehensive memory testing."""
    
    def __init__(self, checkpoint_path: str, output_dir: str = "./comprehensive_memory_analysis"):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test results from individual modules
        self.test_results = {}
        self.test_status = {}
        self.consolidated_report = {}
        
        # Script paths
        self.script_dir = Path(__file__).parent
        self.scripts = {
            'balance_factors': self.script_dir / 'analyze_balance_factors.py',
            'memory_debug': self.script_dir / 'debug_memory_usage.py',
            'memory_comparison': self.script_dir / 'compare_memory_vs_no_memory.py',
            'content_analysis': self.script_dir / 'memory_content_analysis.py',
            'progressive_test': self.script_dir / 'progressive_context_test.py'
        }
        
    def run_comprehensive_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive memory analysis with all testing modules."""
        
        print("=" * 70)
        print("COMPREHENSIVE INFINI-ATTENTION MEMORY ANALYSIS")
        print("=" * 70)
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Analysis mode: {'QUICK' if config.get('quick_test', False) else 'FULL'}")
        print()
        
        start_time = time.time()
        
        # Phase 1: Balance Factor Analysis (always run first - fast and informative)
        print("Phase 1: Analyzing Balance Factors...")
        balance_result = self._run_balance_factor_analysis(config)
        
        # Phase 2: Memory Usage Debugging (if requested)
        if config.get('run_memory_debug', True):
            print("\nPhase 2: Memory Usage Debugging...")
            debug_result = self._run_memory_debugging(config)
        else:
            debug_result = {'skipped': True, 'reason': 'Not requested'}
        
        # Phase 3: Memory vs No-Memory Comparison (if requested)
        if config.get('run_comparison', True):
            print("\nPhase 3: Memory vs No-Memory Comparison...")
            comparison_result = self._run_memory_comparison(config)
        else:
            comparison_result = {'skipped': True, 'reason': 'Not requested'}
        
        # Phase 4: Memory Content Analysis (if requested and not quick test)
        if config.get('run_content_analysis', True) and not config.get('quick_test', False):
            print("\nPhase 4: Memory Content Analysis...")
            content_result = self._run_content_analysis(config)
        else:
            content_result = {'skipped': True, 'reason': 'Quick test mode or not requested'}
        
        # Phase 5: Progressive Context Testing (if requested and not quick test)
        if config.get('run_progressive_test', True) and not config.get('quick_test', False):
            print("\nPhase 5: Progressive Context Testing...")
            progressive_result = self._run_progressive_testing(config)
        else:
            progressive_result = {'skipped': True, 'reason': 'Quick test mode or not requested'}
        
        # Consolidate all results
        all_results = {
            'balance_factors': balance_result,
            'memory_debug': debug_result,
            'memory_comparison': comparison_result,
            'content_analysis': content_result,
            'progressive_test': progressive_result
        }
        
        # Generate consolidated analysis
        print("\n" + "=" * 50)
        print("CONSOLIDATING RESULTS...")
        print("=" * 50)
        
        consolidated = self._consolidate_results(all_results, config)
        
        total_time = time.time() - start_time
        consolidated['analysis_metadata'] = {
            'total_analysis_time': total_time,
            'checkpoint_path': str(self.checkpoint_path),
            'output_directory': str(self.output_dir),
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': config
        }
        
        # Generate comprehensive report
        self._generate_unified_report(consolidated)
        
        # Print final summary
        self._print_final_summary(consolidated, total_time)
        
        return consolidated
    
    def _run_balance_factor_analysis(self, config: Dict) -> Dict[str, Any]:
        """Run balance factor analysis."""
        
        balance_output_dir = self.output_dir / "balance_factors"
        
        try:
            cmd = [
                sys.executable, str(self.scripts['balance_factors']),
                '--checkpoint', str(self.checkpoint_path),
                '--output-dir', str(balance_output_dir)
            ]
            
            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                # Load results
                result_file = balance_output_dir / "balance_factor_report.json"
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    return {'status': 'success', 'data': data, 'stdout': result.stdout}
                else:
                    return {'status': 'partial_success', 'stdout': result.stdout, 'stderr': result.stderr}
            else:
                return {'status': 'failed', 'error': result.stderr, 'stdout': result.stdout}
        
        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'error': 'Balance factor analysis timed out'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _run_memory_debugging(self, config: Dict) -> Dict[str, Any]:
        """Run memory debugging analysis."""
        
        debug_output_dir = self.output_dir / "memory_debug"
        context_lengths = config.get('debug_context_lengths', '1024,2048,4096')
        
        try:
            cmd = [
                sys.executable, str(self.scripts['memory_debug']),
                '--checkpoint', str(self.checkpoint_path),
                '--context-lengths', context_lengths,
                '--num-samples', str(config.get('debug_samples', 3)),
                '--output-dir', str(debug_output_dir)
            ]
            
            if config.get('verbose', False):
                cmd.append('--verbose')
            
            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                # Load results
                result_file = debug_output_dir / "comprehensive_memory_debug_report.json"
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    return {'status': 'success', 'data': data, 'stdout': result.stdout}
                else:
                    return {'status': 'partial_success', 'stdout': result.stdout, 'stderr': result.stderr}
            else:
                return {'status': 'failed', 'error': result.stderr, 'stdout': result.stdout}
        
        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'error': 'Memory debugging timed out'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _run_memory_comparison(self, config: Dict) -> Dict[str, Any]:
        """Run memory vs no-memory comparison."""
        
        comparison_output_dir = self.output_dir / "memory_comparison"
        context_lengths = config.get('comparison_context_lengths', '1024,2048,4096')
        
        try:
            cmd = [
                sys.executable, str(self.scripts['memory_comparison']),
                '--checkpoint', str(self.checkpoint_path),
                '--context-lengths', context_lengths,
                '--samples-per-length', str(config.get('comparison_samples', 5 if config.get('quick_test') else 10)),
                '--output-dir', str(comparison_output_dir)
            ]
            
            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                # Load results
                result_file = comparison_output_dir / "comparison_report.json"
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    return {'status': 'success', 'data': data, 'stdout': result.stdout}
                else:
                    return {'status': 'partial_success', 'stdout': result.stdout, 'stderr': result.stderr}
            else:
                return {'status': 'failed', 'error': result.stderr, 'stdout': result.stdout}
        
        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'error': 'Memory comparison timed out'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _run_content_analysis(self, config: Dict) -> Dict[str, Any]:
        """Run memory content analysis."""
        
        content_output_dir = self.output_dir / "content_analysis"
        context_lengths = config.get('content_context_lengths', '4096')
        
        try:
            cmd = [
                sys.executable, str(self.scripts['content_analysis']),
                '--checkpoint', str(self.checkpoint_path),
                '--context-lengths', context_lengths,
                '--output-dir', str(content_output_dir)
            ]
            
            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                # Load results
                result_file = content_output_dir / "memory_content_overall_summary.json"
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    return {'status': 'success', 'data': data, 'stdout': result.stdout}
                else:
                    return {'status': 'partial_success', 'stdout': result.stdout, 'stderr': result.stderr}
            else:
                return {'status': 'failed', 'error': result.stderr, 'stdout': result.stdout}
        
        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'error': 'Memory content analysis timed out'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _run_progressive_testing(self, config: Dict) -> Dict[str, Any]:
        """Run progressive context testing."""
        
        progressive_output_dir = self.output_dir / "progressive_testing"
        
        try:
            cmd = [
                sys.executable, str(self.scripts['progressive_test']),
                '--checkpoint', str(self.checkpoint_path),
                '--min-context', str(config.get('progressive_min_context', 1024)),
                '--max-context', str(config.get('progressive_max_context', 8192)),
                '--step-size', str(config.get('progressive_step_size', 1024)),
                '--samples-per-length', str(config.get('progressive_samples', 5)),
                '--output-dir', str(progressive_output_dir)
            ]
            
            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            
            if result.returncode == 0:
                # Load results
                result_file = progressive_output_dir / "progressive_context_report.json"
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    return {'status': 'success', 'data': data, 'stdout': result.stdout}
                else:
                    return {'status': 'partial_success', 'stdout': result.stdout, 'stderr': result.stderr}
            else:
                return {'status': 'failed', 'error': result.stderr, 'stdout': result.stdout}
        
        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'error': 'Progressive testing timed out'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _consolidate_results(self, results: Dict[str, Any], config: Dict) -> Dict[str, Any]:
        """Consolidate results from all test modules."""
        
        consolidation = {
            'overall_assessment': {},
            'test_results_summary': {},
            'key_findings': [],
            'recommendations': [],
            'detailed_analysis': {},
            'pass_fail_determination': {}
        }
        
        # Extract key metrics from each test
        test_scores = {}
        
        # Balance factor analysis
        if results['balance_factors']['status'] == 'success':
            bf_data = results['balance_factors']['data']
            if 'conclusion' in bf_data:
                conclusion = bf_data['conclusion']
                test_scores['balance_factors'] = {
                    'score': conclusion.get('confidence_score', 0),
                    'status': 'PASS' if conclusion.get('memory_mechanism_learned', False) else 'FAIL',
                    'confidence': conclusion.get('confidence', 'LOW')
                }
                consolidation['key_findings'].append(f"Balance factors: {conclusion.get('primary_finding', 'Unknown')}")
        
        # Memory debugging
        if results['memory_debug']['status'] == 'success':
            debug_data = results['memory_debug']['data']
            if 'overall_analysis' in debug_data:
                analysis = debug_data['overall_analysis']
                effectiveness = analysis.get('effectiveness_rating', 'UNKNOWN')
                test_scores['memory_debug'] = {
                    'score': 1.0 if 'HIGHLY_ACTIVE' in effectiveness else 0.5 if 'ACTIVE' in effectiveness else 0.0,
                    'status': 'PASS' if 'ACTIVE' in effectiveness else 'FAIL',
                    'confidence': 'HIGH' if analysis.get('memory_activation_rate', 0) > 0.5 else 'MEDIUM'
                }
                consolidation['key_findings'].append(f"Memory usage: {effectiveness}")
        
        # Memory comparison
        if results['memory_comparison']['status'] == 'success':
            comp_data = results['memory_comparison']['data']
            if 'conclusion' in comp_data:
                conclusion = comp_data['conclusion']
                effectiveness = conclusion.get('memory_effectiveness', 'INEFFECTIVE')
                test_scores['memory_comparison'] = {
                    'score': 1.0 if 'HIGHLY' in effectiveness else 0.7 if 'MODERATELY' in effectiveness else 0.3 if 'SOMEWHAT' in effectiveness else 0.0,
                    'status': 'PASS' if 'EFFECTIVE' in effectiveness and effectiveness != 'INEFFECTIVE' else 'FAIL',
                    'confidence': 'HIGH'
                }
                consolidation['key_findings'].append(f"Performance comparison: {effectiveness}")
        
        # Content analysis
        if results['content_analysis']['status'] == 'success':
            content_data = results['content_analysis']['data']
            if 'overall_conclusions' in content_data:
                conclusions = content_data['overall_conclusions']
                avg_accuracy = conclusions.get('average_accuracy_across_contexts', 0)
                test_scores['content_analysis'] = {
                    'score': avg_accuracy,
                    'status': 'PASS' if avg_accuracy > 0.6 else 'FAIL',
                    'confidence': 'HIGH' if conclusions.get('consistency', 'low') == 'high' else 'MEDIUM'
                }
                consolidation['key_findings'].append(f"Information retention: {avg_accuracy:.1%} average accuracy")
        
        # Progressive testing
        if results['progressive_test']['status'] == 'success':
            prog_data = results['progressive_test']['data']
            if 'conclusion' in prog_data:
                conclusion = prog_data['conclusion']
                effectiveness = conclusion.get('memory_effectiveness', 'UNKNOWN')
                scaling = conclusion.get('scaling_behavior', 'UNKNOWN')
                test_scores['progressive_test'] = {
                    'score': 1.0 if effectiveness == 'HIGHLY_EFFECTIVE' else 0.6 if effectiveness == 'MODERATELY_EFFECTIVE' else 0.0,
                    'status': 'PASS' if 'EFFECTIVE' in effectiveness else 'FAIL',
                    'confidence': conclusion.get('confidence_level', 'LOW')
                }
                consolidation['key_findings'].append(f"Progressive scaling: {effectiveness}, {scaling}")
        
        # Calculate overall scores
        successful_tests = [name for name, score in test_scores.items() if score['status'] == 'PASS']
        total_tests = len(test_scores)
        
        if total_tests > 0:
            pass_rate = len(successful_tests) / total_tests
            avg_score = np.mean([score['score'] for score in test_scores.values()])
            high_confidence_tests = sum(1 for score in test_scores.values() if score['confidence'] == 'HIGH')
            
            # Overall determination
            if pass_rate >= 0.8 and avg_score >= 0.7:
                overall_status = 'EXCELLENT'
            elif pass_rate >= 0.6 and avg_score >= 0.5:
                overall_status = 'GOOD'
            elif pass_rate >= 0.4 or avg_score >= 0.3:
                overall_status = 'MODERATE'
            else:
                overall_status = 'POOR'
            
            consolidation['overall_assessment'] = {
                'status': overall_status,
                'pass_rate': pass_rate,
                'average_score': avg_score,
                'successful_tests': successful_tests,
                'total_tests': total_tests,
                'high_confidence_tests': high_confidence_tests,
                'memory_mechanism_working': pass_rate >= 0.5
            }
        else:
            consolidation['overall_assessment'] = {
                'status': 'UNKNOWN',
                'error': 'No successful test results to analyze'
            }
        
        # Generate recommendations
        consolidation['recommendations'] = self._generate_consolidated_recommendations(test_scores, results)
        
        # Pass/Fail determination with reasons
        consolidation['pass_fail_determination'] = self._determine_pass_fail(test_scores, consolidation['overall_assessment'])
        
        consolidation['test_results_summary'] = {
            'individual_test_scores': test_scores,
            'test_execution_status': {name: result['status'] for name, result in results.items()}
        }
        
        consolidation['detailed_analysis'] = results
        
        return consolidation
    
    def _generate_consolidated_recommendations(self, test_scores: Dict, results: Dict) -> List[str]:
        """Generate consolidated recommendations from all test results."""
        
        recommendations = []
        
        # High-level recommendations based on overall performance
        passing_tests = [name for name, score in test_scores.items() if score['status'] == 'PASS']
        failing_tests = [name for name, score in test_scores.items() if score['status'] == 'FAIL']
        
        if len(passing_tests) >= len(failing_tests):
            recommendations.append("Memory mechanism appears to be working - consider optimizing for better performance")
        else:
            recommendations.append("Memory mechanism has significant issues - requires debugging or retraining")
        
        # Specific recommendations based on individual test failures
        if 'balance_factors' in failing_tests:
            recommendations.append("Balance factors indicate poor learning - check training configuration and balance_factor_lr")
        
        if 'memory_debug' in failing_tests:
            recommendations.append("Memory mechanism is not activating properly during inference")
        
        if 'memory_comparison' in failing_tests:
            recommendations.append("Memory provides no performance benefit - investigate training or implementation issues")
        
        if 'content_analysis' in failing_tests:
            recommendations.append("Memory is not retaining meaningful information across segments")
        
        if 'progressive_test' in failing_tests:
            recommendations.append("Memory does not scale well with context length - consider architectural improvements")
        
        # Extract specific recommendations from individual tests
        for test_name, result in results.items():
            if result['status'] == 'success' and 'data' in result:
                if 'recommendations' in result['data']:
                    recommendations.extend(result['data']['recommendations'][:2])  # Limit to top 2
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:10]  # Limit to top 10
    
    def _determine_pass_fail(self, test_scores: Dict, overall_assessment: Dict) -> Dict[str, Any]:
        """Determine overall pass/fail with detailed reasoning."""
        
        # Determine overall pass/fail
        overall_status = overall_assessment.get('status', 'UNKNOWN')
        memory_working = overall_assessment.get('memory_mechanism_working', False)
        
        if overall_status in ['EXCELLENT', 'GOOD']:
            final_determination = 'PASS'
            confidence = 'HIGH'
        elif overall_status == 'MODERATE':
            final_determination = 'CONDITIONAL_PASS'
            confidence = 'MEDIUM'
        else:
            final_determination = 'FAIL'
            confidence = 'HIGH'
        
        # Detailed reasoning
        reasoning = []
        
        if memory_working:
            reasoning.append("Memory mechanism shows evidence of functioning")
        else:
            reasoning.append("Memory mechanism shows little to no evidence of functioning")
        
        pass_count = sum(1 for score in test_scores.values() if score['status'] == 'PASS')
        total_count = len(test_scores)
        
        reasoning.append(f"{pass_count}/{total_count} individual tests passed")
        
        if overall_assessment.get('average_score', 0) > 0.6:
            reasoning.append("Average test scores indicate good performance")
        else:
            reasoning.append("Average test scores indicate poor performance")
        
        # Critical test failures
        critical_tests = ['balance_factors', 'memory_comparison']
        critical_failures = [test for test in critical_tests if test in test_scores and test_scores[test]['status'] == 'FAIL']
        
        if critical_failures:
            reasoning.append(f"Critical tests failed: {', '.join(critical_failures)}")
            if final_determination == 'PASS':
                final_determination = 'CONDITIONAL_PASS'
        
        return {
            'final_determination': final_determination,
            'confidence': confidence,
            'reasoning': reasoning,
            'critical_test_failures': critical_failures if 'critical_failures' in locals() else [],
            'numeric_score': overall_assessment.get('average_score', 0),
            'tests_passed': f"{pass_count}/{total_count}"
        }
    
    def _generate_unified_report(self, consolidated: Dict):
        """Generate unified HTML report."""
        
        # Create comprehensive HTML report
        html_template = self._get_html_template()
        
        # Render report
        template = Template(html_template)
        html_content = template.render(
            analysis=consolidated,
            timestamp=consolidated['analysis_metadata']['analysis_timestamp'],
            checkpoint_path=consolidated['analysis_metadata']['checkpoint_path']
        )
        
        # Save HTML report
        html_path = self.output_dir / "comprehensive_memory_analysis_report.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        # Save JSON report
        json_path = self.output_dir / "comprehensive_memory_analysis_report.json"
        with open(json_path, 'w') as f:
            json.dump(consolidated, f, indent=2)
        
        print(f"Unified report generated:")
        print(f"  HTML: {html_path}")
        print(f"  JSON: {json_path}")
    
    def _get_html_template(self) -> str:
        """Get HTML template for comprehensive report."""
        
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Infini-Attention Memory Analysis Report</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        h3 { color: #7f8c8d; }
        .status-box { padding: 15px; margin: 10px 0; border-radius: 8px; font-weight: bold; }
        .status-pass { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status-fail { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .status-conditional { background-color: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
        .test-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .test-card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; }
        .test-pass { border-left: 5px solid #28a745; }
        .test-fail { border-left: 5px solid #dc3545; }
        .metric { display: inline-block; margin: 10px 15px 10px 0; padding: 8px 12px; background-color: #e9ecef; border-radius: 6px; }
        .finding { margin: 8px 0; padding: 10px; background-color: #f8f9fa; border-left: 4px solid #007bff; }
        .recommendation { margin: 8px 0; padding: 10px; background-color: #fff3cd; border-left: 4px solid #ffc107; }
        .metadata { font-size: 0.9em; color: #6c757d; margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: 600; }
        .progress-bar { width: 100%; height: 20px; background-color: #e9ecef; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background-color: #28a745; transition: width 0.3s ease; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Infini-Attention Memory Analysis Report</h1>
        
        <div class="metadata">
            <strong>Analysis Date:</strong> {{ timestamp }}<br>
            <strong>Checkpoint:</strong> {{ checkpoint_path }}<br>
            <strong>Analysis Duration:</strong> {{ "%.1f"|format(analysis.analysis_metadata.total_analysis_time) }} seconds
        </div>
        
        <h2>📊 Overall Assessment</h2>
        
        {% set overall = analysis.overall_assessment %}
        {% set pass_fail = analysis.pass_fail_determination %}
        
        <div class="status-box 
        {%- if analysis.pass_fail_determination.final_determination == 'PASS' %} status-pass
        {%- elif analysis.pass_fail_determination.final_determination == 'FAIL' %} status-fail
        {%- else %} status-conditional
        {%- endif %}">
            <h3>Final Determination: {{ analysis.pass_fail_determination.final_determination }}</h3>
            <p><strong>Confidence:</strong> {{ analysis.pass_fail_determination.confidence }}</p>
            <p><strong>Numeric Score:</strong> {{ "%.2f"|format(analysis.pass_fail_determination.numeric_score) }}/1.0</p>
            <p><strong>Tests Passed:</strong> {{ analysis.pass_fail_determination.tests_passed }}</p>
        </div>
        
        <div style="margin: 20px 0;">
            <strong>Overall Status:</strong> {{ analysis.overall_assessment.status }}<br>
            <div class="progress-bar" style="margin-top: 10px;">
                <div class="progress-fill" style="width: {{ (analysis.overall_assessment.average_score * 100)|int if analysis.overall_assessment.average_score is defined else 0 }}%;"></div>
            </div>
            <small>{{ "%.1f"|format(analysis.overall_assessment.average_score * 100) if analysis.overall_assessment.average_score is defined else "0.0" }}% Average Score</small>
        </div>
        
        <h3>Reasoning:</h3>
        <ul>
        {% for reason in analysis.pass_fail_determination.reasoning %}
            <li>{{ reason }}</li>
        {% endfor %}
        </ul>
        
        <h2>🔍 Individual Test Results</h2>
        
        <div class="test-grid">
        {% for test_name, score in analysis.test_results_summary.individual_test_scores.items() %}
            <div class="test-card {{ 'test-pass' if score.status == 'PASS' else 'test-fail' }}">
                <h3>{{ test_name.replace('_', ' ').title() }}</h3>
                <div class="metric">Status: <strong>{{ score.status }}</strong></div>
                <div class="metric">Score: <strong>{{ "%.2f"|format(score.score) }}</strong></div>
                <div class="metric">Confidence: <strong>{{ score.confidence }}</strong></div>
            </div>
        {% endfor %}
        </div>
        
        <h2>💡 Key Findings</h2>
        {% for finding in analysis.key_findings %}
            <div class="finding">{{ finding }}</div>
        {% endfor %}
        
        <h2>🛠️ Recommendations</h2>
        {% for recommendation in analysis.recommendations %}
            <div class="recommendation">{{ recommendation }}</div>
        {% endfor %}
        
        <h2>📈 Test Execution Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Test Module</th>
                    <th>Execution Status</th>
                    <th>Result Status</th>
                    <th>Score</th>
                </tr>
            </thead>
            <tbody>
            {% for test_name, exec_status in analysis.test_results_summary.test_execution_status.items() %}
                <tr>
                    <td>{{ test_name.replace('_', ' ').title() }}</td>
                    <td>{{ exec_status.title() }}</td>
                    <td>
                    {% if test_name in analysis.test_results_summary.individual_test_scores %}
                        {{ analysis.test_results_summary.individual_test_scores[test_name].status }}
                    {% else %}
                        N/A
                    {% endif %}
                    </td>
                    <td>
                    {% if test_name in analysis.test_results_summary.individual_test_scores %}
                        {{ "%.2f"|format(analysis.test_results_summary.individual_test_scores[test_name].score) }}
                    {% else %}
                        N/A
                    {% endif %}
                    </td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
        
        {% if analysis.pass_fail_determination.critical_test_failures %}
        <h2>⚠️ Critical Test Failures</h2>
        <div style="background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <strong>The following critical tests failed:</strong>
            <ul>
            {% for failure in analysis.pass_fail_determination.critical_test_failures %}
                <li>{{ failure.replace('_', ' ').title() }}</li>
            {% endfor %}
            </ul>
            <p>These failures indicate fundamental issues with the memory mechanism that require immediate attention.</p>
        </div>
        {% endif %}
        
        <div class="metadata">
            <hr>
            <h3>Analysis Configuration</h3>
            <p><strong>Configuration:</strong> {{ analysis.analysis_metadata.configuration }}</p>
            <p><strong>Output Directory:</strong> {{ analysis.analysis_metadata.output_directory }}</p>
        </div>
        
    </div>
</body>
</html>
        """
    
    def _print_final_summary(self, consolidated: Dict, total_time: float):
        """Print final summary to console."""
        
        print("\n" + "=" * 70)
        print("🎯 COMPREHENSIVE MEMORY ANALYSIS COMPLETE")
        print("=" * 70)
        
        pass_fail = consolidated['pass_fail_determination']
        overall = consolidated['overall_assessment']
        
        # Status with color coding (if terminal supports it)
        status = pass_fail['final_determination']
        if status == 'PASS':
            print(f"✅ FINAL RESULT: {status} (Score: {pass_fail['numeric_score']:.2f}/1.0)")
        elif status == 'FAIL':
            print(f"❌ FINAL RESULT: {status} (Score: {pass_fail['numeric_score']:.2f}/1.0)")
        else:
            print(f"⚠️  FINAL RESULT: {status} (Score: {pass_fail['numeric_score']:.2f}/1.0)")
        
        print(f"📊 Overall Status: {overall['status']}")
        print(f"🎯 Tests Passed: {pass_fail['tests_passed']}")
        print(f"⏱️  Analysis Time: {total_time:.1f} seconds")
        print(f"🔍 Confidence: {pass_fail['confidence']}")
        
        print("\n📋 Key Findings:")
        for i, finding in enumerate(consolidated['key_findings'][:5], 1):
            print(f"  {i}. {finding}")
        
        print("\n💡 Top Recommendations:")
        for i, rec in enumerate(consolidated['recommendations'][:3], 1):
            print(f"  {i}. {rec}")
        
        if pass_fail.get('critical_test_failures'):
            print("\n⚠️  Critical Issues:")
            for failure in pass_fail['critical_test_failures']:
                print(f"  - {failure.replace('_', ' ').title()} test failed")
        
        print(f"\n📁 Detailed results: {self.output_dir}/comprehensive_memory_analysis_report.html")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Infini-Attention Memory Testing")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./comprehensive_memory_analysis",
                       help="Output directory for all analysis results")
    
    # Test selection
    parser.add_argument("--quick-test", action="store_true",
                       help="Run quick test (balance factors + memory debug + comparison only)")
    parser.add_argument("--skip-balance-factors", action="store_true",
                       help="Skip balance factor analysis")
    parser.add_argument("--skip-memory-debug", action="store_true",
                       help="Skip memory usage debugging")
    parser.add_argument("--skip-comparison", action="store_true",
                       help="Skip memory vs no-memory comparison")
    parser.add_argument("--skip-content-analysis", action="store_true",
                       help="Skip memory content analysis")
    parser.add_argument("--skip-progressive-test", action="store_true",
                       help="Skip progressive context testing")
    
    # Test configuration
    parser.add_argument("--debug-context-lengths", type=str, default="1024,2048,4096",
                       help="Context lengths for memory debugging")
    parser.add_argument("--comparison-context-lengths", type=str, default="1024,2048,4096",
                       help="Context lengths for memory comparison")
    parser.add_argument("--content-context-lengths", type=str, default="4096",
                       help="Context lengths for content analysis")
    parser.add_argument("--progressive-max-context", type=int, default=8192,
                       help="Maximum context length for progressive testing")
    
    # Sample sizes
    parser.add_argument("--debug-samples", type=int, default=3,
                       help="Number of samples for memory debugging")
    parser.add_argument("--comparison-samples", type=int, default=10,
                       help="Number of samples per context length for comparison")
    parser.add_argument("--progressive-samples", type=int, default=5,
                       help="Number of samples per context length for progressive testing")
    
    # Other options
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint path {checkpoint_path} does not exist")
        sys.exit(1)
    
    # Build configuration
    config = {
        'quick_test': args.quick_test,
        'run_balance_factors': not args.skip_balance_factors,
        'run_memory_debug': not args.skip_memory_debug,
        'run_comparison': not args.skip_comparison,
        'run_content_analysis': not args.skip_content_analysis and not args.quick_test,
        'run_progressive_test': not args.skip_progressive_test and not args.quick_test,
        'debug_context_lengths': args.debug_context_lengths,
        'comparison_context_lengths': args.comparison_context_lengths,
        'content_context_lengths': args.content_context_lengths,
        'progressive_max_context': args.progressive_max_context,
        'debug_samples': args.debug_samples,
        'comparison_samples': args.comparison_samples if not args.quick_test else 5,
        'progressive_samples': args.progressive_samples,
        'verbose': args.verbose
    }
    
    # Initialize and run comprehensive tester
    tester = ComprehensiveMemoryTester(args.checkpoint, args.output_dir)
    
    try:
        results = tester.run_comprehensive_analysis(config)
        
        # Exit with appropriate code
        final_result = results['pass_fail_determination']['final_determination']
        if final_result == 'PASS':
            sys.exit(0)
        elif final_result == 'CONDITIONAL_PASS':
            sys.exit(1)  # Warning exit code
        else:
            sys.exit(2)  # Failure exit code
    
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError during comprehensive analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()