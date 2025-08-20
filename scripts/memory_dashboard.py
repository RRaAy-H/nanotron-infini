#!/usr/bin/env python3
"""
Infini-Attention Memory Visualization Dashboard

This script creates an interactive web-based dashboard for visualizing
and monitoring infini-attention memory states in real-time or from
saved analysis results.

Usage:
    python scripts/memory_dashboard.py --checkpoint ./checkpoints/model/30000 --port 8080
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import threading
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, callback, ctx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc

# Import nanotron components
import sys
sys.path.append('src')
from nanotron import constants
from nanotron.config import get_config_from_file


class MemoryDashboard:
    """Interactive dashboard for memory visualization and monitoring."""
    
    def __init__(self, checkpoint_path: str = None, results_dir: str = None, port: int = 8050):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.results_dir = Path(results_dir) if results_dir else None
        self.port = port
        
        # Dashboard data
        self.dashboard_data = {
            'balance_factors': None,
            'memory_states': None,
            'performance_data': None,
            'analysis_results': {}
        }
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.setup_layout()
        self.setup_callbacks()
        
    def load_analysis_results(self):
        """Load analysis results from various sources."""
        
        print("Loading analysis results...")
        
        # Try to load from comprehensive analysis directory
        if self.results_dir and self.results_dir.exists():
            self._load_from_results_directory()
        
        # Try to load individual analysis files if no results directory
        if self.checkpoint_path and self.checkpoint_path.exists():
            self._load_from_checkpoint_directory()
        
        print("Analysis results loaded successfully")
    
    def _load_from_results_directory(self):
        """Load results from comprehensive analysis directory."""
        
        results_dir = self.results_dir
        
        # Load comprehensive report
        comp_report = results_dir / "comprehensive_memory_analysis_report.json"
        if comp_report.exists():
            with open(comp_report, 'r') as f:
                self.dashboard_data['analysis_results']['comprehensive'] = json.load(f)
        
        # Load balance factor analysis
        balance_dir = results_dir / "balance_factors"
        if balance_dir.exists():
            balance_report = balance_dir / "balance_factor_report.json"
            if balance_report.exists():
                with open(balance_report, 'r') as f:
                    self.dashboard_data['analysis_results']['balance_factors'] = json.load(f)
        
        # Load memory debugging results
        debug_dir = results_dir / "memory_debug"
        if debug_dir.exists():
            debug_files = list(debug_dir.glob("*_report.json"))
            if debug_files:
                with open(debug_files[0], 'r') as f:
                    self.dashboard_data['analysis_results']['memory_debug'] = json.load(f)
        
        # Load comparison results
        comparison_dir = results_dir / "memory_comparison"
        if comparison_dir.exists():
            comparison_report = comparison_dir / "comparison_report.json"
            if comparison_report.exists():
                with open(comparison_report, 'r') as f:
                    self.dashboard_data['analysis_results']['comparison'] = json.load(f)
        
        # Load progressive test results
        progressive_dir = results_dir / "progressive_testing"
        if progressive_dir.exists():
            progressive_report = progressive_dir / "progressive_context_report.json"
            if progressive_report.exists():
                with open(progressive_report, 'r') as f:
                    self.dashboard_data['analysis_results']['progressive'] = json.load(f)
    
    def _load_from_checkpoint_directory(self):
        """Load basic data from checkpoint directory."""
        
        # Try to load balance factors directly from checkpoint
        try:
            self._load_balance_factors_from_checkpoint()
        except Exception as e:
            print(f"Could not load balance factors from checkpoint: {e}")
    
    def _load_balance_factors_from_checkpoint(self):
        """Load balance factors directly from checkpoint."""
        
        decoder_path = self.checkpoint_path / "model" / "model" / "decoder"
        if not decoder_path.exists():
            return
        
        # Load balance factors
        import safetensors
        from safetensors import safe_open
        
        layer_dirs = sorted([d for d in decoder_path.iterdir() if d.is_dir() and d.name.isdigit()])
        merged_tensors = []
        
        for layer_idx in range(len(layer_dirs)):
            layer_path = decoder_path / str(layer_idx) / "pp_block" / "attn"
            balance_files = list(layer_path.glob("model_balance_factors_*.safetensors"))
            
            if balance_files:
                layer_tensors = []
                for file_path in sorted(balance_files):
                    try:
                        tensor_file = safe_open(file_path, framework="pt", device="cpu")
                        tensor_data = tensor_file.get_tensor("data").to(torch.float32).numpy()
                        layer_tensors.append(tensor_data)
                    except Exception:
                        continue
                
                if layer_tensors:
                    merged_tensor = np.concatenate(layer_tensors)
                    merged_tensors.append(merged_tensor)
        
        if merged_tensors:
            balance_factors = np.array(merged_tensors)
            global_weights = 1 / (1 + np.exp(-balance_factors))  # sigmoid
            
            self.dashboard_data['balance_factors'] = {
                'raw_factors': balance_factors,
                'global_weights': global_weights,
                'shape': balance_factors.shape
            }
    
    def setup_layout(self):
        """Setup the dashboard layout."""
        
        self.app.layout = dbc.Container([
            
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("🧠 Infini-Attention Memory Dashboard", 
                           className="text-primary mb-4"),
                    html.P("Interactive visualization and analysis of memory mechanism performance", 
                           className="lead text-muted")
                ])
            ], className="mb-4"),
            
            # Control Panel
            dbc.Card([
                dbc.CardHeader("📊 Dashboard Controls"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Analysis Type:"),
                            dcc.Dropdown(
                                id='analysis-type-dropdown',
                                options=[
                                    {'label': 'Overview', 'value': 'overview'},
                                    {'label': 'Balance Factors', 'value': 'balance_factors'},
                                    {'label': 'Memory Usage', 'value': 'memory_usage'},
                                    {'label': 'Performance Comparison', 'value': 'comparison'},
                                    {'label': 'Progressive Testing', 'value': 'progressive'},
                                    {'label': 'Content Analysis', 'value': 'content'}
                                ],
                                value='overview'
                            )
                        ], width=4),
                        
                        dbc.Col([
                            html.Label("Update Mode:"),
                            dcc.Dropdown(
                                id='update-mode-dropdown',
                                options=[
                                    {'label': 'Static Analysis', 'value': 'static'},
                                    {'label': 'Auto Refresh', 'value': 'refresh'}
                                ],
                                value='static'
                            )
                        ], width=4),
                        
                        dbc.Col([
                            html.Label("Export Options:"),
                            dbc.ButtonGroup([
                                dbc.Button("📊 Export Charts", id="export-charts-btn", size="sm"),
                                dbc.Button("📄 Generate Report", id="generate-report-btn", size="sm")
                            ])
                        ], width=4)
                    ])
                ])
            ], className="mb-4"),
            
            # Status Cards
            dbc.Row(id="status-cards-row", className="mb-4"),
            
            # Main Visualization Area
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        id="main-loading",
                        type="default",
                        children=html.Div(id="main-visualization-area")
                    )
                ], width=12)
            ]),
            
            # Detailed Analysis Area
            dbc.Row([
                dbc.Col([
                    dcc.Tabs(id="detailed-tabs", value="details-tab", children=[
                        dcc.Tab(label="📈 Detailed Metrics", value="details-tab"),
                        dcc.Tab(label="🔍 Raw Data", value="raw-data-tab"),
                        dcc.Tab(label="💡 Insights", value="insights-tab"),
                        dcc.Tab(label="🛠️ Recommendations", value="recommendations-tab")
                    ]),
                    html.Div(id="detailed-content")
                ], width=12)
            ], className="mt-4"),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=10*1000,  # Update every 10 seconds
                n_intervals=0,
                disabled=True
            ),
            
            # Hidden div to store data
            html.Div(id='dashboard-data-store', style={'display': 'none'})
            
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            Output('dashboard-data-store', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_data_store(n_intervals):
            """Update data store with latest analysis results."""
            return json.dumps(self.dashboard_data)
        
        @self.app.callback(
            [Output('status-cards-row', 'children'),
             Output('main-visualization-area', 'children')],
            [Input('analysis-type-dropdown', 'value'),
             Input('dashboard-data-store', 'children')]
        )
        def update_main_content(analysis_type, dashboard_data_json):
            """Update main dashboard content based on selected analysis type."""
            
            if dashboard_data_json:
                dashboard_data = json.loads(dashboard_data_json)
            else:
                dashboard_data = self.dashboard_data
            
            status_cards = self._create_status_cards(dashboard_data)
            main_viz = self._create_main_visualization(analysis_type, dashboard_data)
            
            return status_cards, main_viz
        
        @self.app.callback(
            Output('detailed-content', 'children'),
            [Input('detailed-tabs', 'value'),
             Input('analysis-type-dropdown', 'value'),
             Input('dashboard-data-store', 'children')]
        )
        def update_detailed_content(tab_value, analysis_type, dashboard_data_json):
            """Update detailed content area."""
            
            if dashboard_data_json:
                dashboard_data = json.loads(dashboard_data_json)
            else:
                dashboard_data = self.dashboard_data
            
            return self._create_detailed_content(tab_value, analysis_type, dashboard_data)
        
        @self.app.callback(
            Output('interval-component', 'disabled'),
            Input('update-mode-dropdown', 'value')
        )
        def toggle_auto_refresh(update_mode):
            """Toggle auto refresh based on update mode."""
            return update_mode == 'static'
    
    def _create_status_cards(self, data: Dict) -> List[dbc.Col]:
        """Create status cards showing key metrics."""
        
        cards = []
        
        # Overall status card
        if 'analysis_results' in data and 'comprehensive' in data['analysis_results']:
            comp_data = data['analysis_results']['comprehensive']
            if 'pass_fail_determination' in comp_data:
                pass_fail = comp_data['pass_fail_determination']
                
                status = pass_fail['final_determination']
                score = pass_fail.get('numeric_score', 0)
                
                if status == 'PASS':
                    card_color = "success"
                    icon = "✅"
                elif status == 'FAIL':
                    card_color = "danger"
                    icon = "❌"
                else:
                    card_color = "warning"
                    icon = "⚠️"
                
                cards.append(
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(f"{icon} {status}", className=f"text-{card_color}"),
                                html.P(f"Score: {score:.2f}/1.0"),
                                html.P(f"Confidence: {pass_fail.get('confidence', 'Unknown')}")
                            ])
                        ], color=card_color, outline=True)
                    ], width=3)
                )
        
        # Balance factors status
        if 'analysis_results' in data and 'balance_factors' in data['analysis_results']:
            bf_data = data['analysis_results']['balance_factors']
            if 'conclusion' in bf_data:
                conclusion = bf_data['conclusion']
                working = conclusion.get('memory_mechanism_learned', False)
                
                cards.append(
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(f"🎯 Balance Factors", className="text-info"),
                                html.P("✅ Learning" if working else "❌ Not Learning"),
                                html.P(f"Confidence: {conclusion.get('confidence', 'Unknown')}")
                            ])
                        ], color="info", outline=True)
                    ], width=3)
                )
        
        # Memory usage status
        if 'analysis_results' in data and 'memory_debug' in data['analysis_results']:
            debug_data = data['analysis_results']['memory_debug']
            if 'overall_analysis' in debug_data:
                analysis = debug_data['overall_analysis']
                effectiveness = analysis.get('effectiveness_rating', 'UNKNOWN')
                
                cards.append(
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(f"🔧 Memory Usage", className="text-primary"),
                                html.P(effectiveness.replace('_', ' ').title()),
                                html.P(f"Activation: {analysis.get('memory_activation_rate', 0):.1%}")
                            ])
                        ], color="primary", outline=True)
                    ], width=3)
                )
        
        # Performance comparison status
        if 'analysis_results' in data and 'comparison' in data['analysis_results']:
            comp_data = data['analysis_results']['comparison']
            if 'conclusion' in comp_data:
                conclusion = comp_data['conclusion']
                effectiveness = conclusion.get('memory_effectiveness', 'UNKNOWN')
                
                cards.append(
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(f"📊 Performance", className="text-secondary"),
                                html.P(effectiveness.replace('_', ' ').title()),
                                html.P(f"Improvement: {conclusion.get('improvement_rate', 0):.1%}")
                            ])
                        ], color="secondary", outline=True)
                    ], width=3)
                )
        
        # Fill remaining cards if needed
        while len(cards) < 4:
            cards.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📋 No Data", className="text-muted"),
                            html.P("Analysis not available"),
                            html.P("Run comprehensive test")
                        ])
                    ], color="light", outline=True)
                ], width=3)
            )
        
        return cards
    
    def _create_main_visualization(self, analysis_type: str, data: Dict) -> html.Div:
        """Create main visualization based on analysis type."""
        
        if analysis_type == 'overview':
            return self._create_overview_viz(data)
        elif analysis_type == 'balance_factors':
            return self._create_balance_factors_viz(data)
        elif analysis_type == 'memory_usage':
            return self._create_memory_usage_viz(data)
        elif analysis_type == 'comparison':
            return self._create_comparison_viz(data)
        elif analysis_type == 'progressive':
            return self._create_progressive_viz(data)
        elif analysis_type == 'content':
            return self._create_content_viz(data)
        else:
            return html.Div([
                dbc.Alert("Selected analysis type not available", color="warning")
            ])
    
    def _create_overview_viz(self, data: Dict) -> html.Div:
        """Create overview visualization."""
        
        figures = []
        
        # Overall performance summary
        if 'analysis_results' in data and 'comprehensive' in data['analysis_results']:
            comp_data = data['analysis_results']['comprehensive']
            
            if 'test_results_summary' in comp_data and 'individual_test_scores' in comp_data['test_results_summary']:
                scores = comp_data['test_results_summary']['individual_test_scores']
                
                # Create radar chart of test scores
                categories = [name.replace('_', ' ').title() for name in scores.keys()]
                values = [score['score'] for score in scores.values()]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Test Scores'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    title="Overall Memory Mechanism Performance",
                    height=400
                )
                
                figures.append(dcc.Graph(figure=fig))
        
        # Test execution status
        if 'analysis_results' in data and 'comprehensive' in data['analysis_results']:
            comp_data = data['analysis_results']['comprehensive']
            
            if 'test_results_summary' in comp_data:
                test_summary = comp_data['test_results_summary']
                
                if 'test_execution_status' in test_summary:
                    exec_status = test_summary['test_execution_status']
                    
                    # Create execution status pie chart
                    status_counts = {}
                    for status in exec_status.values():
                        status_counts[status] = status_counts.get(status, 0) + 1
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=list(status_counts.keys()),
                        values=list(status_counts.values()),
                        title="Test Execution Status"
                    )])
                    
                    fig.update_layout(height=400)
                    figures.append(dcc.Graph(figure=fig))
        
        if not figures:
            return html.Div([
                dbc.Alert("No comprehensive analysis results available. Run comprehensive test first.", 
                         color="info")
            ])
        
        return html.Div(figures)
    
    def _create_balance_factors_viz(self, data: Dict) -> html.Div:
        """Create balance factors visualization."""
        
        figures = []
        
        # Try to load balance factors from analysis results
        balance_data = None
        if 'analysis_results' in data and 'balance_factors' in data['analysis_results']:
            bf_analysis = data['analysis_results']['balance_factors']
            if 'analysis' in bf_analysis and 'basic_statistics' in bf_analysis['analysis']:
                balance_data = bf_analysis
        
        # Or from direct checkpoint loading
        elif 'balance_factors' in data and data['balance_factors']:
            balance_data = data['balance_factors']
        
        if balance_data:
            # Distribution plot
            if 'analysis' in balance_data and 'binned_distribution' in balance_data['analysis']:
                dist_data = balance_data['analysis']['binned_distribution']
                
                fig = go.Figure(data=[
                    go.Bar(x=dist_data['bins'], y=dist_data['percentages'])
                ])
                fig.update_layout(
                    title="Balance Factor Distribution",
                    xaxis_title="Balance Factor Range",
                    yaxis_title="Percentage",
                    height=400
                )
                figures.append(dcc.Graph(figure=fig))
            
            # Heatmap (if raw data available)
            if 'global_weights' in balance_data:
                weights = np.array(balance_data['global_weights'])
                
                fig = go.Figure(data=go.Heatmap(
                    z=weights,
                    colorscale='RdYlBu_r',
                    zmin=0,
                    zmax=1
                ))
                fig.update_layout(
                    title="Balance Factors Heatmap",
                    xaxis_title="Attention Head",
                    yaxis_title="Layer",
                    height=500
                )
                figures.append(dcc.Graph(figure=fig))
        
        if not figures:
            return html.Div([
                dbc.Alert("No balance factor analysis results available.", color="info")
            ])
        
        return html.Div(figures)
    
    def _create_memory_usage_viz(self, data: Dict) -> html.Div:
        """Create memory usage visualization."""
        
        figures = []
        
        if ('analysis_results' in data and 
            'memory_debug' in data['analysis_results'] and
            'results_by_context' in data['analysis_results']['memory_debug']):
            
            debug_data = data['analysis_results']['memory_debug']['results_by_context']
            
            # Memory activation by context length
            context_lengths = []
            activation_rates = []
            
            for context_length, context_data in debug_data.items():
                context_lengths.append(int(context_length))
                
                # Calculate average activation rate for this context
                samples = context_data.get('samples', [])
                if samples:
                    rates = [s.get('memory_usage', {}).get('memory_retrievals', 0) / 
                           max(s.get('memory_usage', {}).get('total_retrievals', 1), 1) 
                           for s in samples if 'memory_usage' in s]
                    if rates:
                        activation_rates.append(np.mean(rates))
                    else:
                        activation_rates.append(0)
                else:
                    activation_rates.append(0)
            
            if context_lengths and activation_rates:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=context_lengths,
                    y=activation_rates,
                    mode='lines+markers',
                    name='Memory Activation Rate'
                ))
                fig.update_layout(
                    title="Memory Activation by Context Length",
                    xaxis_title="Context Length (tokens)",
                    yaxis_title="Activation Rate",
                    height=400
                )
                figures.append(dcc.Graph(figure=fig))
        
        if not figures:
            return html.Div([
                dbc.Alert("No memory usage debug results available.", color="info")
            ])
        
        return html.Div(figures)
    
    def _create_comparison_viz(self, data: Dict) -> html.Div:
        """Create performance comparison visualization."""
        
        figures = []
        
        if ('analysis_results' in data and 
            'comparison' in data['analysis_results'] and
            'detailed_results' in data['analysis_results']['comparison']):
            
            results = data['analysis_results']['comparison']['detailed_results']
            
            context_lengths = []
            with_memory_accs = []
            without_memory_accs = []
            
            for result in results:
                context_lengths.append(result['context_length'])
                with_memory_accs.append(result['statistical_test']['with_memory_mean'])
                without_memory_accs.append(result['statistical_test']['without_memory_mean'])
            
            if context_lengths:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=context_lengths,
                    y=with_memory_accs,
                    mode='lines+markers',
                    name='With Memory',
                    line=dict(color='blue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=context_lengths,
                    y=without_memory_accs,
                    mode='lines+markers',
                    name='Without Memory',
                    line=dict(color='red')
                ))
                
                fig.update_layout(
                    title="Performance: Memory vs No Memory",
                    xaxis_title="Context Length (tokens)",
                    yaxis_title="Accuracy",
                    height=400
                )
                figures.append(dcc.Graph(figure=fig))
                
                # Effect sizes
                effect_sizes = [result['effect_size'] for result in results]
                
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=context_lengths,
                    y=effect_sizes,
                    name='Effect Size'
                ))
                fig2.update_layout(
                    title="Memory Effect Size by Context Length",
                    xaxis_title="Context Length (tokens)",
                    yaxis_title="Effect Size (Cohen's d)",
                    height=400
                )
                figures.append(dcc.Graph(figure=fig2))
        
        if not figures:
            return html.Div([
                dbc.Alert("No performance comparison results available.", color="info")
            ])
        
        return html.Div(figures)
    
    def _create_progressive_viz(self, data: Dict) -> html.Div:
        """Create progressive testing visualization."""
        
        figures = []
        
        if ('analysis_results' in data and 
            'progressive' in data['analysis_results'] and
            'detailed_results' in data['analysis_results']['progressive']):
            
            prog_data = data['analysis_results']['progressive']['detailed_results']
            
            if 'results_with_memory' in prog_data and prog_data['results_with_memory']:
                mem_results = prog_data['results_with_memory']
                
                context_lengths = sorted([int(k) for k in mem_results.keys()])
                accuracies = [mem_results[str(cl)]['statistics']['success_rate'] for cl in context_lengths]
                stds = [mem_results[str(cl)]['statistics']['accuracy_std'] for cl in context_lengths]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=context_lengths,
                    y=accuracies,
                    error_y=dict(type='data', array=stds, visible=True),
                    mode='lines+markers',
                    name='Accuracy (with Memory)'
                ))
                
                fig.update_layout(
                    title="Progressive Context Length Performance",
                    xaxis_title="Context Length (tokens)",
                    yaxis_title="Accuracy",
                    height=400
                )
                figures.append(dcc.Graph(figure=fig))
        
        if not figures:
            return html.Div([
                dbc.Alert("No progressive testing results available.", color="info")
            ])
        
        return html.Div(figures)
    
    def _create_content_viz(self, data: Dict) -> html.Div:
        """Create content analysis visualization."""
        
        figures = []
        
        if ('analysis_results' in data and 
            'content_analysis' in data['analysis_results']):
            # This would need the actual content analysis structure
            pass
        
        if not figures:
            return html.Div([
                dbc.Alert("No content analysis results available.", color="info")
            ])
        
        return html.Div(figures)
    
    def _create_detailed_content(self, tab_value: str, analysis_type: str, data: Dict) -> html.Div:
        """Create detailed content for tabs."""
        
        if tab_value == "details-tab":
            return self._create_detailed_metrics(analysis_type, data)
        elif tab_value == "raw-data-tab":
            return self._create_raw_data_view(analysis_type, data)
        elif tab_value == "insights-tab":
            return self._create_insights_view(analysis_type, data)
        elif tab_value == "recommendations-tab":
            return self._create_recommendations_view(analysis_type, data)
        else:
            return html.Div("Content not available")
    
    def _create_detailed_metrics(self, analysis_type: str, data: Dict) -> html.Div:
        """Create detailed metrics view."""
        
        return html.Div([
            html.H4("📊 Detailed Metrics"),
            html.P("Detailed metrics will be displayed here based on selected analysis type."),
            html.Pre(json.dumps(data.get('analysis_results', {}), indent=2)[:1000] + "...")
        ])
    
    def _create_raw_data_view(self, analysis_type: str, data: Dict) -> html.Div:
        """Create raw data view."""
        
        return html.Div([
            html.H4("🔍 Raw Data"),
            html.P("Raw analysis data for debugging and detailed inspection."),
            html.Pre(json.dumps(data, indent=2)[:2000] + "...", 
                    style={'background-color': '#f8f9fa', 'padding': '10px', 
                           'border-radius': '5px', 'font-size': '12px'})
        ])
    
    def _create_insights_view(self, analysis_type: str, data: Dict) -> html.Div:
        """Create insights view."""
        
        insights = []
        
        if 'analysis_results' in data and 'comprehensive' in data['analysis_results']:
            comp_data = data['analysis_results']['comprehensive']
            if 'key_findings' in comp_data:
                insights = comp_data['key_findings']
        
        if insights:
            insight_cards = []
            for i, insight in enumerate(insights):
                insight_cards.append(
                    dbc.Alert(f"{i+1}. {insight}", color="info", className="mb-2")
                )
            
            return html.Div([
                html.H4("💡 Key Insights"),
                html.Div(insight_cards)
            ])
        else:
            return html.Div([
                html.H4("💡 Key Insights"),
                dbc.Alert("No insights available. Run comprehensive analysis first.", color="warning")
            ])
    
    def _create_recommendations_view(self, analysis_type: str, data: Dict) -> html.Div:
        """Create recommendations view."""
        
        recommendations = []
        
        if 'analysis_results' in data and 'comprehensive' in data['analysis_results']:
            comp_data = data['analysis_results']['comprehensive']
            if 'recommendations' in comp_data:
                recommendations = comp_data['recommendations']
        
        if recommendations:
            rec_cards = []
            for i, rec in enumerate(recommendations):
                rec_cards.append(
                    dbc.Alert(f"{i+1}. {rec}", color="warning", className="mb-2")
                )
            
            return html.Div([
                html.H4("🛠️ Recommendations"),
                html.Div(rec_cards)
            ])
        else:
            return html.Div([
                html.H4("🛠️ Recommendations"),
                dbc.Alert("No recommendations available. Run comprehensive analysis first.", color="info")
            ])
    
    def run(self):
        """Run the dashboard server."""
        
        print(f"Starting Memory Dashboard...")
        print(f"Dashboard will be available at: http://localhost:{self.port}")
        
        # Load analysis results
        self.load_analysis_results()
        
        # Start dashboard
        try:
            # Try to open browser automatically
            threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{self.port}")).start()
        except Exception:
            pass  # Browser opening failed, but dashboard will still work
        
        self.app.run_server(debug=False, host='0.0.0.0', port=self.port)


def main():
    parser = argparse.ArgumentParser(description="Infini-Attention Memory Visualization Dashboard")
    parser.add_argument("--checkpoint", type=str,
                       help="Path to model checkpoint (for loading balance factors)")
    parser.add_argument("--results-dir", type=str,
                       help="Path to analysis results directory")
    parser.add_argument("--port", type=int, default=8050,
                       help="Port for dashboard server")
    
    args = parser.parse_args()
    
    if not args.checkpoint and not args.results_dir:
        print("Error: Must provide either --checkpoint or --results-dir")
        print("Examples:")
        print("  python memory_dashboard.py --checkpoint ./checkpoints/model/30000")
        print("  python memory_dashboard.py --results-dir ./comprehensive_memory_analysis")
        sys.exit(1)
    
    # Validate paths
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint path {checkpoint_path} does not exist")
            sys.exit(1)
    
    if args.results_dir:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"Error: Results directory {results_dir} does not exist")
            sys.exit(1)
    
    print("Infini-Attention Memory Dashboard")
    print("=" * 50)
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    if args.results_dir:
        print(f"Results Directory: {args.results_dir}")
    print(f"Port: {args.port}")
    print()
    
    # Initialize and run dashboard
    dashboard = MemoryDashboard(
        checkpoint_path=args.checkpoint,
        results_dir=args.results_dir,
        port=args.port
    )
    
    try:
        dashboard.run()
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Dashboard error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()