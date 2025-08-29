#!/usr/bin/env python3
"""
Formal Plot Style Configuration for Academic Papers

This module provides consistent styling for all plots in the Infini-Attention
analysis suite, ensuring publication-ready figures with professional formatting.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Academic color palette - colorblind friendly and suitable for papers
ACADEMIC_COLORS = {
    'primary_blue': '#1f77b4',      # Professional blue
    'primary_red': '#d62728',       # Professional red  
    'primary_green': '#2ca02c',     # Professional green
    'primary_orange': '#ff7f0e',    # Professional orange
    'primary_purple': '#9467bd',    # Professional purple
    'primary_brown': '#8c564b',     # Professional brown
    'primary_pink': '#e377c2',      # Professional pink
    'primary_gray': '#7f7f7f',      # Professional gray
    'primary_olive': '#bcbd22',     # Professional olive
    'primary_cyan': '#17becf',      # Professional cyan
    
    # Semantic colors
    'memory_enabled': '#1f77b4',    # Blue for memory enabled
    'memory_disabled': '#d62728',   # Red for memory disabled
    'improvement': '#2ca02c',       # Green for improvements
    'degradation': '#d62728',       # Red for degradations
    'neutral': '#7f7f7f',          # Gray for neutral
    'significant': '#ff7f0e',       # Orange for significant results
    'background': '#f8f9fa',        # Light background
    'grid': '#e9ecef',             # Light grid color
}

# Font configuration for academic papers
ACADEMIC_FONTS = {
    'family': ['Times New Roman', 'Times', 'serif'],
    'size': {
        'small': 10,
        'medium': 12,
        'large': 14,
        'xlarge': 16,
        'title': 18,
        'figure_title': 16,
    }
}

def setup_matplotlib_style():
    """Setup matplotlib with academic paper styling."""
    
    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Configure matplotlib parameters
    mpl.rcParams.update({
        # Font settings
        'font.family': ACADEMIC_FONTS['family'],
        'font.size': ACADEMIC_FONTS['size']['medium'],
        'axes.titlesize': ACADEMIC_FONTS['size']['large'],
        'axes.labelsize': ACADEMIC_FONTS['size']['medium'],
        'xtick.labelsize': ACADEMIC_FONTS['size']['small'],
        'ytick.labelsize': ACADEMIC_FONTS['size']['small'],
        'legend.fontsize': ACADEMIC_FONTS['size']['small'],
        'figure.titlesize': ACADEMIC_FONTS['size']['title'],
        
        # Figure settings
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        # Note: savefig.bbox removed - not a valid rcParam
        'savefig.pad_inches': 0.1,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        
        # Axes settings
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.prop_cycle': mpl.cycler('color', [
            ACADEMIC_COLORS['primary_blue'],
            ACADEMIC_COLORS['primary_red'],
            ACADEMIC_COLORS['primary_green'],
            ACADEMIC_COLORS['primary_orange'],
            ACADEMIC_COLORS['primary_purple'],
            ACADEMIC_COLORS['primary_brown'],
            ACADEMIC_COLORS['primary_pink'],
            ACADEMIC_COLORS['primary_gray'],
        ]),
        
        # Grid settings
        'grid.color': ACADEMIC_COLORS['grid'],
        'grid.linestyle': '-',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.7,
        
        # Line settings
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'lines.markeredgewidth': 1,
        
        # Legend settings
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True,
        'legend.framealpha': 0.9,
        'legend.facecolor': 'white',
        'legend.edgecolor': 'gray',
        
        # Tick settings
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        
        # Error bar settings
        'errorbar.capsize': 4,
    })

def get_plotly_template():
    """Get a custom Plotly template for academic papers."""
    
    template = {
        'layout': {
            'font': {
                'family': ', '.join(ACADEMIC_FONTS['family']),
                'size': ACADEMIC_FONTS['size']['medium'],
                'color': 'black'
            },
            'title': {
                'font': {
                    'size': ACADEMIC_FONTS['size']['figure_title'],
                    'family': ', '.join(ACADEMIC_FONTS['family']),
                    'color': 'black'
                },
                'x': 0.5,
                'xanchor': 'center'
            },
            'paper_bgcolor': 'white',
            'plot_bgcolor': 'white',
            'colorway': [
                ACADEMIC_COLORS['primary_blue'],
                ACADEMIC_COLORS['primary_red'],
                ACADEMIC_COLORS['primary_green'],
                ACADEMIC_COLORS['primary_orange'],
                ACADEMIC_COLORS['primary_purple'],
                ACADEMIC_COLORS['primary_brown'],
                ACADEMIC_COLORS['primary_pink'],
                ACADEMIC_COLORS['primary_gray'],
            ],
            'xaxis': {
                'showgrid': True,
                'gridwidth': 1,
                'gridcolor': ACADEMIC_COLORS['grid'],
                'showline': True,
                'linewidth': 1.2,
                'linecolor': 'black',
                'ticks': 'inside',
                'tickfont': {
                    'size': ACADEMIC_FONTS['size']['small'],
                    'family': ', '.join(ACADEMIC_FONTS['family'])
                },
                'title': {
                    'font': {
                        'size': ACADEMIC_FONTS['size']['medium'],
                        'family': ', '.join(ACADEMIC_FONTS['family'])
                    }
                }
            },
            'yaxis': {
                'showgrid': True,
                'gridwidth': 1,
                'gridcolor': ACADEMIC_COLORS['grid'],
                'showline': True,
                'linewidth': 1.2,
                'linecolor': 'black',
                'ticks': 'inside',
                'tickfont': {
                    'size': ACADEMIC_FONTS['size']['small'],
                    'family': ', '.join(ACADEMIC_FONTS['family'])
                },
                'title': {
                    'font': {
                        'size': ACADEMIC_FONTS['size']['medium'],
                        'family': ', '.join(ACADEMIC_FONTS['family'])
                    }
                }
            },
            'legend': {
                'font': {
                    'size': ACADEMIC_FONTS['size']['small'],
                    'family': ', '.join(ACADEMIC_FONTS['family'])
                },
                'bgcolor': 'rgba(255,255,255,0.9)',
                'bordercolor': 'gray',
                'borderwidth': 1
            },
            'margin': {
                'l': 80,
                'r': 50,
                't': 80,
                'b': 60
            }
        }
    }
    
    return template

def setup_plotly_style():
    """Setup Plotly with academic paper styling."""
    
    # Register the custom template
    pio.templates["academic"] = get_plotly_template()
    pio.templates.default = "academic"

def save_plotly_figure(fig, output_path: Path, html_filename: str, vector_filename: str, 
                      width: int = 1200, height: int = 800, 
                      vector_format: str = 'pdf', include_png: bool = False):
    """
    Save a Plotly figure as HTML and vector format (PDF/EPS) with consistent formatting.
    
    Args:
        fig: Plotly figure object
        output_path: Directory to save files
        html_filename: Name for HTML file (without extension)
        vector_filename: Name for vector file (without extension) 
        width: Figure width in pixels
        height: Figure height in pixels
        vector_format: Vector format ('pdf', 'eps', or 'svg')
        include_png: Also save PNG version for compatibility
    """
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Update figure layout for publication quality
    fig.update_layout(
        width=width,
        height=height,
        font=dict(
            family=', '.join(ACADEMIC_FONTS['family']),
            size=ACADEMIC_FONTS['size']['medium'],
            color='black'
        ),
        title=dict(
            font=dict(
                size=ACADEMIC_FONTS['size']['figure_title'],
                family=', '.join(ACADEMIC_FONTS['family']),
                color='black'
            ),
            x=0.5,
            xanchor='center'
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=80, r=50, t=80, b=60)
    )
    
    saved_files = []
    
    # Save HTML version
    html_path = output_path / f"{html_filename}.html"
    fig.write_html(str(html_path))
    saved_files.append(str(html_path))
    
    # Save vector version (PDF/EPS/SVG)
    try:
        vector_path = output_path / f"{vector_filename}.{vector_format.lower()}"
        
        if vector_format.lower() == 'pdf':
            fig.write_image(str(vector_path), format='pdf', width=width, height=height)
        elif vector_format.lower() == 'eps':
            fig.write_image(str(vector_path), format='eps', width=width, height=height)
        elif vector_format.lower() == 'svg':
            fig.write_image(str(vector_path), format='svg', width=width, height=height)
        else:
            raise ValueError(f"Unsupported vector format: {vector_format}")
            
        saved_files.append(str(vector_path))
        print(f"  Saved vector plot: {vector_path.name}")
        
    except Exception as e:
        print(f"  Warning: Could not save {vector_format.upper()} format: {e}")
        print(f"  This may be due to missing dependencies (kaleido for PDF/EPS)")
        
        # Fallback to PNG if vector format fails
        png_path = output_path / f"{vector_filename}.png"
        try:
            fig.write_image(str(png_path), width=width, height=height, scale=2)
            saved_files.append(str(png_path))
            print(f"  Fallback: Saved PNG format: {png_path.name}")
        except Exception as png_error:
            print(f"  Warning: PNG fallback also failed: {png_error}")
    
    # Optionally save PNG version for compatibility
    if include_png:
        try:
            png_path = output_path / f"{vector_filename}.png"
            fig.write_image(str(png_path), width=width, height=height, scale=2)
            saved_files.append(str(png_path))
            print(f"  Also saved PNG version: {png_path.name}")
        except Exception as e:
            print(f"  Warning: Could not save PNG version: {e}")
    
    return saved_files

def save_matplotlib_figure(fig, output_path: Path, filename: str, 
                          figsize: tuple = (12, 8), vector_format: str = 'pdf',
                          include_png: bool = False, dpi: int = 300):
    """
    Save a matplotlib figure with publication quality vector format.
    
    Args:
        fig: Matplotlib figure object
        output_path: Directory to save file
        filename: Name for output file (without extension)
        figsize: Figure size in inches
        vector_format: Vector format ('pdf', 'eps', or 'svg')
        include_png: Also save PNG version for compatibility
        dpi: Dots per inch for raster formats
    """
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set figure size
    fig.set_size_inches(figsize)
    
    saved_files = []
    
    # Save vector format (primary)
    try:
        vector_path = output_path / f"{filename}.{vector_format.lower()}"
        
        # Configure save parameters based on format
        save_kwargs = {
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none',
            'pad_inches': 0.1
        }
        
        if vector_format.lower() == 'pdf':
            save_kwargs['format'] = 'pdf'
            save_kwargs['metadata'] = {
                'Title': filename.replace('_', ' ').title(),
                'Creator': 'Infini-Attention Analysis Suite',
                'Producer': 'matplotlib'
            }
        elif vector_format.lower() == 'eps':
            save_kwargs['format'] = 'eps'
        elif vector_format.lower() == 'svg':
            save_kwargs['format'] = 'svg'
        else:
            raise ValueError(f"Unsupported vector format: {vector_format}")
        
        fig.savefig(str(vector_path), **save_kwargs)
        saved_files.append(str(vector_path))
        print(f"  Saved vector plot: {vector_path.name}")
        
    except Exception as e:
        print(f"  Warning: Could not save {vector_format.upper()} format: {e}")
        
        # Fallback to PNG if vector format fails
        png_path = output_path / f"{filename}.png"
        try:
            fig.savefig(
                str(png_path),
                dpi=dpi,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                pad_inches=0.1,
                format='png'
            )
            saved_files.append(str(png_path))
            print(f"  Fallback: Saved PNG format: {png_path.name}")
        except Exception as png_error:
            print(f"  Warning: PNG fallback also failed: {png_error}")
    
    # Optionally save PNG version for compatibility
    if include_png:
        try:
            png_path = output_path / f"{filename}.png"
            fig.savefig(
                str(png_path),
                dpi=dpi,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                pad_inches=0.1,
                format='png'
            )
            saved_files.append(str(png_path))
            print(f"  Also saved PNG version: {png_path.name}")
        except Exception as e:
            print(f"  Warning: Could not save PNG version: {e}")
    
    return saved_files[0] if saved_files else None

def create_comparison_colors(values: List[float], threshold: float = 0.0) -> List[str]:
    """
    Create color mapping for comparison values (e.g., improvements vs degradations).
    
    Args:
        values: List of numeric values to color-code
        threshold: Threshold above which values are considered positive
    
    Returns:
        List of color strings
    """
    
    colors = []
    for value in values:
        if value > threshold:
            colors.append(ACADEMIC_COLORS['improvement'])
        elif value < threshold:
            colors.append(ACADEMIC_COLORS['degradation'])
        else:
            colors.append(ACADEMIC_COLORS['neutral'])
    
    return colors

def create_significance_colors(p_values: List[float]) -> List[str]:
    """
    Create color mapping for statistical significance levels.
    
    Args:
        p_values: List of p-values
        
    Returns:
        List of color strings
    """
    
    colors = []
    for p in p_values:
        if p < 0.001:
            colors.append('#006400')  # Dark green - highly significant
        elif p < 0.01:
            colors.append(ACADEMIC_COLORS['improvement'])  # Green - very significant
        elif p < 0.05:
            colors.append(ACADEMIC_COLORS['significant'])  # Orange - significant
        else:
            colors.append(ACADEMIC_COLORS['degradation'])  # Red - not significant
    
    return colors

def create_effect_size_colors(effect_sizes: List[float]) -> List[str]:
    """
    Create color mapping for effect sizes (Cohen's d).
    
    Args:
        effect_sizes: List of effect size values
        
    Returns:
        List of color strings
    """
    
    colors = []
    for es in effect_sizes:
        abs_es = abs(es)
        if abs_es >= 0.8:
            colors.append('#006400')  # Dark green - large effect
        elif abs_es >= 0.5:
            colors.append(ACADEMIC_COLORS['improvement'])  # Green - medium effect
        elif abs_es >= 0.2:
            colors.append(ACADEMIC_COLORS['significant'])  # Orange - small effect
        else:
            colors.append(ACADEMIC_COLORS['degradation'])  # Red - negligible effect
    
    return colors

def add_significance_annotations(fig, x_values: List, y_values: List, p_values: List[float]):
    """
    Add significance annotations to a Plotly figure.
    
    Args:
        fig: Plotly figure object
        x_values: X-axis values for annotations
        y_values: Y-axis values for annotations  
        p_values: P-values for significance testing
    """
    
    annotations = []
    for x, y, p in zip(x_values, y_values, p_values):
        if p < 0.001:
            symbol = '***'
        elif p < 0.01:
            symbol = '**'
        elif p < 0.05:
            symbol = '*'
        else:
            symbol = 'ns'
        
        annotations.append(
            dict(
                x=x,
                y=y + 0.02,  # Slightly above the bar/point
                text=symbol,
                showarrow=False,
                font=dict(size=ACADEMIC_FONTS['size']['small'])
            )
        )
    
    fig.update_layout(annotations=annotations)

# Initialize styles when module is imported
setup_matplotlib_style()
setup_plotly_style()
