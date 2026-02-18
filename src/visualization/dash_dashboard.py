"""
Interactive visualization dashboard with Plotly Dash.

Features:
- Real-time metrics monitoring
- Interactive plots
- Model comparison visualizations
- Training progress tracking
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json


@dataclass
class DashboardConfig:
    """Plotly Dash dashboard configuration."""
    app_title: str = "CRISPR Prediction Pipeline"
    port: int = 8050
    debug: bool = True
    update_interval: int = 5000  # ms


class DashboardBuilder:
    """Build interactive Plotly Dash dashboard."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.app = None
        self.callbacks = []
    
    def create_app(self):
        """Create Dash application."""
        try:
            import dash
            from dash import dcc, html
            
            self.app = dash.Dash(__name__)
            self.app.title = self.config.app_title
            
            self.app.layout = html.Div([
                html.H1("ChromaGuide CRISPR Pipeline Dashboard"),
                dcc.Tabs(id='tabs', children=[
                    self._create_overview_tab(),
                    self._create_metrics_tab(),
                    self._create_comparison_tab(),
                    self._create_training_tab(),
                    self._create_results_tab(),
                ])
            ])
        except ImportError:
            print("Dash not installed. Install with: pip install dash")
    
    def _create_overview_tab(self) -> 'dcc.Tab':
        """Create overview tab."""
        try:
            from dash import dcc, html
            
            return dcc.Tab(label='Overview', children=[
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3("Phase Status"),
                            html.Div(id='phase-status')
                        ], className='card'),
                        
                        html.Div([
                            html.H3("Resource Usage"),
                            html.Div(id='resource-usage')
                        ], className='card'),
                    ], style={'display': 'flex', 'gap': '20px'}),
                ])
            ])
        except:
            return dcc.Tab(label='Overview', children=[html.Div("Tab not available")])
    
    def _create_metrics_tab(self) -> 'dcc.Tab':
        """Create metrics tab."""
        try:
            from dash import dcc, html
            
            return dcc.Tab(label='Metrics', children=[
                html.Div([
                    dcc.Graph(id='loss-curve'),
                    dcc.Graph(id='accuracy-curve'),
                    dcc.Graph(id='f1-curve'),
                ], style={'columnCount': 2})
            ])
        except:
            return dcc.Tab(label='Metrics', children=[html.Div("Tab not available")])
    
    def _create_comparison_tab(self) -> 'dcc.Tab':
        """Create model comparison tab."""
        try:
            from dash import dcc, html
            
            return dcc.Tab(label='Model Comparison', children=[
                html.Div([
                    dcc.Graph(id='model-comparison'),
                    dcc.Graph(id='benchmark-results'),
                ])
            ])
        except:
            return dcc.Tab(label='Comparison', children=[html.Div("Tab not available")])
    
    def _create_training_tab(self) -> 'dcc.Tab':
        """Create training progress tab."""
        try:
            from dash import dcc, html
            
            return dcc.Tab(label='Training', children=[
                html.Div([
                    dcc.Interval(id='training-interval', 
                                interval=self.config.update_interval),
                    dcc.Graph(id='training-progress'),
                    html.Div(id='training-stats')
                ])
            ])
        except:
            return dcc.Tab(label='Training', children=[html.Div("Tab not available")])
    
    def _create_results_tab(self) -> 'dcc.Tab':
        """Create results tab."""
        try:
            from dash import dcc, html
            
            return dcc.Tab(label='Results', children=[
                html.Div([
                    dcc.Graph(id='roc-curve'),
                    dcc.Graph(id='confusion-matrix'),
                    dcc.Graph(id='feature-importance'),
                ])
            ])
        except:
            return dcc.Tab(label='Results', children=[html.Div("Tab not available")])
    
    def add_callback(self, callback_fn):
        """Add callback function."""
        self.callbacks.append(callback_fn)
    
    def run(self, debug: bool = None, port: int = None):
        """Run dashboard server."""
        if self.app is None:
            self.create_app()
        
        debug = debug if debug is not None else self.config.debug
        port = port if port is not None else self.config.port
        
        self.app.run_server(debug=debug, port=port)


class MetricsVisualizer:
    """Visualize training metrics."""
    
    @staticmethod
    def create_loss_plot(history: Dict[str, List[float]]):
        """Create loss curve plot."""
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            if 'train_loss' in history:
                fig.add_trace(go.Scatter(
                    y=history['train_loss'],
                    name='Training Loss',
                    mode='lines'
                ))
            
            if 'val_loss' in history:
                fig.add_trace(go.Scatter(
                    y=history['val_loss'],
                    name='Validation Loss',
                    mode='lines'
                ))
            
            fig.update_layout(
                title='Training Loss Over Time',
                xaxis_title='Epoch',
                yaxis_title='Loss',
                hovermode='x unified'
            )
            
            return fig
        except:
            return None
    
    @staticmethod
    def create_accuracy_plot(history: Dict[str, List[float]]):
        """Create accuracy curve plot."""
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            for metric in ['accuracy', 'val_accuracy', 'r2_score', 'val_r2']:
                if metric in history:
                    fig.add_trace(go.Scatter(
                        y=history[metric],
                        name=metric.replace('_', ' ').title(),
                        mode='lines+markers'
                    ))
            
            fig.update_layout(
                title='Accuracy Metrics Over Time',
                xaxis_title='Epoch',
                yaxis_title='Score',
                hovermode='x unified'
            )
            
            return fig
        except:
            return None
    
    @staticmethod
    def create_model_comparison(results: Dict[str, Dict]):
        """Create model comparison bar chart."""
        try:
            import plotly.graph_objects as go
            
            models = list(results.keys())
            metrics = list(results[models[0]].keys())
            
            fig = go.Figure()
            
            for metric in metrics:
                values = [results[model].get(metric, 0) for model in models]
                fig.add_trace(go.Bar(
                    x=models,
                    y=values,
                    name=metric
                ))
            
            fig.update_layout(
                title='Model Comparison',
                xaxis_title='Model',
                yaxis_title='Score',
                barmode='group'
            )
            
            return fig
        except:
            return None


class ReportExporter:
    """Export interactive reports."""
    
    @staticmethod
    def export_html_report(fig, filepath: Path):
        """Export Plotly figure to HTML."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            fig.write_html(str(filepath))
        except:
            print(f"Failed to export report to {filepath}")
    
    @staticmethod
    def create_multi_page_report(figures: Dict[str, Any], output_dir: Path):
        """Create multi-page HTML report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        html_content = "<html><head><title>CRISPR Analysis Report</title></head><body>"
        
        for page_name, fig in figures.items():
            filename = f"{page_name}.html"
            filepath = output_dir / filename
            ReportExporter.export_html_report(fig, filepath)
            html_content += f'<a href="{filename}">{page_name}</a><br/>'
        
        html_content += "</body></html>"
        
        index_path = output_dir / "index.html"
        with open(index_path, 'w') as f:
            f.write(html_content)
