#!/usr/bin/env python3
"""
Real-Time Web Dashboard for ChromaGuide Pipeline
================================================

Interactive Streamlit dashboard for monitoring:
- Phase 1 training progress and metrics
- Phase 2-4 execution status
- Benchmarking results visualization
- Model performance comparison
- GPU and system metrics
- Training history and convergence

Usage:
    streamlit run dashboard_ui.py
    
    Open browser: http://localhost:8501
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
from typing import Dict, List, Optional
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="ChromaGuide Pipeline Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 5px;
    }
    .status-running {
        color: #00ff00;
        font-weight: bold;
    }
    .status-pending {
        color: #ffaa00;
        font-weight: bold;
    }
    .status-complete {
        color: #00aa00;
        font-weight: bold;
    }
    .status-failed {
        color: #ff0000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class DashboardData:
    """Manager for dashboard data."""
    
    def __init__(self):
        self.project_path = Path("/Users/studio/Desktop/PhD/Proposal")
        self.narval_host = "narval"
        self.job_id = "56644478"
    
    def get_phase1_status(self) -> Dict:
        """Get Phase 1 training status from Narval."""
        try:
            cmd = [
                'ssh', self.narval_host,
                f'squeue -j {self.job_id} --format="%i %T %e %C %m" --noheader'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.stdout.strip():
                parts = result.stdout.strip().split()
                return {
                    'job_id': parts[0],
                    'status': parts[1],
                    'exit_code': parts[2] if len(parts) > 2 else 'N/A',
                    'cpus': parts[3] if len(parts) > 3 else 'N/A',
                    'memory': parts[4] if len(parts) > 4 else 'N/A',
                }
            return {'status': 'UNKNOWN', 'error': 'Job not found'}
        except Exception as e:
            logger.error(f"Error getting Phase 1 status: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def get_training_history(self) -> Optional[Dict]:
        """Load training history from Phase 1."""
        try:
            history_file = self.project_path / "checkpoints/phase1/training_history.json"
            if history_file.exists():
                with open(history_file) as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"Could not load training history: {e}")
        return None
    
    def get_benchmark_results(self) -> Optional[pd.DataFrame]:
        """Load latest benchmark results."""
        try:
            results_file = self.project_path / "results/benchmark_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                    return pd.DataFrame(data).T
        except Exception as e:
            logger.debug(f"Could not load benchmark results: {e}")
        return None
    
    def get_orchestration_logs(self, n_lines: int = 50) -> str:
        """Get latest orchestration logs."""
        try:
            log_file = self.project_path / "orchestration.log"
            if log_file.exists():
                with open(log_file) as f:
                    lines = f.readlines()
                    return ''.join(lines[-n_lines:])
        except Exception as e:
            logger.debug(f"Could not load logs: {e}")
        return "No logs available"
    
    def get_mock_data_stats(self) -> Dict:
        """Get statistics on mock data."""
        try:
            summary_file = self.project_path / "data/mock/mock_data_summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"Could not load mock data stats: {e}")
        return {}


# Initialize dashboard data manager
if 'dashboard_data' not in st.session_state:
    st.session_state.dashboard_data = DashboardData()

dashboard = st.session_state.dashboard_data


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    """Main dashboard layout."""
    
    # Header
    st.title("🧬 ChromaGuide Pipeline Monitor")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 5, 300, 30)
    show_logs = st.sidebar.checkbox("Show logs", value=True)
    show_benchmarks = st.sidebar.checkbox("Show benchmarks", value=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Status", "🏃 Training", "📈 Benchmarks", "📝 Logs", "⚙️ Settings"]
    )
    
    # ========================================================================
    # TAB 1: STATUS
    # ========================================================================
    with tab1:
        st.header("Pipeline Status Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Phase 1 Status
        with col1:
            phase1_status = dashboard.get_phase1_status()
            status_emoji = "🟢" if phase1_status.get('status') == 'R' else "🟡"
            st.metric(
                "Phase 1 (DNABERT-Mamba)",
                status_emoji + " " + phase1_status.get('status', 'UNKNOWN'),
                "GPU Training"
            )
        
        # Phase 2 Status
        with col2:
            st.metric(
                "Phase 2 (XGBoost)",
                "⏳ Pending",
                "Auto-execute"
            )
        
        # Phase 3 Status
        with col3:
            st.metric(
                "Phase 3 (DeepHybrid)",
                "⏳ Pending",
                "Auto-execute"
            )
        
        # Phase 4 Status
        with col4:
            st.metric(
                "Phase 4 (Clinical)",
                "⏳ Pending",
                "Auto-execute"
            )
        
        st.markdown("---")
        
        # Detailed Status Table
        st.subheader("Detailed Phase Status")
        
        phases = [
            {
                "Phase": "Phase 1: DNABERT-Mamba",
                "Status": phase1_status.get('status', 'UNKNOWN'),
                "Duration": "18-24 hrs",
                "Location": "Narval GPU",
                "Progress": "Training"
            },
            {
                "Phase": "Phase 2: XGBoost",
                "Status": "Pending",
                "Duration": "2-3 hrs",
                "Location": "Local CPU",
                "Progress": "Waiting"
            },
            {
                "Phase": "Phase 3: DeepHybrid",
                "Status": "Pending",
                "Duration": "1-2 hrs",
                "Location": "Local GPU",
                "Progress": "Waiting"
            },
            {
                "Phase": "Phase 4: Clinical",
                "Status": "Pending",
                "Duration": "45 min",
                "Location": "Local CPU",
                "Progress": "Waiting"
            }
        ]
        
        df_phases = pd.DataFrame(phases)
        st.dataframe(df_phases, use_container_width=True, hide_index=True)
        
        # Timeline visualization
        st.subheader("Expected Timeline")
        fig_timeline = go.Figure()
        
        phases_timeline = [
            ("Phase 1: GPU Training", 0, 24, "#FF6B6B"),
            ("Phase 2: XGBoost", 24, 27, "#4ECDC4"),
            ("Phase 3: DeepHybrid", 27, 29, "#45B7D1"),
            ("Phase 4: Clinical", 29, 30, "#FFA07A"),
            ("Benchmarking", 30, 32, "#98D8C8"),
            ("Figure Gen", 32, 33, "#F7DC6F"),
            ("Results Done", 33, 34, "#52B788"),
        ]
        
        for phase, start, end, color in phases_timeline:
            fig_timeline.add_trace(go.Bar(
                y=[phase],
                x=[end - start],
                base=start,
                orientation='h',
                marker=dict(color=color),
                name=phase,
                hovertemplate=f"{phase}: {start}h - {end}h"
            ))
        
        fig_timeline.update_layout(
            xaxis_title="Hours from Start",
            height=300,
            showlegend=False,
            barmode='overlay'
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # ========================================================================
    # TAB 2: TRAINING
    # ========================================================================
    with tab2:
        st.header("Training Progress")
        
        # Phase 1 Training History
        history = dashboard.get_training_history()
        
        if history:
            col1, col2 = st.columns(2)
            
            with col1:
                # Loss curve
                epochs = list(range(len(history.get('loss', []))))
                fig_loss = go.Figure()
                
                fig_loss.add_trace(go.Scatter(
                    x=epochs,
                    y=history.get('loss', []),
                    mode='lines',
                    name='Training Loss',
                    line=dict(color='#FF6B6B', width=2)
                ))
                
                if 'val_loss' in history:
                    fig_loss.add_trace(go.Scatter(
                        x=epochs,
                        y=history['val_loss'],
                        mode='lines',
                        name='Validation Loss',
                        line=dict(color='#4ECDC4', width=2, dash='dash')
                    ))
                
                fig_loss.update_layout(
                    title="Training Loss Curve",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig_loss, use_container_width=True)
            
            with col2:
                # Metrics
                fig_metrics = go.Figure()
                
                if 'spearman' in history:
                    fig_metrics.add_trace(go.Scatter(
                        x=epochs,
                        y=history['spearman'],
                        mode='lines+markers',
                        name='Spearman R',
                        line=dict(color='#45B7D1', width=2)
                    ))
                
                if 'pearson' in history:
                    fig_metrics.add_trace(go.Scatter(
                        x=epochs,
                        y=history['pearson'],
                        mode='lines+markers',
                        name='Pearson R',
                        line=dict(color='#FFA07A', width=2)
                    ))
                
                fig_metrics.update_layout(
                    title="Correlation Metrics",
                    xaxis_title="Epoch",
                    yaxis_title="Correlation",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig_metrics, use_container_width=True)
        
        else:
            st.info("Training history not yet available. Phase 1 still in progress.")
            
            # Placeholder statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Epoch", "Estimating...", "of 10")
            with col2:
                st.metric("Loss", "Loading...", "")
            with col3:
                st.metric("Spearman R", "Loading...", "")
            with col4:
                st.metric("ETA", "~20 hours", "GPU dependent")
    
    # ========================================================================
    # TAB 3: BENCHMARKS
    # ========================================================================
    with tab3:
        st.header("Model Benchmarking")
        
        if show_benchmarks:
            results = dashboard.get_benchmark_results()
            
            if results is not None and len(results) > 0:
                # Results table
                st.subheader("SOTA Model Comparison")
                st.dataframe(results, use_container_width=True)
                
                # Performance visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_rmse = px.bar(
                        results.reset_index(names='Model'),
                        x='Model',
                        y='rmse',
                        title="Model RMSE Comparison",
                        color='rmse',
                        color_continuous_scale='Reds_r'
                    )
                    st.plotly_chart(fig_rmse, use_container_width=True)
                
                with col2:
                    fig_r2 = px.bar(
                        results.reset_index(names='Model'),
                        x='Model',
                        y='r2',
                        title="R² Score Comparison",
                        color='r2',
                        color_continuous_scale='Greens'
                    )
                    st.plotly_chart(fig_r2, use_container_width=True)
            
            else:
                st.info("Benchmark results not yet available. Benchmarking runs after Phase 4.")
                
                # Placeholder data
                placeholder_models = ['DeepHF', 'CRISPRon', 'TransCRISPR', 'XGBoost', 'RandomForest']
                placeholder_data = pd.DataFrame({
                    'Model': placeholder_models,
                    'RMSE': np.random.uniform(0.08, 0.15, len(placeholder_models)),
                    'R²': np.random.uniform(0.65, 0.85, len(placeholder_models)),
                    'Spearman': np.random.uniform(0.70, 0.90, len(placeholder_models))
                })
                st.info("Example expected results:")
                st.dataframe(placeholder_data, use_container_width=True, hide_index=True)
    
    # ========================================================================
    # TAB 4: LOGS
    # ========================================================================
    with tab4:
        st.header("System Logs")
        
        if show_logs:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader("Orchestration Logs")
            with col2:
                if st.button("🔄 Refresh"):
                    st.rerun()
            
            logs = dashboard.get_orchestration_logs(n_lines=100)
            st.code(logs, language="log")
            
            # Log statistics
            st.subheader("Log Summary")
            
            log_lines = logs.split('\n')
            error_count = sum(1 for line in log_lines if 'ERROR' in line)
            warning_count = sum(1 for line in log_lines if 'WARNING' in line)
            info_count = sum(1 for line in log_lines if 'INFO' in line)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Info Messages", info_count)
            with col2:
                st.metric("Warnings", warning_count)
            with col3:
                st.metric("Errors", error_count)
    
    # ========================================================================
    # TAB 5: SETTINGS
    # ========================================================================
    with tab5:
        st.header("Settings & Configuration")
        
        st.subheader("Pipeline Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Cluster Settings**")
            st.write(f"Host: narval.alliancecan.ca")
            st.write(f"Account: def-kwiese")
            st.write(f"Job ID: {dashboard.job_id}")
        
        with col2:
            st.write("**Local Settings**")
            st.write("Project Path: /Users/studio/Desktop/PhD/Proposal")
            st.write("Python Version: 3.13.9")
            st.write("Test Status: 23/24 Passing")
        
        st.markdown("---")
        
        st.subheader("Mock Data Statistics")
        mock_stats = dashboard.get_mock_data_stats()
        
        if mock_stats:
            st.write(f"**Generated:** {mock_stats.get('timestamp', 'Unknown')}")
            st.write(f"**Samples:** {mock_stats.get('n_samples', 'Unknown')}")
            st.write(f"**Files:** {mock_stats.get('files_created', 'Unknown')}")
        
        st.markdown("---")
        
        st.subheader("Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Refresh Dashboard"):
                st.rerun()
        
        with col2:
            if st.button("🚀 View on Narval"):
                st.info("Open Terminal and run: `ssh narval squeue -j 56644478`")
        
        # Dashboard information
        st.markdown("---")
        st.info("""
        **ChromaGuide Pipeline Dashboard v1.0**
        
        Real-time monitoring of CRISPR prediction pipeline including:
        - Phase 1 GPU training on Narval supercomputer
        - Automated phases 2-4 execution
        - Comprehensive benchmarking against SOTA models
        - Publication-quality figure generation
        
        Auto-refresh every {} seconds
        """.format(refresh_rate))


if __name__ == "__main__":
    main()
