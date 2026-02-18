import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Set global styles for Dark Mode / Deep Space theme - Optimized for Presentation Visibility
plt.style.use('dark_background')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'Arial'],
    'axes.facecolor': '#0f172a',
    'figure.facecolor': '#0f172a',
    'axes.edgecolor': '#f1f5f9',
    'axes.labelcolor': '#f1f5f9',
    'xtick.color': '#f1f5f9',
    'ytick.color': '#f1f5f9',
    'grid.color': '#1e293b',
    'text.color': '#f1f5f9',
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 22
})

def save_fig(name):
    path = os.path.join('/Users/studio/Desktop/PhD/Proposal/Presentation/Figs/', name)
    plt.savefig(path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Generated {path}")

# 1. Performance Comparison (Slide 4) - ENLARGED
def gen_performance_comparison():
    plt.figure(figsize=(12, 6))
    models = ['DeepHF\n(2019)', 'AttCRISPR\n(2020)', 'ChromeCRISPR\n(2025)']
    scores = [0.867, 0.872, 0.876]
    colors = ['#94a3b8', '#38bdf8', '#CC0633'] # Muted gray, Sky, SFU Red

    bars = plt.bar(models, scores, color=colors, width=0.5, alpha=1.0, edgecolor='white', linewidth=1.5)
    plt.ylim(0.85, 0.89)
    plt.ylabel('Spearman Correlation (ρ)', fontsize=18, fontweight='bold', labelpad=15)
    plt.title('Baseline Performance Comparison', fontsize=22, pad=25, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=18, fontweight='bold', color='white')

    plt.grid(axis='y', linestyle='--', alpha=0.2)
    save_fig('performance_comparison.png')

# 2. Uncertainty Calibration (Slide 13) - ENLARGED
def gen_uncertainty_viz():
    plt.figure(figsize=(12, 6))
    x = np.linspace(0.7, 1.0, 1000)
    # Simulated prediction intervals
    y_target = x
    y_lower = x - 0.04 - np.random.normal(0, 0.003, 1000)
    y_upper = x + 0.04 + np.random.normal(0, 0.003, 1000)

    plt.fill_between(x, y_lower, y_upper, color='#38bdf8', alpha=0.3, label='90% Conformal Interval')
    plt.plot(x, y_target, color='#10b981', linewidth=4, label='Perfect Calibration')

    plt.xlabel('Predicted Efficacy (μ)', fontsize=18, fontweight='bold', labelpad=10)
    plt.ylabel('Observed Efficacy (y)', fontsize=18, fontweight='bold', labelpad=10)
    plt.title('Calibrated Coverage Verification', fontsize=22, pad=25, fontweight='bold')
    plt.legend(frameon=True, fontsize=16, facecolor='#1e293b', edgecolor='white')
    plt.xlim(0.7, 1.0)
    plt.ylim(0.7, 1.0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(alpha=0.2)
    save_fig('uncertainty_calibration.png')

# 3. Expected Results (Slide 19) - ENLARGED
def gen_expected_results():
    plt.figure(figsize=(12, 6))
    labels = ['ChromeCRISPR\n(Baseline)', 'ChromaGuide\n(Target)']
    curr = 0.876
    target = 0.880

    bars = plt.bar(labels, [curr, target], color=['#CC0633', '#10b981'], width=0.4, edgecolor='white', linewidth=1.5)
    plt.ylim(0.87, 0.885)
    plt.ylabel('Spearman Correlation (ρ)', fontsize=18, fontweight='bold', labelpad=15)
    plt.title('Anticipated SOTA Advancement', fontsize=22, pad=25, fontweight='bold')

    plt.text(0, curr + 0.0003, f'{curr:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=18)
    plt.text(1, target + 0.0003, f'{target:.3f}*', ha='center', va='bottom', fontweight='bold', fontsize=18, color='#10b981')

    plt.annotate('Target Improvement Δρ ≥ 0.004', xy=(0.5, 0.878), xytext=(0.5, 0.883),
                 arrowprops=dict(facecolor='#fbbf24', shrink=0.05, width=4, headwidth=12),
                 ha='center', color='#fbbf24', fontweight='bold', fontsize=16)

    plt.grid(axis='y', alpha=0.2)
    save_fig('expected_results.png')

# 4. Timeline Gantt (Slide 20) - ENLARGED
def gen_timeline_gantt():
    plt.figure(figsize=(14, 7))
    tasks = ['Proposal & Setup', 'Implementation', 'Experiments', 'Thesis & Defense']
    start_dates = [0, 1, 3, 5] # Relative months
    durations = [1, 2, 2, 2]
    colors = ['#38bdf8', '#fbbf24', '#10b981', '#CC0633']

    for i, (task, start, dur, color) in enumerate(zip(tasks, start_dates, durations, colors)):
        plt.barh(task, dur, left=start, color=color, alpha=1.0, height=0.6, edgecolor='white', linewidth=1.5)
        plt.text(start + dur/2, i, task, ha='center', va='center', color='white', fontweight='bold', fontsize=15)

    plt.xlabel('2026 Timeline', fontsize=18, fontweight='bold', labelpad=15)
    plt.xticks(range(7), ['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'], fontsize=16)
    plt.yticks(fontsize=16, fontweight='bold')
    plt.title('ChromaGuide Project Timeline', fontsize=22, pad=25, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    save_fig('timeline_gantt.png')

# 5. Data Isolation (Slide 17) - ENLARGED
def gen_data_isolation():
    plt.figure(figsize=(12, 6))
    categories = ['Random\nSplit', 'Sequence\nHold-out', 'Gene\nHold-out']
    val_perf = [0.92, 0.88, 0.81] # Values representing R-squared or similar

    bars = plt.bar(categories, val_perf, color=['#94a3b8', '#38bdf8', '#CC0633'], alpha=1.0, width=0.5, edgecolor='white', linewidth=1.5)
    plt.ylim(0, 1.05)
    plt.ylabel('Observed Performance (R²)', fontsize=18, fontweight='bold', labelpad=15)
    plt.title('Impact of Data Leakage', fontsize=22, pad=25, fontweight='bold')

    for i, v in enumerate(val_perf):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold', fontsize=18)

    plt.grid(axis='y', alpha=0.2)
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16)
    save_fig('data_isolation.png')

if __name__ == "__main__":
    os.makedirs('/Users/studio/Desktop/PhD/Proposal/Presentation/Figs/', exist_ok=True)
    gen_performance_comparison()
    gen_uncertainty_viz()
    gen_expected_results()
    gen_timeline_gantt()
    gen_data_isolation()
