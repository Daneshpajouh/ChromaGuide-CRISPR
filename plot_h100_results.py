import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
import sys

# Usage: python plot_h100_results.py <path_to_log_file>

def parse_log(filepath):
    metrics = []
    with open(filepath, 'r') as f:
        for line in f:
            # Look for: Epoch 1: Loss: 0.8123 | Val: 0.7900 | AUC: 0.5500 | Spearman: 0.0500
            match = re.search(r'Epoch (\d+): Loss: ([\d\.]+) \| Val: ([\d\.]+) \| AUC: ([\d\.]+) \| Spearman: ([-]?[\d\.]+)', line)
            if match:
                metrics.append({
                    'Epoch': int(match.group(1)),
                    'Train_Loss': float(match.group(2)),
                    'Val_Loss': float(match.group(3)),
                    'AUC': float(match.group(4)),
                    'Spearman': float(match.group(5))
                })
    return pd.DataFrame(metrics)

def plot_metrics(df, output_path):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['font.family'] = 'sans-serif'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Spearman (The Key Metric)
    sns.lineplot(data=df, x='Epoch', y='Spearman', ax=ax1, color='tab:green', linewidth=2.5, marker='o')
    ax1.set_title('Spearman Correlation (H100)', fontweight='bold')
    ax1.set_ylabel('Spearman Rho')
    ax1.axhline(0.80, color='red', linestyle='--', label='Target (0.80)')
    ax1.legend()

    # Plot 2: Loss
    sns.lineplot(data=df, x='Epoch', y='Train_Loss', ax=ax2, label='Train', color='tab:blue')
    sns.lineplot(data=df, x='Epoch', y='Val_Loss', ax=ax2, label='Val', color='tab:orange', linestyle='--')
    ax2.set_title('Loss Convergence', fontweight='bold')
    ax2.set_ylabel('Hybrid Loss')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_h100_results.py <log_file>")
        # Create dummy data for testing if no file provided
        print("Generating dummy plot for verification...")
        dummy_data = pd.DataFrame({
            'Epoch': range(1, 11),
            'Train_Loss': [0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.38, 0.36, 0.35, 0.34],
            'Val_Loss': [0.82, 0.72, 0.62, 0.52, 0.47, 0.42, 0.40, 0.38, 0.37, 0.36],
            'AUC': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.82, 0.85, 0.87],
            'Spearman': [0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.75, 0.82, 0.85, 0.88]
        })
        plot_metrics(dummy_data, 'figures/h100_results_dummy.png')
    else:
        df = parse_log(sys.argv[1])
        if df.empty:
            print("No metrics found in log file.")
        else:
            plot_metrics(df, 'figures/h100_results_real.png')
