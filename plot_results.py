import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
import sys
import os

# Function to parse the log file
def parse_log_file(filepath):
    metrics = []
    with open(filepath, 'r') as f:
        content = f.read()

    # Regex to capture Epoch lines
    # Epoch 1: Loss: 0.5911 | Val: 1.2858 | AUC: 0.5833 | Spearman: -0.3243
    pattern = r"Epoch (\d+): Loss: ([\d\.]+) \| Val: ([\d\.]+) \| AUC: ([\d\.]+) \| Spearman: ([-]?[\d\.]+)"

    matches = re.findall(pattern, content)

    for match in matches:
        metrics.append({
            'Epoch': int(match[0]),
            'Train Loss': float(match[1]),
            'Val Loss': float(match[2]),
            'AUC': float(match[3]),
            'Spearman': float(match[4])
        })

    return pd.DataFrame(metrics)

def generate_thesis_figure(df, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)

    # Set academic style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.constrained_layout.use'] = True

    # Create Figure: Dual Axis (Loss + Spearman)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 1. Plot Spearman (Primary - The Goal)
    color_spearman = '#2ca02c' # Green
    sns.lineplot(data=df, x='Epoch', y='Spearman', ax=ax1, color=color_spearman, linewidth=3, label='Spearman Correlation', marker='o', markersize=6)
    ax1.set_ylabel('Spearman Rho', color=color_spearman, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_spearman)
    ax1.set_ylim(-0.2, 0.6) # Focus range
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # 2. Plot Loss (Secondary - The Cost)
    ax2 = ax1.twinx()
    color_loss = '#1f77b4' # Blue
    sns.lineplot(data=df, x='Epoch', y='Train Loss', ax=ax2, color=color_loss, linewidth=2, linestyle='--', label='Train Loss', alpha=0.7)
    sns.lineplot(data=df, x='Epoch', y='Val Loss', ax=ax2, color='#d62728', linewidth=2, linestyle=':', label='Val Loss', alpha=0.7)
    ax2.set_ylabel('Hybrid Loss', color='gray', fontweight='bold', rotation=270, labelpad=15)
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(0, 1.5)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, framealpha=0.9)

    # Title
    plt.title('Training Dynamics on Mac Studio (MPS)\nValidating Safe Config (dt_min=0.001)', fontsize=14, pad=15)

    # Save
    base_name = os.path.join(output_dir, "local_training_dynamics")
    plt.savefig(f"{base_name}.pdf", dpi=300)
    plt.savefig(f"{base_name}.png", dpi=300)
    print(f"Generated: {base_name}.png")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to latest known log for convenience
        default_log = "logs/local_training/m3_run_1765852175.log"
        print(f"No file specified. Using latest detected: {default_log}")
        log_file = default_log
    else:
        log_file = sys.argv[1]

    if not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}")
        sys.exit(1)

    df = parse_log_file(log_file)
    if df.empty:
        print("Error: No metrics found in log file.")
    else:
        print(f"Parsed {len(df)} epochs.")
        print(df.tail())
        generate_thesis_figure(df)
