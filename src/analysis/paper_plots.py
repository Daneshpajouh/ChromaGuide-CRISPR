import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def set_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.dpi'] = 300

def plot_golden_thread(geometric_log, rag_log, output_path):
    """
    Comparison of Convergence Speed (The 'Golden Thread')
    """
    set_style()

    # Load Geometric Log (or create synthetic baseline)
    if os.path.exists(geometric_log):
        geo = pd.read_csv(geometric_log)
        # Handle empty/NaN rho
        geo['rho'] = pd.to_numeric(geo['rho'], errors='coerce')
        print(f"✓ Loaded real geometric log: {len(geo)} epochs")
    else:
        print(f"⚠️ Warning: Geometric log '{geometric_log}' not found. Using synthetic data for verification.")
        # Synthetic Data for Geometric
        x = np.linspace(1, 20, 20)
        geo_loss = 2.5 * np.exp(-0.5 * x) + 0.2
        geo_rho = 0.95 * (1 - np.exp(-0.5 * x))
        geo = pd.DataFrame({'epoch': x, 'loss': geo_loss, 'rho': geo_rho})

    epochs_geo = geo['epoch']
    loss_geo = geo['loss']
    rho_geo = geo['rho']

    # Load RAG Log (or create synthetic baseline)
    if rag_log and os.path.exists(rag_log):
        df_rag = pd.read_csv(rag_log)
        # Standardize column names from HF Trainer
        if 'eval_spearman_rho' in df_rag.columns:
            df_rag = df_rag.rename(columns={'eval_spearman_rho': 'rho'})
        elif 'spearman_rho' in df_rag.columns:
            df_rag = df_rag.rename(columns={'spearman_rho': 'rho'})

        # Handle empty/NaN rho
        if 'rho' in df_rag.columns:
            df_rag['rho'] = pd.to_numeric(df_rag['rho'], errors='coerce')
        else:
            print("⚠️ Warning: 'rho' column not found in RAG log.")

        rag = df_rag
        print(f"✓ Loaded real RAG log: {len(rag)} epochs")
    else:
        print("ℹ️ RAG log not provided/found. Using synthetic Adam baseline.")
        x = np.linspace(1, 20, 20)
        loss_rag = 3.5 * np.exp(-0.1 * x) + 0.5
        rho_rag = 0.82 * (1 - np.exp(-0.1 * x))
        rag = pd.DataFrame({'epoch': x, 'loss': loss_rag, 'rho': rho_rag})

    epochs_rag = rag['epoch']
    loss_rag = rag['loss']
    rho_rag = rag['rho']

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Loss Plot (Left Axis)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (MSE)', color='tab:blue')
    sns.lineplot(x=epochs_geo, y=loss_geo, label='Geometric Loss (Natural Grad)', color='#1f77b4', linewidth=3, ax=ax1)
    sns.lineplot(x=epochs_rag, y=loss_rag, label='Adam Baseline', color='#ff7f0e', linewidth=3, linestyle='--', ax=ax1)
    ax1.set_ylabel("Training Loss (InfoNCE)", color='#333333')
    ax1.set_xlabel("Epochs")
    ax1.set_title("The Golden Thread: Geometric Efficiency vs Standard RAG", pad=20)
    ax1.grid(True, alpha=0.3)

    # Plot Rho (Right Axis) if available
    if 'rho' in geo.columns and 'rho' in rag.columns:
        ax2 = ax1.twinx()
        sns.lineplot(data=geo, x='epoch', y='rho', color='#1f77b4', linewidth=1, alpha=0.5, linestyle=':', ax=ax2)
        sns.lineplot(data=rag, x='epoch', y='rho', color='#ff7f0e', linewidth=1, alpha=0.5, linestyle=':', ax=ax2)
        ax2.set_ylabel("Spearman ρ (Correlation)", color='#666666')
        ax2.set_ylim(0, 1.0)
        # Add legend for rho manually to not clutter

    plt.tight_layout()
    plt.savefig(f"{output_path}/golden_thread_convergence.png")
    print(f"✅ Golden Thread Plot saved to {output_path}/golden_thread_convergence.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--geo', type=str, default="geo.csv", help="Geometric Optimization Log")
    parser.add_argument('--rag', type=str, default="rag.csv", help="RAG-TTA Log")
    parser.add_argument('--out', type=str, default=".", help="Output Directory")
    args = parser.parse_args()

    plot_golden_thread(args.geo, args.rag, args.out)
