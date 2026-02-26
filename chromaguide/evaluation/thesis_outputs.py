"""Generate thesis-ready figures and tables.

Creates:
    1. Main results table (Table 1)
    2. Ablation results table (Table 2)
    3. Scatter plots: predicted vs actual efficacy
    4. Calibration plots
    5. Conformal coverage plots
    6. Gate value heatmaps (interpretability)
    7. Backbone comparison radar chart
    8. Training curves
    9. Bootstrap CI forest plot
"""
from __future__ import annotations
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_all(results_dir: str) -> None:
    """Generate all thesis-ready outputs."""
    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating thesis outputs...")
    
    # Load all evaluation results
    eval_files = list(results_dir.glob("eval_*.json"))
    all_results = {}
    for f in eval_files:
        with open(f) as fh:
            all_results[f.stem] = json.load(fh)
    
    if not all_results:
        logger.warning("No evaluation results found. Run experiments first.")
        return
    
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set thesis-quality defaults
        plt.rcParams.update({
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        })
    except ImportError:
        logger.error("matplotlib/seaborn not available")
        return
    
    # 1. Generate main results table
    _generate_main_table(all_results, tables_dir)
    
    # 2. Generate scatter plots
    _generate_scatter_plots(results_dir, figures_dir)
    
    # 3. Generate calibration plots
    _generate_calibration_plots(results_dir, figures_dir)
    
    # 4. Generate conformal coverage plots
    _generate_coverage_plots(results_dir, figures_dir)
    
    # 5. Generate training curves
    _generate_training_curves(results_dir, figures_dir)
    
    logger.info(f"Thesis outputs saved to {figures_dir} and {tables_dir}")


def _generate_main_table(all_results: dict, tables_dir: Path) -> None:
    """Generate Table 1: Main experimental results."""
    rows = []
    
    for name, metrics in all_results.items():
        row = {
            "Model": name.replace("eval_", "").replace("_splitA", ""),
            "Spearman ρ": f"{metrics.get('spearman_rho', 0):.3f}",
            "Pearson r": f"{metrics.get('pearson_r', 0):.3f}",
            "MSE": f"{metrics.get('mse', 0):.4f}",
            "nDCG@10": f"{metrics.get('ndcg_10', 0):.3f}",
            "ECE": f"{metrics.get('ece', 0):.4f}",
            "Coverage": f"{metrics.get('conformal_coverage', 0):.3f}",
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save as CSV
    df.to_csv(tables_dir / "table1_main_results.csv", index=False)
    
    # Save as LaTeX
    latex = df.to_latex(index=False, escape=False, column_format="l" + "c" * (len(df.columns) - 1))
    with open(tables_dir / "table1_main_results.tex", "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Main experimental results on Split A (gene-held-out).}\n")
        f.write("\\label{tab:main_results}\n")
        f.write(latex)
        f.write("\\end{table}\n")
    
    logger.info(f"  Table 1: {len(rows)} models")


def _generate_scatter_plots(results_dir: Path, figures_dir: Path) -> None:
    """Generate predicted vs actual scatter plots."""
    import matplotlib.pyplot as plt
    
    pred_files = list(results_dir.glob("predictions_*.npz"))
    
    for f in pred_files:
        data = np.load(f)
        name = f.stem.replace("predictions_", "")
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(data["test_y"], data["test_mu"], alpha=0.3, s=10, c="#1f77b4")
        ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Perfect")
        
        from scipy.stats import spearmanr
        rho, _ = spearmanr(data["test_y"], data["test_mu"])
        ax.set_xlabel("True Efficacy")
        ax.set_ylabel("Predicted Efficacy")
        ax.set_title(f"{name}\n(Spearman ρ = {rho:.3f})")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        
        plt.savefig(figures_dir / f"scatter_{name}.pdf")
        plt.savefig(figures_dir / f"scatter_{name}.png")
        plt.close()
    
    logger.info(f"  Scatter plots: {len(pred_files)} models")


def _generate_calibration_plots(results_dir: Path, figures_dir: Path) -> None:
    """Generate calibration (reliability) diagrams."""
    import matplotlib.pyplot as plt
    
    pred_files = list(results_dir.glob("predictions_*.npz"))
    
    for f in pred_files:
        data = np.load(f)
        name = f.stem.replace("predictions_", "")
        
        y_true = data["test_y"]
        y_pred = data["test_mu"]
        
        # Bin predictions
        n_bins = 15
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_means_pred = []
        bin_means_true = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_means_pred.append(y_pred[mask].mean())
                bin_means_true.append(y_true[mask].mean())
                bin_counts.append(mask.sum())
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), height_ratios=[3, 1])
        
        # Calibration curve
        ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
        ax1.bar(bin_means_pred, bin_means_true, width=0.05, alpha=0.6, color="#2ca02c")
        ax1.scatter(bin_means_pred, bin_means_true, c="#d62728", zorder=3, s=30)
        ax1.set_xlabel("Mean Predicted")
        ax1.set_ylabel("Mean Observed")
        ax1.set_title(f"Calibration: {name}")
        ax1.legend()
        
        # Histogram of predictions
        ax2.hist(y_pred, bins=30, color="#1f77b4", alpha=0.7)
        ax2.set_xlabel("Predicted Efficacy")
        ax2.set_ylabel("Count")
        
        plt.tight_layout()
        plt.savefig(figures_dir / f"calibration_{name}.pdf")
        plt.savefig(figures_dir / f"calibration_{name}.png")
        plt.close()


def _generate_coverage_plots(results_dir: Path, figures_dir: Path) -> None:
    """Generate conformal prediction coverage plots."""
    import matplotlib.pyplot as plt
    
    pred_files = list(results_dir.glob("predictions_*.npz"))
    
    for f in pred_files:
        data = np.load(f)
        name = f.stem.replace("predictions_", "")
        
        if "lower" not in data or "upper" not in data:
            continue
        
        y_true = data["test_y"]
        lower = data["lower"]
        upper = data["upper"]
        mu = data["test_mu"]
        
        # Sort by predicted value
        sort_idx = np.argsort(mu)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        n_show = min(200, len(y_true))
        idx = np.linspace(0, len(sort_idx) - 1, n_show, dtype=int)
        
        x = np.arange(n_show)
        ax.fill_between(x, lower[sort_idx[idx]], upper[sort_idx[idx]], 
                        alpha=0.3, color="#1f77b4", label="90% PI")
        ax.scatter(x, y_true[sort_idx[idx]], s=8, c="#d62728", alpha=0.6, label="True", zorder=3)
        ax.plot(x, mu[sort_idx[idx]], color="#1f77b4", linewidth=1, label="Predicted")
        
        covered = ((y_true >= lower) & (y_true <= upper)).mean()
        ax.set_title(f"Conformal Prediction: {name} (Coverage: {covered:.1%})")
        ax.set_xlabel("Sample (sorted by prediction)")
        ax.set_ylabel("Efficacy")
        ax.legend()
        
        plt.savefig(figures_dir / f"conformal_{name}.pdf")
        plt.savefig(figures_dir / f"conformal_{name}.png")
        plt.close()


def _generate_training_curves(results_dir: Path, figures_dir: Path) -> None:
    """Generate training loss and metric curves."""
    import matplotlib.pyplot as plt
    
    # Look for training history files
    history_files = list(results_dir.glob("*history*.json"))
    
    if not history_files:
        return
    
    for f in history_files:
        with open(f) as fh:
            history = json.load(fh)
        
        name = f.stem
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        if "train_loss" in history:
            ax1.plot(history["train_loss"], label="Train Loss")
        if "val_loss" in history:
            ax1.plot(history["val_loss"], label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Curves")
        ax1.legend()
        
        if "spearman" in history:
            ax2.plot(history["spearman"], label="Val Spearman ρ", color="#2ca02c")
            ax2.axhline(y=0.91, color="r", linestyle="--", label="Target (0.91)")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Spearman ρ")
            ax2.set_title("Validation Performance")
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(figures_dir / f"training_{name}.pdf")
        plt.savefig(figures_dir / f"training_{name}.png")
        plt.close()
