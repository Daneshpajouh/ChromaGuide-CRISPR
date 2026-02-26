#!/usr/bin/env python3
"""Collect and summarize results from all ChromaGuide experiments.

Usage:
    python collect_results.py [--results-dir RESULTS_DIR]
"""
import json
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='results')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Collect all results
    all_results = []
    for result_file in sorted(results_dir.glob('*/results.json')):
        try:
            with open(result_file) as f:
                r = json.load(f)
            all_results.append(r)
        except Exception as e:
            print(f"Error reading {result_file}: {e}")
    
    if not all_results:
        print("No results found!")
        return
    
    print("=" * 80)
    print("ChromaGuide Experiment Results Summary")
    print("=" * 80)
    print(f"\nTotal experiments: {len(all_results)}")
    
    # Group by backbone × split
    groups = defaultdict(list)
    for r in all_results:
        key = (r['backbone'], r['split'])
        groups[key].append(r)
    
    # Print table
    print("\n" + "-" * 80)
    print(f"{'Backbone':<25} {'Split':<8} {'Seeds':<6} {'Spearman ρ':<18} {'Pearson r':<18} {'ECE':<12} {'Coverage':<12}")
    print("-" * 80)
    
    for (backbone, split), results in sorted(groups.items()):
        spearmans = [r['test_metrics']['spearman'] for r in results]
        pearsons = [r['test_metrics']['pearson'] for r in results]
        eces = [r['test_metrics']['ece'] for r in results]
        coverages = [r['conformal']['coverage'] for r in results if r['conformal']['coverage'] > 0]
        
        sp_mean = np.mean(spearmans)
        sp_std = np.std(spearmans) if len(spearmans) > 1 else 0
        pe_mean = np.mean(pearsons)
        pe_std = np.std(pearsons) if len(pearsons) > 1 else 0
        ece_mean = np.mean(eces)
        ece_std = np.std(eces) if len(eces) > 1 else 0
        
        if coverages:
            cov_mean = np.mean(coverages)
            cov_std = np.std(coverages) if len(coverages) > 1 else 0
            cov_str = f"{cov_mean:.3f}±{cov_std:.3f}"
        else:
            cov_str = "N/A"
        
        print(f"{backbone:<25} {split:<8} {len(results):<6} "
              f"{sp_mean:.4f}±{sp_std:.4f}   "
              f"{pe_mean:.4f}±{pe_std:.4f}   "
              f"{ece_mean:.4f}±{ece_std:.4f} "
              f"{cov_str}")
    
    print("-" * 80)
    
    # Best results per split
    print("\n" + "=" * 80)
    print("BEST RESULTS PER SPLIT")
    print("=" * 80)
    
    for split in ['A', 'B', 'C']:
        split_results = [r for r in all_results if r['split'] == split]
        if not split_results:
            continue
        
        best = max(split_results, key=lambda r: r['test_metrics']['spearman'])
        print(f"\nSplit {split} (best Spearman):")
        print(f"  Backbone:  {best['backbone']}")
        print(f"  Seed:      {best['seed']}")
        print(f"  Spearman:  {best['test_metrics']['spearman']:.4f}")
        print(f"  Pearson:   {best['test_metrics']['pearson']:.4f}")
        print(f"  ECE:       {best['test_metrics']['ece']:.4f}")
        print(f"  Coverage:  {best['conformal']['coverage']:.4f}")
        print(f"  Time:      {best['training_time_seconds']/60:.1f} min")
    
    # Thesis targets check
    print("\n" + "=" * 80)
    print("THESIS TARGETS CHECK")
    print("=" * 80)
    
    targets = {
        'Spearman ≥ 0.91 (Split A)': lambda r: r['split'] == 'A' and r['test_metrics']['spearman'] >= 0.91,
        'ECE < 0.05': lambda r: r['test_metrics']['ece'] < 0.05,
        'Coverage 90%±2%': lambda r: r['conformal']['coverage'] > 0 and abs(r['conformal']['coverage'] - 0.90) < 0.02,
    }
    
    for target_name, check_fn in targets.items():
        passing = [r for r in all_results if check_fn(r)]
        if passing:
            best = max(passing, key=lambda r: r['test_metrics']['spearman'])
            print(f"\n✓ {target_name}")
            print(f"  Best: {best['backbone']} (seed {best['seed']})")
            print(f"  Spearman={best['test_metrics']['spearman']:.4f}, ECE={best['test_metrics']['ece']:.4f}, Coverage={best['conformal']['coverage']:.4f}")
        else:
            # Find closest
            if 'Spearman' in target_name:
                split_a = [r for r in all_results if r['split'] == 'A']
                if split_a:
                    closest = max(split_a, key=lambda r: r['test_metrics']['spearman'])
                    print(f"\n✗ {target_name}")
                    print(f"  Closest: {closest['backbone']} (seed {closest['seed']}) = {closest['test_metrics']['spearman']:.4f}")
            else:
                print(f"\n✗ {target_name}: No experiments pass this target yet")
    
    # Save CSV summary
    csv_path = results_dir / 'summary.csv'
    with open(csv_path, 'w') as f:
        f.write('backbone,split,seed,spearman,pearson,mse,mae,ece,coverage,avg_width,time_min,gpu,best_epoch\n')
        for r in sorted(all_results, key=lambda x: (x['backbone'], x['split'], x['seed'])):
            f.write(f"{r['backbone']},{r['split']},{r['seed']},"
                    f"{r['test_metrics']['spearman']:.6f},"
                    f"{r['test_metrics']['pearson']:.6f},"
                    f"{r['test_metrics']['mse']:.6f},"
                    f"{r['test_metrics']['mae']:.6f},"
                    f"{r['test_metrics']['ece']:.6f},"
                    f"{r['conformal']['coverage']:.6f},"
                    f"{r['conformal']['avg_width']:.6f},"
                    f"{r['training_time_seconds']/60:.1f},"
                    f"{r.get('gpu', 'unknown')},"
                    f"{r['best_epoch']}\n")
    print(f"\nCSV summary saved to: {csv_path}")


if __name__ == '__main__':
    main()
