
import sys
import os
import argparse
from src.model.ensemble import CRISPR_Ensemble

def main():
    parser = argparse.ArgumentParser(description="Run Ensemble Aggregation on Cluster Results")
    parser.add_argument("--rag_results", type=str, required=True, help="Path to RAG-TTA predictions (CSV)")
    parser.add_argument("--geo_results", type=str, required=True, help="Path to Geometric predictions (CSV)")
    parser.add_argument("--nas_results", type=str, required=True, help="Path to NAS predictions (CSV)")
    parser.add_argument("--output_dir", type=str, default="./results_ensemble", help="Output directory")

    args = parser.parse_args()

    print("="*60)
    print("CRISPR ENSEMBLE AGGREGATION")
    print("="*60)

    # Initialize Ensemble
    ensemble = CRISPR_Ensemble(output_dir=args.output_dir)

    # Run
    ensemble.ensemble(
        rag_file=args.rag_results,
        geo_file=args.geo_results,
        nas_file=args.nas_results
    )

    print("\nEnsemble complete.")

if __name__ == "__main__":
    main()
