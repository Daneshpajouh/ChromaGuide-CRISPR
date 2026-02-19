import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.sota_comparison import SOTABenchmark

def main():
    bench = SOTABenchmark()
    baselines = bench.BASELINES

    print("="*60)
    print("CHROMAGUIDE V2: COMPETITIVE LANDSCAPE")
    print("="*60)
    print(f"{'Model':<15} | {'Year':<5} | {'Metric (ρ)':<10} | {'Methodology'}")
    print("-" * 60)

    sorted_baselines = sorted(baselines.values(), key=lambda x: x.primary_metric, reverse=True)

    for b in sorted_baselines:
        print(f"{b.model_name:<15} | {b.year:<5} | {b.primary_metric:<10.3f} | {b.notes[:50]}...")

    target = 0.911
    print("-" * 60)
    print(f"TARGET TO BEAT: {target:.3f} (CCL/MoFF)")
    print("="*60)

    # Generate a dummy placeholder results file for the front-end if needed
    placeholder_results = {
        "status": "TRAINING_IN_PROGRESS",
        "job_id": "56718038",
        "target": target,
        "baselines_count": len(baselines)
    }

    with open("current_target_status.json", "w") as f:
        json.dump(placeholder_results, f, indent=4)
    print("Saved current_target_status.json")

if __name__ == "__main__":
    main()
