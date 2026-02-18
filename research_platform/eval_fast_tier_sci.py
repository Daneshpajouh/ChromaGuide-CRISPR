import os
import time
import subprocess

def run_scientific_benchmark(model_path, prompt):
    print(f"[*] Scientific Benchmarking: {model_path}")

    cmd = [
        "python3", "-m", "mlx_lm.generate",
        "--model", model_path,
        "--prompt", prompt,
        "--max-tokens", "1000",
        "--temp", "0.0"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr}")

if __name__ == "__main__":
    model_path = "/Volumes/Elements/edison_models/fast/r1_1.5b"
    prompt = """
    Phase: PhD Ideation
    Topic: Geometric Biothermodynamics
    Task: Explain the relationship between the curvature of the statistical manifold and the thermodynamic efficiency of a microscopic heat engine.
    Use Chain-of-Thought reasoning.
    """
    run_scientific_benchmark(model_path, prompt)
