import os
import time
import subprocess
import json

def run_benchmark(model_path, prompt):
    print(f"[*] Benchmarking Model: {model_path}")
    print(f"[*] Prompt: {prompt}\n")

    cmd = [
        "python3", "-m", "mlx_lm.generate",
        "--model", model_path,
        "--prompt", prompt,
        "--max-tokens", "1000",
        "--temp", "0.0"
    ]

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()

        duration = end_time - start_time
        output = result.stdout

        # Estimate tokens/sec (very rough based on output length / duration)
        # mlx_lm usually prints the actual speed at the end, let's extract it if possible
        print(output)
        print(f"\n[+] Full Cycle Time: {duration:.2f}s")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr}")

if __name__ == "__main__":
    model_path = "/Volumes/Elements/edison_models/fast/r1_1.5b"
    prompt = "If there are 3 killers in a room and you kill one of them, how many killers are left in the room? Think step by step."
    run_benchmark(model_path, prompt)
