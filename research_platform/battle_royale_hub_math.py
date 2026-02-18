import os
import time
import subprocess
import json

def run_mlx_eval(model_path, model_name):
    # Check if path exists
    if not os.path.exists(model_path):
        print(f"❌ Skip: {model_name} (Path not found: {model_path})")
        return None

    results = []
    prompts = [
        "Solve for x: 3x + 12 = 5x - 4. Show steps.",
        "Draw a LaTeX table for a 3x3 identity matrix.",
        "Describe the architectural constraints of a microscopic heat engine using the second law of thermodynamics."
    ]

    print(f"\n🚀 BATTLE ROYALE ROUND 2: {model_name}")
    print("-" * 50)

    for i, prompt in enumerate(prompts):
        print(f"[*] Task {i+1}: {prompt[:50]}...")

        cmd = [
            "python3", "-m", "mlx_lm.generate",
            "--model", model_path,
            "--prompt", prompt,
            "--max-tokens", "500",
            "--temp", "0.0"
        ]

        start = time.time()
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            end = time.time()
            output = res.stdout

            tps = 0.0
            if "Generation:" in output:
                try:
                    parts = output.split("Generation:")[1].split("tokens-per-sec")[0].strip().split(",")
                    if len(parts) > 1:
                        tps = float(parts[1].strip())
                except:
                    pass

            # Logic check for the math problem
            math_pass = False
            if i == 0:
                if "x = 8" in output or "x=8" in output:
                    math_pass = True

            results.append({
                "task": i+1,
                "tps": tps,
                "latency": end - start,
                "math_pass": math_pass,
                "output_snippet": output[-300:].strip()
            })

        except Exception as e:
            print(f"❌ Error during eval: {e}")

    return results

if __name__ == "__main__":
    # STRICT SSD POLICY: All paths must be in /Volumes/Elements
    models = {
        "Gemma-3-4B-MLX": "/Volumes/Elements/edison_models/eval/gemma_3_4b",
        "Qwen2.5-Math-1.5B-MLX": "/Volumes/Elements/edison_models/eval/qwen_math_1.5b",
        "Granite-4.0-Small-MLX": "/Volumes/Elements/edison_models/eval/granite_4_small"
    }

    final_report = {}
    for name, path in models.items():
        res = run_mlx_eval(path, name)
        if res:
            final_report[name] = res

    with open("/Users/studio/Desktop/PhD/Proposal/research_platform/battle_royale_math_results.json", "w") as f:
        json.dump(final_report, f, indent=4)

    print("\n✅ Math & Architecture Evaluations Complete. Results saved to battle_royale_math_results.json")
