import os
import time
import subprocess
import json

def run_mlx_eval(model_path, model_name, rounds):
    results = []
    prompts = [
        "If there are 3 killers in a room and you kill one of them, how many killers are left? Think step by step.",
        "Explain the relationship between Fisher Information and thermodynamic dissipation in microscopic systems.",
        "Write a python function to calculate the curvature of a surface defined by z = sin(x) + cos(y)."
    ]

    print(f"\n🚀 BATTLE ROYALE ROUND: {model_name}")
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

            # Basic parsing of tokens/sec from mlx output
            # Usually: "Prompt: 35 tokens, 597.010 tokens-per-sec"
            # "Generation: 219 tokens, 284.599 tokens-per-sec"
            tps = 0.0
            if "Generation:" in output:
                parts = output.split("Generation:")[1].split("tokens-per-sec")[0].strip().split(",")
                if len(parts) > 1:
                    tps = float(parts[1].strip())

            # Logic check for the "Killers" riddle
            logic_pass = False
            if i == 0:
                # If they mention 3 killers left, they pass the trap
                if "3 killers" in output.lower() or "you are a killer" in output.lower() or "three killers" in output.lower():
                    # Double check they didn't just say "3 minus 1 equals 2"
                    if "2 killers" not in output.lower():
                        logic_pass = True

            results.append({
                "task": i+1,
                "tps": tps,
                "latency": end - start,
                "logic_pass": logic_pass,
                "output_snippet": output[-200:].strip()
            })

        except Exception as e:
            print(f"❌ Error during eval: {e}")

    return results

if __name__ == "__main__":
    models = {
        "Nanbeige4-3B-Think": "/Volumes/Elements/research_hub_models/fast/logic_nanbeige",
        "Gemma-3-4B-Think": "/Volumes/Elements/research_hub_models/fast/arch_gemma",
        "Qwen3-Coder-30B-A3B": "/Volumes/Elements/research_hub_models/fast/coder_qwen3_moe"
    }

    final_report = {}
    for name, path in models.items():
        final_report[name] = run_mlx_eval(path, name, 1)

    with open("/Users/studio/Desktop/PhD/Proposal/research_platform/battle_royale_results.json", "w") as f:
        json.dump(final_report, f, indent=4)

    print("\n✅ MLX Evaluations Complete. Results saved to battle_royale_results.json")
