import json
import sys
from pathlib import Path

def generate_latex_table(results):
    rho = results.get('final_gold_rho', 0.0)
    target = 0.911
    improvement = rho - target

    status = "✓ SOTA ACHIEVED" if rho >= target else "❌ BELOW SOTA"

    latex = f"""
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lcccc}}
\\hline
\\textbf{{Model}} & \\textbf{{Modality}} & \\textbf{{Spearman $\\rho$}} & \\textbf{{Dataset}} & \\textbf{{Status}} \\\\
\\hline
CCL/MoFF (2025) & Seq+Epi & 0.911 & DeepHF & Baseline \\\\
ChromaGuide V1 & Seq & 0.872 & DeepHF & Previous \\\\
\\textbf{{ChromaGuide V2}} & \\textbf{{Seq (Beta Reg)}} & \\textbf{{{rho:.3f}}} & \\textbf{{DeepHF}} & \\textbf{{{status}}} \\\\
\\hline
\\end{{tabular}}
\\caption{{Final Comparison against State-of-the-Art (SOTA) on DeepHF Gold Set.}}
\\label{{tab:final_results}}
\\end{{table}}
"""
    return latex

def main():
    res_path = Path("results_v2_production.json")
    if not res_path.exists():
        # Check if remote file is available via ssh
        print("Results file not found locally. Waiting for cluster job...")
        return

    with open(res_path, "r") as f:
        data = json.load(f)

    print("\n" + "="*40)
    print("PRODUCTION RESULTS ANALYSIS")
    print("="*40)
    print(f"Final Spearman Rho: {data['final_gold_rho']:.4f}")
    print(f"Target Rho:         0.9110")
    print(f"Delta:              {data['final_gold_rho'] - 0.911:.4f}")
    print(f"Target Reached:     {data['target_reached']}")
    print("-" * 40)

    latex = generate_latex_table(data)
    print("LATEX TABLE SNIPPET:")
    print(latex)

    with open("FINAL_RESULTS_SNIPPET.tex", "w") as f:
        f.write(latex)
    print("Saved to FINAL_RESULTS_SNIPPET.tex")

if __name__ == "__main__":
    main()
