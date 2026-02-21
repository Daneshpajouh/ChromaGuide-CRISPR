import json
import os
import glob

def main():
    results_dir = "results/ablation_modality"
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} not found.")
        return

    # Files follow ablation_{modality}.json
    files = glob.glob(os.path.join(results_dir, "ablation_*.json"))

    summary = {}
    for f in files:
        mod = os.path.basename(f).replace("ablation_", "").replace(".json", "")
        with open(f, "r") as jf:
            data = json.load(jf)
            # Find max rho across epochs
            if 'epochs' in data:
                max_rho = max([e.get('rho', 0) for e in data['epochs']])
                summary[mod] = max_rho
            else:
                summary[mod] = data.get('final_rho', 0)

    print("\n\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lc}")
    print("\\hline")
    print("\\textbf{Modality Combination} & \\textbf{Spearman $\\rho$} \\\\")
    print("\\hline")

    # Priority order for display
    order = ["seq_only", "seq_dnase", "seq_h3k4me3", "seq_ctcf", "full_multimodal"]
    for mod in order:
        val = summary.get(mod, "TBD")
        label = mod.replace("_", "+").upper()
        if isinstance(val, float):
            print(f"{label} & {val:.3f} \\\\")
        else:
            print(f"{label} & {val} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Ablation Study: Contribution of Individual Epigenomic Modalities.}")
    print("\\label{tab:ablation_modality}")
    print("\\end{table}")

if __name__ == "__main__":
    main()
