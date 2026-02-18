
import os

def fix_ch8():
    path = "/Users/studio/Desktop/PhD/Proposal/Chapter_8_Complete.tex"
    with open(path, 'r') as f:
        lines = f.readlines()

    # Identify the block to remove at start
    start_idx = -1
    end_idx = -1

    # Look for the misplaced block near the top
    for i, line in enumerate(lines[:100]):
        if "\\subsection{Risk-Based Guide Ranking}" in line:
            start_idx = i
            # Find end of figure
            for j in range(i, i+30):
                if "\\end{figure}" in lines[j]:
                    end_idx = j
                    break
            break

    if start_idx != -1 and end_idx != -1:
        print(f"Ch8: Found misplaced block at lines {start_idx}-{end_idx}")
        # Extract the content to move (we have the text, but let's use a fresh string to be clean)
        # Actually, let's just remove it. I know what to insert.
        del lines[start_idx:end_idx+1]

    # Find insertion point
    insert_idx = -1
    for i, line in enumerate(lines):
        if "\\subsection{Regulatory Pathway to FDA Approval}" in line:
            insert_idx = i
            break

    if insert_idx != -1:
        print(f"Ch8: Inserting at line {insert_idx}")
        block = [
            "\\subsection{Risk-Based Guide Ranking}\n",
            "\n",
            "\\begin{figure}[h!]\n",
            "    \\centering\n",
            "    \\includegraphics[width=1.0\\textwidth]{figures/fig_8_3.png}\n",
            "    \\caption[Clinical Dashboard UI Mockup]{Wireframe of the Clinical Decision Support Dashboard. The interface ranks guides by a composite Quality Score (Efficiency - Risk), highlighting \"Safe Recommendation\" guides in green and \"High Risk\" guides in red. Confidence intervals are visually displayed.}\n",
            "    \\label{fig:clinical_ui}\n",
            "\\end{figure}\n",
            "\n"
        ]
        lines[insert_idx:insert_idx] = block

    with open(path, 'w') as f:
        f.writelines(lines)
    print("Ch8 fixed.")

def fix_ch11():
    path = "/Users/studio/Desktop/PhD/Proposal/Chapter_11_Complete.tex"
    with open(path, 'r') as f:
        lines = f.readlines()

    # Find ASCII art block
    start_idx = -1
    end_idx = -1

    for i, line in enumerate(lines):
        if "Pareto Frontier: Accuracy vs Inference Latency" in line:
            # This is inside the block. Find boundaries.
            # Scan backwards for \begin{figure}
            for j in range(i, i-20, -1):
                if "\\begin{figure}" in lines[j]:
                    start_idx = j
                    break
            # Scan forwards for \end{figure}
            for j in range(i, i+40):
                if "\\end{figure}" in lines[j]:
                    end_idx = j
                    break
            break

    if start_idx != -1 and end_idx != -1:
        print(f"Ch11: Removing ASCII art at lines {start_idx}-{end_idx}")
        del lines[start_idx:end_idx+1]

    with open(path, 'w') as f:
        f.writelines(lines)
    print("Ch11 fixed.")

def fix_ch12():
    path = "/Users/studio/Desktop/PhD/Proposal/Chapter_12_Complete.tex"
    with open(path, 'r') as f:
        lines = f.readlines()

    # Remove misplaced fig_12_1
    start_idx = -1
    end_idx = -1

    for i, line in enumerate(lines):
        if "Figure 12.1: CRISPRO-MAMBA-X System Overview" in line:
            # Found the text header
            start_idx = i
        if "\\label{fig:system_overview}" in line:
             # Find end figure
             for j in range(i, i+5):
                 if "\\end{figure}" in lines[j]:
                     end_idx = j
                     break

    if start_idx != -1 and end_idx != -1:
        print(f"Ch12: Removing misplaced block at lines {start_idx}-{end_idx}")
        del lines[start_idx:end_idx+1]

    # Insert fig_12_1 at correct location
    insert_idx = -1
    for i, line in enumerate(lines):
        if "\\subsection{Global Health and Health Equity Considerations}" in line:
            # Insert after the Opportunities subsection header or list?
            # Looking for 443 approx.
            # There is an enumerate list.
            pass

    # Refine search
    for i, line in enumerate(lines):
        if "Global Health Impact Map" in line: # Check if already there?
             # If correct figure is there, skip.
             pass

    # Try finding "Opportunities"
    for i, line in enumerate(lines):
        if "\\subsubsection{Opportunities}" in line:
            insert_idx = i + 2 # After \begin{enumerate} usually?
            break

    if insert_idx != -1:
        print(f"Ch12: Inserting at line {insert_idx}")
        block = [
            "\n",
            "\\begin{figure}[h!]\n",
            "    \\centering\n",
            "    \\includegraphics[width=1.0\\textwidth]{figures/fig_12_1.png}\n",
            "    \\caption[Global Health Impact Map]{Heatmap of genetic disease burden overlaid with potential CRISPRO deployment sites. The map highlights high-burden regions (e.g., Sub-Saharan Africa for Sickle Cell) where democratized access to optimized gene editing could have the greatest humanitarian impact.}\n",
            "    \\label{fig:global_impact}\n",
            "\\end{figure}\n",
            "\n"
        ]
        lines[insert_idx:insert_idx] = block

    with open(path, 'w') as f:
        f.writelines(lines)
    print("Ch12 fixed.")

if __name__ == "__main__":
    fix_ch8()
    fix_ch11()
    fix_ch12()
