# REFINED PhD PROPOSAL PRESENTATION PROMPT: HIGH-FIDELITY ARCHITECT

## 🚀 THE MISSION
Act as a **Senior Frontend Engineer, Presentation Designer, and PhD Committee Member**. Your goal is to generate a **single-file, high-fidelity HTML presentation** for a PhD proposal defense. The design must be **pixel-perfect for PDF conversion (16:9 ratio)** and feel premium, academically rigorous, and modern.

## 🎨 DESIGN SYSTEM (CSS BOILERPLATE)
The presentation must use a "Deep Space" theme with SFU Red accents. Generate the HTML following this structure:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-body: #020617;
            --bg-slide: #0f172a;
            --bg-card: #1e293b;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --accent-main: #38bdf8; /* Sky Blue */
            --accent-sfu: #CC0633;  /* SFU Red */
            --accent-green: #10b981;
            --accent-amber: #fbbf24;
            --border: #1e293b;
        }
        body {
            margin: 0; padding: 0; background-color: var(--bg-body);
            font-family: 'Inter', Arial, sans-serif; display: flex;
            flex-direction: column; align-items: center; color: var(--text-primary);
            -webkit-print-color-adjust: exact; print-color-adjust: exact;
        }
        .slide {
            width: 960px; height: 540px; margin-bottom: 40px;
            box-sizing: border-box; overflow: hidden; display: flex;
            flex-direction: column;
            background-color: var(--bg-slide); border: 1px solid var(--border);
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5);
            position: relative;
            padding: 40px 60px;
            -webkit-print-color-adjust: exact; print-color-adjust: exact;
        }
        header { border-left: 4px solid var(--accent-sfu); padding-left: 20px; margin-bottom: 30px; }
        h1 { font-size: 28px; margin: 0; color: var(--text-primary); text-transform: uppercase; letter-spacing: 1px; }
        .content { flex: 1; font-size: 18px; line-height: 1.6; }
        .footer {
            position: absolute; bottom: 20px; width: calc(100% - 120px);
            display: flex; justify-content: space-between;
            font-size: 12px; color: var(--text-secondary); opacity: 0.7;
        }
        .card {
            background: var(--bg-card); border-radius: 12px; padding: 20px;
            border: 1px solid rgba(255,255,255,0.05); margin-bottom: 20px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }
        .accent { color: var(--accent-main); font-weight: 600; }
        .sfu-red { color: var(--accent-sfu); font-weight: 700; }
        ul { list-style: none; padding-left: 0; }
        li { margin-bottom: 12px; position: relative; padding-left: 25px; }
        li::before { content: "→"; position: absolute; left: 0; color: var(--accent-sfu); }
        .img-container {
            width: 100%; height: 250px; background: rgba(0,0,0,0.2);
            border: 1px dashed var(--border); display: flex; align-items: center;
            justify-content: center; border-radius: 8px; margin: 10px 0;
            font-size: 14px; font-style: italic; color: var(--text-secondary);
        }
        .grid { display: grid; gap: 20px; grid-template-columns: repeat(2, 1fr); }
        .split { display: flex; gap: 30px; align-items: center; }

        /* PRINT ENGINE - PIXEL PERFECT FOR 10" x 5.625" (16:9) */
        @page { size: 10in 5.625in; margin: 0; }
        @media print {
            body { background-color: transparent; }
            .slide { margin-bottom: 0; border: none; page-break-after: always; box-shadow: none; }
        }
    </style>
</head>
<body>
    <!-- Template Loop for Slides -->
</body>
</html>
```

## 📝 CONTENT OBJECTIVES
Topic: **ChromaGuide: A Multi-Modal Deep Learning Framework for Comprehensive CRISPR-Cas9 Guide RNA Design and Efficacy Prediction**

### GLOBAL CONSTRAINTS:
1. **Academic Rigor**: Use numbered citations [1, 2, 3...] linked to a final Reference slide.
2. **Key Corrections**:
   - ChromeCRISPR is a **bioRxiv preprint (April 16, 2025)**.
   - Nobel Prize in Chemistry was **2020** (Doudna & Charpentier).
   - FDA approvals: UK (Nov 2023), US (Dec 2023/Jan 2024), EU (Dec 2023).
3. **Visual Balance**: Max 5-7 bullet points. Use cards for grouping.
4. **Imagery**: Reference local paths for existing figures:
   - `/Users/studio/Desktop/PhD/Proposal/Presentation/ChromaGuide__A_Multi_Modal_Deep_Learning_Framework_for_Comprehensive_CRISPR_Cas9_Guide_RNA_Design_and_Efficacy_Prediction/Figs/ChromaGuide.png`
   - `/Users/studio/Desktop/PhD/Proposal/Presentation/ChromaGuide__A_Multi_Modal_Deep_Learning_Framework_for_Comprehensive_CRISPR_Cas9_Guide_RNA_Design_and_Efficacy_Prediction/Figs/CRISPR-Cas9.png`
   - `/Users/studio/Desktop/PhD/Proposal/Presentation/ChromaGuide__A_Multi_Modal_Deep_Learning_Framework_for_Comprehensive_CRISPR_Cas9_Guide_RNA_Design_and_Efficacy_Prediction/Figs/Data.png`

## 📊 SLIDE STRUCTURE (23 SLIDES)
1. **Title Slide**: Student: Amirhossein Daneshpajouh, SFU Red theme.
2. **Clinical Impact**: 2020 Nobel, FDA 2023/24 updates.
3. **Core Problem**: Sequence-centric bias, lack of uncertainty, generalization failure.
4. **Prior Success (ChromeCRISPR)**: bioRxiv 2025 results (ρ = 0.876).
5. **Research Gap**: Context Blindness, Uncertainty Vacuum, Evaluation Leakage.
6. **Research Questions (RQ1-3)**: Multi-modal fusion, Off-target risk, Integrated ranking.
7. **Expected Contributions**: CS/ML, CompBio, Bioinformatics, Genomics impacts.
8. **ChromaGuide Architecture**: Use `ChromaGuide.png`.
9. **CRISPR-Cas9 Mechanism**: Use `CRISPR-Cas9.png`.
10. **Input Data**: Sequence encoding + Epigenomic tracks (ENCODE).
11. **Module 1 (Sequence)**: CNN-GRU vs. Mamba-SSM (selective state spaces).
12. **Module 1 (Epigenomics)**: Fusion layer + MINE/CLUB non-redundancy objective.
13. **Module 1 (Uncertainty)**: Beta Regression + Conformal Prediction (Weighted).
14. **Module 2 (Off-Target)**: Candidate enumeration + Site-level scoring.
15. **Module 3 (Integration)**: Balancing Efficiency vs. Specificity (Formula: w_on·μ_on - w_off·R).
16. **Datasets & Baselines**: DeepHF (Wang 2019 Nat Commun), CRISPRon, etc.
17. **Leakage-Controlled Eval**: Use `Data.png` for SPLIT strategies (Gene/Dataset/Cell-line).
18. **Metrics**: Spearman, Coverage Calibration, Statistical Rigor.
19. **Expected Results**: Target ρ ≥ 0.880, 90% coverage band.
20. **Timeline**: 6-month plan (Feb - Aug 2026).
21. **Broader Impacts**: Lab to Clinic pathway + Responsible AI.
22. **Summary**: Context-Aware, Uncertainty-Quantified sgRNA Design.
23. **References**: Consolidated list [1-15] using APA format.

## 🖨️ PDF CONVERSION COMMAND
```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --headless --disable-gpu --print-to-pdf="ChromaGuide_Proposal.pdf" --no-pdf-header-footer "presentation.html"
```

## ⚡ TASK
**Generate the full `presentation.html` file now.** Ensure every slide contains high-quality academic content, professional layout utilities, and clear placeholders for charts. Make it ready for a PhD committee.
