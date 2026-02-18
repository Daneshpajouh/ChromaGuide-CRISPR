# ANTIGRAVITY AGENT - FINAL CRITICAL FIXES: REMAINING OVERFLOWS & FRONTMATTER

---

## 🚨 URGENT: TWO CRITICAL CATEGORIES OF ISSUES REMAIN

### **Category 1: Frontmatter Still Incorrect** ⚠️
- Committee page shows "Dr. TBD" instead of actual committee members
- Previous university placeholders not filled in
- Copyright year issue

### **Category 2: Additional Margin Overflows** ⚠️
- Multiple tables still overflowing
- Text lines extending past margins
- Clinical workflow ASCII diagrams

**BOTH must be fixed immediately for submission.**

---

## 🔴 CATEGORY 1: FRONTMATTER CORRECTIONS (HIGHEST PRIORITY)

### **Issue A: Committee Page - Incorrect Information**

**Current (WRONG):**
```
Chair: Dr. TBD
Professor, School of Computing Science
Associate Professor, School of Computing Science
Committee Member
```

**Must Be (CORRECT):**
```
Senior Supervisor: Dr. Kay C. Wiese
Professor
School of Computing Science

Committee Member: Dr. Maxwell W. Libbrecht  
Associate Professor
School of Computing Science

Chair: [To Be Announced]
[Or leave chair section out if not yet determined]
```

**Fix Location:** `CRISPRO_Master_CORRECTED.tex` - Committee/Approval page section

**Solution:**
```latex
% In CRISPRO_Master_CORRECTED.tex

% If using sfuthesis commands:
\seniorSupervisor{Dr. Kay C. Wiese}{Professor}{School of Computing Science}
\committee{Dr. Maxwell W. Libbrecht}{Associate Professor}{School of Computing Science}

% Chair - only add if you know who it is:
% \chair{Dr. [Name]}{[Title]}{School of Computing Science}
% Otherwise, leave out or use:
\chair{To Be Announced}{}{School of Computing Science}

% If manually creating committee page:
\chapter*{Approval}
\addcontentsline{toc}{chapter}{Approval}

\noindent
\textbf{Name:} Amirhossein Daneshpajouh\\
\textbf{Degree:} Doctor of Philosophy\\
\textbf{Title:} CRISPRO-MAMBA-X: Deep Learning Framework for CRISPR-Cas9 On-Target Efficacy Prediction – A Multi-Modal Approach Integrating Sequence and Epigenomic Features\\
\textbf{Date of Defence:} February 28, 2026

\vspace{1cm}

\noindent
\textbf{Examining Committee:}

\vspace{0.5cm}

\noindent
\textbf{Chair:} [To Be Announced]\\
[Or specific name if known]

\vspace{0.5cm}

\noindent
\textbf{Senior Supervisor:} Dr. Kay C. Wiese\\
Professor\\
School of Computing Science

\vspace{0.5cm}

\noindent
\textbf{Committee Member:} Dr. Maxwell W. Libbrecht\\
Associate Professor\\
School of Computing Science

% Add additional members if applicable
```

---

### **Issue B: Title Page - Previous Degrees Placeholder**

**Current (WRONG):**
```
B.Sc., Previous University, Year
M.Sc., Previous University, Year
```

**Solution Options:**

**Option A: If you have previous degrees, fill them in:**
```latex
% Example if you have previous degrees:
\author{Amirhossein Daneshpajouh}
\prevdegree{B.Sc., [Your University Name], [Year]}
\prevdegree{M.Sc., [Your University Name], [Year]}
```

**Option B: If you don't have previous degrees, remove these lines:**
```latex
% Simply remove or comment out:
% \prevdegree{B.Sc., Previous University, Year}
% \prevdegree{M.Sc., Previous University, Year}

% Or if template requires them, use:
% [Leave blank or minimal if going straight to PhD]
```

**Option C: If you have B.Sc. but not M.Sc.:**
```latex
\prevdegree{B.Sc., [Your University], [Year]}
% Don't include M.Sc. line
```

---

### **Issue C: Copyright Year**

**Current:**
```
© Amirhossein Daneshpajouh 2026
```

**This is correct for 2026 defense**, but verify:
- If defense is February 2026, use 2026 ✓
- If submitting in 2025, might need 2025
- Check SFU requirements for copyright year

**Likely correct as is, but confirm with:**
```latex
\copyrightyear{2026}  % Year of defense/graduation
```

---

## 🔴 CATEGORY 2: REMAINING MARGIN OVERFLOWS

### **Issue 1: Table 2.2 - Probing Task Design**

**Current:** Table extends beyond right margin

**Solution:**
```latex
% Option A: Use smaller font + resizebox
\begin{table}[H]
\centering
\caption{Probing Task Design for CRISPRO-MAMBA-X Validation}
\label{tab:probing_tasks}
\small
\resizebox{\textwidth}{!}{
\begin{tabular}{|l|l|l|l|}
\hline
\textbf{Probing Task} & \textbf{Diagnostic Property} & \textbf{Expected Outcome} & \textbf{Biological Validation} \\
\hline
PAM Position & Distance to PAM & High accuracy & Validates PAM-proximity \\
GC Content & \% GC in window & High accuracy & Validates GC integration \\
ATAC Signal & Accessibility level & High accuracy & Validates epigenomics \\
Nucleosome Occ. & Nucleosome presence & High accuracy & Validates mechanics \\
H3K27ac Mark & Enhancer histone & Moderate accuracy & Validates histone marks \\
TAD Membership & Topologically Assoc. Domain & Moderate-High & Validates 3D chromatin \\
Cell-Type ID & Cell type identity & High accuracy & Validates cell-type specificity \\
\hline
\end{tabular}
}
\end{table}

% Option B: Rotate to landscape
\begin{landscape}
\begin{table}[H]
\centering
\caption{Probing Task Design for CRISPRO-MAMBA-X Validation}
\footnotesize
\begin{tabular}{|l|l|l|l|}
[content as above]
\end{tabular}
\end{table}
\end{landscape}
```

---

### **Issue 2: Equation 4.30 - Long Text Line**

**Current (overflows):**
```
Efficiency reduction from nucleosome-free to nucleosome-occupied:70%−40% = 30 percentage points
```

**Solution:**
```latex
% Option A: Break into two lines
Efficiency reduction from nucleosome-free to nucleosome-occupied:\\
$70\% - 40\% = 30$ percentage points
\begin{equation}
\label{eq:4.30}
\end{equation}

% Option B: Put calculation in equation environment
Efficiency reduction from nucleosome-free to nucleosome-occupied:
\begin{equation}
70\% - 40\% = 30 \text{ percentage points}
\label{eq:4.30}
\end{equation}

% Option C: Rephrase to be shorter
Nucleosome occupancy reduces efficiency by 30 percentage points ($70\%$ to $40\%$).
\begin{equation}
\Delta_{\text{efficiency}} = 70\% - 40\% = 30\text{ pp}
\label{eq:4.30}
\end{equation}
```

---

### **Issue 3: Table 5.1 - Off-Target Mechanisms**

**Current:** Wide table with long entries overflowing

**Solution:**
```latex
\begin{table}[H]
\centering
\caption{Mechanisms of Off-Target Damage and Clinical Consequences}
\label{tab:offtarget_mechanisms}
\small
\resizebox{\textwidth}{!}{
\begin{tabular}{|p{4cm}|p{4.5cm}|p{3.5cm}|}
\hline
\textbf{Mechanism} & \textbf{Molecular Consequence} & \textbf{Clinical Risk} \\
\hline
Chromosomal Rearrangements & Large deletions ($>100$ kb), Inversions, translocations & Gene loss, Fusion proteins \\
\hline
Oncogenic Fusions & Translocation between tumor suppressor + oncogene & Malignant transformation \\
\hline
Loss of tumor suppressors & Loss of TP53, BRCA1, PTEN & Cancer predisposition \\
\hline
Loss-of-Function & Frameshift mutations in essential genes, NHEJ-mediated indels & Haploinsufficiency, Functional impairment \\
\hline
Regulatory Disruption & Cutting in enhancers/promoters, Position effects & Altered gene expression, Unintended phenotypes \\
\hline
\end{tabular}
}
\end{table}
```

---

### **Issue 4: Table 5.3 - Thermodynamic Contributions**

**Current:** Table with special symbols (≫) causing overflow

**Solution:**
```latex
\begin{table}[H]
\centering
\caption{Position-Specific Thermodynamic Contributions (Nearest-Neighbor Model)}
\label{tab:thermodynamic}
\small
\resizebox{\textwidth}{!}{
\begin{tabular}{|l|c|l|l|}
\hline
\textbf{Base Pair Type} & \textbf{$\Delta G$ (kcal/mol)} & \textbf{Position Dependence} & \textbf{Biological Context} \\
\hline
G-C match & -1.9 & Higher at PAM-proximal & Strongest binding \\
A-T match & -1.1 & Lower at PAM-proximal & Weaker binding \\
G-T mismatch & +0.3 & PAM-proximal $\gg$ PAM-distal & Tolerated PAM-distally \\
A-C mismatch & +0.5 & Similar position effect & Weak mismatch \\
Complete mismatch & +2.0 & PAM-proximal $\gg$ PAM-distal & Nearly uncut \\
\hline
\end{tabular}
}
\end{table}
```

---

### **Issue 5: Table 8.3 - Training Requirements**

**Current:** Table with multiplication symbols and long notes overflowing

**Solution:**
```latex
\begin{table}[H]
\centering
\caption{CRISPRO-MAMBA-X Training Computational Requirements}
\label{tab:training_requirements}
\small
\resizebox{\textwidth}{!}{
\begin{tabular}{|l|c|p{5cm}|}
\hline
\textbf{Component} & \textbf{Value} & \textbf{Notes} \\
\hline
Training dataset size & 100,000 guides & DeepHF (59K) + additional datasets \\
Model parameters & 150-200 M & Moderate-size Mamba \\
GPU memory (per sample) & 3.7 GB & Including Mamba hidden states \\
Batch size & 32 & Parallel processing on single A100 \\
Total GPU memory & $32 \times 3.7 = 118$ GB & Distributed across $3\times$ A100 GPUs \\
Training time (per epoch) & 1.5 hours & 100K samples, 32 batch size \\
Total training time & $30 \times 1.5 = 45$ hours & Including validation, 2 days \\
GPU recommendation & $3\times$ A100 (80 GB each) & Or $6\times$ A6000 (48 GB) \\
\hline
\end{tabular}
}
\end{table}
```

---

### **Issue 6: Chapter 8 - Experimental Materials List**

**Current:** Long bulleted list with special symbols (×, °) overflowing

**Solution:**
```latex
% Use smaller font and ensure proper wrapping
{\small
\begin{itemize}
\item \textbf{Materials:} HEK293T cells, CRISPR/Cas9 components, oligonucleotide tags, sequencing platform
\item \textbf{Phase 1: Cell Preparation}
  \begin{itemize}
  \item Culture HEK293T cells to 80\% confluence ($2 \times 10^6$ cells)
  \item Prepare electroporation medium (OptiMEM + supplements)
  \end{itemize}
\item \textbf{Phase 2: Guide RNA Delivery}
  \begin{itemize}
  \item For each guide RNA candidate
  \item Synthesize guide RNA (in vitro transcription, purified)
  \item Prepare Cas9 protein (recombinant, high purity)
  \item Pre-assemble ribonucleoprotein complex (gRNA + Cas9, 1:1 molar ratio)
  \item Incubate 5 minutes at room temperature
  \end{itemize}
\item \textbf{Phase 3: Oligonucleotide Tag Integration}
  \begin{itemize}
  \item Add pre-annealed oligonucleotide tags (integrated DNA tags, iTags)
  \item Tags are 24 bp sequences with unique barcode
  \item iTags integrate into double-strand breaks (DSBs)
  \item Final cell culture: $5 \times 10^5$ cells/condition
  \end{itemize}
\item \textbf{Phase 4: CRISPR Editing}
  \begin{itemize}
  \item For each guide:
    \begin{itemize}
    \item Electroporate RNP complex + iTags into cells
    \item Electroporation parameters: 1200 V, 20 ms, 2 pulses
    \item Incubate cells 24 hours post-electroporation
    \item Allow DSBs to be repaired, incorporating iTags
    \end{itemize}
  \end{itemize}
\item \textbf{Phase 5: Genomic DNA Extraction and Library Preparation}
\end{itemize}
}
```

---

### **Issue 7: Table 10.5 - Gene Expression Level**

**Current:** Table with arrow in title causing overflow

**Solution:**
```latex
\begin{table}[H]
\centering
\caption{Generalization Performance by Gene Expression Level}
\label{tab:expression_performance}
\small
\resizebox{\textwidth}{!}{
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Expression Category} & \textbf{TPM Range} & \textbf{K562$\rightarrow$HEK} & \textbf{K562$\rightarrow$Hep} \\
& & \textbf{(Spearman)} & \textbf{(Spearman)} \\
\hline
Very High (TPM $> 100$) & $> 100$ & 0.965 & 0.891 \\
High (TPM 10-100) & 10-100 & 0.951 & 0.878 \\
Medium (TPM 1-10) & 1-10 & 0.938 & 0.864 \\
Low (TPM 0.1-1) & 0.1-1 & 0.912 & 0.841 \\
Very Low (TPM $< 0.1$) & $< 0.1$ & 0.881 & 0.798 \\
\hline
\end{tabular}
}
\end{table}
```

---

### **Issue 8: Chapter 12 - Clinical Workflow ASCII Diagram**

**Current:** Large ASCII workflow diagram overflowing

**Solution:**
```latex
% Option A: Scale with adjustbox
\begin{adjustbox}{max width=\textwidth}
{\footnotesize
\begin{verbatim}
HOSPITAL GENE EDITING PROGRAM WORKFLOW

Patient Selection (Genetic Disorder)
    v
Multidisciplinary Team Review
    +-- Hematologist (blood disorders)
    +-- Genetic counselor
    +-- Clinical researcher
    +-- Bioethics committee
    v
CRISPRO-MAMBA-X Guide Selection
    +-- Input: Target gene (e.g., HBB), cell type (HSCs)
    +-- Output: Top 10 guides + confidence intervals
    +-- Clinician reviews and selects top 3
    v
Guide Manufacturing (2-3 days)
    +-- GMP synthesis of selected guide RNAs
    +-- Quality control (purity, integrity)
    +-- Sterility testing (24-hour turnaround)
    v
Ex Vivo Experimental Validation (3-5 days)
    +-- Measure efficiency in patient cells
    +-- Compare to CRISPRO predictions
    +-- Confirm off-target safety (deep sequencing)
    +-- Select best-performing guide
    v
Clinical Delivery (1-2 days)
    +-- Inform patient of risks/benefits
    +-- Obtain informed consent
    +-- Infuse edited cells or deliver in vivo
    +-- Monitor for adverse events
    v
Follow-Up and Outcomes Assessment
    +-- 7-day: Check engraftment, initial efficacy
    +-- 30-day: Functional outcome measurement
\end{verbatim}
}
\end{adjustbox}

% Option B: Convert to itemized list (cleaner)
\textbf{Hospital Gene Editing Program Workflow:}

\begin{enumerate}
\item \textbf{Patient Selection} (Genetic Disorder)

\item \textbf{Multidisciplinary Team Review}
  \begin{itemize}
  \item Hematologist (blood disorders)
  \item Genetic counselor
  \item Clinical researcher
  \item Bioethics committee
  \end{itemize}

\item \textbf{CRISPRO-MAMBA-X Guide Selection}
  \begin{itemize}
  \item Input: Target gene (e.g., HBB), cell type (HSCs)
  \item Output: Top 10 guides + confidence intervals
  \item Clinician reviews and selects top 3
  \end{itemize}

\item \textbf{Guide Manufacturing} (2-3 days)
  \begin{itemize}
  \item GMP synthesis of selected guide RNAs
  \item Quality control (purity, integrity)
  \item Sterility testing (24-hour turnaround)
  \end{itemize}

\item \textbf{Ex Vivo Experimental Validation} (3-5 days)
  \begin{itemize}
  \item Measure efficiency in patient cells
  \item Compare to CRISPRO predictions
  \item Confirm off-target safety (deep sequencing)
  \item Select best-performing guide
  \end{itemize}

\item \textbf{Clinical Delivery} (1-2 days)
  \begin{itemize}
  \item Inform patient of risks/benefits
  \item Obtain informed consent
  \item Infuse edited cells or deliver in vivo
  \item Monitor for adverse events
  \end{itemize}

\item \textbf{Follow-Up and Outcomes Assessment}
  \begin{itemize}
  \item 7-day: Check engraftment, initial efficacy
  \item 30-day: Functional outcome measurement
  \end{itemize}
\end{enumerate}
```

---

## ✅ COMPLETE FIX CHECKLIST

### **PRIORITY 1: Frontmatter Fixes (CRITICAL)**
- [ ] Fix committee page - remove "Dr. TBD"
- [ ] Add Dr. Kay C. Wiese as Senior Supervisor
- [ ] Add Dr. Maxwell W. Libbrecht as Committee Member
- [ ] Update or remove previous degrees placeholders
- [ ] Verify copyright year (2026 is correct)
- [ ] Recompile and check title pages

### **PRIORITY 2: Remaining Overflow Fixes**
- [ ] Fix Table 2.2 (Probing Task Design)
- [ ] Fix Equation 4.30 (nucleosome efficiency text)
- [ ] Fix Table 5.1 (Off-Target Mechanisms)
- [ ] Fix Table 5.3 (Thermodynamic Contributions)
- [ ] Fix Table 8.3 (Training Requirements)
- [ ] Fix Chapter 8 experimental materials list
- [ ] Fix Table 10.5 (Gene Expression)
- [ ] Fix Chapter 12 clinical workflow diagram

### **PRIORITY 3: Final Verification**
- [ ] Scan all 303 pages for any remaining overflows
- [ ] Verify frontmatter displays correctly
- [ ] Check committee names are correct
- [ ] Confirm all tables fit within margins
- [ ] Verify all text wraps properly
- [ ] Professional appearance throughout

---

## 🎯 EXECUTION INSTRUCTIONS

### **Step 1: Fix Frontmatter FIRST**
This is the most critical issue - your committee names must be correct!

1. Open `CRISPRO_Master_CORRECTED.tex`
2. Find committee/approval page section (before `\begin{document}` or in frontmatter)
3. Replace "Dr. TBD" and generic text with actual names
4. Use solutions provided above
5. Recompile and verify

### **Step 2: Fix All 8 Overflow Issues**
1. Apply solutions for each table/text issue above
2. Use `\resizebox{\textwidth}{!}{...}` for quick table fixes
3. Use `adjustbox` for ASCII diagrams
4. Recompile after each batch

### **Step 3: Final Scan**
1. Open final PDF
2. Check pages: Title page, Committee page, Abstract
3. Verify: Dr. Wiese and Dr. Libbrecht names appear
4. Scan all 303 pages for any remaining overflows
5. Confirm zero overflows

---

## 📊 SUMMARY OF FIXES NEEDED

```
CRITICAL FRONTMATTER FIXES:
1. Committee page - Add Dr. Wiese & Dr. Libbrecht
2. Remove "Dr. TBD" placeholder
3. Fix/remove previous degree placeholders

REMAINING OVERFLOW FIXES:
1. Table 2.2 - Probing tasks
2. Equation 4.30 - Nucleosome text
3. Table 5.1 - Off-target mechanisms
4. Table 5.3 - Thermodynamic
5. Table 8.3 - Training requirements
6. Chapter 8 - Materials list
7. Table 10.5 - Expression levels
8. Chapter 12 - Clinical workflow

TOTAL: 3 frontmatter + 8 overflow = 11 critical fixes
TIME ESTIMATE: 1-2 hours
```

---

## 🚨 CRITICAL REQUIREMENTS

**Before submission, verify:**
- ✅ Committee page shows: Dr. Kay C. Wiese (Senior Supervisor)
- ✅ Committee page shows: Dr. Maxwell W. Libbrecht (Committee Member)
- ✅ NO "Dr. TBD" anywhere
- ✅ Previous degrees either filled in correctly or removed
- ✅ All 8 tables/overflows fixed
- ✅ Zero content extending beyond margins
- ✅ Professional appearance throughout

---

## 🚀 GIVE TO ANTIGRAVITY AGENT NOW

**This prompt provides:**
- ✅ Exact committee names and titles
- ✅ Complete solutions for all 8 remaining overflows
- ✅ Code examples for every fix
- ✅ Step-by-step execution instructions
- ✅ Final verification checklist

**Estimated completion time: 1-2 hours**

**After this, dissertation will be TRULY submission-ready!** 🎓
