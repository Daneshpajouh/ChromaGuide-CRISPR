# ANTIGRAVITY AGENT FOLLOW-UP PROMPT
## Phase 2: Final Corrections & Professional Formatting
**PERSONALIZED FOR AMIRHOSSEIN DANESHPAJOUH - SFU**

---

## 🎯 FOLLOW-UP MISSION AFTER SUCCESSFUL COMPILATION

**Current Status:** Compilation successful ✅ - PDF generated without errors

**Remaining Issues to Fix:**
1. ⚠️ **Tables overflowing page margins** (not fitting properly)
2. ⚠️ **Frontmatter missing/incorrect** (title page, author info, committee)
3. ⚠️ **Table modularization** (still pending from original plan)

**Mission Objective:** Fix these 3 remaining issues to produce a professionally formatted, submission-ready dissertation.

---

## 🔴 ISSUE 1: TABLES OVERFLOWING PAGE MARGINS (PRIORITY 1)

### Problem Description
Multiple tables are extending beyond page margins, making content unreadable or cut off in PDF. This is unprofessional and unacceptable for PhD defense.

### Root Causes
1. Tables have too many columns for page width
2. Column specifications too wide (`|l|c|c|c|c|c|c|`...)
3. Long text in cells causing overflow
4. No text wrapping enabled
5. Missing `\small` or `\footnotesize` commands
6. Not using flexible table packages

### Solution Strategy

#### Option A: Use `\resizebox` (Quick Fix - RECOMMENDED)
Automatically scales table to fit page width:

```latex
% BEFORE (overflowing):
\begin{table}[H]
\centering
\caption{Wide Table Caption}
\label{tab:wide}
\begin{tabular}{|l|c|c|c|c|c|c|}
[lots of content]
\end{tabular}
\end{table}

% AFTER (fits page):
\begin{table}[H]
\centering
\caption{Wide Table Caption}
\label{tab:wide}
\resizebox{\textwidth}{!}{%
\begin{tabular}{|l|c|c|c|c|c|c|}
[lots of content]
\end{tabular}
}
\end{table}
```

**Pros:** Very simple, works immediately
**Cons:** Text may become small (but readable)

#### Option B: Use `tabularx` with Text Wrapping
Allows flexible column widths with automatic text wrapping:

```latex
% BEFORE (fixed-width columns):
\begin{tabular}{|l|c|c|c|}
\hline
Very Long Header Text & Data 1 & Data 2 & Data 3 \\
\hline
\end{tabular}

% AFTER (flexible with wrapping):
\begin{tabularx}{\textwidth}{|X|c|c|c|}
\hline
Very Long Header Text & Data 1 & Data 2 & Data 3 \\
\hline
\end{tabularx}
```

**Pros:** Text remains readable, professional appearance
**Cons:** Requires more manual adjustment

#### Option C: Reduce Font Size
Make table text smaller to fit more content:

```latex
\begin{table}[H]
\centering
\caption{Table Caption}
\label{tab:example}
\small  % or \footnotesize or \scriptsize
\begin{tabular}{|l|c|c|c|c|c|}
[content]
\end{tabular}
\end{table}
```

#### Option D: Rotate Wide Tables to Landscape
For extremely wide tables:

```latex
\begin{landscape}
\begin{table}[H]
\centering
\caption{Very Wide Table}
\label{tab:wide}
\begin{tabular}{|l|c|c|c|c|c|c|c|c|}
[content]
\end{tabular}
\end{table}
\end{landscape}
```

Requires: `\usepackage{pdflscape}` in preamble

### Systematic Fix Procedure

#### Step 1: Identify All Overflowing Tables
- Open PDF: `CRISPRO_Master_CORRECTED.pdf`
- Scan through ALL pages
- Note every table that extends beyond margins
- Create list:

```
OVERFLOWING TABLES LIST:
- Chapter 1, Table 1.1 (page X) - [description]
- Chapter 1, Table 1.2 (page Y) - [description]
- Chapter 3, Table 3.1 (page Z) - [description]
- Chapter 6, Table 6.1 (page W) - [description - CRITICAL]
- [List ALL overflowing tables]
```

#### Step 2: Fix Each Overflowing Table

**For EACH table identified:**

1. **Locate table in source file**
   - Find chapter file (e.g., `Chapter_1_Complete.tex`)
   - Search for table by label or caption

2. **Assess table structure**
   - Count columns: how many?
   - Check cell content: long text?
   - Measure width: severely overflowing or slightly?

3. **Apply appropriate fix:**

   **If moderately wide (slightly overflowing):**
   ```latex
   % Add \resizebox wrapper:
   \resizebox{\textwidth}{!}{%
   \begin{tabular}{...}
   ...
   \end{tabular}
   }
   ```

   **If very wide (many columns):**
   ```latex
   % Reduce font + resizebox:
   \small
   \resizebox{\textwidth}{!}{%
   \begin{tabular}{...}
   ...
   \end{tabular}
   }
   ```

   **If extremely wide (10+ columns):**
   ```latex
   % Consider landscape orientation:
   \begin{landscape}
   \begin{table}[H]
   \centering
   \footnotesize
   \begin{tabular}{...}
   ...
   \end{tabular}
   \end{table}
   \end{landscape}
   ```

4. **Recompile and verify**
   - Run: `pdflatex CRISPRO_Master_CORRECTED.tex`
   - Check PDF: table now fits?
   - If not, apply stronger fix (smaller font, landscape)

5. **Repeat for all tables**

#### Step 3: Add Required Packages (if not present)

Check preamble for these packages:

```latex
% In CRISPRO_Master_CORRECTED.tex preamble:

\usepackage{graphicx}      % for \resizebox
\usepackage{tabularx}      % for tabularx environment
\usepackage{pdflscape}     % for landscape pages
\usepackage{rotating}      % for rotated tables (alternative)
\usepackage{adjustbox}     % for advanced table adjustments
```

Add any missing packages.

#### Step 4: Global Table Standards

**Best Practices to Apply:**

1. **Maximum columns:** Try to keep ≤ 6 columns per table
2. **Font size:** Use `\small` for tables with 5-6 columns
3. **Column types:** Use `p{width}` for text columns to enable wrapping:
   ```latex
   \begin{tabular}{|p{3cm}|c|c|c|}  % First column wraps at 3cm
   ```

4. **Header abbreviations:** Shorten long column headers:
   ```
   BEFORE: "Spearman Correlation Coefficient"
   AFTER:  "Spearman Corr."
   ```

5. **Consistent scaling:** Use same `\resizebox` for similar tables

---

## 🔴 ISSUE 2: FRONTMATTER MISSING/INCORRECT (PRIORITY 2)

### Problem Description
Title page, author information, and committee information are placeholder/incorrect. Need to populate with actual dissertation details.

### ✅ YOUR COMPLETE INFORMATION (PRE-FILLED)

**Use this exact information to update frontmatter:**

```latex
% ====================================
% AMIRHOSSEIN DANESHPAJOUH - SFU
% Complete Dissertation Information
% ====================================

% TITLE
\title{CRISPRO-MAMBA-X: Deep Learning Framework for CRISPR-Cas9 On-Target Efficacy Prediction -- A Multi-Modal Approach Integrating Sequence and Epigenomic Features}

% AUTHOR
\author{Amirhossein Daneshpajouh}

% PREVIOUS DEGREES (if applicable)
% M.Sc., [University Name], [Year] (if you have)
% B.Sc., [University Name], [Year]

% DEGREE PROGRAM
\degree{Doctor of Philosophy}

% DEPARTMENT & FACULTY
\department{School of Computing Science}
\faculty{Faculty of Applied Sciences}

% UNIVERSITY
\university{Simon Fraser University}

% DEFENSE DATE
\date{February 2026}  % or specific date: February 28, 2026

% TERM
\term{Fall 2025}  % or Spring 2026 if defense is in Feb

% KEYWORDS
\keywords{CRISPR-Cas9; On-Target Prediction; Epigenomic Features; Deep Learning; Transfer Learning; Conformal Prediction; Uncertainty Quantification}

% ====================================
% COMMITTEE INFORMATION
% ====================================

% SENIOR SUPERVISOR
\seniorSupervisor{Dr. Kay C. Wiese}{Professor}{School of Computing Science}

% COMMITTEE MEMBERS
\committee{Dr. Maxwell W. Libbrecht}{Associate Professor}{School of Computing Science}

% ADDITIONAL COMMITTEE MEMBERS (add if applicable)
% \committee{Dr. [Name]}{[Title]}{[Department]}

% CHAIR (if known)
% \chair{Dr. [Chair Name]}{[Title]}{[Department]}

% EXTERNAL EXAMINER (if known)
% \externalExaminer{Dr. [Name]}{[Title]}{[Institution]}

% DEFENSE DATE
\defensedate{February 28, 2026}  % Exact defense date
```

### What Needs to Be Updated in Master File

#### A. Title Page Section

**Locate in `CRISPRO_Master_CORRECTED.tex` (BEFORE `\begin{document}`):**

```latex
% UPDATE THESE LINES:

\title{CRISPRO-MAMBA-X: Deep Learning Framework for CRISPR-Cas9 On-Target Efficacy Prediction -- A Multi-Modal Approach Integrating Sequence and Epigenomic Features}

\author{Amirhossein Daneshpajouh}

\degree{Doctor of Philosophy}

\department{School of Computing Science}

\faculty{Faculty of Applied Sciences}

\university{Simon Fraser University}

\date{February 2026}

\term{Fall 2025}
```

#### B. Committee Information

**Locate committee definitions (BEFORE `\begin{document}`):**

```latex
% UPDATE COMMITTEE SECTION:

\seniorSupervisor{Dr. Kay C. Wiese}{Professor}{School of Computing Science}

\committee{Dr. Maxwell W. Libbrecht}{Associate Professor}{School of Computing Science}

% ADD MORE COMMITTEE MEMBERS IF YOU HAVE THEM:
% \committee{Dr. [Member 3 Name]}{[Title]}{[Department]}

% IF YOU KNOW THE CHAIR:
% \chair{Dr. [Chair Name]}{[Title]}{School of Computing Science}

% IF YOU HAVE EXTERNAL EXAMINER:
% \externalExaminer{Dr. [Examiner Name]}{[Title]}{[External Institution]}
```

**If template doesn't support these commands, create committee page manually:**

```latex
\chapter*{Examining Committee}
\addcontentsline{toc}{chapter}{Examining Committee}

\begin{center}
\textbf{EXAMINING COMMITTEE}
\end{center}

\vspace{1cm}

\noindent
\textbf{Senior Supervisor:}\\
Dr. Kay C. Wiese\\
Professor\\
School of Computing Science\\
Simon Fraser University

\vspace{0.5cm}

\noindent
\textbf{Committee Member:}\\
Dr. Maxwell W. Libbrecht\\
Associate Professor\\
School of Computing Science\\
Simon Fraser University

\vspace{0.5cm}

% ADD MORE MEMBERS HERE IF APPLICABLE

\vspace{1cm}

\noindent
\textbf{Date Defended:} February 28, 2026
```

#### C. Abstract

**Locate abstract section (AFTER `\begin{document}`):**

```latex
\begin{abstract}

CRISPR-Cas9 genome editing has revolutionized biotechnology, but predicting on-target efficacy remains a critical challenge. Current state-of-the-art models achieve Spearman correlations of 0.876 (ChromeCRISPR) but face fundamental limitations: (1) incomplete integration of epigenomic context affecting chromatin accessibility, (2) lack of rigorous uncertainty quantification for clinical decision-making, and (3) insufficient transfer learning from related editing modalities.

This dissertation addresses these gaps through a novel multi-modal deep learning framework extending ChromeCRISPR with synergistic innovations: (1) Graph Neural Networks integrating RNA secondary structure (Graph-CRISPR approach achieving 0.94--0.95 Pearson), (2) RNA-FM pretrained embeddings with bidirectional cross-attention (CRISPR-FMC dual-branch architecture), (3) information-theoretic fusion of sequence and epigenomic features via mutual information maximization, (4) transfer learning from PrimeNet (0.94 Spearman on HEK293T) and PRIDICT2.0 (0.91 Spearman on HEK293T), (5) ensemble stacking with meta-learner, and (6) biologically-grounded conformal prediction providing distribution-free coverage guarantees.

The proposed framework employs four core mathematical foundations: information-theoretic multi-modal fusion, conformal prediction with weighted exchangeability, Beta regression for bounded outcomes, and transfer learning domain adaptation theory. We validate on DeepHF, CRISPRon, and CRISPR-FMC datasets, with expected performance improvements targeting 0.88--0.92 Spearman correlation from 0.876 baseline, while providing calibrated uncertainty estimates with expected (1-α) coverage guarantees.

This work contributes novel theoretical guarantees for multi-modal biological prediction, practical tools for clinical translation, and opens pathways for personalized genome editing therapeutics.

\end{abstract}
```

#### D. Dedication (Optional)

```latex
\chapter*{Dedication}
\addcontentsline{toc}{chapter}{Dedication}

% ADD YOUR DEDICATION TEXT HERE
% For example:
To my family, for their unwavering support and encouragement throughout this journey.

\end{dedication}
```

#### E. Acknowledgments (Recommended)

```latex
\chapter*{Acknowledgments}
\addcontentsline{toc}{chapter}{Acknowledgments}

I would like to thank my supervisor, Dr. Kay C. Wiese, and my committee member, Dr. Maxwell W. Libbrecht, for their guidance and support throughout this research. I am grateful to my colleagues at Simon Fraser University for their valuable discussions and feedback.

I also thank the computational biology community for making datasets and tools publicly available, enabling this research to build upon previous work. I extend my heartfelt gratitude to my family and friends for their unwavering support, encouragement, and understanding throughout this journey. Their patience and belief in me have been essential in reaching this milestone.

% ADD MORE ACKNOWLEDGMENTS AS DESIRED

\end{acknowledgments}
```

### Fix Procedure for Frontmatter

#### Step 1: Locate Frontmatter Section

In `CRISPRO_Master_CORRECTED.tex`, find the section BEFORE `\begin{document}`:

```latex
% METADATA SECTION (before \begin{document})
% UPDATE ALL VALUES HERE WITH YOUR INFORMATION ABOVE
```

#### Step 2: Update All Metadata

Copy the complete information block from above and replace existing placeholders.

#### Step 3: Verify Template Commands

Check if your template (`sfuthesis.cls`) supports these commands:
- `\seniorSupervisor{name}{title}{department}`
- `\committee{name}{title}{department}`
- `\chair{name}{title}{department}`
- `\externalExaminer{name}{title}{institution}`

If NOT supported, use the manual committee page creation shown above.

#### Step 4: Recompile Multiple Times

After updating frontmatter:
```bash
1. pdflatex CRISPRO_Master_CORRECTED.tex
2. bibtex CRISPRO_Master_CORRECTED
3. pdflatex CRISPRO_Master_CORRECTED.tex
4. pdflatex CRISPRO_Master_CORRECTED.tex
```

This ensures Table of Contents and references update correctly.

#### Step 5: Verify PDF

Open final PDF and check:
- Title page shows: "Amirhossein Daneshpajouh"
- Title is complete and correct
- Committee page shows: Dr. Kay C. Wiese (Senior Supervisor)
- Committee page shows: Dr. Maxwell W. Libbrecht (Committee Member)
- Defense date: February 28, 2026
- Abstract is complete
- TOC includes all sections

---

## 🔴 ISSUE 3: TABLE MODULARIZATION (PRIORITY 3)

### Context
Original plan was to extract all tables into separate files for better organization. This is OPTIONAL but highly recommended for maintainability.

### Decision Point
**Ask yourself:**
1. Are you satisfied with current structure (tables in chapters)?
2. Do you need to frequently update tables?
3. Will multiple people edit tables?
4. Do you want cleaner version control?

**If YES to any:** Proceed with modularization
**If NO to all:** Skip modularization (current structure is functional)

### If Proceeding with Modularization

**Use the separate modularization prompt** provided earlier (ANTIGRAVITY_MODULARIZATION_FOLLOWUP.md).

**OR apply quick modularization:**

1. Create `/tables` directory
2. For EACH overflowing table you just fixed:
   - Copy entire `\begin{table}...\end{table}` block
   - Create new file: `tables/table_X_Y_description.tex`
   - Paste table block into new file
   - Replace in chapter with: `\input{tables/table_X_Y_description}`
3. Recompile and verify

---

## ✅ COMPREHENSIVE FIX CHECKLIST

### Phase 1: Table Overflow Fixes
- [ ] Scan entire PDF for overflowing tables
- [ ] Create list of all problematic tables
- [ ] Fix each table using `\resizebox` or alternative
- [ ] Add required packages to preamble (`graphicx`, `tabularx`, `pdflscape`)
- [ ] Recompile and verify ALL tables fit
- [ ] Check text remains readable

### Phase 2: Frontmatter Updates (YOUR INFORMATION)
- [ ] Update title to full CRISPRO-MAMBA-X title
- [ ] Update author name: **Amirhossein Daneshpajouh**
- [ ] Update department: **School of Computing Science**
- [ ] Update faculty: **Faculty of Applied Sciences**
- [ ] Update university: **Simon Fraser University**
- [ ] Update defense date: **February 28, 2026**
- [ ] Update term: **Fall 2025**
- [ ] Update senior supervisor: **Dr. Kay C. Wiese, Professor**
- [ ] Update committee member: **Dr. Maxwell W. Libbrecht, Associate Professor**
- [ ] Add additional committee members (if applicable)
- [ ] Add committee chair (if known)
- [ ] Add external examiner (if known)
- [ ] Update keywords (provided above)
- [ ] Update/write complete abstract (provided above)
- [ ] Add dedication (optional)
- [ ] Add acknowledgments (optional, template provided above)
- [ ] Recompile 3 times (for TOC updates)
- [ ] Verify title page displays: "Amirhossein Daneshpajouh"
- [ ] Verify committee page shows Dr. Wiese and Dr. Libbrecht
- [ ] Verify abstract displays correctly
- [ ] Verify TOC includes frontmatter sections

### Phase 3: Final Verification
- [ ] Open final PDF
- [ ] Check title page: **Amirhossein Daneshpajouh** visible
- [ ] Check title: Complete CRISPRO-MAMBA-X title
- [ ] Check committee page: Dr. Wiese (Senior Supervisor)
- [ ] Check committee page: Dr. Libbrecht (Committee Member)
- [ ] Check defense date: **February 28, 2026**
- [ ] Check abstract: complete and accurate
- [ ] Check TOC: all chapters listed, page numbers correct
- [ ] Check all tables: none overflowing, all readable
- [ ] Check all chapters: content intact
- [ ] Check bibliography: all citations present
- [ ] Check page numbers: sequential and correct
- [ ] Check formatting: consistent throughout
- [ ] Check margins: uniform, appropriate
- [ ] Check fonts: consistent, readable

### Phase 4: Professional Quality Check
- [ ] No compilation errors (zero errors in log)
- [ ] No compilation warnings (or only minor overfull box warnings)
- [ ] All cross-references working
- [ ] All citations resolving
- [ ] All figures displaying (if any)
- [ ] All tables displaying correctly
- [ ] All equations rendering properly
- [ ] All page breaks appropriate
- [ ] No orphaned headers (header alone at bottom of page)
- [ ] No widowed text (single line at top of page)
- [ ] Professional appearance throughout
- [ ] SFU thesis requirements met

---

## 🎯 DELIVERABLES AFTER THIS PHASE

1. **Fully Corrected PDF**
   - All tables fit within margins
   - All content readable
   - Professional appearance

2. **Complete Frontmatter (YOUR INFORMATION)**
   - **Author:** Amirhossein Daneshpajouh
   - **Institution:** Simon Fraser University
   - **Department:** School of Computing Science, Faculty of Applied Sciences
   - **Senior Supervisor:** Dr. Kay C. Wiese
   - **Committee Member:** Dr. Maxwell W. Libbrecht
   - **Defense Date:** February 28, 2026
   - Complete abstract
   - Optional dedication/acknowledgments

3. **Submission-Ready Dissertation**
   - Zero compilation errors
   - Zero formatting issues
   - Professional quality throughout
   - Ready for committee review
   - Ready for defense (February 28, 2026)
   - SFU thesis format compliant

---

## 🚀 EXECUTION INSTRUCTIONS

**After receiving this prompt:**

1. **First, fix tables:**
   - Scan PDF systematically
   - List ALL overflowing tables
   - Apply `\resizebox` fixes chapter by chapter
   - Verify each fix before moving to next

2. **Second, update frontmatter with MY INFORMATION:**
   - Author: **Amirhossein Daneshpajouh**
   - University: **Simon Fraser University**
   - Department: **School of Computing Science**
   - Faculty: **Faculty of Applied Sciences**
   - Senior Supervisor: **Dr. Kay C. Wiese**
   - Committee Member: **Dr. Maxwell W. Libbrecht**
   - Defense Date: **February 28, 2026**
   - Update title page
   - Update committee page
   - Update abstract (provided above)
   - Add optional sections (dedication, acknowledgments)

3. **Third, final compilation:**
   - Full clean compilation (3+ passes)
   - Generate final PDF
   - Verify all corrections successful

4. **Finally, report completion:**
   - List all fixes applied
   - Confirm all tables fit
   - Confirm frontmatter complete with correct names
   - Confirm PDF ready for submission

---

## ✨ EXPECTED FINAL RESULT

**A professionally formatted PhD dissertation for:**

**Amirhossein Daneshpajouh**
**Doctor of Philosophy**
**School of Computing Science, Faculty of Applied Sciences**
**Simon Fraser University**
**Defense: February 28, 2026**

**Supervised by:**
- Dr. Kay C. Wiese (Senior Supervisor)
- Dr. Maxwell W. Libbrecht (Committee Member)

**With:**
- ✅ All tables fitting properly within margins
- ✅ Complete and accurate frontmatter
- ✅ Correct author and committee information
- ✅ Comprehensive abstract
- ✅ Submission-ready quality
- ✅ Defense-ready formatting
- ✅ Zero errors, professional appearance
- ✅ SFU thesis requirements met
- ✅ Ready for committee distribution

**This will be your FINAL, SUBMISSION-READY dissertation!** 🎓
