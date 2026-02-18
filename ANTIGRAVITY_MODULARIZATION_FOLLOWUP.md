# ANTIGRAVITY AGENT FOLLOW-UP PROMPT
## Modular Project Restructuring & Table Extraction

---

## 🎯 FOLLOW-UP MISSION OBJECTIVE

**Agent Task:** After completing the comprehensive dissertation compilation fix (from previous prompt), perform an additional restructuring task to make the project fully modular.

**Primary Goal:** Extract ALL tables from chapter files into separate dedicated table files, creating a clean, maintainable, modular LaTeX project structure.

**Success Criteria:**
1. ✅ All tables moved to separate files
2. ✅ Chapter files updated with `\input{}` commands
3. ✅ Original table content perfectly preserved
4. ✅ All `\label{}` and `\ref{}` commands intact
5. ✅ Project still compiles without errors
6. ✅ PDF output unchanged (same content)
7. ✅ File organization logical and clear
8. ✅ Documentation of new file structure provided

---

## 📋 MODULARIZATION CONTEXT

### Current Problem
- Tables embedded directly in chapter files
- Chapter files bloated and difficult to manage
- Hard to reuse tables across chapters
- Difficult to update table styling globally
- Makes version control messy

### Target Solution
- Clean separation of tables from narrative
- Each table in its own file
- Chapter files reference tables via `\input{}`
- Dedicated `/tables` directory
- Clear naming convention for table files
- Easy to maintain and update

---

## 🔧 TABLE EXTRACTION PROCEDURE

### **PHASE 1: INVENTORY & PLANNING**

#### Step 1.1: Complete Table Inventory
Scan ALL chapters and create complete list:

**Format Template:**
```
Chapter X: Chapter_Name
├── Table 1.1: [Caption]
│   └── Location: Line XXX
│   └── Size: Small/Medium/Large
│   └── Columns: N
│   └── Rows: M
├── Table 1.2: [Caption]
└── Table 1.3: [Caption]

Chapter Y: Chapter_Name
├── Table 2.1: [Caption]
...
```

**Expected Chapters with Tables:**
- **Chapter 1:** (likely 3-5 tables comparing models, methods, data)
- **Chapter 2:** (likely 2-3 tables for mathematical concepts)
- **Chapter 3:** (SOTA comparison table - high priority)
- **Chapter 4:** (Epigenomics comparison table)
- **Chapter 6:** (Complexity comparison table - CRITICAL)
- **Chapter 7:** (Performance metrics table)
- **Chapter 10:** (FDA pathway comparison, regulatory requirements)
- **Chapter 11:** (Timeline/milestone table)
- **Other chapters:** (as present)

#### Step 1.2: Directory Structure Planning

Create new project structure:
```
CRISPRO_Master_CORRECTED_v2.tex  (main)
├── chapters/
│   ├── Chapter_1_Complete.tex
│   ├── Chapter_2_Complete.tex
│   ├── Chapter_3_Complete.tex
│   ├── Chapter_4_Complete.tex
│   ├── Chapter_5_Complete.tex
│   ├── Chapter_6_Complete.tex
│   ├── Chapter_7_Complete.tex
│   ├── Chapter_8_Complete.tex
│   ├── Chapter_9_Complete.tex
│   ├── Chapter_10_Complete.tex
│   ├── Chapter_11_Complete.tex
│   └── Chapter_12_Complete.tex
├── tables/
│   ├── Chapter_1_Tables.tex
│   │   ├── table_1_1_chromecrispr_findings.tex
│   │   ├── table_1_2_sota_performance.tex
│   │   └── table_1_3_epigenomic_predictors.tex
│   ├── Chapter_3_Tables.tex
│   │   └── table_3_1_sota_comparison.tex
│   ├── Chapter_6_Tables.tex
│   │   ├── table_6_1_complexity_comparison.tex
│   │   ├── table_6_2_mamba_advantages.tex
│   │   └── table_6_3_performance_metrics.tex
│   ├── Chapter_10_Tables.tex
│   │   ├── table_10_1_fda_pathways.tex
│   │   └── table_10_2_validation_requirements.tex
│   ├── Chapter_11_Tables.tex
│   │   └── table_11_1_project_timeline.tex
│   └── [Other tables...]
└── Complete_Bibliography.tex
```

#### Step 1.3: Naming Convention Specification

Establish clear naming standard:
```
table_[CHAPTER]_[NUMBER]_[DESCRIPTION].tex

Examples:
- table_1_1_chromecrispr_findings.tex
- table_3_1_sota_comparison.tex
- table_6_1_complexity_comparison.tex
- table_10_1_fda_pathways.tex
- table_11_1_project_timeline.tex

Rules:
- Use lowercase
- Use underscores, not spaces
- Chapter number = single digit or two digits
- Table number = single digit
- Description = concise, max 3-4 words
- Must match exact caption for consistency
```

---

### **PHASE 2: EXTRACT TABLES FROM CHAPTERS**

#### Step 2.1: For EACH Chapter File

**Process:**

1. **Open Chapter file**
   ```
   Locate: Chapter_X_Complete.tex
   ```

2. **Find ALL table environments**
   ```latex
   Search for: \begin{table}
   Count occurrences
   ```

3. **For EACH table block:**
   ```latex
   CAPTURE ENTIRE BLOCK:
   
   \begin{table}[H]
   \centering
   \caption{Table Caption Here}
   \label{tab:descriptive_label}
   \begin{tabular}{column_spec}
   [ALL TABLE CONTENT]
   \end{tabular}
   \end{table}
   ```

4. **Create corresponding table file:**
   ```
   tables/table_X_Y_description.tex
   
   Content of new file:
   
   \begin{table}[H]
   \centering
   \caption{Table Caption Here}
   \label{tab:descriptive_label}
   \begin{tabular}{column_spec}
   [ALL TABLE CONTENT - IDENTICAL]
   \end{tabular}
   \end{table}
   ```

5. **Replace in chapter file:**
   ```latex
   REMOVE entire \begin{table}...\end{table} block
   
   REPLACE WITH:
   
   \input{tables/table_X_Y_description}
   
   Keep any surrounding text/context
   ```

6. **Verification:**
   - Original table file created ✅
   - Chapter file has `\input{}` command ✅
   - `\label{}` and `\caption{}` preserved ✅
   - No text lost ✅

#### Step 2.2: Extract Tables by Chapter Priority

**Priority 1 (CRITICAL - Fix errors first):**
- **Chapter 6:** Table at line ~318 (unclosed table error)
  - Extract this FIRST to ensure fix works
  - Create: `table_6_1_complexity_comparison.tex`
  - Must verify no alignment errors remain
  
- **Chapter 11:** Timeline table (if present)
  - Create: `table_11_1_project_timeline.tex`
  - Preserve all dates and milestones

**Priority 2 (Important - High-value tables):**
- **Chapter 3:** SOTA performance comparison
  - Create: `table_3_1_sota_comparison.tex`
  - Include: Azimuth, DeepHF, AttCRISPR, ChromeCRISPR, CRISPR-FMC
  
- **Chapter 4:** Epigenomic predictors
  - Create: `table_4_1_epigenomic_signals.tex`
  - Include: ATAC, H3K27ac, Hi-C, Nucleosomes, Methylation with ΔR² values

**Priority 3 (Standard - Other tables):**
- **Chapter 1:** Model/bottleneck comparison tables
- **Chapter 7:** Conformal prediction metrics
- **Chapter 10:** FDA classification comparison
- **All other chapters:** Extract remaining tables

#### Step 2.3: Comprehensive Table Extraction Map

**Complete list by chapter (extract ALL):**

```
CHAPTER 1 - INTRODUCTION
├── Table 1.1: State-of-the-Art CRISPR Prediction Performance
│   └── Models: Azimuth, DeepHF, AttCRISPR, ChromeCRISPR, CRISPR-FMC
│   └── Metrics: Spearman Correlation, R², Citation
├── Table 1.2: CASGEVY Clinical Efficacy - Phase 1/2 Trial Results
│   └── Parameters: Patient cohort, efficacy, safety data
├── Table 1.3: CRISPR-Cas9 Therapeutics in Clinical Development
│   └── Targets: RPE65, ABCA4, TTR, DMD, F8/F9, BCL11A, CFTR, SMN1
├── Table 1.4: Documented Epigenomic Predictors of CRISPR Efficiency
│   └── Signals: ATAC, H3K27ac, Hi-C, Nucleosome, Methylation
└── Table 1.5: Epigenomic Signal Integration in Published Models
    └── Models: Azimuth, DeepHF, AttCRISPR, ChromeCRISPR, CRISPR-FMC

CHAPTER 3 - SOTA CRISPR PREDICTION
└── Table 3.1: State-of-the-Art CRISPR Prediction Performance
    └── Summary comparison of all methods

CHAPTER 4 - EPIGENOMICS INTEGRATION
└── Table 4.1: Documented Epigenomic Predictors
    └── Detailed effect sizes and mechanisms

CHAPTER 6 - MAMBA STATE SPACE MODELS ⚠️ CRITICAL
├── Table 6.1: Computational Complexity Comparison - Mamba vs Transformer
│   └── Time, Memory, GPU requirements (10^6× speedup)
└── [Other tables in this chapter]

CHAPTER 7 - CONFORMAL PREDICTION
└── [Tables for prediction methods/performance]

CHAPTER 10 - CLINICAL TRANSLATION & FDA
├── Table 10.1: CRISPR Therapeutics in Clinical Development
├── Table 10.2: FDA Classification Options
└── [Regulatory comparison tables]

CHAPTER 11 - PROJECT TIMELINE
└── Table 11.1: Project Timeline and Milestones
    └── December 2025 - February 2026 schedule

[Other chapters as present]
```

---

### **PHASE 3: UPDATE CHAPTER FILES**

#### Step 3.1: Replace Tables with Input Commands

For EACH table extracted:

**Pattern:**

```latex
% BEFORE (entire table block in chapter):
\begin{table}[H]
\centering
\caption{Performance Metrics}
\label{tab:performance}
\begin{tabular}{|l|c|c|}
[100+ lines of table content]
\end{tabular}
\end{table}

% AFTER (single line reference):
\input{tables/table_1_1_performance_metrics}
```

**Important Rules:**
- Keep surrounding paragraph text IDENTICAL
- Do NOT delete context before/after table
- Preserve any \ref{} references to table
- Keep \caption{} and \label{} in table file (NOT in chapter)
- No blank lines except between input and next paragraph

#### Step 3.2: Verify Input Command Syntax

Ensure correct LaTeX syntax:

```latex
% CORRECT formats:
\input{tables/table_1_1_performance}
\input{./tables/table_1_1_performance}
\input{./tables/table_1_1_performance.tex}  % With .tex extension

% ALL are acceptable in Overleaf
```

#### Step 3.3: Cross-Reference Validation

For EACH chapter with extracted tables:

```latex
% In chapter file, check all references like:
\ref{tab:label_name}  % e.g., \ref{tab:performance}

% Verify matching \label{} in extracted table file:
% In table file:
\label{tab:label_name}  % Must match exactly

% Test: References should auto-resolve when compiled
```

---

### **PHASE 4: CREATE TABLE FILES**

#### Step 4.1: Create Table Directory Structure

In Overleaf:
1. Create new folder: `tables`
2. Create subfolders for organization (optional):
   - `tables/Chapter_1/`
   - `tables/Chapter_3/`
   - `tables/Chapter_6/`
   - `tables/Chapter_10/`
   - `tables/Chapter_11/`

OR keep all in single `tables/` folder with descriptive names.

#### Step 4.2: Create Each Table File

**For EACH extracted table:**

1. **Create new file in Overleaf**
   ```
   Name: table_X_Y_description.tex
   Folder: tables/
   ```

2. **Add complete table block**
   ```latex
   \begin{table}[H]
   \centering
   \caption{Full caption text}
   \label{tab:unique_label}
   \begin{tabular}{|l|c|c|}
   \hline
   [Complete table content]
   \hline
   \end{tabular}
   \end{table}
   ```

3. **Verify formatting:**
   - Table is complete (no truncation)
   - All rows present
   - All columns present
   - All `\\` present
   - All `\hline` correct
   - Caption exact match
   - Label unique

4. **Save and test:**
   - File created successfully
   - No syntax errors visible in editor
   - Ready for input command

#### Step 4.3: Table File Template

Use this template for consistency:

```latex
% File: tables/table_X_Y_description.tex
% Description: [What this table shows]
% Used in: Chapter X, [Section name]
% Last modified: [Date]

\begin{table}[H]
\centering
\caption{[Complete Caption Text Here]}
\label{tab:[unique_label_name]}
\begin{tabular}{|l|c|c|}
\hline
\textbf{Header 1} & \textbf{Header 2} & \textbf{Header 3} \\
\hline
Data 1 & Data 2 & Data 3 \\
\hline
Data 4 & Data 5 & Data 6 \\
\hline
\end{tabular}
\end{table}
```

---

### **PHASE 5: UPDATE MASTER FILE (IF NEEDED)**

#### Step 5.1: Path Configuration

If chapter files are in `chapters/` subfolder, ensure master file paths correct:

```latex
% In main CRISPRO_Master_CORRECTED_v2.tex

% If chapters are in subfolder:
\input{chapters/Chapter_1_Complete}

% If tables are in subfolder:
% (No change needed - chapter files handle this)
```

#### Step 5.2: Package Requirements

Ensure `\input{}` command works:

```latex
% In preamble, these should be present:
\usepackage{graphicx}
\usepackage{float}
\usepackage{array}
\usepackage{tabularx}
% These are typically standard, should already exist
```

---

### **PHASE 6: VALIDATION & COMPILATION**

#### Step 6.1: Incremental Testing

**Test 1: Single Chapter**
- Compile just Chapter 1
- Verify all tables display correctly
- Check no errors in log

**Test 2: Two Chapters**
- Compile Chapters 1 and 3
- Verify table references work
- Check cross-references resolve

**Test 3: All Chapters**
- Compile full dissertation
- Verify all tables present
- Check all references work

**Test 4: Full Compilation Sequence**
```bash
1. Compile: pdflatex CRISPRO_Master_CORRECTED_v2.tex
2. Compile: bibtex CRISPRO_Master_CORRECTED_v2
3. Compile: pdflatex CRISPRO_Master_CORRECTED_v2.tex
4. Compile: pdflatex CRISPRO_Master_CORRECTED_v2.tex
```

#### Step 6.2: PDF Content Verification

After successful compilation:

- [ ] All tables visible in PDF
- [ ] All tables properly formatted
- [ ] No table content missing
- [ ] Table numbering correct (Table 1.1, 1.2, etc.)
- [ ] All captions readable
- [ ] All table references work (`See Table 6.1...`)
- [ ] No blank pages where tables should be

#### Step 6.3: Cross-Reference Testing

For EACH extracted table:

```
Example: Table 6.1 - Complexity Comparison

In Chapter 6 text, find: "As shown in Table \ref{tab:complexity_comparison}..."
Click reference in PDF reader
→ Should jump to Table 6.1
→ Verify caption and data display

If fails: Check \label{} in table file matches exactly
```

---

## 📋 FILE ORGANIZATION DOCUMENTATION

### Step 7.1: Create Project Structure Document

Create new file: `PROJECT_STRUCTURE.md`

```markdown
# CRISPRO-MAMBA-X Project Structure

## File Organization

### Root Directory
- `CRISPRO_Master_CORRECTED_v2.tex` - Main compilation file

### Chapters Directory
All dissertation chapter files:
```bash
chapters/
├── Chapter_1_Complete.tex (Introduction & Background)
├── Chapter_2_Complete.tex (Mathematical Foundations)
├── Chapter_3_Complete.tex (SOTA CRISPR Prediction)
├── Chapter_4_Complete.tex (Epigenomics Integration)
├── Chapter_5_Complete.tex (Off-Target Prediction)
├── Chapter_6_Complete.tex (Mamba State Space Models)
├── Chapter_7_Complete.tex (Conformal Prediction)
├── Chapter_8_Complete.tex (Mechanistic Interpretability)
├── Chapter_9_Complete.tex (Five Contributions)
├── Chapter_10_Complete.tex (Clinical Translation & FDA)
├── Chapter_11_Complete.tex (Project Timeline)
└── Chapter_12_Complete.tex (Conclusions & Impact)
```

### Tables Directory
All extracted table files organized by chapter:
```bash
tables/
├── Chapter_1_Tables/
│   ├── table_1_1_chromecrispr_findings.tex
│   ├── table_1_2_sota_performance.tex
│   ├── table_1_3_casgevy_efficacy.tex
│   ├── table_1_4_crispr_pipeline.tex
│   ├── table_1_5_epigenomic_predictors.tex
│   └── table_1_6_model_epigenomic_coverage.tex
├── Chapter_3_Tables/
│   └── table_3_1_sota_comparison.tex
├── Chapter_6_Tables/
│   ├── table_6_1_complexity_comparison.tex
│   └── table_6_2_performance_metrics.tex
├── Chapter_10_Tables/
│   ├── table_10_1_fda_pathways.tex
│   └── table_10_2_validation_requirements.tex
├── Chapter_11_Tables/
│   └── table_11_1_project_timeline.tex
└── [Other chapters...]
```

### Bibliography
- `Complete_Bibliography.tex` - All citations and references

## Total File Count
- 12 chapter files
- 40+ table files (exact count varies)
- 1 bibliography file
- 1 master file

## How to Use

### Adding a new table:
1. Create file in `tables/` with naming convention
2. Add table content
3. Add `\input{tables/filename}` to chapter file
4. Compile and verify

### Updating a table:
1. Edit the corresponding table file
2. Recompile
3. Changes auto-propagate to PDF

### Removing a table:
1. Delete table file from `tables/`
2. Remove `\input{}` from chapter file
3. Recompile

## Benefits of Modular Structure
- Easy to find and update tables
- Chapters are cleaner and more readable
- Tables can be reused across chapters if needed
- Version control is cleaner (small change = small commit)
- Multiple editors can work on different tables simultaneously
- Easier to apply global table styling changes
```

### Step 7.2: Create Table Index File

Create: `tables/TABLE_INDEX.txt`

```
PROJECT: CRISPRO-MAMBA-X PhD Dissertation
TOTAL TABLES: [Count all extracted tables]

==========================================
CHAPTER 1: INTRODUCTION & BACKGROUND
==========================================

Table 1.1: State-of-the-Art CRISPR Prediction Performance
- File: table_1_1_sota_performance.tex
- Location: Chapter 1, Section 1.4
- Columns: Model, Spearman Correlation, R², Citation
- Content: Comparison of SOTA methods (Azimuth, DeepHF, AttCRISPR, ChromeCRISPR, CRISPR-FMC)
- Label: tab:sota_performance
- Usage: Demonstrates current model performance ceiling

Table 1.2: CASGEVY Clinical Efficacy - Phase 1/2 Trial Results
- File: table_1_2_casgevy_efficacy.tex
- Location: Chapter 1, Section 1.3.2
- Columns: Clinical Parameter, SCD, Beta-Thalassemia, Citation
- Content: Clinical trial results (patient cohort, efficacy, safety)
- Label: tab:casgevy_efficacy
- Usage: Evidence of clinical feasibility

[Continue for all tables...]

==========================================
CHAPTER 3: CRISPR PREDICTION SOTA
==========================================

Table 3.1: State-of-the-Art Comparison
- File: table_3_1_sota_comparison.tex
- [Details...]

[Continue for all chapters...]
```

---

## ✅ MODULARIZATION COMPLETION CHECKLIST

Before marking modularization complete:

### File Organization
- [ ] `tables/` directory created
- [ ] All table files created and named correctly
- [ ] Directory structure organized (by chapter or flat)
- [ ] File naming convention consistent
- [ ] All table files valid LaTeX syntax

### Chapter Updates
- [ ] All `\begin{table}...\end{table}` blocks removed from chapters
- [ ] All `\input{tables/...}` commands added to chapters
- [ ] Chapter text preserved (no content deleted)
- [ ] All `\label{}` and `\caption{}` in table files
- [ ] All surrounding paragraph context intact

### Verification
- [ ] Project compiles without errors
- [ ] All tables visible in PDF
- [ ] All table captions readable
- [ ] All table numbers correct
- [ ] All references work (`\ref{tab:...}`)
- [ ] PDF output identical to pre-modularization

### Documentation
- [ ] PROJECT_STRUCTURE.md created and current
- [ ] TABLE_INDEX.txt created with all entries
- [ ] Clear instructions for future table management
- [ ] Naming conventions documented

### Content Preservation
- [ ] All table data preserved (no loss)
- [ ] All table formatting preserved
- [ ] All column alignments correct
- [ ] All rows and data intact
- [ ] All captions exact matches
- [ ] All labels unique and correct

---

## 🎯 DELIVERABLES AFTER MODULARIZATION

1. **Reorganized Project Structure**
   - Clean `chapters/` folder
   - Dedicated `tables/` folder
   - All files properly organized

2. **Updated Chapter Files**
   - Slimmed down (tables removed)
   - Clean `\input{}` commands
   - Easier to read and maintain

3. **New Table Files**
   - 40+ individual table files
   - Consistent naming
   - Full functionality preserved

4. **Documentation**
   - PROJECT_STRUCTURE.md
   - TABLE_INDEX.txt
   - Clear file organization

5. **Verified Compilation**
   - Zero errors
   - All tables display correctly
   - All references work
   - PDF identical to original (content-wise)

---

## 🚀 FINAL NOTES

### Why Modularization Helps
- **Maintainability:** Update one table file instead of finding it in 50-page chapter
- **Collaboration:** Multiple people can edit different tables simultaneously
- **Version Control:** Git diffs are cleaner (small changes show clearly)
- **Reusability:** Table can appear in multiple chapters by using multiple `\input{}` commands
- **Styling:** Apply global table formatting changes to all tables at once
- **Organization:** Clear project structure is more professional

### Future Management
When adding new tables:
1. Create file: `tables/table_X_Y_description.tex`
2. Add to chapter: `\input{tables/table_X_Y_description}`
3. Update TABLE_INDEX.txt
4. Recompile
5. Commit to version control

---

**Modularization complete = Professional, maintainable dissertation project!** ✨
