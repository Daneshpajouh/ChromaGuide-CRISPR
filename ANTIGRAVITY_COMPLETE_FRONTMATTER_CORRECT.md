# ANTIGRAVITY AGENT - COMPLETE FRONTMATTER FIX WITH CORRECT DEGREE INFO

---

## 🎯 COMPLETE FRONTMATTER CORRECTION - ALL INFORMATION PROVIDED

This prompt provides **ALL correct information** to fix the frontmatter completely, including your actual previous degree.

---

## ✅ YOUR COMPLETE INFORMATION (VERIFIED)

```latex
% ====================================================================
% COMPLETE FRONTMATTER METADATA - AMIRHOSSEIN DANESHPAJOUH
% ====================================================================

% Title
\title{CRISPRO-MAMBA-X: Deep Learning Framework for CRISPR-Cas9 On-Target Efficacy Prediction -- A Multi-Modal Approach Integrating Sequence and Epigenomic Features}

% Author
\author{Amirhossein Daneshpajouh}

% Previous Degrees - YOUR ACTUAL DEGREE
\previousdegrees{%
    B.Sc., Islamic Azad University, 2020}

% Current Degree
\degree{Doctor of Philosophy}

% Department and Faculty
\department{School of Computing Science}
\faculty{Faculty of Applied Sciences}

% University
\university{Simon Fraser University}

% Dates
\date{February 2026}
\term{Fall 2025}
\copyrightyear{2026}

% Keywords - ACTUAL keywords
\keywords{CRISPR-Cas9; On-Target Prediction; Epigenomic Features; Deep Learning; Transfer Learning; Conformal Prediction; Uncertainty Quantification}

% Committee Information
\chair{To Be Announced}{}{School of Computing Science}
\seniorSupervisor{Dr. Kay C. Wiese}{Professor}{School of Computing Science}
\committee{Dr. Maxwell W. Libbrecht}{Associate Professor}{School of Computing Science}

% ====================================================================
```

---

## 🔴 CRITICAL ISSUES TO FIX (FROM PDF SCREENSHOTS)

### **Issue 1: Committee Page (Page 1) - BROKEN FORMATTING**

**Current Problem:**
```
Dr. Kay C. WieseProfessor, School of Computing Science Associate Professor, School of
Computing Science Chair: To Be Announced
```

All text is mashed together with no proper spacing or line breaks.

**Solution:**

The template commands are clearly NOT working properly. You MUST manually create the committee page.

**REPLACE the automatic committee page generation with this manual version:**

```latex
% After \begin{document}, create manual committee page:

\thispagestyle{empty}
\begin{center}
\vspace*{2cm}
{\Large\textbf{Approval}}
\end{center}

\vspace{1.5cm}

\noindent
\textbf{Name:} Amirhossein Daneshpajouh\\[0.3cm]
\textbf{Degree:} Doctor of Philosophy\\[0.3cm]
\textbf{Title:} CRISPRO-MAMBA-X: Deep Learning Framework for CRISPR-Cas9 On-Target Efficacy Prediction -- A Multi-Modal Approach Integrating Sequence and Epigenomic Features\\[0.5cm]
\textbf{Date of Defence:} February 28, 2026

\vspace{1.5cm}

\noindent
\textbf{Examining Committee:}

\vspace{1cm}

\noindent
\textbf{Chair:} To Be Announced\\
School of Computing Science

\vspace{0.8cm}

\noindent
\textbf{Senior Supervisor:}\\
Dr. Kay C. Wiese\\
Professor\\
School of Computing Science

\vspace{0.8cm}

\noindent
\textbf{Committee Member:}\\
Dr. Maxwell W. Libbrecht\\
Associate Professor\\
School of Computing Science

\clearpage
```

---

### **Issue 2: Title Page (Page 2) - Previous Degrees**

**Current Problem:**
Shows placeholder: "B.Sc., Previous University, Year"

**Solution:**
Replace with your actual degree:

```latex
\previousdegrees{%
    B.Sc., Islamic Azad University, 2020}
```

**Location:** In preamble (before `\begin{document}`)

---

### **Issue 3: Keywords Page (Page 3) - Placeholder**

**Current Problem:**
Shows: "Keywords: KEYWORDS"

**Solution:**
```latex
\keywords{CRISPR-Cas9; On-Target Prediction; Epigenomic Features; Deep Learning; Transfer Learning; Conformal Prediction; Uncertainty Quantification}
```

**Location:** In preamble (before `\begin{document}`)

---

## 🔧 COMPLETE FIX PROCEDURE

### **Step 1: Update Preamble (Before \begin{document})**

Locate and update these commands:

```latex
% COMPLETE PREAMBLE METADATA
% Copy this entire block and replace existing metadata

\title{CRISPRO-MAMBA-X: Deep Learning Framework for CRISPR-Cas9 On-Target Efficacy Prediction -- A Multi-Modal Approach Integrating Sequence and Epigenomic Features}

\author{Amirhossein Daneshpajouh}

\previousdegrees{%
    B.Sc., Islamic Azad University, 2020}

\degree{Doctor of Philosophy}

\department{School of Computing Science}

\faculty{Faculty of Applied Sciences}

\university{Simon Fraser University}

\date{February 2026}

\term{Fall 2025}

\copyrightyear{2026}

\keywords{CRISPR-Cas9; On-Target Prediction; Epigenomic Features; Deep Learning; Transfer Learning; Conformal Prediction; Uncertainty Quantification}

% Keep these for metadata, but we'll override the committee page manually
\chair{To Be Announced}{}{School of Computing Science}
\seniorSupervisor{Dr. Kay C. Wiese}{Professor}{School of Computing Science}
\committee{Dr. Maxwell W. Libbrecht}{Associate Professor}{School of Computing Science}
```

---

### **Step 2: Fix Committee Page (After \begin{document})**

**Find the section that generates the committee/approval page** and replace it with:

```latex
\begin{document}

% ====================================================================
% MANUALLY CREATED COMMITTEE PAGE (Template version is broken)
% ====================================================================

\thispagestyle{empty}
\begin{center}
\vspace*{2cm}
{\Large\textbf{Approval}}
\end{center}

\vspace{1.5cm}

\noindent
\textbf{Name:} Amirhossein Daneshpajouh\\[0.3cm]
\textbf{Degree:} Doctor of Philosophy\\[0.3cm]
\textbf{Title:} CRISPRO-MAMBA-X: Deep Learning Framework for CRISPR-Cas9 On-Target Efficacy Prediction -- A Multi-Modal Approach Integrating Sequence and Epigenomic Features\\[0.5cm]
\textbf{Date of Defence:} February 28, 2026

\vspace{1.5cm}

\noindent
\textbf{Examining Committee:}

\vspace{1cm}

\noindent
\textbf{Chair:} To Be Announced\\
School of Computing Science

\vspace{0.8cm}

\noindent
\textbf{Senior Supervisor:}\\
Dr. Kay C. Wiese\\
Professor\\
School of Computing Science

\vspace{0.8cm}

\noindent
\textbf{Committee Member:}\\
Dr. Maxwell W. Libbrecht\\
Associate Professor\\
School of Computing Science

\clearpage

% ====================================================================
% Continue with automatic title page, abstract, etc.
% ====================================================================

% The rest of the frontmatter should work automatically now
```

---

### **Step 3: Verify Title Page**

The title page should now show:

```
CRISPRO-MAMBA-X: Deep Learning Framework
for CRISPR-Cas9 On-Target Efficacy Prediction
– A Multi-Modal Approach Integrating
Sequence and Epigenomic Features

by

Amirhossein Daneshpajouh

B.Sc., Islamic Azad University, 2020

Dissertation Submitted in Partial Fulfillment of the
Requirements for the Degree of
Doctor of Philosophy

in the

School of Computing Science
Faculty of Applied Sciences

© Amirhossein Daneshpajouh 2026
SIMON FRASER UNIVERSITY
Fall 2025
```

---

### **Step 4: Verify Keywords Page**

Should show:

```
Keywords: CRISPR-Cas9; On-Target Prediction; Epigenomic Features; 
Deep Learning; Transfer Learning; Conformal Prediction; 
Uncertainty Quantification
```

NOT: "Keywords: KEYWORDS"

---

## ✅ COMPLETE VERIFICATION CHECKLIST

### **Page 1: Approval/Committee Page**
- [ ] "Approval" heading centered at top
- [ ] "Name: Amirhossein Daneshpajouh" visible
- [ ] "Degree: Doctor of Philosophy" visible
- [ ] Title shown (truncated is OK for this page)
- [ ] "Date of Defence: February 28, 2026" visible
- [ ] "Examining Committee:" heading visible
- [ ] "Chair: To Be Announced" on separate lines
- [ ] "Senior Supervisor:" label visible
- [ ] "Dr. Kay C. Wiese" on own line
- [ ] "Professor" on own line
- [ ] "School of Computing Science" on own line
- [ ] "Committee Member:" label visible
- [ ] "Dr. Maxwell W. Libbrecht" on own line
- [ ] "Associate Professor" on own line
- [ ] "School of Computing Science" on own line
- [ ] **NO text mashed together**
- [ ] Clear spacing between each committee member
- [ ] Professional appearance

### **Page 2: Title Page**
- [ ] Full title displayed (with line breaks)
- [ ] "by" on its own line
- [ ] "Amirhossein Daneshpajouh" displayed
- [ ] "B.Sc., Islamic Azad University, 2020" displayed
- [ ] **NO "Previous University, Year" placeholder**
- [ ] "Doctor of Philosophy" visible
- [ ] "School of Computing Science" visible
- [ ] "Faculty of Applied Sciences" visible
- [ ] "© Amirhossein Daneshpajouh 2026" at bottom
- [ ] "SIMON FRASER UNIVERSITY" at bottom
- [ ] "Fall 2025" at bottom

### **Page 3: Keywords Page**
- [ ] Actual keywords listed (not "KEYWORDS")
- [ ] Semicolon-separated
- [ ] All 7 keywords present:
  - CRISPR-Cas9
  - On-Target Prediction
  - Epigenomic Features
  - Deep Learning
  - Transfer Learning
  - Conformal Prediction
  - Uncertainty Quantification

### **Page 4: Abstract**
- [ ] Complete abstract text
- [ ] Proper paragraphs
- [ ] No truncation
- [ ] Professional formatting

---

## 📊 SUMMARY OF CHANGES

```
METADATA UPDATES:
1. Previous degrees: Islamic Azad University, 2020 ✓
2. Keywords: 7 actual keywords (not placeholder) ✓
3. Committee: Proper names and titles ✓

FORMATTING FIXES:
1. Manual committee page (template broken) ✓
2. Proper spacing and line breaks ✓
3. All committee members visible ✓
4. No mashed-together text ✓

INFORMATION VERIFIED:
- Author: Amirhossein Daneshpajouh ✓
- B.Sc.: Islamic Azad University, 2020 ✓
- University: Simon Fraser University ✓
- Department: School of Computing Science ✓
- Faculty: Faculty of Applied Sciences ✓
- Senior Supervisor: Dr. Kay C. Wiese ✓
- Committee Member: Dr. Maxwell W. Libbrecht ✓
- Defense Date: February 28, 2026 ✓
```

---

## 🎯 EXECUTION INSTRUCTIONS

1. **Open `CRISPRO_Master_CORRECTED.tex`**

2. **Update preamble** (before `\begin{document}`):
   - Replace `\previousdegrees{}` with actual degree info
   - Replace `\keywords{KEYWORDS}` with actual keywords
   - Keep all other metadata as provided above

3. **Fix committee page** (after `\begin{document}`):
   - Find where committee/approval page is generated
   - Replace with manual version provided above
   - This bypasses the broken template formatting

4. **Recompile:**
   ```bash
   pdflatex CRISPRO_Master_CORRECTED.tex
   pdflatex CRISPRO_Master_CORRECTED.tex
   ```

5. **Verify PDF:**
   - Check first 4 pages match checklists above
   - Confirm no mashed-together text
   - Confirm all information correct

---

## ✨ EXPECTED RESULT

**After this fix, your dissertation will have:**

✅ **Professional committee page** with proper spacing
✅ **Correct previous degree:** B.Sc., Islamic Azad University, 2020
✅ **Actual keywords** (not placeholder)
✅ **All committee members** listed properly:
   - Chair: To Be Announced
   - Senior Supervisor: Dr. Kay C. Wiese
   - Committee Member: Dr. Maxwell W. Libbrecht
✅ **No formatting disasters**
✅ **Ready for committee distribution**
✅ **Ready for defense February 28, 2026**

---

## 🚨 CRITICAL NOTE

**The template's automatic committee page generation is BROKEN.** This is why you're seeing mashed-together text. The manual override provided above bypasses the broken template and creates a properly formatted page.

**Do NOT try to fix the template commands** - they're fundamentally broken. Use the manual page creation instead.

---

## 🚀 READY TO EXECUTE

This prompt provides:
- ✅ Your actual previous degree (Islamic Azad University, 2020)
- ✅ All correct metadata
- ✅ Manual committee page (bypasses broken template)
- ✅ Complete verification checklists
- ✅ Clear execution instructions

**Give this to Antigravity Agent to fix all frontmatter issues with your correct information!**

**Time: 30-45 minutes**
**Result: Professional, submission-ready frontmatter**
