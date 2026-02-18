# MANUAL DATASET DOWNLOAD INSTRUCTIONS

Due to SSL certificate issues with automated downloads, please manually download the datasets:

## Option 1: Quick Start (Use CRISPRoffT Mini)

You already have the mini dataset. Let's expand it first:

```bash
cd /Users/studio/Desktop/PhD/Proposal
python3 emergency_dnabert2_train.py
```

This will train on your existing 960 samples as a quick test.

## Option 2: Download Full Datasets (Recommended for Breakthrough)

### 1. CRISPRon Dataset (24k samples) - HIGHEST PRIORITY

**Method A: Using curl (bypass SSL)**
```bash
cd /Users/studio/Desktop/PhD/Proposal/data/raw
mkdir -p /Users/studio/Desktop/PhD/Proposal/data/raw

curl -k -L "https://github.com/JasonChen-Spotec/CRISPRon/raw/master/data/all_data.csv" -o CRISPRon.csv
```

**Method B: Manual Browser Download**
1. Visit: https://github.com/JasonChen-Spotec/CRISPRon
2. Navigate to: data/all_data.csv
3. Click "Download Raw File"
4. Save to: `/Users/studio/Desktop/PhD/Proposal/data/raw/CRISPRon.csv`

### 2. Wang 2019 Dataset (13k samples)

```bash
cd /Users/studio/Desktop/PhD/Proposal/data/raw
curl -k -L "https://github.com/Peppags/CNN-SVR/raw/master/data/Wang_dataset.csv" -o Wang2019.csv
```

### 3. DeepSpCas9 Dataset (13k samples)

```bash
cd /Users/studio/Desktop/PhD/Proposal/data/raw
curl -k -L "https://github.com/Jm-Kwak/DeepSpCas9/raw/master/data/DeepSpCas9_data.csv" -o DeepSpCas9.csv
```

### 4. Alternative: Use Git Clone

```bash
cd /tmp
git clone https://github.com/JasonChen-Spotec/CRISPRon.git
cp CRISPRon/data/all_data.csv /Users/studio/Desktop/PhD/Proposal/data/raw/CRISPRon.csv

git clone https://github.com/Peppags/CNN-SVR.git
cp CNN-SVR/data/Wang_dataset.csv /Users/studio/Desktop/PhD/Proposal/data/raw/Wang2019.csv

git clone https://github.com/Jm-Kwak/DeepSpCas9.git
cp DeepSpCas9/data/DeepSpCas9_data.csv /Users/studio/Desktop/PhD/Proposal/data/raw/DeepSpCas9.csv
```

## After Downloading

Once you have the files in `/Users/studio/Desktop/PhD/Proposal/data/raw/`, verify:

```bash
cd /Users/studio/Desktop/PhD/Proposal/data/raw
ls -lh *.csv
```

Then run the merge script again:

```bash
cd /Users/studio/Desktop/PhD/Proposal
python3 download_datasets.py
```

This will merge all datasets with Z-score normalization and create:
`/Users/studio/Desktop/PhD/Proposal/data/merged_crispr_data.csv`

## Quick Training Test (While Downloading)

Don't wait! Start training on mini dataset now:

```bash
cd /Users/studio/Desktop/PhD/Proposal
python3 emergency_dnabert2_train.py
```

Expected time on Mac Studio: ~30 mins for 10 epochs
Target: Spearman ρ > 0.4 (on mini data)

Once full data is merged, rerun for ρ > 0.75!
