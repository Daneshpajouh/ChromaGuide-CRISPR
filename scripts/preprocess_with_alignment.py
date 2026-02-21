import pandas as pd
import subprocess
import os
import pyBigWig
import numpy as np
from pathlib import Path
import json

def main():
    # ADJUST PATHS FOR NARVAL
    root = Path("/home/amird/chromaguide_experiments")
    data_path = root / "data/real/merged.csv"
    out_path = root / "data/real/merged_with_epi.csv"
    index_path = root / "data/reference/GRCh38_noalt_as/GRCh38_noalt_as"

    # Files
    bw_files = {
        "dnase": root / "data/real/raw/encode/dnase_signal.bigWig",
        "h3k4me3": root / "data/real/raw/encode/h3k4me3_signal.bigWig",
        "h3k27ac": root / "data/real/raw/encode/h3k27ac_signal.bigWig"
    }

    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print(f"Loading {data_path}...")
    df = pd.read_csv(data_path)

    # 1. Create Fasta for alignment
    fasta_path = root / "temp_sequences.fasta"
    print(f"Writing sequences to {fasta_path}...")
    with open(fasta_path, "w") as f:
        for i, row in df.iterrows():
            # 20bp guide + 3bp PAM
            seq = str(row['sequence'])
            f.write(f">seq_{i}\n{seq}\n")

    # 2. Run Bowtie2
    print("Running Bowtie2 alignment...")
    sam_path = root / "temp_alignment.sam"
    # -U: unpaired, -k 1: only first hit, --very-sensitive
    # module load bowtie2 must be done before running this script or within the cmd
    cmd = f"bowtie2 -x {index_path} -U {fasta_path} -S {sam_path} -k 1 --very-sensitive --threads 8 --no-hd --no-sq"
    print(f"Executing: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Bowtie2 failed. Ensure index exists at {index_path}")
        return

    # 3. Parse SAM
    print("Parsing alignment results...")
    coords = {}
    if os.path.exists(sam_path):
        with open(sam_path, "r") as f:
            for line in f:
                cols = line.split("\t")
                if len(cols) < 4: continue
                qname = cols[0]
                flag = int(cols[1])
                if flag & 4: continue # Unmapped

                try:
                    idx = int(qname.split("_")[1])
                    chrom = cols[2]
                    pos = int(cols[3]) # 1-based start
                    coords[idx] = (chrom, pos)
                except:
                    continue

    # 4. Extract Signals
    print(f"Mapping complete. Found coordinates for {len(coords)}/{len(df)} sequences.")
    print("Extracting signals from BigWig files...")

    # Open BigWigs
    try:
        bws = {k: pyBigWig.open(str(v)) for k, v in bw_files.items()}
    except Exception as e:
        print(f"Error opening BigWig files: {e}")
        return

    results = []
    for idx, row in df.iterrows():
        row_data = row.to_dict()
        if idx in coords:
            chrom, pos = coords[idx]
            # Window +/- 5kb for 10kb total
            start = max(0, pos - 5000)
            end = pos + 5000

            row_data["chrom"] = chrom
            row_data["start"] = start
            row_data["end"] = end

            # Extract 100 bins
            for name, bw in bws.items():
                try:
                    # stats returns a list of means per bin
                    vals = bw.stats(chrom, start, end, nBins=100)
                    # Replace None with 0.0
                    vals = [float(v) if v is not None else 0.0 for v in vals]
                    # Store as comma-separated string for CSV or keep as list for JSON/Pickle
                    # For ChromaGuide, we'll store as string to keep CSV simple
                    row_data[f"{name}_signal"] = ",".join(map(str, vals))
                except Exception as e:
                    row_data[f"{name}_signal"] = ",".join(["0.0"] * 100)
        else:
            # Fallback for unmapped
            row_data["chrom"] = "unmapped"
            row_data["start"] = 0
            row_data["end"] = 0
            for name in bws:
                row_data[f"{name}_signal"] = ",".join(["0.0"] * 100)

        results.append(row_data)
        if idx % 10000 == 0:
            print(f"Processed {idx} samples...")

    # 5. Save enriched data
    new_df = pd.DataFrame(results)
    new_df.to_csv(out_path, index=False)
    print(f"SUCCESS. Saved enriched dataset with real signals to {out_path}")

    # Cleanup
    if os.path.exists(fasta_path): os.remove(fasta_path)
    if os.path.exists(sam_path): os.remove(sam_path)

if __name__ == "__main__":
    main()
