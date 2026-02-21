import pandas as pd
import pyBigWig
import subprocess
import os
import h5py
import numpy as np
import argparse
import tempfile
from tqdm import tqdm

def align_sequences(sequences, bowtie2_index, threads=8):
    """Align sequences using Bowtie2 and return a dictionary mapping sequence to coordinates."""
    unique_seqs = list(set(sequences))
    print(f"Aligning {len(unique_seqs)} unique sequences...")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        for i, seq in enumerate(unique_seqs):
            f.write(f">seq_{i}\n{seq}\n")
        fasta_path = f.name

    sam_path = fasta_path.replace('.fasta', '.sam')

    # Run Bowtie2
    # -f: fasta input, -U: unpaired, -k 1: report first valid alignment, --end-to-end: no soft clipping
    cmd = [
        "bowtie2", "-x", bowtie2_index, "-f", "-U", fasta_path,
        "-S", sam_path, "-p", str(threads), "-k", "1", "--end-to-end", "--very-sensitive"
    ]

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    coord_map = {}
    with open(sam_path, 'r') as f:
        for line in f:
            if line.startswith('@'):
                continue
            parts = line.split('\t')
            qname = parts[0]
            flag = int(parts[1])
            if flag & 4: # unmapped
                continue
            chrom = parts[2]
            pos = int(parts[3]) # 1-based
            seq_idx = int(qname.split('_')[1])
            seq = unique_seqs[seq_idx]

            # CRISPR sequences are typically 20bp.
            # SAM pos is the leftmost mapping position.
            coord_map[seq] = (chrom, pos, pos + 20)

    os.remove(fasta_path)
    os.remove(sam_path)
    return coord_map

def extract_signals(df, coord_map, bw_paths, window_size=5000, num_bins=100):
    """Extract binned BigWig signals for each coordinate."""
    print(f"Extracting binned signals ({num_bins} bins, +/- {window_size}bp)...")

    # Open BigWig files
    bw_objs = {name: pyBigWig.open(path) for name, path in bw_paths.items()}
    track_names = sorted(bw_paths.keys()) # Ensure consistent order

    # Storage for valid data
    all_epigenomics = []
    all_sequences = []
    all_efficiencies = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        seq = row['sequence']
        if seq in coord_map:
            chrom, start, end = coord_map[seq]
            center = (start + end) // 2

            # Define window
            w_start = max(0, center - window_size)
            w_end = center + window_size

            try:
                sample_tracks = []
                for name in track_names:
                    bw = bw_objs[name]
                    # Get binned values directly from pyBigWig if possible, or average
                    # bw.stats(chrom, start, end, type="mean", nBins=num_bins) is very efficient
                    vals = bw.stats(chrom, w_start, w_end, type="mean", nBins=num_bins)
                    # Replace None with 0.0
                    vals = [v if v is not None else 0.0 for v in vals]
                    sample_tracks.append(np.log1p(vals))

                all_epigenomics.append(sample_tracks)
                all_sequences.append(seq)
                all_efficiencies.append(row['efficiency'])

            except Exception as e:
                continue

    for bw in bw_objs.values():
        bw.close()

    return np.array(all_sequences), np.array(all_efficiencies), np.array(all_epigenomics)

def main():
    parser = argparse.ArgumentParser(description="Preprocess epigenomics for ChromaGuide")
    parser.add_argument("--input", required=True, help="Path to merged.csv")
    parser.add_argument("--bowtie2_index", required=True, help="Path to Bowtie2 index")
    parser.add_argument("--bw_dnase", required=True, help="Path to DNase BigWig")
    parser.add_argument("--bw_h3k4me3", required=True, help="Path to H3K4me3 BigWig")
    parser.add_argument("--bw_h3k27ac", required=True, help="Path to H3K27ac BigWig")
    parser.add_argument("--output", required=True, help="Path to output HDF5 file")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads for Bowtie2")
    parser.add_argument("--window", type=int, default=5000, help="Window size around center")
    parser.add_argument("--bins", type=int, default=100, help="Number of bins")

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    coord_map = align_sequences(df['sequence'].tolist(), args.bowtie2_index, args.threads)

    bw_paths = {
        'dnase': args.bw_dnase,
        'h3k4me3': args.bw_h3k4me3,
        'h3k27ac': args.bw_h3k27ac
    }

    seqs, effs, epi = extract_signals(df, coord_map, bw_paths, args.window, args.bins)

    print(f"Saving {len(seqs)} multimodal samples to {args.output}...")
    with h5py.File(args.output, "w") as f:
        f.create_dataset("sequences", data=seqs.astype('S'))
        f.create_dataset("efficiencies", data=effs.astype(np.float32))
        f.create_dataset("epigenomics", data=epi.astype(np.float32))

    print("DONE.")

if __name__ == "__main__":
    main()
