import os
import requests
import pandas as pd
from pathlib import Path

def download_file(url, target_path):
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✓ Saved to {target_path}")
        return True
    else:
        print(f"✗ Failed to download {url} (Status: {response.status_code})")
        return False

def main():
    base_dir = Path("data/real")
    base_dir.mkdir(parents=True, exist_ok=True)

    # DeepHF (using a known raw URL if possible)
    # The original paper used 3 cell lines
    base_url = "https://raw.githubusercontent.com/izhangcd/DeepHF/master/data"
    files = ["DeepHF_training_data_with_epigenetic_features.csv"]

    for f in files:
        url = f"{base_url}/{f}"
        download_file(url, base_dir / f)

    # CRISPRon (Alternative source)
    # CRISPRon is often found in these repos
    crispron_url = "https://raw.githubusercontent.com/MaximilianHaeussler/crispor/master/data/crispron/crispron_test_data.csv"
    download_file(crispron_url, base_dir / "crispron_test.csv")

if __name__ == "__main__":
    main()
