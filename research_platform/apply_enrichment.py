import json
import os
import re

MANIFEST_FILES = [
    "/Users/studio/Desktop/PhD/Proposal/research_platform/GLOBAL_MODEL_MANIFEST.md",
    "/Users/studio/Desktop/PhD/Proposal/research_platform/SPECIALIZED_MANIFEST.md"
]

CACHE_FILE = "/Users/studio/Desktop/PhD/Proposal/research_platform/scans/METADATA_CACHE.json"

def load_cache():
    with open(CACHE_FILE, 'r') as f:
        return json.load(f)

def update_manifest(filepath, cache):
    if not os.path.exists(filepath): return

    with open(filepath, 'r') as f:
        lines = f.readlines()

    output_lines = []
    in_table = False

    for line in lines:
        if line.strip().startswith("|"):
            if "Model" in line and "likes" in line.lower():
                in_table = True
                if "Organization" in line:
                    new_header = "| Organization | Latest Model | Params | Disk | Arch | Version | Likes | Tags |"
                else:
                    new_header = "| Model | Params | Disk | Arch | Version | Likes | Org | Tags |"
                output_lines.append(new_header + "\n")
                continue

            if "---" in line and in_table:
                # Count separators based on header
                if "Organization" in output_lines[-1]:
                    output_lines.append("|---|---|---|---|---|---|---|---|\n")
                else:
                    output_lines.append("|---|---|---|---|---|---|---|---|\n")
                continue

            if in_table:
                # Extract Model ID from link: [name](https://huggingface.co/id)
                match = re.search(r"https://huggingface.co/([^\)\s\|]+)", line)
                if match:
                    mid = match.group(1)
                    meta = cache.get(mid, {})
                    params = meta.get('params', 'Unknown')
                    disk = meta.get('disk_gb', 'Unknown')
                    arch = meta.get('arch', 'Unknown')
                    version = meta.get('version', 'N/A')

                    # Split current row and strip whitespace
                    cols = [c.strip() for c in line.split("|") if c.strip() != ""]
                    # If the row starts with empty | (split behavior), first item might be empty
                    # Realistically splitting "| A | B |" -> ['', ' A ', ' B ', '']
                    all_cols = [c.strip() for c in line.split("|")]
                    clean_cols = [c for c in all_cols if c] # This might skip empty cells though.

                    # Safer split
                    raw_cols = line.strip().strip("|").split("|")
                    strip_cols = [c.strip() for c in raw_cols]

                    if "GLOBAL" in filepath.upper():
                        # Org | Model | Params | Likes | Tags (Old Cols)
                        # We want: Org | Model | Params | Disk | Arch | Version | Likes | Tags
                        org = strip_cols[0]
                        model_link = strip_cols[1]
                        # strip_cols[2] was params
                        likes = strip_cols[3]
                        tags = strip_cols[4]
                        new_row = f"| {org} | {model_link} | {params} | {disk} | {arch} | {version} | {likes} | {tags} |"
                    else:
                        # Model | Likes | Org | Tags (Old Cols)
                        # We want: Model | Params | Disk | Arch | Version | Likes | Org | Tags
                        model_link = strip_cols[0]
                        likes = strip_cols[1]
                        org = strip_cols[2]
                        tags = strip_cols[3]
                        new_row = f"| {model_link} | {params} | {disk} | {arch} | {version} | {likes} | {org} | {tags} |"

                    output_lines.append(new_row + "\n")
                else:
                    output_lines.append(line)
            else:
                output_lines.append(line)
        else:
            in_table = False
            output_lines.append(line)

    with open(filepath, 'w') as f:
        f.writelines(output_lines)

def main():
    cache = load_cache()
    for f in MANIFEST_FILES:
        print(f"Updating {f}...")
        update_manifest(f, cache)
    print("Updates done.")

if __name__ == "__main__":
    main()
