import re
import os

target_files = [
    "Chapter_1_Complete.tex",
    "Chapter_3_Complete.tex",
    "Chapter_6_Complete.tex",
    "Chapter_9_Complete.tex",
    "Chapter_11_Complete.tex",
    "Chapter_12_Complete.tex"
]

base_dir = "/Users/studio/Desktop/PhD/Proposal"
tables_dir = os.path.join(base_dir, "tables")

def modularize_file(filename):
    filepath = os.path.join(base_dir, filename)
    with open(filepath, 'r') as f:
        content = f.read()

    # Regex to capture table environments
    # Using non-greedy match with DOTALL
    # We loop to handle multiple tables sequentially

    new_content = ""
    last_pos = 0
    table_count = 0

    pattern = re.compile(r'(\\begin\{table\}.*?\\end\{table\})', re.DOTALL)

    matches = list(pattern.finditer(content))

    if not matches:
        print(f"No tables found in {filename}")
        return

    # Process matches in reverse order to avoid index shifting issues if we were replacing in place,
    # but since we are rebuilding the string or using replacements, let's just do sequential string replacement?
    # Actually, simpler: iterate matches, replace context.
    # Safe approach: usage string substitution on the full content is risky if duplicates exist.
    # Better: Reconstruct string.

    reconstructed = ""
    current_pos = 0

    for match in matches:
        start, end = match.span()
        table_content = match.group(1)

        # Append text before this table
        reconstructed += content[current_pos:start]

        # Determine filename
        label_match = re.search(r'\\label\{([^}]+)\}', table_content)
        if label_match:
            label = label_match.group(1).replace(':', '_').replace('-', '_')
            tab_filename = f"{label}.tex"
        else:
            table_count += 1
            # Extract chapter number from filename
            chap_num = re.search(r'Chapter_(\d+)', filename).group(1)
            tab_filename = f"ch{chap_num}_table_{table_count}.tex"

        tab_filepath = os.path.join(tables_dir, tab_filename)

        # Write table content to new file
        with open(tab_filepath, 'w') as f_tab:
            f_tab.write(table_content)

        print(f"Extracted table to {tab_filename}")

        # Append input command
        reconstructed += f"\\input{{tables/{os.path.splitext(tab_filename)[0]}}}"

        current_pos = end

    # Append remaining text
    reconstructed += content[current_pos:]

    # Write back modified chapter file
    with open(filepath, 'w') as f:
        f.write(reconstructed)

    print(f"Updated {filename}")

for f in target_files:
    print(f"Processing {f}...")
    modularize_file(f)
