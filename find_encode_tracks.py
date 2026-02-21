import requests
import json

def get_encode_files(target=None, assay=None):
    base_url = "https://www.encodeproject.org/search/?"
    params = {
        "type": "File",
        "file_format": "bigWig",
        "biosample_ontology.term_name": "K562",
        "assembly": "GRCh38",
        "status": "released",
        "limit": "all",
        "format": "json"
    }
    if target:
        params["target.label"] = target
    if assay:
        params["assay_title"] = assay

    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return []

    data = response.json()
    results = []
    for item in data.get("@graph", []):
        output_type = item.get("output_type", "")
        # Filter for signal-like output types
        if output_type in ["fold change over control", "signal p-value", "read-depth normalized signal"]:
            results.append({
                "accession": item.get("accession"),
                "output_type": output_type,
                "href": "https://www.encodeproject.org" + item.get("href"),
                "experiment": item.get("dataset")
            })
    return results

targets = ["H3K4me1", "H3K4me3", "H3K27ac"]
assays = ["DNase-seq"]

print("### DNase-seq (GRCh38) ###")
for f in get_encode_files(assay="DNase-seq")[:5]:
    print(f"{f['accession']} ({f['output_type']}): {f['href']}")

for target in targets:
    print(f"\n### {target} (GRCh38) ###")
    for f in get_encode_files(target=target)[:5]:
        print(f"{f['accession']} ({f['output_type']}): {f['href']}")
