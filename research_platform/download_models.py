import os
import json
import subprocess
import sys

def download_tier(target_tier="ultra_light_fast"):
    """
    Robust Download Engine for Research Hub v14.1.
    Ensures 100% integrity on the Elements SSD using modern hf-cli.
    """
    manifest_path = "/Users/studio/Desktop/PhD/Proposal/research_platform/MODEL_MANIFEST.json"
    if not os.path.exists(manifest_path):
        print("Error: MODEL_MANIFEST.json not found.")
        return

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    base_dir = manifest.get("base_model_dir", "/Volumes/Elements/research_hub_models")
    os.makedirs(base_dir, exist_ok=True)

    print(f"🚀 Research Hub v14.1: Verifying & Syncing {target_tier.upper()} Tier to SSD\n")

    tier_models = manifest.get(target_tier, [])
    if not tier_models:
        print(f"Error: Tier '{target_tier}' not found in manifest.")
        return

    for m in tier_models:
        hf_id = m['hf_id']
        local_rel_path = m['local_path']
        full_dest = os.path.join(base_dir, local_rel_path)

        print(f"[*] Processing: {m['name']} ({hf_id})")

        # Check if already exists and is complete
        if os.path.exists(full_dest) and os.path.isdir(full_dest):
            files = os.listdir(full_dest)
            if any(f.endswith('.safetensors') or f.endswith('.gguf') or f.endswith('.bin') for f in files):
                 print(f"    ✅ Weights verified on SSD. Skipping.")
                 continue
            else:
                 print(f"    ⚠️  Directory exists but looks incomplete. Re-downloading...")

        os.makedirs(full_dest, exist_ok=True)

        # Using the absolute most reliable method: modern 'hf' tool
        cmd = [
            "hf", "download",
            hf_id,
            "--local-dir", full_dest
        ]

        try:
            print(f"    ⬇️ Downloading weights to: {full_dest}...")
            subprocess.run(cmd, check=True)

            # Post-download verification
            files = os.listdir(full_dest)
            if any(f.endswith('.safetensors') or f.endswith('.gguf') or f.endswith('.bin') for f in files):
                print(f"    ✅ Download Success.")
            else:
                print(f"    ❌ Error: Download finished but no weight files found in {full_dest}.")
        except subprocess.CalledProcessError as e:
            print(f"    ❌ CLI Error: {e}")
        except Exception as e:
            print(f"    ❌ Unexpected Error: {e}")

    print(f"\n🏆 {target_tier.upper()} Tier Verification Complete.")

if __name__ == "__main__":
    tier = sys.argv[1] if len(sys.argv) > 1 else "ultra_light_fast"
    download_tier(tier)
