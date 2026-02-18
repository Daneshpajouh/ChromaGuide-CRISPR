import sys
from huggingface_hub import HfApi

def list_repo_files(repo_id):
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id)
        print(f"Files in {repo_id}:")
        for f in files:
            print(f" - {f}")
    except Exception as e:
        print(f"Error listing {repo_id}: {e}")

if __name__ == "__main__":
    for rid in sys.argv[1:]:
        list_repo_files(rid)
