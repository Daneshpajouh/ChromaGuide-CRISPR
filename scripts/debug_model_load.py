import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import os

MODEL_PATH = "zhihan1996/DNABERT-2-117M"
LOCAL_CACHE = "/home/amird/.cache/huggingface/hub"
os.environ["HF_HOME"] = "/home/amird/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = LOCAL_CACHE
os.environ["HF_HUB_OFFLINE"] = "1"

def test():
    # Force default device to CPU to avoid 'meta' device bug in DNABERT-2
    torch.set_default_device('cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

        print("Loading config...")
        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
        config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # MONKEY PATCH START
        print("Applying monkey patch for DNABERT-2 alibi tensor...")
        import transformers.models.auto.modeling_auto
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        # Get the class dynamically
        BertEncoder = get_class_from_dynamic_module("bert_layers.BertEncoder", MODEL_PATH)

        original_rebuild = BertEncoder.rebuild_alibi_tensor
        def patched_rebuild(self, size, device=None):
            # Force device to CPU if it's None or if we're in this weird state
            if device is None:
                device = torch.device('cpu')
            return original_rebuild(self, size, device)

        BertEncoder.rebuild_alibi_tensor = patched_rebuild
        print("Monkey patch applied.")
        # MONKEY PATCH END

        print("Loading model...")
        model = AutoModel.from_pretrained(MODEL_PATH, config=config, trust_remote_code=True, low_cpu_mem_usage=False)
        model = model.to(device)
        print("Model loaded successfully!")

        # Test forward pass
        seq = "ATGC" * 10
        tokens = tokenizer([seq], return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**tokens)
        print(f"Forward pass success! Output shape: {out.last_hidden_state.shape}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
