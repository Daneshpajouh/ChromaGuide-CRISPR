
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append("/Users/studio/Desktop/PhD/Proposal")
from src.model.crispr_rag_tta import CRISPR_RAG_Head

def test_retrieval_local():
    print("Testing RAG Retrieval Logic Locally (FAISS CPU)...")

    # 1. Setup Dummy Data for Memory Bank
    print("Initializing Memory Bank (1000 samples)...")
    d_model = 768
    n_samples = 1000

    # Random vectors simulating encoded training guides
    memory_keys = torch.randn(n_samples, d_model)
    memory_values = torch.rand(n_samples, 1) # Efficiency scores 0-1

    # 2. Initialize RAG Head
    rag_head = CRISPR_RAG_Head(d_model=d_model, memory_size=n_samples, k_neighbors=5)

    # 3. Build Index
    print("Building FAISS Index...")
    rag_head.initialize_memory(memory_keys, memory_values)

    # 4. Test Retrieval
    print("Querying RAG Head...")
    query = torch.randn(10, d_model) # Batch of 10 queries

    retrieved_values, attention_weights = rag_head(query)

    print(f"Retrieved Values Shape: {retrieved_values.shape}")
    print(f"Attention Weights Shape: {attention_weights.shape}")

    # Checks
    assert retrieved_values.shape == (10, 1)
    assert attention_weights.shape == (10, 5) # k=5

    # Check values range
    if  retrieved_values.max() <= 1.0 and retrieved_values.min() >= 0.0:
        print("✅ Retrieval values in correct range [0, 1]")
    else:
        print("❌ Retrieval values out of range!")

    print("✅ RAG logic verified successfully.")

if __name__ == "__main__":
    try:
        test_retrieval_local()
    except ImportError as e:
        print(f"❌ Failed to import FAISS or dependencies: {e}")
        print("Note: Ensure faiss-cpu is installed locally.")
