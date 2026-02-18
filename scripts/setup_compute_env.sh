#!/bin/bash
# Environment setup for SLURM jobs
# Ensures compute nodes can find pre-cached HuggingFace models

export HF_HOME="/home/amird/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/amird/.cache/huggingface/hub"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# Output environment setup status
if [ -d "$HF_HOME" ]; then
    echo "✓ HuggingFace cache found: $HF_HOME"
    if [ -d "$TRANSFORMERS_CACHE" ]; then
        MODEL_COUNT=$(find "$TRANSFORMERS_CACHE" -name "config.json" 2>/dev/null | wc -l)
        echo "  - Models cached: $MODEL_COUNT"
    fi
else
    echo "⚠ HuggingFace cache not found at $HF_HOME"
fi

# Verify CUDA
if [ -f "/usr/bin/nvidia-smi" ] || command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
else
    echo "⚠ No NVIDIA GPU detected"
fi
