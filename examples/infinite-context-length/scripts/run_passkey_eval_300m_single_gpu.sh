#!/bin/bash
# Low memory passkey evaluation for 300M Infini-Attention model
set -e

CHECKPOINT_PATH="${1:-./checkpoints/fineweb_4gpu_300m_infini/10000}"
CONTEXT_LENGTH="${2:-1024}"
NUM_SAMPLES="${3:-10}"  # Very small for testing
GPU_DEVICE="${4:-6}"  # GPU device to use (default: 6)

# Create results directory
SAVE_DIR="./results/passkey_300m_lowmem_$(date +%Y%m%d_%H%M%S)"
mkdir -p $SAVE_DIR

echo "=========================================================="
echo "LOW-MEMORY PASSKEY EVALUATION - 300M INFINI-ATTENTION"
echo "=========================================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Context Length: $CONTEXT_LENGTH tokens"
echo "Samples per depth: $NUM_SAMPLES"
echo "GPU Device: $GPU_DEVICE"
echo "Results: $SAVE_DIR"
echo "=========================================================="

# Validate checkpoint
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

# Check for local datasets first, fallback to HuggingFace
LOCAL_1K_DATASET="./llama3-1024-passkey-retrieval-eval"
LOCAL_16K_DATASET="./llama3-16k-passkey-retrieval-eval"

if [ "$CONTEXT_LENGTH" -le 1024 ]; then
    # Check for directory or direct parquet file
    if [ -d "$LOCAL_1K_DATASET" ] || [ -f "$LOCAL_1K_DATASET/train-00000-of-00001.parquet" ]; then
        DATASET="$LOCAL_1K_DATASET"
        echo "Using local 1K dataset: $DATASET"
    else
        DATASET="nanotron/llama3-1024-passkey-retrieval-eval"
        echo "Using HuggingFace 1K dataset: $DATASET"
    fi
else
    # Check for directory or direct parquet file
    if [ -d "$LOCAL_16K_DATASET" ] || [ -f "$LOCAL_16K_DATASET/train-00000-of-00001.parquet" ]; then
        DATASET="$LOCAL_16K_DATASET"
        echo "Using local 16K dataset: $DATASET"
    else
        DATASET="nanotron/llama3-16k-passkey-retrieval-eval"
        echo "Using HuggingFace 16K dataset: $DATASET"
    fi
fi

# Clear GPU memory
echo "Clearing GPU memory..."
nvidia-smi --gpu-reset > /dev/null 2>&1 || true

# Set memory-efficient environment
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE  # Use specified GPU
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

echo "Starting low-memory evaluation on GPU $GPU_DEVICE..."

torchrun --nproc_per_node=1 \
    examples/infinite-context-length/scripts/run_passkey_eval.py \
    --ckpt-path $CHECKPOINT_PATH \
    --save_path $SAVE_DIR \
    --eval_dataset_path $DATASET \
    --num_shots 0 \
    --num_digits 0 \
    --seed 42 \
    --num_samples $NUM_SAMPLES \
    --max-new-tokens 15 \
    --dp 1 \
    --tp 1 \
    --pp 1

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================================="
    echo "LOW-MEMORY EVALUATION COMPLETED!"
    echo "=========================================================="
    echo "Results saved to: $SAVE_DIR"
    
    # Quick analysis
    if [ -f "examples/infinite-context-length/scripts/analyze_passkey_results.py" ]; then
        echo "Running analysis..."
        python examples/infinite-context-length/scripts/analyze_passkey_results.py $SAVE_DIR
    fi
else
    echo "ERROR: Evaluation failed"
    exit 1
fi