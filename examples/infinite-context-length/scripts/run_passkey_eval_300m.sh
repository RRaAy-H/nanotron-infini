#!/bin/bash
# Passkey evaluation for 300M Infini-Attention model
set -e  # Exit on any error

CHECKPOINT_PATH="${1:-./checkpoints/fineweb_4gpu_300m_infini/15000}"  # latest checkpoint
CONTEXT_LENGTH="${2:-1024}"  # Start with 1K for testing, can go up to 8192
NUM_SAMPLES="${3:-25}"       # Reduced to save memory

# Create results directory with timestamp
SAVE_DIR="./results/passkey_300m_$(date +%Y%m%d_%H%M%S)"

# Validate checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint path does not exist: $CHECKPOINT_PATH"
    echo "Available checkpoints:"
    ls -la ./checkpoints/fineweb_4gpu_300m_infini/ 2>/dev/null || echo "  No checkpoints found"
    exit 1
fi

# Create results directory
mkdir -p $SAVE_DIR

echo "=========================================================="
echo "PASSKEY EVALUATION FOR 300M INFINI-ATTENTION MODEL"
echo "=========================================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Context Length: $CONTEXT_LENGTH tokens"
echo "Samples per depth: $NUM_SAMPLES"
echo "Segment Length: 1024 (from config)"
echo "Results will be saved to: $SAVE_DIR"
echo "=========================================================="

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
elif [ "$CONTEXT_LENGTH" -le 16384 ]; then
    # Check for directory or direct parquet file
    if [ -d "$LOCAL_16K_DATASET" ] || [ -f "$LOCAL_16K_DATASET/train-00000-of-00001.parquet" ]; then
        DATASET="$LOCAL_16K_DATASET"
        echo "Using local 16K dataset: $DATASET"
    else
        DATASET="nanotron/llama3-16k-passkey-retrieval-eval"
        echo "Using HuggingFace 16K dataset: $DATASET"
    fi
else
    echo "ERROR: Context length $CONTEXT_LENGTH not supported by pre-built datasets"
    echo "Available datasets support up to 16K tokens"
    exit 1
fi

# Set environment variables for distributed training
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=6,7

# Run evaluation using the same parallelism as training (4 DP)
echo "Starting evaluation..."
echo "Command: torchrun --nproc_per_node=4 examples/infinite-context-length/scripts/run_passkey_eval.py"
echo ""

torchrun --nproc_per_node=2 \
    examples/infinite-context-length/scripts/run_passkey_eval.py \
    --ckpt-path $CHECKPOINT_PATH \
    --save_path $SAVE_DIR \
    --eval_dataset_path $DATASET \
    --num_shots 0 \
    --num_digits 0 \
    --seed 42 \
    --num_samples $NUM_SAMPLES \
    --max-new-tokens 15 \
    --dp 2 \
    --tp 1 \
    --pp 1

# Check if evaluation completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================================="
    echo "EVALUATION COMPLETED SUCCESSFULLY!"
    echo "=========================================================="
    
    # Count result files
    RESULT_FILES=$(ls $SAVE_DIR/*.pkl 2>/dev/null | wc -l)
    echo "Generated $RESULT_FILES result files"
    
    # Run analysis if results exist
    if [ $RESULT_FILES -gt 0 ]; then
        echo "Analyzing results..."
        echo ""
        
        # Check if analysis script exists
        if [ -f "examples/infinite-context-length/scripts/analyze_passkey_results.py" ]; then
            python examples/infinite-context-length/scripts/analyze_passkey_results.py $SAVE_DIR
        else
            echo "Analysis script not found. Results saved to: $SAVE_DIR"
            echo "Result files:"
            ls -la $SAVE_DIR/*.pkl
        fi
    else
        echo "WARNING: No result files generated. Check logs for errors."
    fi
    
    echo ""
    echo "Results saved to: $SAVE_DIR"
    
else
    echo ""
    echo "ERROR: Evaluation failed. Check logs above for details."
    exit 1
fi