#!/bin/bash

# Comprehensive memory test runner with automatic environment setup
# This script automatically sets up the distributed PyTorch environment
# and runs the comprehensive memory test

set -e  # Exit on any error

echo "================================================================="
echo "Infini-Attention Comprehensive Memory Test Runner"
echo "================================================================="

# Check if checkpoint path is provided
if [ $# -eq 0 ]; then
    echo "Error: No checkpoint path provided"
    echo "Usage: $0 <checkpoint_path> [additional_args...]"
    echo "       $0 <checkpoint_path> --summary    # Quick summary test"
    echo "Example: $0 /data1/infini-attn/infini-llama/nanotron-infini/checkpoints/fineweb_4gpu_300m_infini/30000"
    echo "         $0 /path/to/checkpoint --summary"
    exit 1
fi

CHECKPOINT_PATH="$1"
shift  # Remove first argument, keep the rest

# Check if checkpoint path exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint path '$CHECKPOINT_PATH' does not exist"
    exit 1
fi

echo "Checkpoint path: $CHECKPOINT_PATH"
echo ""

# Set up distributed PyTorch environment for single GPU
echo "Setting up distributed PyTorch environment..."
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0

echo "Environment variables:"
echo "  MASTER_ADDR=$MASTER_ADDR"
echo "  MASTER_PORT=$MASTER_PORT"
echo "  WORLD_SIZE=$WORLD_SIZE"
echo "  RANK=$RANK"
echo "  LOCAL_RANK=$LOCAL_RANK"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

# Check Python and dependencies
echo "Checking Python environment..."
python --version || { echo "Error: Python not found"; exit 1; }

echo "Checking dependencies..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" || { echo "Error: PyTorch not found"; exit 1; }
python -c "import sklearn; print('✓ scikit-learn available')" 2>/dev/null || {
    echo "⚠ Warning: scikit-learn not found. Installing..."
    pip install scikit-learn || { echo "Error: Failed to install scikit-learn"; exit 1; }
}

echo ""

# Check if summary mode requested
if [[ "$*" == *"--summary"* ]]; then
    echo "Running quick summary test..."
    echo "================================================================="
    # Remove --summary from args and pass the rest
    args_without_summary=$(echo "$@" | sed 's/--summary//')
    python scripts/memory_summary_test.py --checkpoint "$CHECKPOINT_PATH" $args_without_summary
else
    echo "Starting comprehensive memory analysis..."
    echo "================================================================="
    python scripts/test_memory_comprehensive.py --checkpoint "$CHECKPOINT_PATH" "$@"
fi

exit_code=$?

echo ""
echo "================================================================="
if [ $exit_code -eq 0 ]; then
    echo "✅ Comprehensive memory analysis completed successfully!"
elif [ $exit_code -eq 1 ]; then
    echo "⚠️  Comprehensive memory analysis completed with warnings"
else
    echo "❌ Comprehensive memory analysis failed"
fi
echo "================================================================="

exit $exit_code
