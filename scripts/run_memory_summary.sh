#!/bin/bash

# Quick Memory Summary Test Runner
# Runs essential memory tests and outputs a concise summary

set -e

echo "================================================================="
echo "Infini-Attention Memory Summary Test"
echo "================================================================="

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <checkpoint_path> [options]"
    echo ""
    echo "Options:"
    echo "  --extreme      Test with very long contexts (32K, 64K, 128K)"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/checkpoint                    # Standard test (8K, 16K, 32K)"
    echo "  $0 /path/to/checkpoint --extreme          # Extreme test (32K, 64K, 128K)"
    exit 1
fi

CHECKPOINT_PATH="$1"
shift  # Remove checkpoint path, keep other args

# Check checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint path '$CHECKPOINT_PATH' does not exist"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT_PATH"
echo ""

# Set up environment
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# Check dependencies
echo "Checking dependencies..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" || { echo "Error: PyTorch not found"; exit 1; }
python -c "import sklearn; print('✓ scikit-learn available')" 2>/dev/null || {
    echo "Installing scikit-learn..."
    pip install scikit-learn
}

echo ""

# Run summary test
echo "Running memory summary test..."
python scripts/memory_summary_test.py --checkpoint "$CHECKPOINT_PATH" "$@"

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "✅ Memory mechanism: EXCELLENT/GOOD"
elif [ $exit_code -eq 1 ]; then
    echo "⚠️  Memory mechanism: MODERATE"
else
    echo "❌ Memory mechanism: POOR"
fi

exit $exit_code
