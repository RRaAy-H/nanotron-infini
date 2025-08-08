#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/run_direct_training.sh

# Change to project root directory
cd "$(dirname "$0")"
ROOT_DIR="$(pwd)"

# Set Python path to include project root and src directory
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:$PYTHONPATH"

# Run the direct training script
python scripts/run_direct_training.py "$@"
