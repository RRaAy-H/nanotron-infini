#!/usr/bin/env bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/run_infini_llama_training.sh

# Master script to run the Infini-Llama training with proper error handling
# This script will:
# 1. Check the dataset and prepare it if needed
# 2. Run the fixed training script

set -e  # Exit on any error

# Set up environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "======= Infini-Llama Training Pipeline ======="
echo "Current directory: $(pwd)"

# Data directory to use
DATA_DIR="/data1/dataset/HuggingFaceFW/processes/tiny"
echo "Using data directory: $DATA_DIR"

# Step 1: Check and prepare the dataset
echo ""
echo "Step 1: Checking and preparing dataset..."
python prepare_tiny_dataset.py --data-dir "$DATA_DIR"

if [ $? -ne 0 ]; then
    echo "Dataset preparation failed. Exiting."
    exit 1
fi

echo "Dataset preparation completed successfully."

# Step 2: Run the training script
echo ""
echo "Step 2: Running the training script..."
./train_tiny_infini_fixed.sh

if [ $? -ne 0 ]; then
    echo "Training failed. Exiting."
    exit 1
fi

echo "Training completed successfully!"
echo "========================================"
echo "The entire pipeline has completed successfully!"
