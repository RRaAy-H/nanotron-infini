#!/usr/bin/env bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/run_training_with_correct_paths.sh

# Master script to run the Infini-Llama training with the correct paths
# and working directory for your setup

set -e  # Exit on any error

# Set the correct working directory
WORKING_DIR="/home/data/daal_insight/fiery/Infini-attention/nanotron-infini"

# Go to the working directory
cd "$WORKING_DIR" || { echo "Cannot change to working directory $WORKING_DIR"; exit 1; }
echo "Changed to working directory: $(pwd)"

# Set up environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "======= Infini-Llama Training Pipeline ======="
echo "Current directory: $(pwd)"
echo "Python path: $PYTHONPATH"

# Data directory to use
DATA_DIR="/data1/dataset/HuggingFaceFW/processed/tiny"
echo "Using data directory: $DATA_DIR"

# Create directories if they don't exist
mkdir -p tensorboard_logs
mkdir -p infini_llama_checkpoints
mkdir -p models

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

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null
then
    echo "CUDA is available. Using GPU for training."
    GPU_FLAG=""
else
    echo "CUDA is not available. Using CPU for training."
    GPU_FLAG="--cpu-only"
fi

# Run the fixed training script with the custom configuration
echo "Starting training with fixed script..."
python train_infini_llama_fixed.py \
    --config-file custom_tiny_infini_config.yaml \
    --data-dir "$DATA_DIR" \
    --tensorboard-dir tensorboard_logs \
    --use-gpu-dataloader \
    --verbose \
    $GPU_FLAG

if [ $? -ne 0 ]; then
    echo "Training failed. Exiting."
    exit 1
fi

echo "Training completed successfully!"
echo "========================================"
echo "The entire pipeline has completed successfully!"
echo "Checkpoints saved in: $(pwd)/infini_llama_checkpoints"
echo "TensorBoard logs saved in: $(pwd)/tensorboard_logs"
