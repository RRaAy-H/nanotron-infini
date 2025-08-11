#!/usr/bin/env bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/train_tiny_infini_fixed.sh

# Script to train the Infini-Llama model on the tiny dataset with the fixed training script

echo "===== Infini-Llama Training Script with Fixed Dependencies ====="

# Make script exit on any error
set -e

# Set up environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "Current directory: $(pwd)"
echo "Python path: $PYTHONPATH"

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null
then
    echo "CUDA is available. Using GPU for training."
    GPU_FLAG=""
else
    echo "CUDA is not available. Using CPU for training."
    GPU_FLAG="--cpu-only"
fi

# Create directories if they don't exist
mkdir -p tensorboard_logs
mkdir -p models

# Make sure the script is executable
chmod +x train_infini_llama_fixed.py

# Data directory to use
DATA_DIR="/data1/dataset/HuggingFaceFW/processed/tiny"
echo "Using data directory: $DATA_DIR"

# Run the fixed training script with the custom configuration
echo "Starting training with fixed script..."
python train_infini_llama_fixed.py \
    --config-file custom_tiny_infini_config.yaml \
    --data-dir "$DATA_DIR" \
    --tensorboard-dir tensorboard_logs \
    --use-gpu-dataloader \
    --verbose \
    $GPU_FLAG

# Check if the training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with error code $?."
    exit 1
fi
