#!/bin/bash
# Script to run single Infini-Attention model training

# Set paths
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src

# Check if data directory is provided
DATA_DIR=${1:-tiny_test_data/preprocessed_*}
GPU_ID=${2:-0}
CONFIG_FILE="scripts/config/tiny_test_config.yaml"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TB_DIR="tensorboard_logs/infini_${TIMESTAMP}"

# Check if data needs to be preprocessed
if [[ ! -d "$DATA_DIR" || "$DATA_DIR" == *"*"* ]]; then
    echo "Preprocessed data not found or wildcard provided. Running preprocessing first..."
    mkdir -p tiny_test_data
    python scripts/preprocessing/preprocess_data_fixed.py \
        --config-file $CONFIG_FILE \
        --output-dir tiny_test_data \
        --gpu-id $GPU_ID
    
    # Find the latest preprocessed directory
    DATA_DIR=$(find tiny_test_data -name "preprocessed_*" -type d | sort | tail -n 1)
    echo "Using preprocessed data from: $DATA_DIR"
fi

# Run training with Infini-Attention enabled
echo "Starting training with Infini-Attention..."
echo "Using GPU $GPU_ID"
echo "Using data directory: $DATA_DIR"

CUDA_VISIBLE_DEVICES=$GPU_ID python train_infini_llama.py \
    --config-file $CONFIG_FILE \
    --gpu-device cuda:0 \
    --tensorboard-dir $TB_DIR \
    --use-gpu-dataloader \
    --data-dir $DATA_DIR \
    --verbose

echo "Training complete. TensorBoard logs saved to $TB_DIR"
echo "To view TensorBoard logs: tensorboard --logdir $TB_DIR"
