#!/bin/bash
# Script to train a single Infini-Attention model

set -e  # Exit on error

echo "=========================================================="
echo "Training a single Infini-Attention model"
echo "=========================================================="

# Configuration
CONFIG_FILE="scripts/config/tiny_test_config.yaml"
PREPROCESS_SCRIPT="scripts/preprocessing/preprocess_data_fixed.py"
TRAIN_SCRIPT="train_infini_llama.py"
OUTPUT_DIR="tiny_test_data"
GPU_DEVICE="cuda:0"  # Change to the GPU you want to use

# Create timestamp for unique directory names
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TB_DIR="tensorboard_logs/infini_${TIMESTAMP}"

echo "Configuration:"
echo "- Config file: ${CONFIG_FILE}"
echo "- Preprocessing script: ${PREPROCESS_SCRIPT}"
echo "- Training script: ${TRAIN_SCRIPT}"
echo "- Output directory: ${OUTPUT_DIR}"
echo "- GPU device: ${GPU_DEVICE}"
echo "- TensorBoard logs: ${TB_DIR}"
echo ""

# Step 1: Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Step 2: Run preprocessing
echo "=========================================================="
echo "Step 1: Preprocessing data..."
echo "=========================================================="
python ${PREPROCESS_SCRIPT} \
    --config-file ${CONFIG_FILE} \
    --output-dir ${OUTPUT_DIR} \
    --gpu-id 0

# Get the most recent preprocessed data directory
LATEST_DATA=$(find ${OUTPUT_DIR} -type d -name "preprocessed_*" | sort -r | head -n 1)
if [ -z "${LATEST_DATA}" ]; then
    echo "Error: No preprocessed data directory found."
    exit 1
fi

echo "Using preprocessed data from: ${LATEST_DATA}"

# Step 3: Train the model with Infini-Attention
echo "=========================================================="
echo "Step 2: Training model with Infini-Attention..."
echo "=========================================================="
python ${TRAIN_SCRIPT} \
    --config-file ${CONFIG_FILE} \
    --gpu-device ${GPU_DEVICE} \
    --tensorboard-dir ${TB_DIR} \
    --use-gpu-dataloader \
    --data-dir ${LATEST_DATA} \
    --verbose

echo "=========================================================="
echo "Training complete!"
echo "- Check TensorBoard logs in: ${TB_DIR}"
echo "=========================================================="

# Launch TensorBoard
echo "Starting TensorBoard..."
tensorboard --logdir=${TB_DIR} --port=6006 &
echo "TensorBoard started at http://localhost:6006"
