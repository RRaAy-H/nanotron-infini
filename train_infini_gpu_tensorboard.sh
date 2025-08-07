#!/bin/bash
# Script for running Infini-Llama training with TensorBoard on a single GPU
set -e  # Exit on error

# Configuration
BASE_DIR="/home/data/daal_insight/fiery/Infini-attention/nanotron-infini"
CONFIG_FILE="custom_infini_config_gpu.yaml"
TENSORBOARD_DIR="tensorboard_logs"

# Display header
echo "======================================================================"
echo "🚀 Infini-Llama Training with TensorBoard (Single GPU Version)"
echo "======================================================================"

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: NVIDIA GPU driver not found. Make sure CUDA is properly installed."
    exit 1
fi

# Display GPU information
echo "📊 GPU Information:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/  /'
echo ""

# Create directories if they don't exist
echo "🔧 Setting up directories..."
mkdir -p "${BASE_DIR}/checkpoints"
mkdir -p "${BASE_DIR}/${TENSORBOARD_DIR}"

# Set up environment variables for PyTorch distributed
echo "🔧 Configuring distributed environment for single GPU..."
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR="localhost"
export MASTER_PORT="29500"
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Start TensorBoard in the background
echo "🔄 Starting TensorBoard server..."
tensorboard --logdir="${BASE_DIR}/${TENSORBOARD_DIR}" --host=0.0.0.0 --port=6006 &
TB_PID=$!
echo "  TensorBoard running with PID ${TB_PID}"
echo "  Access TensorBoard at http://localhost:6006"

# Trap signals to kill TensorBoard when the script exits
trap "echo '🛑 Stopping TensorBoard...'; kill ${TB_PID} 2>/dev/null || true" EXIT INT TERM

# Print configuration info
echo "⚙️ Configuration:"
echo "  Config file: ${CONFIG_FILE}"
echo "  Base directory: ${BASE_DIR}"
echo "  TensorBoard logs: ${TENSORBOARD_DIR}"
echo ""

# Run the training script
echo "🔥 Starting training..."
cd "${BASE_DIR}"
python train_gpu_with_tensorboard.py --config-file "${CONFIG_FILE}" --tensorboard-dir "${TENSORBOARD_DIR}"

# Script finished successfully
echo "✅ Training completed successfully!"
