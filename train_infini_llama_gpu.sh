#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/train_infini_llama_gpu.sh

# Usage: ./train_infini_llama_gpu.sh [GPU_ID] [TENSORBOARD_DIR]
# Example: ./train_infini_llama_gpu.sh 0 tensorboard_logs

# Default values
GPU_ID=${1:-0}  # Default to GPU 0 if not specified
TENSORBOARD_DIR=${2:-"/Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/tensorboard_logs"}  # Default directory for TensorBoard logs
CONFIG_FILE="custom_infini_config_gpu.yaml"

# Set environment variables
export CUDA_VISIBLE_DEVICES=$GPU_ID
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Create TensorBoard directory if it doesn't exist
mkdir -p $TENSORBOARD_DIR

echo "=========================================="
echo "Starting Infini-Llama GPU training with TensorBoard"
echo "=========================================="
echo "Using GPU ID: $GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "TensorBoard logs will be saved to: $TENSORBOARD_DIR"
echo "Configuration file: $CONFIG_FILE"
echo "=========================================="

# Install dependencies if needed
echo "Checking and installing required dependencies..."
pip install -e . 2>/dev/null || echo "nanotron already installed"
pip install torch>=1.13.1 flash-attn>=2.5.0 datasets transformers huggingface_hub pyarrow pandas tensorboard torchvision tqdm

# Check if data is prepared
if [ ! -d "/Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/data/processed" ]; then
  echo "Preparing dataset..."
  python prepare_data.py
else
  echo "Dataset already prepared."
fi

# Start TensorBoard in the background
echo "Starting TensorBoard..."
tensorboard --logdir=$TENSORBOARD_DIR --port=6006 &
TENSORBOARD_PID=$!
echo "TensorBoard started with PID: $TENSORBOARD_PID"
echo "View TensorBoard at http://localhost:6006"

# Wait a moment for TensorBoard to start
sleep 3

# Run training with one GPU
echo "Starting training..."
python -u train_gpu_with_tensorboard.py --config-file $CONFIG_FILE --tensorboard-dir $TENSORBOARD_DIR

# Capture the exit code of the training script
TRAINING_EXIT_CODE=$?

# Clean up TensorBoard process when training is done
echo "Stopping TensorBoard process (PID: $TENSORBOARD_PID)..."
kill $TENSORBOARD_PID 2>/dev/null || echo "TensorBoard process already terminated"

# Show training status
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
  echo "Training completed successfully."
else
  echo "Training exited with code $TRAINING_EXIT_CODE."
fi

echo "=========================================="
echo "Training finished."
echo "TensorBoard logs are available in: $TENSORBOARD_DIR"
echo "To view logs later: tensorboard --logdir=$TENSORBOARD_DIR"
echo "=========================================="
