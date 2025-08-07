#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/train_infini_llama_gpu.sh

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Directory for TensorBoard logs
TENSORBOARD_DIR="/Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/tensorboard_logs"

# Create TensorBoard directory if it doesn't exist
mkdir -p $TENSORBOARD_DIR

echo "=========================================="
echo "Starting training with GPU and TensorBoard"
echo "=========================================="
echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "TensorBoard logs will be saved to: $TENSORBOARD_DIR"
echo "=========================================="

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
python -u train_gpu_with_tensorboard.py --config-file custom_infini_config_gpu.yaml --tensorboard-dir $TENSORBOARD_DIR

# Clean up TensorBoard process when training is done
kill $TENSORBOARD_PID
echo "Training completed and TensorBoard stopped."
