#!/bin/bash
# Script to run both Infini-Attention and baseline models in parallel

# Set the current date and time for log directory names
timestamp=$(date +"%Y%m%d_%H%M%S")

# Check if data directory is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <preprocessed_data_dir>"
    echo "Example: $0 tiny_test_data/preprocessed_20250808_123456"
    exit 1
fi

data_dir=$1
config_file="scripts/config/tiny_test_config.yaml"

# Make sure to run from the project root directory
cd "$(dirname "$0")/.."

# Ensure PYTHONPATH includes the project root and src directories
export PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/src"

# Create log directories
mkdir -p tensorboard_logs

# Start Infini-Attention model training on GPU 0
echo "Starting training with Infini-Attention on GPU 0..."
python train_infini_llama.py \
    --config-file "$config_file" \
    --gpu-device cuda:0 \
    --tensorboard-dir "tensorboard_logs/infini_${timestamp}" \
    --use-gpu-dataloader \
    --data-dir "$data_dir" &

# Wait a few seconds to ensure proper GPU memory allocation
sleep 5

# Start baseline model (without Infini-Attention) on GPU 1
echo "Starting training with baseline model on GPU 1..."
python train_infini_llama.py \
    --config-file "$config_file" \
    --gpu-device cuda:1 \
    --tensorboard-dir "tensorboard_logs/baseline_${timestamp}" \
    --use-gpu-dataloader \
    --disable-infini-attn \
    --data-dir "$data_dir" &

# Wait for both processes to complete
wait

echo "Both training processes completed!"
echo "You can view the results in TensorBoard:"
echo "  tensorboard --logdir tensorboard_logs"
