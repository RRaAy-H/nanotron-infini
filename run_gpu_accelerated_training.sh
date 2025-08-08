#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/run_gpu_accelerated_training.sh

# This script runs the Infini-Llama training with GPU-accelerated data processing
# It specifically sets up to use GPU 0 for both data processing and training

# Set default values
CONFIG_FILE="custom_infini_config_gpu.yaml"
TENSORBOARD_DIR="tensorboard_logs"
GPU_DEVICE="0"  # Default to first GPU
USE_FLASH_ATTENTION=true

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --tensorboard-dir|-t)
            TENSORBOARD_DIR="$2"
            shift 2
            ;;
        --gpu|-g)
            GPU_DEVICE="$2"
            shift 2
            ;;
        --disable-flash-attn)
            USE_FLASH_ATTENTION=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --config, -c FILE      Configuration file (default: custom_infini_config_gpu.yaml)"
            echo "  --tensorboard-dir, -t  Directory for TensorBoard logs (default: tensorboard_logs)"
            echo "  --gpu, -g NUM          GPU device number to use (default: 0)"
            echo "  --disable-flash-attn   Disable Flash Attention"
            echo "  --help, -h             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set up environment for training
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE

# Print GPU info
echo "Using GPU device: $GPU_DEVICE"
echo "Running nvidia-smi to confirm GPU selection:"
nvidia-smi

# Build command
CMD="python train_infini_llama.py --config-file $CONFIG_FILE --tensorboard-dir $TENSORBOARD_DIR --gpu-device cuda:0 --use-gpu-dataloader"

# Add Flash Attention flag if needed
if [ "$USE_FLASH_ATTENTION" = false ]; then
    CMD="$CMD --disable-flash-attn"
fi

# Print the command
echo "Running command: $CMD"
echo "Starting training..."

# Run the training script
$CMD
