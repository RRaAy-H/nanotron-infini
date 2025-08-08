#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/run_training.sh

# Wrapper script to run the consolidated training script with common configurations

# Default values
CONFIG_FILE="custom_infini_config_gpu.yaml"
TENSORBOARD_DIR="tensorboard_logs"
USE_CPU=0
DISABLE_FLASH=0
NUM_GPUS=1
DISTRIBUTED=0

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
        --cpu)
            USE_CPU=1
            shift
            ;;
        --no-flash)
            DISABLE_FLASH=1
            shift
            ;;
        --gpus|-g)
            NUM_GPUS="$2"
            if [[ $NUM_GPUS -gt 1 ]]; then
                DISTRIBUTED=1
            fi
            shift 2
            ;;
        --distributed|-d)
            DISTRIBUTED=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --config, -c FILE      Configuration file (default: custom_infini_config_gpu.yaml)"
            echo "  --tensorboard-dir, -t  Directory for TensorBoard logs (default: tensorboard_logs)"
            echo "  --cpu                  Use CPU-only mode"
            echo "  --no-flash             Disable Flash Attention"
            echo "  --gpus, -g NUM         Number of GPUs to use (default: 1)"
            echo "  --distributed, -d      Force distributed training mode"
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

# Build the command
CMD=""

# Set up distributed training if needed
if [[ $DISTRIBUTED -eq 1 ]]; then
    # Use torchrun for distributed training
    export CUDA_DEVICE_MAX_CONNECTIONS=1  # important for some distributed operations
    CMD="torchrun --nproc_per_node=$NUM_GPUS"
fi

# Add the main script
CMD="$CMD python train_infini_llama.py --config-file $CONFIG_FILE"

# Add optional arguments
if [[ -n "$TENSORBOARD_DIR" ]]; then
    CMD="$CMD --tensorboard-dir $TENSORBOARD_DIR"
fi

if [[ $USE_CPU -eq 1 ]]; then
    CMD="$CMD --cpu-only"
fi

if [[ $DISABLE_FLASH -eq 1 ]]; then
    CMD="$CMD --disable-flash-attn"
fi

# Print the command to be executed
echo "Executing: $CMD"

# Run the command
eval $CMD
