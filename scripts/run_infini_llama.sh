#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/run_infini_llama.sh

# Infini-Llama Training Pipeline
# This script provides a simple interface to run the Infini-Llama training pipeline

# Default values
CONFIG_FILE="custom_infini_config_gpu.yaml"
OUTPUT_DIR="processed_data"
DATA_DIR=""
TENSORBOARD_DIR="tensorboard_logs"
GPU_DEVICE="0"
USE_FLASH_ATTENTION=true
USE_GPU_DATALOADER=true
PREPROCESS_ONLY=false
TRAIN_ONLY=false
CPU_ONLY=false
SEED=42

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Display help message
show_help() {
    echo "Infini-Llama Training Pipeline"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --config, -c FILE         Configuration file (default: custom_infini_config_gpu.yaml)"
    echo "  --output-dir, -o DIR      Directory to save preprocessed data (default: processed_data)"
    echo "  --data-dir, -d DIR        Directory containing preprocessed data (defaults to output-dir if not specified)"
    echo "  --tensorboard-dir, -t DIR Directory for TensorBoard logs (default: tensorboard_logs)"
    echo "  --gpu, -g NUM             GPU device number to use (default: 0)"
    echo "  --cpu-only                Use CPU only for training"
    echo "  --disable-flash-attn      Disable Flash Attention"
    echo "  --disable-gpu-dataloader  Disable GPU-accelerated data processing"
    echo "  --preprocess-only         Only run preprocessing step"
    echo "  --train-only              Only run training step (requires preprocessed data)"
    echo "  --seed NUM                Random seed for reproducibility (default: 42)"
    echo "  --help, -h                Show this help message"
    echo ""
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data-dir|-d)
            DATA_DIR="$2"
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
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --disable-flash-attn)
            USE_FLASH_ATTENTION=false
            shift
            ;;
        --disable-gpu-dataloader)
            USE_GPU_DATALOADER=false
            shift
            ;;
        --preprocess-only)
            PREPROCESS_ONLY=true
            shift
            ;;
        --train-only)
            TRAIN_ONLY=true
            shift
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set DATA_DIR to OUTPUT_DIR if not specified
if [ -z "$DATA_DIR" ]; then
    DATA_DIR="$OUTPUT_DIR"
fi

# Construct command for run_infini_llama.py
CMD="python ${SCRIPT_DIR}/run_infini_llama.py --config-file ${CONFIG_FILE} --output-dir ${OUTPUT_DIR} --data-dir ${DATA_DIR} --seed ${SEED}"

# Add options based on parameters
if [ "$CPU_ONLY" = true ]; then
    CMD="${CMD} --cpu-only"
fi

if [ "$USE_FLASH_ATTENTION" = false ]; then
    CMD="${CMD} --disable-flash-attn"
fi

if [ "$USE_GPU_DATALOADER" = true ]; then
    CMD="${CMD} --use-gpu-dataloader"
fi

if [ "$PREPROCESS_ONLY" = true ]; then
    CMD="${CMD} --preprocess-only"
fi

if [ "$TRAIN_ONLY" = true ]; then
    CMD="${CMD} --train-only"
fi

if [ -n "$TENSORBOARD_DIR" ]; then
    CMD="${CMD} --tensorboard-dir ${TENSORBOARD_DIR}"
fi

if [ -n "$GPU_DEVICE" ]; then
    CMD="${CMD} --gpu-device ${GPU_DEVICE}"
fi

# Print the command being run
echo "Running: $CMD"

# Execute the command
eval "$CMD"

# Check the exit status
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "Error: Command failed with status $STATUS"
    exit $STATUS
fi

echo "Infini-Llama training pipeline completed successfully!"
