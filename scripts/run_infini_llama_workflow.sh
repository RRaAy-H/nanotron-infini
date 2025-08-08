#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/run_infini_llama_workflow.sh

# Infini-Llama Complete Workflow Script
# This script runs the complete Infini-Llama workflow:
# 1. Preprocess the data
# 2. Train the model using the preprocessed data

# Set default values
CONFIG_FILE="custom_infini_config_gpu.yaml"
DATA_DIR="preprocessed_data"
TENSORBOARD_DIR="tensorboard_logs"
GPU_DEVICE="0"
USE_FLASH_ATTENTION=true
USE_INFINI_ATTENTION=true
SKIP_PREPROCESSING=false
PREPROCESSING_BATCH_SIZE=2048
DISTRIBUTED=false
NUM_NODES=1

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --config|-c)
            CONFIG_FILE="$2"
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
        --disable-flash-attn)
            USE_FLASH_ATTENTION=false
            shift
            ;;
        --disable-infini-attn)
            USE_INFINI_ATTENTION=false
            shift
            ;;
        --skip-preprocessing)
            SKIP_PREPROCESSING=true
            shift
            ;;
        --preprocessing-batch-size)
            PREPROCESSING_BATCH_SIZE="$2"
            shift 2
            ;;
        --distributed)
            DISTRIBUTED=true
            shift
            ;;
        --num-nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --config, -c FILE                Configuration file (default: custom_infini_config_gpu.yaml)"
            echo "  --data-dir, -d DIR               Directory for preprocessed data (default: preprocessed_data)"
            echo "  --tensorboard-dir, -t DIR        Directory for TensorBoard logs (default: tensorboard_logs)"
            echo "  --gpu, -g NUM                    GPU device number to use (default: 0)"
            echo "  --disable-flash-attn             Disable Flash Attention"
            echo "  --disable-infini-attn            Disable Infini-Attention and use standard attention only (baseline model)"
            echo "  --skip-preprocessing             Skip the preprocessing stage (use if already done)"
            echo "  --preprocessing-batch-size SIZE  Batch size for preprocessing (default: 2048)"
            echo "  --distributed                    Enable distributed training"
            echo "  --num-nodes NUM                  Number of nodes for distributed training (default: 1)"
            echo "  --help, -h                       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create directories if they don't exist
mkdir -p "$DATA_DIR"
mkdir -p "$TENSORBOARD_DIR"

# Set up GPU environment
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE

# Print GPU info
echo "Using GPU device: $GPU_DEVICE"
echo "Running nvidia-smi to confirm GPU selection:"
nvidia-smi

# Preprocessing stage
if [ "$SKIP_PREPROCESSING" = false ]; then
    echo "===== STAGE 1: PREPROCESSING DATA ====="
    echo "Using configuration: $CONFIG_FILE"
    echo "Output directory: $DATA_DIR"
    
    PREPROCESS_CMD="python scripts/preprocessing/preprocess_data.py \
        --config-file $CONFIG_FILE \
        --output-dir $DATA_DIR \
        --gpu-id $GPU_DEVICE \
        --batch-size $PREPROCESSING_BATCH_SIZE"
    
    echo "Running preprocessing command: $PREPROCESS_CMD"
    eval $PREPROCESS_CMD
    
    # Check if preprocessing was successful
    if [ $? -ne 0 ]; then
        echo "Error: Preprocessing failed! Check the logs above for details."
        exit 1
    fi
    
    echo "Preprocessing completed successfully!"
    # Get the latest preprocessed data directory from the 'latest_preprocessed' file
    if [ -f "$DATA_DIR/latest_preprocessed" ]; then
        LATEST_DATA_DIR=$(cat "$DATA_DIR/latest_preprocessed")
        echo "Latest preprocessed data directory: $LATEST_DATA_DIR"
    else
        echo "Warning: Could not find latest preprocessed data reference."
        # Try to find the most recent directory
        LATEST_DATA_DIR=$(find "$DATA_DIR" -type d -name "preprocessed_*" | sort | tail -n 1)
        if [ -z "$LATEST_DATA_DIR" ]; then
            echo "Error: No preprocessed data found in $DATA_DIR"
            exit 1
        fi
        echo "Using most recent directory found: $LATEST_DATA_DIR"
    fi
else
    echo "Skipping preprocessing stage as requested."
    # Try to find the most recent directory if skipping preprocessing
    if [ -f "$DATA_DIR/latest_preprocessed" ]; then
        LATEST_DATA_DIR=$(cat "$DATA_DIR/latest_preprocessed")
    else
        LATEST_DATA_DIR=$(find "$DATA_DIR" -type d -name "preprocessed_*" | sort | tail -n 1)
    fi
    
    if [ -z "$LATEST_DATA_DIR" ]; then
        echo "Error: No preprocessed data found in $DATA_DIR. Did you run preprocessing before?"
        exit 1
    fi
    echo "Using preprocessed data from: $LATEST_DATA_DIR"
fi

# Training stage
echo ""
echo "===== STAGE 2: TRAINING MODEL ====="
echo "Using configuration: $CONFIG_FILE"
echo "Using preprocessed data from: $LATEST_DATA_DIR"

# Build base training command
TRAIN_BASE_CMD="python scripts/training/train_infini_llama.py \
    --config-file $CONFIG_FILE \
    --data-dir $LATEST_DATA_DIR \
    --tensorboard-dir $TENSORBOARD_DIR \
    --gpu-device cuda:0 \
    --use-gpu-dataloader"

# Add Flash Attention flag if needed
if [ "$USE_FLASH_ATTENTION" = false ]; then
    TRAIN_BASE_CMD="$TRAIN_BASE_CMD --disable-flash-attn"
fi

# Add Infini-Attention flag if needed
if [ "$USE_INFINI_ATTENTION" = false ]; then
    TRAIN_BASE_CMD="$TRAIN_BASE_CMD --disable-infini-attn"
fi

# Use torchrun for distributed training if requested
if [ "$DISTRIBUTED" = true ]; then
    TRAIN_CMD="torchrun --nproc_per_node=$(nvidia-smi --list-gpus | wc -l) --nnodes=$NUM_NODES $TRAIN_BASE_CMD"
else
    TRAIN_CMD="$TRAIN_BASE_CMD"
fi

echo "Running training command: $TRAIN_CMD"
eval $TRAIN_CMD

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "Error: Training failed! Check the logs above for details."
    exit 1
fi

echo "Training completed successfully!"
echo "TensorBoard logs saved to: $TENSORBOARD_DIR"
echo "You can view training metrics with: tensorboard --logdir $TENSORBOARD_DIR"
