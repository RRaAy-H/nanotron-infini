#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/parquet_training_workflow.sh
# 
# Training workflow script for Infini-Llama models using parquet files
# This script handles the processing of parquet files and training with Infini-Attention
#
# Usage:
# ./parquet_training_workflow.sh --parquet-data /path/to/parquet/files --config-file custom_infini_config_gpu.yaml

# Source the flexible training workflow to use common functions and variables
source "$(dirname "$0")/flexible_training_workflow.sh" --help > /dev/null 2>&1 || true

# Default values specific to parquet processing
PARQUET_DATA="/Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/data"
OUTPUT_DIR="preprocessed_data"
CONFIG_FILE="scripts/config/tiny_test_config.yaml"
TOKENIZER="meta-llama/Llama-2-7b-hf"
MAX_SEQ_LENGTH=2048
GPU_ID=0
DISABLE_INFINI_ATTN=false
USE_GPU_DATALOADER=true
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --parquet-data)
            PARQUET_DATA="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config-file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER="$2"
            shift 2
            ;;
        --max-seq-length)
            MAX_SEQ_LENGTH="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --disable-infini-attn)
            DISABLE_INFINI_ATTN=true
            shift
            ;;
        --no-gpu-dataloader)
            USE_GPU_DATALOADER=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Parquet Training Workflow for Infini-Llama models"
            echo ""
            echo "Usage: ./parquet_training_workflow.sh [options]"
            echo ""
            echo "Options:"
            echo "  --parquet-data PATH       Path to parquet files directory (default: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/data)"
            echo "  --output-dir PATH         Directory to save preprocessed data (default: preprocessed_data)"
            echo "  --config-file PATH        Path to configuration YAML file (default: scripts/config/tiny_test_config.yaml)"
            echo "  --tokenizer NAME          Tokenizer name or path (default: meta-llama/Llama-2-7b-hf)"
            echo "  --max-seq-length NUM      Maximum sequence length for tokenization (default: 2048)"
            echo "  --gpu-id ID               GPU ID to use (default: 0)"
            echo "  --disable-infini-attn     Disable Infini-Attention (run baseline model)"
            echo "  --no-gpu-dataloader       Disable GPU-accelerated dataloader"
            echo "  --verbose                 Enable verbose logging"
            echo "  --help                    Show this help message and exit"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Set project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Set Python path
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PYTHONPATH"

# Print configuration
echo "-------------------------------------"
echo "Infini-Llama Parquet Training Configuration:"
echo "-------------------------------------"
echo "Parquet data path: $PARQUET_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Config file: $CONFIG_FILE"
echo "Tokenizer: $TOKENIZER"
echo "Max sequence length: $MAX_SEQ_LENGTH"
echo "GPU ID: $GPU_ID"
echo "Infini-Attention: $([ "$DISABLE_INFINI_ATTN" = true ] && echo "disabled" || echo "enabled")"
echo "Using GPU dataloader: $([ "$USE_GPU_DATALOADER" = true ] && echo "yes" || echo "no")"
echo "Verbose logging: $([ "$VERBOSE" = true ] && echo "yes" || echo "no")"
echo "-------------------------------------"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process parquet data
echo "Processing parquet data from: $PARQUET_DATA"
# Create timestamp for preprocessed data directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Run parquet data loader script
if [[ -f "$PROJECT_ROOT/scripts/parquet_data_loader.py" ]]; then
    PARQUET_CMD="python \"$PROJECT_ROOT/scripts/parquet_data_loader.py\" \
        --data-dir \"$PARQUET_DATA\" \
        --output-dir \"$OUTPUT_DIR\" \
        --tokenizer \"$TOKENIZER\" \
        --max-seq-length \"$MAX_SEQ_LENGTH\""
        
    # Add no-gpu flag if requested
    if [[ "$USE_GPU_DATALOADER" = false ]]; then
        PARQUET_CMD="$PARQUET_CMD --no-gpu"
    fi
    
    # Execute the command
    eval $PARQUET_CMD && PREPROCESSED_DATA=$(find "$OUTPUT_DIR" -name "preprocessed_*" -type d | sort | tail -n 1)
else
    echo "Error: Parquet data loader script not found at $PROJECT_ROOT/scripts/parquet_data_loader.py"
    exit 1
fi

if [[ -z "$PREPROCESSED_DATA" ]]; then
    echo "Error: Parquet data processing failed or no output directory was created."
    exit 1
fi

echo "Parquet data processing complete: $PREPROCESSED_DATA"

# Set up tensorboard directory
TENSORBOARD_DIR="tensorboard_logs/$([ "$DISABLE_INFINI_ATTN" = true ] && echo "baseline" || echo "infini")_$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$TENSORBOARD_DIR"

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Set up training environment variables for single GPU mode
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR="localhost"
export MASTER_PORT=29500

# Ensure the wrapper script exists
WRAP_SCRIPT="$PROJECT_ROOT/scripts/wrapper_script.py"
if [[ ! -f "$WRAP_SCRIPT" ]]; then
    echo "Error: Wrapper script not found at $WRAP_SCRIPT"
    exit 1
fi
chmod +x "$WRAP_SCRIPT"

# Build training command
TRAIN_CMD="python \"$WRAP_SCRIPT\" \
    --config-file \"$CONFIG_FILE\" \
    --data-dir \"$PREPROCESSED_DATA\" \
    --gpu-id \"$GPU_ID\" \
    --tensorboard-dir \"$TENSORBOARD_DIR\""

# Add optional flags
if [[ "$DISABLE_INFINI_ATTN" = true ]]; then
    TRAIN_CMD="$TRAIN_CMD --disable-infini-attn"
fi

if [[ "$USE_GPU_DATALOADER" = true ]]; then
    TRAIN_CMD="$TRAIN_CMD --use-gpu-dataloader"
fi

if [[ "$VERBOSE" = true ]]; then
    TRAIN_CMD="$TRAIN_CMD --verbose"
fi

# Run training
echo "Starting training with command:"
echo "$TRAIN_CMD"
echo "-------------------------------------"
eval $TRAIN_CMD

# Print completion message
echo "-------------------------------------"
echo "Training completed!"
echo "Model type: $([ "$DISABLE_INFINI_ATTN" = true ] && echo "baseline" || echo "infini")"
echo "TensorBoard logs saved to: $TENSORBOARD_DIR"
echo "To view training progress: tensorboard --logdir $TENSORBOARD_DIR"
echo "-------------------------------------"
