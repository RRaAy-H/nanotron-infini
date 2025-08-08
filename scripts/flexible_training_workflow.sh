#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/flexible_training_workflow.sh
# 
# Flexible training workflow script for Infini-Llama models
# This script allows flexible configuration of:
# - Data source (raw data path)
# - Output directory for preprocessed data
# - Config file to use
# - Whether to enable or disable Infini-Attention
# - Which GPU(s) to use
#
# Usage examples:
# ./flexible_training_workflow.sh --raw-data /data1/dataset/HuggingFaceFW/processed/tiny --config-file scripts/config/tiny_test_config.yaml
# ./flexible_training_workflow.sh --preprocessed-data tiny_test_data/preprocessed_20240808_123456 --config-file scripts/config/tiny_test_config.yaml --disable-infini-attn
# ./flexible_training_workflow.sh --raw-data /data1/dataset/HuggingFaceFW/processed/pile --config-file custom_infini_config_gpu.yaml --gpu 1

# Set paths - adjust the project root to point to the repository root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default values
RAW_DATA=""
PREPROCESSED_DATA=""
CONFIG_FILE="scripts/config/tiny_test_config.yaml"
OUTPUT_DIR="preprocessed_data"
DISABLE_INFINI_ATTN=false
GPU_ID=0
TENSORBOARD_DIR="tensorboard_logs/train_$(date +"%Y%m%d_%H%M%S")"
USE_GPU_DATALOADER=true
FORCE_PREPROCESS=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --raw-data)
            RAW_DATA="$2"
            shift 2
            ;;
        --preprocessed-data)
            PREPROCESSED_DATA="$2"
            shift 2
            ;;
        --config-file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --disable-infini-attn)
            DISABLE_INFINI_ATTN=true
            shift
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --tensorboard-dir)
            TENSORBOARD_DIR="$2"
            shift 2
            ;;
        --no-gpu-dataloader)
            USE_GPU_DATALOADER=false
            shift
            ;;
        --force-preprocess)
            FORCE_PREPROCESS=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Flexible training workflow for Infini-Llama models"
            echo ""
            echo "Usage: ./flexible_training_workflow.sh [options]"
            echo ""
            echo "Options:"
            echo "  --raw-data PATH           Path to raw data directory (for preprocessing)"
            echo "  --preprocessed-data PATH  Path to already preprocessed data directory"
            echo "  --config-file PATH        Path to configuration YAML file (default: scripts/config/tiny_test_config.yaml)"
            echo "  --output-dir PATH         Directory to save preprocessed data (default: preprocessed_data)"
            echo "  --disable-infini-attn     Disable Infini-Attention (run baseline model)"
            echo "  --gpu ID                  GPU ID to use (default: 0)"
            echo "  --tensorboard-dir PATH    Directory for TensorBoard logs"
            echo "  --no-gpu-dataloader       Disable GPU-accelerated dataloader"
            echo "  --force-preprocess        Force preprocessing even if data exists"
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

# Set Python path to include project root and src directory
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PYTHONPATH"

# Check if we have either raw data or preprocessed data
if [[ -z "$RAW_DATA" ]] && [[ -z "$PREPROCESSED_DATA" ]]; then
    echo "Error: Either --raw-data or --preprocessed-data must be specified."
    echo "Use --help for usage information."
    exit 1
fi

# Validate config file
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Set model type for logs and output naming
MODEL_TYPE="infini"
if [[ "$DISABLE_INFINI_ATTN" = true ]]; then
    MODEL_TYPE="baseline"
    TENSORBOARD_DIR="tensorboard_logs/baseline_$(date +"%Y%m%d_%H%M%S")"
else
    TENSORBOARD_DIR="tensorboard_logs/infini_$(date +"%Y%m%d_%H%M%S")"
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TENSORBOARD_DIR"

# Print configuration
echo "-------------------------------------"
echo "Infini-Llama Training Configuration:"
echo "-------------------------------------"
echo "Model type: $MODEL_TYPE"
echo "Config file: $CONFIG_FILE"
echo "GPU ID: $GPU_ID"
echo "TensorBoard dir: $TENSORBOARD_DIR"
echo "Using GPU dataloader: $USE_GPU_DATALOADER"
echo "Verbose logging: $VERBOSE"
echo "-------------------------------------"

# Preprocessing step
if [[ -n "$RAW_DATA" ]]; then
    # Check if we need to preprocess
    NEED_PREPROCESS=true
    
    # If we have a specified preprocessed data path and force is not enabled,
    # check if it exists and skip preprocessing
    if [[ -n "$PREPROCESSED_DATA" ]] && [[ "$FORCE_PREPROCESS" = false ]]; then
        if [[ -d "$PREPROCESSED_DATA" ]]; then
            echo "Using existing preprocessed data: $PREPROCESSED_DATA"
            NEED_PREPROCESS=false
        else
            echo "Specified preprocessed data directory doesn't exist: $PREPROCESSED_DATA"
            echo "Will preprocess from raw data."
        fi
    fi
    
    # Preprocess if needed
    if [[ "$NEED_PREPROCESS" = true ]]; then
        echo "Preprocessing data from: $RAW_DATA"
        # Create timestamp for preprocessed data directory
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        PREPROCESS_OUTPUT_DIR="$OUTPUT_DIR/preprocessed_$TIMESTAMP"
        
        # Run preprocessing script
        if [[ -f "$PROJECT_ROOT/scripts/preprocessing/preprocess_data_fixed.py" ]]; then
            PREPROCESS_CMD="python \"$PROJECT_ROOT/scripts/preprocessing/preprocess_data_fixed.py\" \
                --config-file \"$CONFIG_FILE\" \
                --output-dir \"$OUTPUT_DIR\" \
                --gpu-id \"$GPU_ID\""
                
            # Only add verbose flag if verbose is true
            if [[ "$VERBOSE" = true ]]; then
                PREPROCESS_CMD="$PREPROCESS_CMD --verbose"
            fi
            
            # Execute the command
            eval $PREPROCESS_CMD && PREPROCESSED_DATA=$(find "$OUTPUT_DIR" -name "preprocessed_*" -type d | sort | tail -n 1)
        else
            echo "Error: Preprocessing script not found."
            exit 1
        fi
        
        if [[ -z "$PREPROCESSED_DATA" ]]; then
            echo "Error: Preprocessing failed or no output directory was created."
            exit 1
        fi
        
        echo "Data preprocessing complete: $PREPROCESSED_DATA"
    fi
else
    # Ensure preprocessed data exists
    if [[ ! -d "$PREPROCESSED_DATA" ]]; then
        echo "Error: Specified preprocessed data directory doesn't exist: $PREPROCESSED_DATA"
        exit 1
    fi
fi

# Build training command
TRAIN_CMD="python $PROJECT_ROOT/train_infini_llama.py \
    --config-file \"$CONFIG_FILE\" \
    --gpu-device cuda:0 \
    --tensorboard-dir \"$TENSORBOARD_DIR\" \
    --data-dir \"$PREPROCESSED_DATA\""

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

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Run training
echo "Starting training with command:"
echo "$TRAIN_CMD"
echo "-------------------------------------"
eval $TRAIN_CMD

# Print completion message
echo "-------------------------------------"
echo "Training completed!"
echo "Model type: $MODEL_TYPE"
echo "TensorBoard logs saved to: $TENSORBOARD_DIR"
echo "To view training progress: tensorboard --logdir $TENSORBOARD_DIR"
echo "-------------------------------------"
