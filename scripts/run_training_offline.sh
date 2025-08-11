#!/bin/bash
# run_training_offline.sh - Wrapper script to run training in offline mode
# This script combines offline mode with the option to disable Flash Attention

# Set paths - adjust the project root to point to the repository root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "======================================================="
echo "Infini-Llama Training - Offline Mode Wrapper"
echo "======================================================="
echo "This script runs training in offline mode with optional Flash Attention disabling"
echo

# Default values
DISABLE_FLASH=false
GPU_ID=0
CONFIG_FILE="scripts/config/tiny_test_config.yaml"
RAW_DATA=""
PREPROCESSED_DATA=""
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --disable-flash-attn)
            DISABLE_FLASH=true
            shift
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --config-file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --raw-data)
            RAW_DATA="$2"
            shift 2
            ;;
        --preprocessed-data)
            PREPROCESSED_DATA="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Usage: ./scripts/run_training_offline.sh [options]"
            echo
            echo "Options:"
            echo "  --disable-flash-attn      Disable Flash Attention (for compatibility issues)"
            echo "  --gpu ID                  GPU ID to use (default: 0)"
            echo "  --config-file FILE        Config file to use (default: scripts/config/tiny_test_config.yaml)"
            echo "  --raw-data PATH           Path to raw data for preprocessing"
            echo "  --preprocessed-data PATH  Path to preprocessed data (skips preprocessing step)"
            echo "  --verbose                 Enable verbose logging"
            echo "  --help                    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Apply offline mode environment variables
source ./scripts/enable_offline_mode.sh

# Build command
CMD="./scripts/flexible_training_workflow.sh --offline-mode"

# Add Flash Attention disabled flag if requested
if [[ "$DISABLE_FLASH" = true ]]; then
    echo "Flash Attention will be disabled"
    CMD="$CMD --disable-flash-attn"
fi

# Add GPU selection
CMD="$CMD --gpu $GPU_ID"

# Add config file
CMD="$CMD --config-file $CONFIG_FILE"

# Add data paths if provided
if [[ -n "$RAW_DATA" ]]; then
    CMD="$CMD --raw-data $RAW_DATA"
fi

if [[ -n "$PREPROCESSED_DATA" ]]; then
    CMD="$CMD --preprocessed-data $PREPROCESSED_DATA"
fi

# Add verbose flag if requested
if [[ "$VERBOSE" = true ]]; then
    CMD="$CMD --verbose"
fi

# Display the command
echo "Running command:"
echo "$CMD"
echo
echo "Starting training in offline mode..."
echo "======================================================="

# Execute the command
eval "$CMD"
