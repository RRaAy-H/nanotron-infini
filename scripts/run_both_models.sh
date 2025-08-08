#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/run_both_models.sh
#
# This script runs both the baseline model and the Infini-Attention model in parallel
# on different GPUs for easy comparison

# Set paths - adjust the project root to point to the repository root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default values
PREPROCESSED_DATA=""
CONFIG_FILE="scripts/config/tiny_test_config.yaml"
GPU_ID_BASELINE=0
GPU_ID_INFINI=1
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --preprocessed-data)
            PREPROCESSED_DATA="$2"
            shift 2
            ;;
        --config-file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --gpu-baseline)
            GPU_ID_BASELINE="$2"
            shift 2
            ;;
        --gpu-infini)
            GPU_ID_INFINI="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Run both baseline and Infini-Attention models in parallel"
            echo ""
            echo "Usage: ./run_both_models.sh [options]"
            echo ""
            echo "Options:"
            echo "  --preprocessed-data PATH  Path to preprocessed data directory"
            echo "  --config-file PATH        Path to configuration YAML file (default: scripts/config/tiny_test_config.yaml)"
            echo "  --gpu-baseline ID         GPU ID for baseline model (default: 0)"
            echo "  --gpu-infini ID           GPU ID for Infini-Attention model (default: 1)"
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

# Check if preprocessed data is specified
if [[ -z "$PREPROCESSED_DATA" ]]; then
    echo "Error: --preprocessed-data must be specified."
    echo "Use --help for usage information."
    exit 1
fi

# Validate config file
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create timestamp for TensorBoard logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASELINE_TB_DIR="tensorboard_logs/baseline_${TIMESTAMP}"
INFINI_TB_DIR="tensorboard_logs/infini_${TIMESTAMP}"

# Create TensorBoard directories
mkdir -p "$BASELINE_TB_DIR"
mkdir -p "$INFINI_TB_DIR"

# Print configuration
echo "-------------------------------------"
echo "Running Both Models in Parallel:"
echo "-------------------------------------"
echo "Config file: $CONFIG_FILE"
echo "Preprocessed data: $PREPROCESSED_DATA"
echo "Baseline model GPU: $GPU_ID_BASELINE"
echo "Infini-Attention model GPU: $GPU_ID_INFINI"
echo "Baseline TensorBoard: $BASELINE_TB_DIR"
echo "Infini-Attention TensorBoard: $INFINI_TB_DIR"
echo "Verbose logging: $VERBOSE"
echo "-------------------------------------"

# Run baseline model in background
echo "Starting baseline model training on GPU $GPU_ID_BASELINE..."
BASELINE_CMD="./scripts/flexible_training_workflow.sh \
    --preprocessed-data \"$PREPROCESSED_DATA\" \
    --config-file \"$CONFIG_FILE\" \
    --gpu $GPU_ID_BASELINE \
    --tensorboard-dir \"$BASELINE_TB_DIR\" \
    --disable-infini-attn"

if [[ "$VERBOSE" = true ]]; then
    BASELINE_CMD="$BASELINE_CMD --verbose"
fi

# Run Infini-Attention model in background
echo "Starting Infini-Attention model training on GPU $GPU_ID_INFINI..."
INFINI_CMD="./scripts/flexible_training_workflow.sh \
    --preprocessed-data \"$PREPROCESSED_DATA\" \
    --config-file \"$CONFIG_FILE\" \
    --gpu $GPU_ID_INFINI \
    --tensorboard-dir \"$INFINI_TB_DIR\""

if [[ "$VERBOSE" = true ]]; then
    INFINI_CMD="$INFINI_CMD --verbose"
fi

# Run both in parallel
eval "$BASELINE_CMD" > baseline_training.log 2>&1 &
BASELINE_PID=$!
echo "Baseline training started with PID: $BASELINE_PID (logging to baseline_training.log)"

eval "$INFINI_CMD" > infini_training.log 2>&1 &
INFINI_PID=$!
echo "Infini-Attention training started with PID: $INFINI_PID (logging to infini_training.log)"

echo "-------------------------------------"
echo "Both models are now training in parallel."
echo "To monitor progress, use:"
echo "  tail -f baseline_training.log"
echo "  tail -f infini_training.log"
echo ""
echo "To visualize training metrics:"
echo "  tensorboard --logdir_spec baseline:$BASELINE_TB_DIR,infini:$INFINI_TB_DIR"
echo "-------------------------------------"

# Wait for both processes to finish
wait $BASELINE_PID
BASELINE_EXIT=$?
wait $INFINI_PID
INFINI_EXIT=$?

echo "-------------------------------------"
echo "Training completed:"
echo "Baseline model exit code: $BASELINE_EXIT"
echo "Infini-Attention model exit code: $INFINI_EXIT"
echo "TensorBoard logs:"
echo "  Baseline: $BASELINE_TB_DIR"
echo "  Infini-Attention: $INFINI_TB_DIR"
echo "-------------------------------------"
