#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/run_comparison.sh
#
# This script runs two training sessions: one with Infini-Attention enabled and one with baseline attention.
# It helps compare the performance and results between the two approaches.

set -e  # Exit on error

# Default values
CONFIG_FILE="custom_infini_config_gpu.yaml"
USE_GPU_DATALOADER=true
DISABLE_FLASH_ATTN=false
TENSORBOARD_DIR="tensorboard_logs"
GPU_DEVICE="0"
VERBOSE=false
CHECKPOINT_DIR="checkpoints"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --config|-c)
      CONFIG_FILE="$2"
      shift
      shift
      ;;
    --no-gpu-dataloader)
      USE_GPU_DATALOADER=false
      shift
      ;;
    --disable-flash-attn)
      DISABLE_FLASH_ATTN=true
      shift
      ;;
    --tensorboard-dir|-t)
      TENSORBOARD_DIR="$2"
      shift
      shift
      ;;
    --gpu|-g)
      GPU_DEVICE="$2"
      shift
      shift
      ;;
    --verbose|-v)
      VERBOSE=true
      shift
      ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
      shift
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --config, -c FILE         Configuration file to use (default: custom_infini_config_gpu.yaml)"
      echo "  --no-gpu-dataloader       Disable GPU-accelerated data loading"
      echo "  --disable-flash-attn      Disable Flash Attention"
      echo "  --tensorboard-dir, -t DIR Directory for TensorBoard logs (default: tensorboard_logs)"
      echo "  --gpu, -g NUM             GPU device number to use (default: 0)"
      echo "  --verbose, -v             Enable verbose output"
      echo "  --checkpoint-dir DIR      Directory for saving checkpoints (default: checkpoints)"
      echo "  --help, -h                Show this help message"
      echo ""
      echo "This script runs two training sessions in sequence:"
      echo "1. Infini-Attention model (with memory enabled)"
      echo "2. Baseline model (with infini attention memory disabled)"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help to see available options"
      exit 1
      ;;
  esac
done

# Create directories
mkdir -p "$TENSORBOARD_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Set timestamp for unique run naming
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
INFINI_TB_DIR="${TENSORBOARD_DIR}/infini_${TIMESTAMP}"
BASELINE_TB_DIR="${TENSORBOARD_DIR}/baseline_${TIMESTAMP}"
INFINI_CP_DIR="${CHECKPOINT_DIR}/infini_${TIMESTAMP}"
BASELINE_CP_DIR="${CHECKPOINT_DIR}/baseline_${TIMESTAMP}"

# Construct common arguments
COMMON_ARGS="--config-file ${CONFIG_FILE} --gpu-device cuda:${GPU_DEVICE}"

if [ "$USE_GPU_DATALOADER" = true ]; then
  COMMON_ARGS="${COMMON_ARGS} --use-gpu-dataloader"
fi

if [ "$DISABLE_FLASH_ATTN" = true ]; then
  COMMON_ARGS="${COMMON_ARGS} --disable-flash-attn"
fi

if [ "$VERBOSE" = true ]; then
  COMMON_ARGS="${COMMON_ARGS} --verbose"
fi

# Create config file directories in case they don't exist
CONFIG_DIR=$(dirname "$CONFIG_FILE")
mkdir -p "$CONFIG_DIR"

echo "======================================================================"
echo "Starting comparison training for Infini-Attention vs. Baseline models"
echo "======================================================================"
echo "Configuration:"
echo "  Config file:          ${CONFIG_FILE}"
echo "  GPU device:           ${GPU_DEVICE}"
echo "  GPU data loader:      ${USE_GPU_DATALOADER}"
echo "  Flash Attention:      $(if [ "$DISABLE_FLASH_ATTN" = true ]; then echo "Disabled"; else echo "Enabled"; fi)"
echo "  TensorBoard logs dir: ${TENSORBOARD_DIR}"
echo "  Checkpoints dir:      ${CHECKPOINT_DIR}"
echo "======================================================================"
echo ""

# Run Infini-Attention model training
echo "======================================================================"
echo "Starting Infini-Attention model training"
echo "======================================================================"
echo "TensorBoard logs: ${INFINI_TB_DIR}"
echo "Checkpoints: ${INFINI_CP_DIR}"
echo ""

python train_infini_llama.py ${COMMON_ARGS} --tensorboard-dir "${INFINI_TB_DIR}"

echo ""
echo "Infini-Attention training completed"
echo ""

# Run Baseline model training
echo "======================================================================"
echo "Starting Baseline model training (Infini-Attention disabled)"
echo "======================================================================"
echo "TensorBoard logs: ${BASELINE_TB_DIR}"
echo "Checkpoints: ${BASELINE_CP_DIR}"
echo ""

python train_infini_llama.py ${COMMON_ARGS} --tensorboard-dir "${BASELINE_TB_DIR}" --disable-infini-attn

echo ""
echo "Baseline training completed"
echo ""

echo "======================================================================"
echo "Training comparison completed successfully!"
echo "======================================================================"
echo "You can compare results using TensorBoard:"
echo "tensorboard --logdir=${TENSORBOARD_DIR}"
echo ""
echo "Infini-Attention logs:  ${INFINI_TB_DIR}"
echo "Baseline model logs:    ${BASELINE_TB_DIR}"
echo "======================================================================"
