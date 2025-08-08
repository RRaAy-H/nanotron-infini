#!/bin/bash
# run_both_models.sh
# Script to run both Infini-Attention and baseline models in parallel on different GPUs

set -e  # Exit on error

# Default values
CONFIG_FILE="scripts/config/tiny_test_config.yaml"
DATA_DIR="tiny_test_data/preprocessed_*"  # Will be expanded by glob
INFINI_GPU="0"
BASELINE_GPU="1"

# Create timestamp for unique tensorboard logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --config|-c)
      CONFIG_FILE="$2"
      shift
      shift
      ;;
    --data-dir|-d)
      DATA_DIR="$2"
      shift
      shift
      ;;
    --infini-gpu)
      INFINI_GPU="$2"
      shift
      shift
      ;;
    --baseline-gpu)
      BASELINE_GPU="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "======================================================================"
echo "Running comparison between Infini-Attention and baseline models"
echo "======================================================================"
echo "Configuration file: $CONFIG_FILE"
echo "Data directory: $DATA_DIR"
echo "Infini-Attention GPU: $INFINI_GPU"
echo "Baseline GPU: $BASELINE_GPU"
echo ""

# Install missing dependencies if needed
echo "Installing required dependencies..."
pip install packaging dacite pyyaml numpy tqdm

echo "======================================================================"
echo "Starting Infini-Attention model training (GPU $INFINI_GPU)"
echo "======================================================================"
echo "To monitor progress, check tensorboard logs in tensorboard_logs/infini_$TIMESTAMP"
echo ""

# Run Infini-Attention model on GPU 0 in a separate terminal
CUDA_VISIBLE_DEVICES=$INFINI_GPU python train_infini_llama.py \
  --config-file $CONFIG_FILE \
  --gpu-device cuda:0 \
  --tensorboard-dir tensorboard_logs/infini_$TIMESTAMP \
  --use-gpu-dataloader \
  --data-dir $DATA_DIR &

INFINI_PID=$!

echo "======================================================================"
echo "Starting baseline model training (GPU $BASELINE_GPU)"
echo "======================================================================"
echo "To monitor progress, check tensorboard logs in tensorboard_logs/baseline_$TIMESTAMP"
echo ""

# Run baseline model on GPU 1 in a separate terminal
CUDA_VISIBLE_DEVICES=$BASELINE_GPU python train_infini_llama.py \
  --config-file $CONFIG_FILE \
  --gpu-device cuda:0 \
  --tensorboard-dir tensorboard_logs/baseline_$TIMESTAMP \
  --use-gpu-dataloader \
  --disable-infini-attn \
  --data-dir $DATA_DIR &

BASELINE_PID=$!

echo "======================================================================"
echo "Both training processes are running in the background"
echo "Infini-Attention PID: $INFINI_PID"
echo "Baseline PID: $BASELINE_PID"
echo "======================================================================"
echo ""
echo "To view training progress:"
echo "tensorboard --logdir tensorboard_logs"
echo ""
echo "To kill training processes:"
echo "kill $INFINI_PID $BASELINE_PID"
echo ""
echo "To wait for training to complete, press CTRL+C when done"

# Wait for both processes
wait $INFINI_PID $BASELINE_PID
