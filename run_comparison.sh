#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/run_comparison.sh

# This script runs both Infini-Attention and baseline model training in parallel
# using separate GPUs to compare their performance

# Check if the required arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <config_file> <data_dir> [additional_args...]"
    echo "Example: $0 scripts/config/tiny_test_config.yaml tiny_test_data/preprocessed_20231215_120000"
    exit 1
fi

CONFIG_FILE=$1
DATA_DIR=$2
shift 2  # Remove the first two arguments, leaving any additional args

# Generate timestamps for unique TensorBoard directories
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
INFINI_TB_DIR="tensorboard_logs/infini_${TIMESTAMP}"
BASELINE_TB_DIR="tensorboard_logs/baseline_${TIMESTAMP}"

# Run Infini-Attention model on GPU 0 in the background
echo "Starting Infini-Attention model training on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python train_infini_llama.py \
    --config-file ${CONFIG_FILE} \
    --gpu-device cuda:0 \
    --tensorboard-dir ${INFINI_TB_DIR} \
    --use-gpu-dataloader \
    --data-dir ${DATA_DIR} \
    "$@" > infini_training_${TIMESTAMP}.log 2>&1 &

INFINI_PID=$!
echo "Infini-Attention model training started with PID ${INFINI_PID}"

# Wait a moment to stagger the startup
sleep 5

# Run baseline model on GPU 1
echo "Starting baseline model training on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python train_infini_llama.py \
    --config-file ${CONFIG_FILE} \
    --gpu-device cuda:0 \
    --tensorboard-dir ${BASELINE_TB_DIR} \
    --use-gpu-dataloader \
    --disable-infini-attn \
    --data-dir ${DATA_DIR} \
    "$@" > baseline_training_${TIMESTAMP}.log 2>&1 &

BASELINE_PID=$!
echo "Baseline model training started with PID ${BASELINE_PID}"

# Display information about monitoring
echo ""
echo "Training processes started in the background."
echo "You can monitor the logs using:"
echo "  tail -f infini_training_${TIMESTAMP}.log"
echo "  tail -f baseline_training_${TIMESTAMP}.log"
echo ""
echo "To monitor with TensorBoard, run:"
echo "  tensorboard --logdir=tensorboard_logs"
echo ""
echo "Process IDs:"
echo "  Infini-Attention model: ${INFINI_PID}"
echo "  Baseline model: ${BASELINE_PID}"
echo ""
echo "To check if the processes are still running:"
echo "  ps -p ${INFINI_PID},${BASELINE_PID}"

# Note: The script will exit but the training processes will continue running
# in the background. Use the process IDs to manage them if needed.
