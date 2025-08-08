#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/run_tiny_test.sh
#
# This script runs a tiny test of preprocessing and training with Infini-Llama
# to validate the workflow with minimal data and compute resources.

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Directory setup
TEST_CONFIG="${SCRIPT_DIR}/config/tiny_test_config.yaml"
DATA_SOURCE="/data1/dataset/HuggingFaceFW/processed/tiny"
OUTPUT_DIR="${ROOT_DIR}/tiny_test_data"
TENSORBOARD_DIR="${ROOT_DIR}/tensorboard_logs/tiny_test"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TENSORBOARD_DIR"

echo "======================================================================"
echo "Starting tiny test workflow for Infini-Llama"
echo "======================================================================"
echo "This will run a minimal test with a tiny dataset to validate the workflow"
echo ""

# Step 1: Preprocessing
echo "======================================================================"
echo "STEP 1: Running preprocessing on tiny dataset"
echo "======================================================================"

echo "Using tiny dataset at: $DATA_SOURCE"

"${SCRIPT_DIR}/run_infini_llama.sh" \
  --config "$TEST_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --preprocess-only

# Check if preprocessing was successful
if [ $? -ne 0 ]; then
  echo "Error: Preprocessing failed. Exiting."
  exit 1
fi

echo ""
echo "Preprocessing completed successfully!"
echo ""

# Step 2: Training with Infini-Attention
echo "======================================================================"
echo "STEP 2: Running training with Infini-Attention"
echo "======================================================================"

"${SCRIPT_DIR}/run_infini_llama.sh" \
  --config "$TEST_CONFIG" \
  --data-dir "$OUTPUT_DIR" \
  --tensorboard-dir "${TENSORBOARD_DIR}/infini" \
  --train-only \
  --gpu 0

# Check if training was successful
if [ $? -ne 0 ]; then
  echo "Error: Training with Infini-Attention failed. Exiting."
  exit 1
fi

echo ""
echo "Infini-Attention training completed successfully!"
echo ""

# Step 3: Training without Infini-Attention (baseline)
echo "======================================================================"
echo "STEP 3: Running training without Infini-Attention (baseline)"
echo "======================================================================"

# We'll use the train_infini_llama.py script directly with the --disable-infini-attn flag
cd "$ROOT_DIR"
python train_infini_llama.py \
  --config-file "$TEST_CONFIG" \
  --gpu-device cuda:0 \
  --tensorboard-dir "${TENSORBOARD_DIR}/baseline" \
  --use-gpu-dataloader \
  --disable-infini-attn \
  --data-dir "$OUTPUT_DIR"

# Check if baseline training was successful
if [ $? -ne 0 ]; then
  echo "Error: Baseline training failed. Exiting."
  exit 1
fi

echo ""
echo "Baseline training completed successfully!"
echo ""

echo "======================================================================"
echo "Tiny test workflow completed successfully!"
echo "======================================================================"
echo "You can compare the results in TensorBoard:"
echo "tensorboard --logdir=${TENSORBOARD_DIR}"
echo ""
echo "Infini-Attention logs: ${TENSORBOARD_DIR}/infini"
echo "Baseline model logs:   ${TENSORBOARD_DIR}/baseline"
echo "======================================================================"
