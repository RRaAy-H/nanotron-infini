#!/bin/bash
# Run Infini-Llama training with TensorBoard without Flash Attention

# Set path variables
export NANOTRON_ROOT="/home/data/daal_insight/fiery/Infini-attention/nanotron-infini"
export TENSORBOARD_DIR="${NANOTRON_ROOT}/checkpoints/tensorboard_logs"
export CONFIG_FILE="custom_infini_config_gpu.yaml"

# Create directories if they don't exist
mkdir -p "${TENSORBOARD_DIR}"

# Ensure CUDA device settings are correct
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0  # Use the first GPU

# Set distributed training environment variables
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR="localhost"
export MASTER_PORT="29500"

# Set the environment variable to disable Flash Attention
export DISABLE_FLASH_ATTN=1

# Set up logging
LOG_DIR="${NANOTRON_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_FILE="${LOG_DIR}/training_no_flash_${TIMESTAMP}.log"

# Print environment information
echo "============================================================"
echo "Starting Infini-Llama training WITHOUT Flash Attention"
echo "============================================================"
echo "NANOTRON_ROOT: ${NANOTRON_ROOT}"
echo "TENSORBOARD_DIR: ${TENSORBOARD_DIR}"
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "DISABLE_FLASH_ATTN: ${DISABLE_FLASH_ATTN}"
echo "============================================================"

# Function to handle cleanup on script exit
cleanup() {
    echo "Stopping TensorBoard process (PID: ${TB_PID})..."
    kill ${TB_PID} 2>/dev/null
}

# Register the cleanup function
trap cleanup EXIT

# Start TensorBoard in background
tensorboard --logdir="${TENSORBOARD_DIR}" --port=6006 &
TB_PID=$!
echo "TensorBoard started with PID: ${TB_PID}"
echo "TensorBoard URL: http://localhost:6006"

# Sleep to ensure TensorBoard has started
sleep 2

# Run the training script
echo "Starting training..."
python "${NANOTRON_ROOT}/train_gpu_with_tensorboard.py" \
    --config-file "${NANOTRON_ROOT}/${CONFIG_FILE}" \
    --tensorboard-dir "${TENSORBOARD_DIR}" \
    --disable-flash-attn 2>&1 | tee "${LOG_FILE}"

# Check if training completed successfully
TRAINING_EXIT_CODE=${PIPESTATUS[0]}
if [ ${TRAINING_EXIT_CODE} -eq 0 ]; then
    echo "==========================================="
    echo "Training finished successfully."
    echo "TensorBoard logs are available in: ${TENSORBOARD_DIR}"
    echo "To view logs later: tensorboard --logdir=${TENSORBOARD_DIR}"
    echo "==========================================="
else
    echo "==========================================="
    echo "Training exited with code ${TRAINING_EXIT_CODE}"
    echo "Check the log file: ${LOG_FILE}"
    echo "==========================================="
fi
