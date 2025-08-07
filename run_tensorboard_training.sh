#!/bin/bash
# This script sets up the environment for training with TensorBoard and runs the training script

# Set the base directory
BASEDIR="/home/data/daal_insight/fiery/Infini-attention/nanotron-infini"
CONFIG_FILE="custom_infini_config_gpu.yaml"
TB_DIR="${BASEDIR}/tensorboard_logs"

# Set environment variables for distributed training
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR="localhost"
export MASTER_PORT="29500"

# Set CUDA environment variables 
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "Starting Infini-Llama training with TensorBoard..."
echo "Configuration: ${CONFIG_FILE}"
echo "TensorBoard logs directory: ${TB_DIR}"

# Start TensorBoard in the background
tensorboard --logdir=${TB_DIR} --host=0.0.0.0 --port=6006 &
TENSORBOARD_PID=$!
echo "TensorBoard started with PID: ${TENSORBOARD_PID}"
echo "TensorBoard available at http://localhost:6006"

# Trap to ensure TensorBoard is stopped if the script is interrupted
trap "echo 'Stopping TensorBoard process (PID: ${TENSORBOARD_PID})...'; kill ${TENSORBOARD_PID} 2>/dev/null || true" EXIT

# Run the training script
python ${BASEDIR}/train_gpu_with_tensorboard.py \
    --config-file ${BASEDIR}/${CONFIG_FILE} \
    --tensorboard-dir ${TB_DIR}

# Return the exit code from the training script
exit_code=$?

echo "Training exited with code ${exit_code}."
exit ${exit_code}
