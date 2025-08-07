#!/bin/bash
# Complete script to fix Flash Attention issues and run training
# Created: August 7, 2025

# Determine the base directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANOTRON_ROOT="$SCRIPT_DIR"
CONFIG_FILE="${1:-custom_infini_config_gpu.yaml}"
TENSORBOARD_DIR="${NANOTRON_ROOT}/tensorboard_logs/runs_$(date +%Y%m%d_%H%M%S)"

# Create necessary directories
mkdir -p "$TENSORBOARD_DIR"
mkdir -p "${NANOTRON_ROOT}/logs"

# Setup log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${NANOTRON_ROOT}/logs/training_${TIMESTAMP}.log"

echo "========================================================================"
echo "🚀 Starting Infini-Llama Training with Flash Attention Fix"
echo "========================================================================"
echo "📁 Base directory: $NANOTRON_ROOT"
echo "📄 Configuration: $CONFIG_FILE"
echo "📊 TensorBoard logs: $TENSORBOARD_DIR"
echo "📝 Log file: $LOG_FILE"
echo "========================================================================"

# Define cleanup function for graceful exit
cleanup() {
    echo "🧹 Cleaning up resources..."
    if [ -n "$TB_PID" ]; then
        echo "  ⏹️  Stopping TensorBoard (PID: $TB_PID)"
        kill $TB_PID 2>/dev/null || true
    fi
    echo "  ✅ Cleanup complete"
}
trap cleanup EXIT INT TERM

# 1. Set up environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0  # Use first GPU

# Ensure distributed environment variables are properly set
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR="localhost"
export MASTER_PORT="29500"

# Disable Flash Attention to avoid issues
export DISABLE_FLASH_ATTN=1

echo "⚙️  Environment variables configured:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  RANK: $RANK"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  LOCAL_RANK: $LOCAL_RANK"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  DISABLE_FLASH_ATTN: $DISABLE_FLASH_ATTN"

# 2. Apply the Flash Attention patch
echo "🔧 Applying Flash Attention patch..."
cd "$NANOTRON_ROOT"
if python patch_flash_attention.py; then
    echo "  ✅ Flash Attention patch applied successfully"
else
    echo "  ⚠️  Warning: Flash Attention patch failed, but we'll continue with training"
fi

# 3. Start TensorBoard
echo "📊 Starting TensorBoard..."
tensorboard --logdir="$TENSORBOARD_DIR" --port=6006 &
TB_PID=$!
echo "  ✅ TensorBoard started (PID: $TB_PID)"
echo "  🌐 TensorBoard URL: http://localhost:6006"

# Wait for TensorBoard to initialize
sleep 3

# 4. Run the training script with Flash Attention disabled
echo "🚀 Starting training..."
echo "  Command: python train_gpu_with_tensorboard.py --config-file $CONFIG_FILE --tensorboard-dir $TENSORBOARD_DIR --disable-flash-attn"

# Run the script and capture its output in the log file
python train_gpu_with_tensorboard.py \
    --config-file "$CONFIG_FILE" \
    --tensorboard-dir "$TENSORBOARD_DIR" \
    --disable-flash-attn 2>&1 | tee "$LOG_FILE"

# Get the exit code from the training script
TRAINING_EXIT_CODE=${PIPESTATUS[0]}

# Report the result
echo ""
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "🎉 Training completed successfully!"
else
    echo "❌ Training exited with error code: $TRAINING_EXIT_CODE"
fi

echo "========================================================================"
echo "📊 TensorBoard logs available at: $TENSORBOARD_DIR"
echo "📝 Training log saved to: $LOG_FILE"
echo "========================================================================"

exit $TRAINING_EXIT_CODE
