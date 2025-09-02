set -e  # Exit on error

# Configuration
CHECKPOINT_PATH="${1:-./checkpoints/fineweb_4gpu_300m_infini/30000}"
CONFIG_FILE="passkey_finetune_300m_optimized_infini_config.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file not found: $CONFIG_FILE"
    exit 1
fi
echo "Configuration file found: $CONFIG_FILE"

# Check if training data exists
TRAINING_DATA_DIR="/data1/infini-attn/infini-llama/nanotron-infini/finetuning"
if [ ! -d "$TRAINING_DATA_DIR" ]; then
    echo "ERROR: Training data directory not found: $TRAINING_DATA_DIR"
    exit 1
fi
echo "Training data found: $TRAINING_DATA_DIR"

# Check if run_train.py exists
if [ ! -f "run_train.py" ]; then
    echo "ERROR: Training script not found: run_train.py"
    echo "Make sure you're in the correct nanotron directory"
    exit 1
fi
echo "Training script found: run_train.py"
echo ""

# Step 1: Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Please specify a valid checkpoint path as the first argument"
    echo "Usage: ./run_passkey_finetune_300m.sh [checkpoint_path]"
    exit 1
fi

# Step 2: Update config with checkpoint path
echo ""
echo "Step 1: Updating config with checkpoint path..."
echo "=========================================================="

# Update the resume_checkpoint_path in the config
echo "Updating resume_checkpoint_path to: $CHECKPOINT_PATH"
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s|resume_checkpoint_path: .*|resume_checkpoint_path: $CHECKPOINT_PATH|g" "$CONFIG_FILE"
else
    # Linux
    sed -i "s|resume_checkpoint_path: .*|resume_checkpoint_path: $CHECKPOINT_PATH|g" "$CONFIG_FILE"
fi

echo "Configuration updated with checkpoint: $CHECKPOINT_PATH"
echo ""

# Step 2: Run the finetuning
echo "Step 2: Starting finetuning..."
echo "=========================================================="

# Set environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_DISABLED=true

NUM_GPUS=4

# Create checkpoint directory (matches optimized config)
mkdir -p ./checkpoints/passkey_finetune_300m_optimized

# Run training
echo "Running finetuning with $NUM_GPUS GPUs..."

# Set Python path to use the current directory's src
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
echo "Using PYTHONPATH: $PYTHONPATH"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_endpoint=localhost:29401 \
    run_train.py \
    --config-file "$CONFIG_FILE"

# Check if training completed
if [ $? -eq 0 ]; then
    echo ""
    echo "Checkpoints saved to: ./checkpoints/passkey_finetune_300m_optimized/"
    echo ""
    
    # Show checkpoint directory contents
    echo "Checkpoint directory contents:"
    ls -la ./checkpoints/passkey_finetune_300m_optimized/ 2>/dev/null || echo "Check the checkpoint directory"
    
else
    echo ""
    echo "ERROR: Finetuning failed. Check the logs above for details."
    exit 1
fi