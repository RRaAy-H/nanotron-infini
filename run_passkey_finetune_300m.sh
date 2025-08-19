set -e  # Exit on error

# Configuration
CHECKPOINT_PATH="${1:-./checkpoints/fineweb_4gpu_300m_infini/30000}"

echo "=========================================================="
echo "PASSKEY FINETUNING FOR 300M INFINI-ATTENTION MODEL"
echo "=========================================================="
echo "Base checkpoint: $CHECKPOINT_PATH"
echo "Using pre-generated dataset: /data1/infini-attn/infini-llama/nanotron-infini/finetuning/"
echo "Sequence length: 10240 tokens (~10K)"
echo "=========================================================="

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
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s|resume_checkpoint_path: .*|resume_checkpoint_path: $CHECKPOINT_PATH|g" passkey_finetune_300m_simple_config.yaml
else
    # Linux
    sed -i "s|resume_checkpoint_path: .*|resume_checkpoint_path: $CHECKPOINT_PATH|g" passkey_finetune_300m_simple_config.yaml
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

NUM_GPUS=4

# Create checkpoint directory
mkdir -p ./checkpoints/passkey_finetune_300m

# Run training
echo "Running finetuning with $NUM_GPUS GPUs..."

# Set Python path to use the current directory's src
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
echo "Using PYTHONPATH: $PYTHONPATH"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_endpoint=localhost:29401 \
    run_train.py \
    --config-file passkey_finetune_300m_simple_config.yaml

# Check if training completed
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================================="
    echo "FINETUNING COMPLETED SUCCESSFULLY!"
    echo "=========================================================="
    echo ""
    echo "Checkpoints saved to: ./checkpoints/passkey_finetune_300m_simple/"
    echo ""
    ls -la ./checkpoints/passkey_finetune_300m_simple/ 2>/dev/null || echo "Check the checkpoint directory"

else
    echo ""
    echo "ERROR: Finetuning failed. Check the logs above for details."
    exit 1
fi