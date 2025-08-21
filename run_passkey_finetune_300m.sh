set -e  # Exit on error

# Configuration
CHECKPOINT_PATH="${1:-./checkpoints/fineweb_4gpu_300m_infini/30000}"
CONFIG_FILE="passkey_finetune_300m_simple_config.yaml"

# Pre-flight checks
echo "=========================================================="
echo "PRE-FLIGHT CHECKS"
echo "=========================================================="

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file not found: $CONFIG_FILE"
    exit 1
fi
echo "✓ Configuration file found: $CONFIG_FILE"

# Check if training data exists
TRAINING_DATA_DIR="/data1/infini-attn/infini-llama/nanotron-infini/finetuning"
if [ ! -d "$TRAINING_DATA_DIR" ]; then
    echo "ERROR: Training data directory not found: $TRAINING_DATA_DIR"
    exit 1
fi
echo "✓ Training data found: $TRAINING_DATA_DIR"

# Check if run_train.py exists
if [ ! -f "run_train.py" ]; then
    echo "ERROR: Training script not found: run_train.py"
    echo "Make sure you're in the correct nanotron directory"
    exit 1
fi
echo "✓ Training script found: run_train.py"
echo ""

echo "=========================================================="
echo "OPTIMIZED PASSKEY FINETUNING FOR 300M INFINI-ATTENTION MODEL"
echo "=========================================================="
echo "Base checkpoint: $CHECKPOINT_PATH"
echo "Using pre-generated dataset: /data1/infini-attn/infini-llama/nanotron-infini/finetuning/"
echo "Sequence length: 4096 tokens (optimized for passkey tasks)"
echo "Segment length: 64 tokens (ULTRA-AGGRESSIVE memory triggering)"
echo "Balance factor LR: 0.05 (help deeper layers learn memory usage)"
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
    echo "=========================================================="
    echo "OPTIMIZED PASSKEY FINETUNING COMPLETED SUCCESSFULLY!"
    echo "=========================================================="
    echo ""
    echo "Checkpoints saved to: ./checkpoints/passkey_finetune_300m_optimized/"
    echo ""
    echo "Next steps:"
    echo "1. Monitor training progress:"
    echo "   MASTER_ADDR=localhost MASTER_PORT=29500 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \\"
    echo "   python monitor_passkey_finetuning.py --training-dir ./checkpoints/passkey_finetune_300m_optimized"
    echo ""
    echo "2. Test specific checkpoint:"
    echo "   MASTER_ADDR=localhost MASTER_PORT=29500 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \\"
    echo "   python scripts/passkey_memory_tracer.py --checkpoint ./checkpoints/passkey_finetune_300m_optimized/1000"
    echo ""
    
    # Show checkpoint directory contents
    echo "Checkpoint directory contents:"
    ls -la ./checkpoints/passkey_finetune_300m_optimized/ 2>/dev/null || echo "Check the checkpoint directory"
    
    echo ""
    echo "🎯 OPTIMIZATION SUMMARY:"
    echo "   ✓ segment_length: 64 (ULTRA-AGGRESSIVE memory triggering)"
    echo "   ✓ balance_factor_lr: 0.05 (deeper layer memory learning)"  
    echo "   ✓ learning_rate: 0.00005 (optimized for fine-tuning)"
    echo "   ✓ sequence_length: 4096 (focused on passkey contexts)"
    echo ""
    echo "Expected improvements (30400 steps total):"
    echo "   - Steps 1-1000: Balance factors start adjusting rapidly"
    echo "   - Steps 1000-5000: Memory usage improves dramatically in deeper layers"
    echo "   - Steps 5000-15000: Passkey retrieval accuracy improves significantly"  
    echo "   - Steps 15000-25000: Fine-tuning and optimization"
    echo "   - Steps 25000-30400: Target >90% passkey success rate"

else
    echo ""
    echo "ERROR: Finetuning failed. Check the logs above for details."
    exit 1
fi