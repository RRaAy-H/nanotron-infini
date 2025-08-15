set -e  # Exit on error

# Configuration
CHECKPOINT_PATH="${1:-./checkpoints/fineweb_4gpu_300m_infini/30000}" 
NUM_EXAMPLES="${2:-2000}"  # Number of training examples to generate
SEED="${3:-42}"

echo "=========================================================="
echo "PASSKEY FINETUNING FOR 300M INFINI-ATTENTION MODEL"
echo "=========================================================="
echo "Base checkpoint: $CHECKPOINT_PATH"
echo "Training examples: $NUM_EXAMPLES"
echo "Sequence length: 10240 tokens (~10K)"
echo "Training steps: 500"
echo "Seed: $SEED"
echo "=========================================================="

# Step 1: Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Please specify a valid checkpoint path as the first argument"
    echo "Usage: ./run_passkey_finetune_300m.sh [checkpoint_path] [num_examples] [seed]"
    exit 1
fi

# Step 2: Generate the passkey finetuning dataset
echo ""
echo "Step 1: Generating passkey finetuning dataset..."
echo "=========================================================="

python generate_passkey_finetune_data.py \
    --tokenizer_path lvwerra/the-tokenizer-v1 \
    --num_examples $NUM_EXAMPLES \
    --target_length 10240 \
    --save_path ./passkey_finetune_data_10k \
    --seed $SEED

if [ ! -f "./passkey_finetune_data_10k.parquet" ]; then
    echo "ERROR: Dataset generation failed. Parquet file not found."
    exit 1
fi

echo ""
echo "Dataset generated successfully!"
echo ""

# Step 3: Update config with correct checkpoint path
echo "Step 2: Updating configuration with checkpoint path..."
echo "=========================================================="

# Create a temporary config with the correct checkpoint path
cp passkey_finetune_300m_config.yaml passkey_finetune_300m_config_temp.yaml

# Update the resume_checkpoint_path in the config
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s|resume_checkpoint_path: .*|resume_checkpoint_path: $CHECKPOINT_PATH|g" passkey_finetune_300m_config_temp.yaml
else
    # Linux
    sed -i "s|resume_checkpoint_path: .*|resume_checkpoint_path: $CHECKPOINT_PATH|g" passkey_finetune_300m_config_temp.yaml
fi

echo "Configuration updated with checkpoint: $CHECKPOINT_PATH"
echo ""

# Step 4: Run the finetuning
echo "Step 3: Starting finetuning..."
echo "=========================================================="

# Set environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=16

# Determine number of GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "Detected $NUM_GPUS GPUs"
    
    # Adjust parallelism if needed
    if [ $NUM_GPUS -lt 4 ]; then
        echo "WARNING: Config expects 4 GPUs but only $NUM_GPUS detected"
        echo "You may need to adjust the 'dp' parameter in the config"
        
        # Update dp in config based on available GPUs
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s|dp: 4|dp: $NUM_GPUS|g" passkey_finetune_300m_config_temp.yaml
        else
            sed -i "s|dp: 4|dp: $NUM_GPUS|g" passkey_finetune_300m_config_temp.yaml
        fi
        echo "Updated config to use $NUM_GPUS GPUs"
    fi
else
    NUM_GPUS=1
    echo "nvidia-smi not found, assuming 1 GPU"
fi

# Create checkpoint directory
mkdir -p ./checkpoints/passkey_finetune_300m

# Run training
echo "Running finetuning with $NUM_GPUS GPUs..."
echo "Command: torchrun --nproc_per_node=$NUM_GPUS run_train.py --config-file passkey_finetune_300m_config_temp.yaml"
echo ""

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --rdzv_endpoint=localhost:29401 \
    run_train.py \
    --config-file passkey_finetune_300m_config_temp.yaml

# Check if training completed
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================================="
    echo "FINETUNING COMPLETED SUCCESSFULLY!"
    echo "=========================================================="
    echo ""
    echo "Checkpoints saved to: ./checkpoints/passkey_finetune_300m/"
    echo ""
    echo "Final checkpoint should be at step 500:"
    ls -la ./checkpoints/passkey_finetune_300m/ 2>/dev/null || echo "Check the checkpoint directory"
    
    # Clean up temp config
    rm passkey_finetune_300m_config_temp.yaml
    
    echo ""
    echo "Next steps:"
    echo "1. Evaluate the model using the passkey eval script:"
    echo "   ./examples/infinite-context-length/scripts/run_passkey_eval_300m.sh ./checkpoints/passkey_finetune_300m/500 10240"
    echo ""

else
    echo ""
    echo "ERROR: Finetuning failed. Check the logs above for details."
    exit 1
fi