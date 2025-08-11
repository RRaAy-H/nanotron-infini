#!/usr/bin/env bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/train_tiny_infini.sh

# Script to train the Infini-Llama model on the tiny dataset

# Set up environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null
then
    echo "CUDA is available. Using GPU for training."
    GPU_FLAG=""
else
    echo "CUDA is not available. Using CPU for training."
    GPU_FLAG="--cpu-only"
fi

# Create tensorboard directory if it doesn't exist
mkdir -p tensorboard_logs

# Run the training script with the custom configuration
python train_infini_llama.py \
    --config-file custom_tiny_infini_config.yaml \
    --data-dir /data1/dataset/HuggingFaceFW/processes/tiny \
    --tensorboard-dir tensorboard_logs \
    --use-gpu-dataloader \
    $GPU_FLAG

# Check if the training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with error code $?."
fi
