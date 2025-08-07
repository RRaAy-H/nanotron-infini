#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/train_infini_llama_cpu.sh

cd /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini

echo "Creating a compatible environment..."
# Create conda environment if it doesn't exist
if ! conda env list | grep -q "infi-llama"; then
    echo "Creating conda environment infi-llama with Python 3.10..."
    conda create -y -n infi-llama python=3.10
    echo "Please run: conda activate infi-llama"
    echo "Then run this script again."
    exit 0
fi

echo "Installing dependencies..."
pip install -e .
pip install datasets transformers huggingface_hub pyarrow pandas

# Prepare the dataset
echo "Preparing dataset..."
python prepare_data.py

echo "Starting training..."
# Force CPU mode
export CUDA_VISIBLE_DEVICES=""

# Run the CPU training script
python run_train_cpu.py --config-file custom_infini_config_cpu.yaml
