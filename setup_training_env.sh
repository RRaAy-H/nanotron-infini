#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/setup_training_env.sh

# Setup script for Infini-Llama training environment

# Base working directory
export NANOTRON_ROOT="/home/data/daal_insight/fiery/Infini-attention/nanotron-infini"

# Data directory
export DATA_DIR="/data1/dataset/HuggingFaceFW/processed"

echo "Setting up Infini-Llama training environment..."
echo "Base directory: $NANOTRON_ROOT"
echo "Data directory: $DATA_DIR"

# Make scripts executable
chmod +x "$NANOTRON_ROOT/train_infini_llama_gpu.sh"
chmod +x "$NANOTRON_ROOT/train_infini_llama_cpu.sh"
chmod +x "$NANOTRON_ROOT/verify_config.py"

# Create necessary directories
mkdir -p "$NANOTRON_ROOT/checkpoints"
mkdir -p "$NANOTRON_ROOT/tensorboard_logs"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
  echo "Warning: Data directory does not exist at $DATA_DIR"
  echo "Please ensure your data is available before starting training."
else
  echo "Data directory verified: $DATA_DIR"
fi

# Install required dependencies
echo "Installing required dependencies..."
pip install -e "$NANOTRON_ROOT" 2>/dev/null || echo "nanotron already installed"
pip install torch>=1.13.1 flash-attn>=2.5.0 datasets transformers huggingface_hub pyarrow pandas tensorboard torchvision tqdm pyyaml

echo "Environment setup complete!"
echo ""
echo "To start GPU training with TensorBoard:"
echo "  ./train_infini_llama_gpu.sh [GPU_ID] [TENSORBOARD_DIR]"
echo ""
echo "To start CPU training (for testing):"
echo "  ./train_infini_llama_cpu.sh"
echo ""
echo "To verify configurations:"
echo "  python verify_config.py --config-file custom_infini_config_gpu.yaml"
