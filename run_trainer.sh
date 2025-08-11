#!/bin/bash
# Script to run Infini-Llama training with trainer

# Default values
CONFIG_FILE="custom_infini_config_gpu.yaml"
DATA_DIR="data"
TOKENIZER="meta-llama/Llama-2-7b-hf"
OUTPUT_DIR="checkpoints/$(date +%Y%m%d_%H%M%S)"
NUM_GPUS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config-file)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --tokenizer)
      TOKENIZER="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --num-gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
  echo "Error: Data directory '$DATA_DIR' not found!"
  exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: Config file '$CONFIG_FILE' not found!"
  exit 1
fi

echo "-------------------------------------"
echo "Starting Infini-Llama training"
echo "Config file: $CONFIG_FILE"
echo "Data directory: $DATA_DIR"
echo "Tokenizer: $TOKENIZER"
echo "Output directory: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "-------------------------------------"

# Run the training script with torchrun for multi-GPU support
if [ "$NUM_GPUS" -gt 1 ]; then
  # Multi-GPU training
  torchrun --nproc_per_node="$NUM_GPUS" train_with_trainer.py \
    --config-file "$CONFIG_FILE" \
    --data-dir "$DATA_DIR" \
    --tokenizer-path "$TOKENIZER" \
    --output-dir "$OUTPUT_DIR" \
    --auto-detect-flash-attn \
    --num-workers 4
else
  # Single GPU or CPU training
  python train_with_trainer.py \
    --config-file "$CONFIG_FILE" \
    --data-dir "$DATA_DIR" \
    --tokenizer-path "$TOKENIZER" \
    --output-dir "$OUTPUT_DIR" \
    --auto-detect-flash-attn \
    --num-workers 4
fi

# Check exit status
if [ $? -eq 0 ]; then
  echo "-------------------------------------"
  echo "Training completed successfully!"
  echo "Model checkpoints saved to: $OUTPUT_DIR"
  echo "To monitor training progress: tensorboard --logdir $OUTPUT_DIR"
  echo "-------------------------------------"
else
  echo "-------------------------------------"
  echo "Training failed! Check logs for errors."
  echo "-------------------------------------"
fi
