#!/bin/bash
# Script to run Infini-Llama training with trainer

# Default values
CONFIG_FILE="custom_infini_config_gpu.yaml"
DATA_DIR="data"
TOKENIZER="meta-llama/Llama-2-7b-hf"
OUTPUT_DIR="checkpoints/$(date +%Y%m%d_%H%M%S)"
NUM_GPUS=1
DEBUG_MODE=false
DISABLE_FLASH_ATTN=false
INSTALL_DEPS=false
MICRO_BATCH_SIZE=""
NUM_WORKERS=4

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
    --debug)
      DEBUG_MODE=true
      shift
      ;;
    --disable-flash-attn)
      DISABLE_FLASH_ATTN=true
      shift
      ;;
    --install-deps)
      INSTALL_DEPS=true
      shift
      ;;
    --micro-batch-size)
      MICRO_BATCH_SIZE="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directory and logs directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# Define log file paths
LOG_FILE="$OUTPUT_DIR/logs/training_$(date +%Y%m%d_%H%M%S).log"
ERROR_LOG="$OUTPUT_DIR/logs/error_$(date +%Y%m%d_%H%M%S).log"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
  echo "Error: Data directory '$DATA_DIR' not found!" | tee -a "$ERROR_LOG"
  exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: Config file '$CONFIG_FILE' not found!" | tee -a "$ERROR_LOG"
  exit 1
fi

# Install dependencies if needed
if [ "$INSTALL_DEPS" = true ]; then
  echo "Installing required dependencies..." | tee -a "$LOG_FILE"
  pip install -q einops transformers datasets torch accelerate tensorboard | tee -a "$LOG_FILE"
  echo "Dependencies installed successfully." | tee -a "$LOG_FILE"
fi

echo "-------------------------------------" | tee -a "$LOG_FILE"
echo "Starting Infini-Llama training" | tee -a "$LOG_FILE"
echo "Config file: $CONFIG_FILE" | tee -a "$LOG_FILE"
echo "Data directory: $DATA_DIR" | tee -a "$LOG_FILE"
echo "Tokenizer: $TOKENIZER" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Number of GPUs: $NUM_GPUS" | tee -a "$LOG_FILE"
echo "Debug mode: $DEBUG_MODE" | tee -a "$LOG_FILE"
echo "Disable Flash Attention: $DISABLE_FLASH_ATTN" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "-------------------------------------" | tee -a "$LOG_FILE"

# Set environment variables to help prevent memory issues
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false

# Check GPU memory status
echo "GPU Memory Status:" | tee -a "$LOG_FILE"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv | tee -a "$LOG_FILE"

# Generate training command with appropriate options
TRAIN_CMD="train_with_trainer.py \
  --config-file \"$CONFIG_FILE\" \
  --data-dir \"$DATA_DIR\" \
  --tokenizer-path \"$TOKENIZER\" \
  --output-dir \"$OUTPUT_DIR\" \
  --auto-detect-flash-attn"

# Add optional arguments
if [ "$DISABLE_FLASH_ATTN" = true ]; then
  TRAIN_CMD+=" --disable-flash-attn"
fi

if [ -n "$MICRO_BATCH_SIZE" ]; then
  TRAIN_CMD+=" --micro-batch-size $MICRO_BATCH_SIZE"
fi

TRAIN_CMD+=" --num-workers $NUM_WORKERS"

echo "Generated command: $TRAIN_CMD" | tee -a "$LOG_FILE"

# Run in debug mode if requested
if [ "$DEBUG_MODE" = true ]; then
  echo "Running in debug mode with reduced memory usage..." | tee -a "$LOG_FILE"
  
  # Use smaller micro batch size and fewer workers to debug
  DEBUG_TRAIN_CMD=$(echo "$TRAIN_CMD" | sed 's/--num-workers [0-9]*/--num-workers 1/' | sed 's/--micro-batch-size [0-9]*/--micro-batch-size 1/')
  # Add debug flag to the Python script
  DEBUG_TRAIN_CMD+=" --debug"
  
  if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Running multi-GPU debug command..." | tee -a "$LOG_FILE"
    eval "CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 $DEBUG_TRAIN_CMD" 2>&1 | tee -a "$LOG_FILE"
  else
    echo "Running single-GPU debug command..." | tee -a "$LOG_FILE"
    eval "CUDA_VISIBLE_DEVICES=0 python -u $DEBUG_TRAIN_CMD" 2>&1 | tee -a "$LOG_FILE"
  fi
else
  # Regular run
  if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Running multi-GPU training..." | tee -a "$LOG_FILE"
    eval "torchrun --nproc_per_node=$NUM_GPUS $TRAIN_CMD" 2>&1 | tee -a "$LOG_FILE"
  else
    echo "Running single-GPU training..." | tee -a "$LOG_FILE"
    eval "python -u $TRAIN_CMD" 2>&1 | tee -a "$LOG_FILE"
  fi
fi

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "-------------------------------------" | tee -a "$LOG_FILE"
  echo "Training completed successfully!" | tee -a "$LOG_FILE"
  echo "Model checkpoints saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
  echo "Training logs saved to: $LOG_FILE" | tee -a "$LOG_FILE"
  echo "To monitor training progress: tensorboard --logdir $OUTPUT_DIR" | tee -a "$LOG_FILE"
  echo "-------------------------------------" | tee -a "$LOG_FILE"
else
  echo "-------------------------------------" | tee -a "$LOG_FILE" "$ERROR_LOG"
  echo "Training failed with exit code: $EXIT_CODE" | tee -a "$LOG_FILE" "$ERROR_LOG"
  echo "Error logs saved to: $ERROR_LOG" | tee -a "$LOG_FILE" "$ERROR_LOG"
  echo "Possible solutions to segmentation fault:" | tee -a "$LOG_FILE" "$ERROR_LOG"
  echo "1. Install dependencies: ./run_trainer.sh --install-deps ..." | tee -a "$LOG_FILE" "$ERROR_LOG"
  echo "2. Try debug mode: ./run_trainer.sh --debug ..." | tee -a "$LOG_FILE" "$ERROR_LOG"
  echo "3. Disable Flash Attention: ./run_trainer.sh --disable-flash-attn ..." | tee -a "$LOG_FILE" "$ERROR_LOG"
  echo "4. Reduce batch size: ./run_trainer.sh --micro-batch-size 1 ..." | tee -a "$LOG_FILE" "$ERROR_LOG"
  echo "5. Check GPU memory with 'nvidia-smi'" | tee -a "$LOG_FILE" "$ERROR_LOG"
  echo "-------------------------------------" | tee -a "$LOG_FILE" "$ERROR_LOG"
  
  # Show the last few lines of the log file to help diagnose the issue
  echo "Last 20 lines of log file:" | tee -a "$ERROR_LOG"
  tail -n 20 "$LOG_FILE" | tee -a "$ERROR_LOG"
fi

echo "Training script completed. All outputs logged to $LOG_FILE"
