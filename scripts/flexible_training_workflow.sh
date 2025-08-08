#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/flexible_training_workflow.sh
# 
# Flexible training workflow script for Infini-Llama models
# This script allows flexible configuration of:
# - Data source (raw data path)
# - Output directory for preprocessed data
# - Config file to use
# - Whether to enable or disable Infini-Attention
# - Which GPU(s) to use
#
# Note: Includes fixes for tensor memory layout issues and training logs path
#
# Usage examples:
# ./flexible_training_workflow.sh --raw-data /data1/dataset/HuggingFaceFW/processed/tiny --config-file scripts/config/tiny_test_config.yaml
# ./flexible_training_workflow.sh --preprocessed-data tiny_test_data/preprocessed_20240808_123456 --config-file scripts/config/tiny_test_config.yaml --disable-infini-attn
# ./flexible_training_workflow.sh --raw-data /data1/dataset/HuggingFaceFW/processed/pile --config-file custom_infini_config_gpu.yaml --gpu 1

# Set paths - adjust the project root to point to the repository root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default values
RAW_DATA=""
PREPROCESSED_DATA=""
CONFIG_FILE="scripts/config/tiny_test_config.yaml"
OUTPUT_DIR="preprocessed_data"
DISABLE_INFINI_ATTN=false
GPU_ID=0
TENSORBOARD_DIR="tensorboard_logs/train_$(date +"%Y%m%d_%H%M%S")"
USE_GPU_DATALOADER=true
FORCE_PREPROCESS=false
DISABLE_FUSED_ADAM=true  # Set to true to avoid fused Adam optimizer errors
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --raw-data)
            RAW_DATA="$2"
            shift 2
            ;;
        --preprocessed-data)
            PREPROCESSED_DATA="$2"
            shift 2
            ;;
        --config-file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --disable-infini-attn)
            DISABLE_INFINI_ATTN=true
            shift
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --run-both-models)
            RUN_BOTH_MODELS=true
            shift
            ;;
        --enable-fused-adam)
            DISABLE_FUSED_ADAM=false
            shift
            ;;
        --tensorboard-dir)
            TENSORBOARD_DIR="$2"
            shift 2
            ;;
        --no-gpu-dataloader)
            USE_GPU_DATALOADER=false
            shift
            ;;
        --force-preprocess)
            FORCE_PREPROCESS=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Flexible training workflow for Infini-Llama models"
            echo ""
            echo "Usage: ./flexible_training_workflow.sh [options]"
            echo ""
            echo "Options:"
            echo "  --raw-data PATH           Path to raw data directory (for preprocessing)"
            echo "  --preprocessed-data PATH  Path to already preprocessed data directory"
            echo "  --config-file PATH        Path to configuration YAML file (default: scripts/config/tiny_test_config.yaml)"
            echo "  --output-dir PATH         Directory to save preprocessed data (default: preprocessed_data)"
            echo "  --disable-infini-attn     Disable Infini-Attention (run baseline model)"
            echo "  --gpu ID                  GPU ID to use (default: 0)"
            echo "  --run-both-models         Run both Infini-Attention and baseline models (requires 2+ GPUs)"
            echo "  --enable-fused-adam       Enable fused Adam optimizer (default: disabled to avoid errors)"
            echo "  --tensorboard-dir PATH    Directory for TensorBoard logs"
            echo "  --no-gpu-dataloader       Disable GPU-accelerated dataloader"
            echo "  --force-preprocess        Force preprocessing even if data exists"
            echo "  --verbose                 Enable verbose logging"
            echo "  --help                    Show this help message and exit"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Set Python path to include project root and src directory
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PYTHONPATH"

# Define the wrapper script path early so we can do checks
WRAP_SCRIPT="$PROJECT_ROOT/scripts/wrapper_script.py"

# Check if the wrapper script exists
if [[ ! -f "$WRAP_SCRIPT" ]]; then
    echo "Warning: Wrapper script not found at $WRAP_SCRIPT"
    echo "Checking for script creation..."
    
    # Check if we can create the wrapper script
    if [[ -d "$PROJECT_ROOT/scripts" ]]; then
        echo "Creating wrapper script at: $WRAP_SCRIPT"
        # Create a simple wrapper script that imports patches and runs the training script
        cat > "$WRAP_SCRIPT" << EOF
#!/usr/bin/env python
# filepath: $WRAP_SCRIPT
import os
import sys

# Add project paths
project_root = "$PROJECT_ROOT"
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "scripts"))

# Try to import patches
try:
    import preimport
    print("Adam optimizer patches applied successfully")
except ImportError as e:
    print(f"Warning: Failed to import pre-import patches: {e}")

# Run the training script
script_path = os.path.join(project_root, "scripts", "run_direct_training.py")
if not os.path.exists(script_path):
    print(f"Error: Training script not found at: {script_path}")
    sys.exit(1)

# Execute the training script with arguments
os.execvp("python", ["python", script_path] + sys.argv[1:])
EOF
        # Make it executable
        chmod +x "$WRAP_SCRIPT"
        echo "Wrapper script created successfully"
    else
        echo "Cannot create wrapper script: scripts directory not found"
        echo "Will fall back to direct execution"
    fi
else
    echo "Found wrapper script at: $WRAP_SCRIPT"
    chmod +x "$WRAP_SCRIPT"  # Ensure it's executable
fi

# Check if we have either raw data or preprocessed data
if [[ -z "$RAW_DATA" ]] && [[ -z "$PREPROCESSED_DATA" ]]; then
    echo "Error: Either --raw-data or --preprocessed-data must be specified."
    echo "Use --help for usage information."
    exit 1
fi

# Validate config file
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Set model type for logs and output naming
MODEL_TYPE="infini"
if [[ "$DISABLE_INFINI_ATTN" = true ]]; then
    MODEL_TYPE="baseline"
    TENSORBOARD_DIR="tensorboard_logs/baseline_$(date +"%Y%m%d_%H%M%S")"
else
    TENSORBOARD_DIR="tensorboard_logs/infini_$(date +"%Y%m%d_%H%M%S")"
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TENSORBOARD_DIR"

# Print configuration
echo "-------------------------------------"
echo "Infini-Llama Training Configuration:"
echo "-------------------------------------"
echo "Model type: $MODEL_TYPE"
echo "Config file: $CONFIG_FILE"
echo "GPU ID: $GPU_ID"
echo "TensorBoard dir: $TENSORBOARD_DIR"
echo "Using GPU dataloader: $USE_GPU_DATALOADER"
echo "Verbose logging: $VERBOSE"
echo "-------------------------------------"

# Preprocessing step
if [[ -n "$RAW_DATA" ]]; then
    # Check if we need to preprocess
    NEED_PREPROCESS=true
    
    # If we have a specified preprocessed data path and force is not enabled,
    # check if it exists and skip preprocessing
    if [[ -n "$PREPROCESSED_DATA" ]] && [[ "$FORCE_PREPROCESS" = false ]]; then
        if [[ -d "$PREPROCESSED_DATA" ]]; then
            echo "Using existing preprocessed data: $PREPROCESSED_DATA"
            NEED_PREPROCESS=false
        else
            echo "Specified preprocessed data directory doesn't exist: $PREPROCESSED_DATA"
            echo "Will preprocess from raw data."
        fi
    fi
    
    # Preprocess if needed
    if [[ "$NEED_PREPROCESS" = true ]]; then
        echo "Preprocessing data from: $RAW_DATA"
        # Create timestamp for preprocessed data directory
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        PREPROCESS_OUTPUT_DIR="$OUTPUT_DIR/preprocessed_$TIMESTAMP"
        
        # Run preprocessing script
        if [[ -f "$PROJECT_ROOT/scripts/preprocessing/preprocess_data_fixed.py" ]]; then
            PREPROCESS_CMD="python \"$PROJECT_ROOT/scripts/preprocessing/preprocess_data_fixed.py\" \
                --config-file \"$CONFIG_FILE\" \
                --output-dir \"$OUTPUT_DIR\" \
                --gpu-id \"$GPU_ID\""
                
            # Only add verbose flag if verbose is true
            if [[ "$VERBOSE" = true ]]; then
                PREPROCESS_CMD="$PREPROCESS_CMD --verbose"
            fi
            
            # Execute the command
            eval $PREPROCESS_CMD && PREPROCESSED_DATA=$(find "$OUTPUT_DIR" -name "preprocessed_*" -type d | sort | tail -n 1)
        else
            echo "Error: Preprocessing script not found."
            exit 1
        fi
        
        if [[ -z "$PREPROCESSED_DATA" ]]; then
            echo "Error: Preprocessing failed or no output directory was created."
            exit 1
        fi
        
        echo "Data preprocessing complete: $PREPROCESSED_DATA"
    fi
else
    # Ensure preprocessed data exists
    if [[ ! -d "$PREPROCESSED_DATA" ]]; then
        echo "Error: Specified preprocessed data directory doesn't exist: $PREPROCESSED_DATA"
        exit 1
    fi
fi

# Build training command using our wrapper script
# Make sure to provide the full path to the wrapper script
if [[ -f "$WRAP_SCRIPT" ]]; then
    TRAIN_CMD="python \"$WRAP_SCRIPT\" \
        --config-file \"$CONFIG_FILE\" \
        --data-dir \"$PREPROCESSED_DATA\" \
        --gpu-id \"$GPU_ID\" \
        --tensorboard-dir \"$TENSORBOARD_DIR\""
else
    # Fallback to direct execution if wrapper script doesn't exist
    echo "WARNING: Wrapper script not found, falling back to direct execution"
    TRAIN_CMD="python \"$PROJECT_ROOT/scripts/run_direct_training.py\" \
        --config-file \"$CONFIG_FILE\" \
        --data-dir \"$PREPROCESSED_DATA\" \
        --gpu-id \"$GPU_ID\" \
        --tensorboard-dir \"$TENSORBOARD_DIR\""
fi

# Add optional flags
if [[ "$DISABLE_INFINI_ATTN" = true ]]; then
    TRAIN_CMD="$TRAIN_CMD --disable-infini-attn"
fi

if [[ "$USE_GPU_DATALOADER" = true ]]; then
    TRAIN_CMD="$TRAIN_CMD --use-gpu-dataloader"
fi

if [[ "$VERBOSE" = true ]]; then
    TRAIN_CMD="$TRAIN_CMD --verbose"
fi

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Set up environment variables for proper path resolution
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PYTHONPATH"

# Set up training logs directory
export TRAINING_LOGS_DIR="$PROJECT_ROOT/training_logs"
mkdir -p "$TRAINING_LOGS_DIR"
# Ensure write permissions
chmod -R 755 "$TRAINING_LOGS_DIR"
echo "Training logs will be saved to: $TRAINING_LOGS_DIR"

# Set up training logs directory
export TRAINING_LOGS_DIR="$PROJECT_ROOT/training_logs"
mkdir -p "$TRAINING_LOGS_DIR"
# Ensure write permissions
chmod -R 755 "$TRAINING_LOGS_DIR"
echo "Training logs will be saved to: $TRAINING_LOGS_DIR"

# Temporarily modify the config if needed to disable fused Adam and ensure weight_decay is set
CONFIG_TEMP="${CONFIG_FILE%.yaml}_temp.yaml"

if [[ "$DISABLE_FUSED_ADAM" = true ]]; then
    cat "$CONFIG_FILE" | sed 's/torch_adam_is_fused: true/torch_adam_is_fused: false/g' > "$CONFIG_TEMP"
    echo "Created temporary config with fused Adam disabled to avoid optimizer errors"
else
    # Just copy the file without changes to continue the process
    cp "$CONFIG_FILE" "$CONFIG_TEMP"
fi

# Check if weight_decay is missing or None in the config and add it
# See docs/WEIGHT_DECAY_FIX.md for details on this fix
if ! grep -q "weight_decay:" "$CONFIG_TEMP" || grep -q "weight_decay: *null" "$CONFIG_TEMP"; then
    # Add or replace weight_decay with default value 0.01
    sed -i.bak '/optimizer:/,/zero_stage:/ s/\(weight_decay: *\)null/\10.01/' "$CONFIG_TEMP"
    if ! grep -q "weight_decay:" "$CONFIG_TEMP"; then
        # If weight_decay is completely missing, add it before zero_stage
        sed -i.bak '/optimizer:/,/zero_stage:/ s/\(zero_stage:.*\)/weight_decay: 0.01\n  \1/' "$CONFIG_TEMP"
    fi
    echo "Added default weight_decay: 0.01 to config to avoid NoneType errors in optimizer"
fi

CONFIG_FILE="$CONFIG_TEMP"

# Configure Infini-Attention constants
python -c "
from dataclasses import dataclass, field
import sys
sys.path.append('$PROJECT_ROOT/src')
from nanotron import constants

@dataclass
class InfiniAttentionConfig:
    segment_length: int = 64
    turn_on_memory: bool = True
    balance_init_type: str = 'zeros'
    balance_act_type: str = 'orig_sigmoid'
    balance_factor_lr: float = 0.001
    logging: bool = False
    logging_interval: int = 100
    log_grad: bool = False
    log_segment_acts: bool = False

@dataclass
class Config:
    infini_attention: InfiniAttentionConfig = field(default_factory=InfiniAttentionConfig)

# Set up the configuration
constants.CONFIG = Config()
print('Infini attention constants configured successfully!')
"

# Fix Flash Attention warnings if the fix script exists
if [[ -f "$PROJECT_ROOT/scripts/fix_flash_attention_warnings.py" ]]; then
    echo "Attempting to fix Flash Attention warnings..."
    python "$PROJECT_ROOT/scripts/fix_flash_attention_warnings.py" || true
fi

# Ensure the pre-import script with Adam optimizer patches is used in training
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PYTHONPATH"

# Apply the Adam optimizer patch directly before training
echo "Applying Adam optimizer patch to fix weight_decay=None issue"
if [[ -f "$PROJECT_ROOT/scripts/apply_adam_patch.sh" ]]; then
    bash "$PROJECT_ROOT/scripts/apply_adam_patch.sh"
    echo "Adam patch script executed"
else
    echo "Creating temporary Adam patch"
    python -c "
import torch.optim.adam
original_adam = torch.optim.adam.adam
def patched_adam(*args, **kwargs):
    if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
        kwargs['weight_decay'] = 0.0
    if len(args) >= 4 and args[3] is None:
        args = list(args)
        args[3] = 0.0
        args = tuple(args)
    return original_adam(*args, **kwargs)
torch.optim.adam.adam = patched_adam
print('Applied temporary Adam patch')
"
fi

# Use the permanent wrapper script instead of creating a temporary one
WRAP_SCRIPT="$PROJECT_ROOT/scripts/wrapper_script.py"

# Check if the wrapper script exists
if [[ ! -f "$WRAP_SCRIPT" ]]; then
    echo "Error: Wrapper script not found at $WRAP_SCRIPT"
    echo "Please make sure the wrapper_script.py file exists in the scripts directory"
    exit 1
fi

# Ensure the wrapper script is executable
chmod +x "$WRAP_SCRIPT"
echo "Using wrapper script with Adam optimizer patches: $WRAP_SCRIPT"

# Debug output to verify the wrapper script exists
echo "Wrapper script path: $WRAP_SCRIPT"
if [[ -f "$WRAP_SCRIPT" ]]; then
    echo "Confirmed wrapper script exists"
else
    echo "WARNING: Wrapper script not found at this path, training will fail!"
fi

# Suppress Flash Attention warnings as a fallback
export PYTHONWARNINGS="ignore::FutureWarning"
echo "Suppressed Flash Attention FutureWarnings"

# Check if we should run both models
if [[ "$RUN_BOTH_MODELS" = true ]]; then
    # Check if we have at least 2 GPUs
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    if [[ $GPU_COUNT -lt 2 ]]; then
        echo "Warning: Running both models requires at least 2 GPUs, but only $GPU_COUNT detected."
        echo "Will run models sequentially on GPU $GPU_ID instead."
        RUN_BOTH_MODELS=false
    else
        echo "-------------------------------------"
        echo "Running both models in parallel:"
        echo "Infini-Attention model on GPU 0"
        echo "Baseline model on GPU 1"
        echo "-------------------------------------"
        
        # Create separate tensorboard dirs
        INFINI_TB_DIR="tensorboard_logs/infini_$(date +"%Y%m%d_%H%M%S")"
        BASELINE_TB_DIR="tensorboard_logs/baseline_$(date +"%Y%m%d_%H%M%S")"
        mkdir -p "$INFINI_TB_DIR"
        mkdir -p "$BASELINE_TB_DIR"
        
        # Create separate log directories
        INFINI_LOG_DIR="$TRAINING_LOGS_DIR/infini_$(date +"%Y%m%d_%H%M%S")"
        BASELINE_LOG_DIR="$TRAINING_LOGS_DIR/baseline_$(date +"%Y%m%d_%H%M%S")"
        mkdir -p "$INFINI_LOG_DIR"
        mkdir -p "$BASELINE_LOG_DIR"
        
        # Build commands for both models using our wrapper script
        if [[ -f "$WRAP_SCRIPT" ]]; then
            INFINI_CMD="CUDA_VISIBLE_DEVICES=0 TRAINING_LOGS_DIR=$INFINI_LOG_DIR python \"$WRAP_SCRIPT\" \
                --config-file \"$CONFIG_FILE\" \
                --data-dir \"$PREPROCESSED_DATA\" \
                --gpu-id 0 \
                --tensorboard-dir \"$INFINI_TB_DIR\""
            
            BASELINE_CMD="CUDA_VISIBLE_DEVICES=1 TRAINING_LOGS_DIR=$BASELINE_LOG_DIR python \"$WRAP_SCRIPT\" \
                --config-file \"$CONFIG_FILE\" \
                --data-dir \"$PREPROCESSED_DATA\" \
                --gpu-id 0 \
                --disable-infini-attn \
                --tensorboard-dir \"$BASELINE_TB_DIR\""
        else
            # Fallback to direct execution if wrapper script doesn't exist
            echo "WARNING: Wrapper script not found, falling back to direct execution for parallel training"
            INFINI_CMD="CUDA_VISIBLE_DEVICES=0 TRAINING_LOGS_DIR=$INFINI_LOG_DIR python \"$PROJECT_ROOT/scripts/run_direct_training.py\" \
                --config-file \"$CONFIG_FILE\" \
                --data-dir \"$PREPROCESSED_DATA\" \
                --gpu-id 0 \
                --tensorboard-dir \"$INFINI_TB_DIR\""
            
            BASELINE_CMD="CUDA_VISIBLE_DEVICES=1 TRAINING_LOGS_DIR=$BASELINE_LOG_DIR python \"$PROJECT_ROOT/scripts/run_direct_training.py\" \
                --config-file \"$CONFIG_FILE\" \
                --data-dir \"$PREPROCESSED_DATA\" \
                --gpu-id 0 \
                --disable-infini-attn \
                --tensorboard-dir \"$BASELINE_TB_DIR\""
        fi
        
        # Add optional flags to both commands
        if [[ "$USE_GPU_DATALOADER" = true ]]; then
            INFINI_CMD="$INFINI_CMD --use-gpu-dataloader"
            BASELINE_CMD="$BASELINE_CMD --use-gpu-dataloader"
        fi
        
        if [[ "$VERBOSE" = true ]]; then
            INFINI_CMD="$INFINI_CMD --verbose"
            BASELINE_CMD="$BASELINE_CMD --verbose"
        fi
        
        # Run both commands in parallel
        echo "Starting Infini-Attention model training..."
        eval "$INFINI_CMD" > infini_training.log 2>&1 &
        INFINI_PID=$!
        echo "Infini-Attention training started with PID: $INFINI_PID"
        
        echo "Starting Baseline model training..."
        eval "$BASELINE_CMD" > baseline_training.log 2>&1 &
        BASELINE_PID=$!
        echo "Baseline training started with PID: $BASELINE_PID"
        
        # Wait for both processes to complete
        echo "Waiting for both training processes to complete..."
        wait $INFINI_PID
        INFINI_STATUS=$?
        wait $BASELINE_PID
        BASELINE_STATUS=$?
        
        # Print completion message
        echo "-------------------------------------"
        echo "Parallel training completed!"
        echo "Infini-Attention model status: $INFINI_STATUS (0 = success)"
        echo "Baseline model status: $BASELINE_STATUS (0 = success)"
        echo "Infini-Attention TensorBoard logs: $INFINI_TB_DIR"
        echo "Baseline TensorBoard logs: $BASELINE_TB_DIR"
        echo "Infini-Attention training logs: $INFINI_LOG_DIR"
        echo "Baseline training logs: $BASELINE_LOG_DIR"
        echo "Log files: infini_training.log and baseline_training.log"
        echo "To compare training progress: tensorboard --logdir_spec=infini:$INFINI_TB_DIR,baseline:$BASELINE_TB_DIR"
        echo "-------------------------------------"
    fi
fi

# Run single model if not running both
if [[ "$RUN_BOTH_MODELS" != true ]]; then
    echo "Starting training with command:"
    echo "$TRAIN_CMD"
    
    # Debug output to verify command construction
    echo "Debug - Checking command components:"
    echo "  Wrapper script: $WRAP_SCRIPT"
    echo "  Config file: $CONFIG_FILE"
    echo "  Data directory: $PREPROCESSED_DATA"
    
    echo "-------------------------------------"
    eval $TRAIN_CMD
    
    # Print completion message
    echo "-------------------------------------"
    echo "Training completed!"
    echo "Model type: $MODEL_TYPE"
    echo "TensorBoard logs saved to: $TENSORBOARD_DIR"
    echo "To view training progress: tensorboard --logdir $TENSORBOARD_DIR"
    echo "-------------------------------------"
fi
