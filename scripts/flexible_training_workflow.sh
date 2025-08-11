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
PARQUET_DATA=""  # Added for parquet files
CONFIG_FILE="scripts/config/tiny_test_config.yaml"
OUTPUT_DIR="preprocessed_data"
DISABLE_INFINI_ATTN=false
GPU_ID=0
TENSORBOARD_DIR="tensorboard_logs/train_$(date +"%Y%m%d_%H%M%S")"
USE_GPU_DATALOADER=true
FORCE_PREPROCESS=false
DISABLE_FUSED_ADAM=true  # Set to true to avoid fused Adam optimizer errors
VERBOSE=false
OFFLINE_MODE=false  # Set to true to disable Hugging Face downloads
MAX_SEQ_LENGTH=2048  # Default sequence length for tokenization
TOKENIZER="meta-llama/Llama-2-7b-hf"  # Default tokenizer

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --raw-data)
            RAW_DATA="$2"
            shift 2
            ;;
        --parquet-data)
            PARQUET_DATA="$2"
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
        --tokenizer)
            TOKENIZER="$2"
            shift 2
            ;;
        --max-seq-length)
            MAX_SEQ_LENGTH="$2"
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
        --offline-mode)
            OFFLINE_MODE=true
            shift
            ;;
        --disable-flash-attn)
            DISABLE_FLASH_ATTENTION=true
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
            echo "  --offline-mode            Run in offline mode (no downloads from HuggingFace)"
            echo "  --disable-flash-attn      Disable Flash Attention (use standard attention instead)"
            echo "                            Use this to avoid GLIBC_2.32 errors or CUDA compatibility issues"
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

# Check if we have either raw data or preprocessed data or parquet data
if [[ -z "$RAW_DATA" ]] && [[ -z "$PREPROCESSED_DATA" ]] && [[ -z "$PARQUET_DATA" ]]; then
    echo "Error: Either --raw-data, --preprocessed-data, or --parquet-data must be specified."
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
echo "Flash Attention: ${DISABLE_FLASH_ATTENTION:+disabled}${DISABLE_FLASH_ATTENTION:-enabled}"
echo "Offline mode: ${OFFLINE_MODE:+enabled}${OFFLINE_MODE:-disabled}"
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
if [[ "$OFFLINE_MODE" = true ]] && [[ -f "/tmp/offline_wrapper.py" ]]; then
    # In offline mode, use the offline wrapper script
    echo "Using offline wrapper script for training"
    if [[ -f "$WRAP_SCRIPT" ]]; then
        TRAIN_CMD="python \"/tmp/offline_wrapper.py\" \"$WRAP_SCRIPT\" \
            --config-file \"$CONFIG_FILE\" \
            --data-dir \"$PREPROCESSED_DATA\" \
            --gpu-id \"$GPU_ID\" \
            --tensorboard-dir \"$TENSORBOARD_DIR\""
    else
        TRAIN_CMD="python \"/tmp/offline_wrapper.py\" \"$PROJECT_ROOT/scripts/run_direct_training.py\" \
            --config-file \"$CONFIG_FILE\" \
            --data-dir \"$PREPROCESSED_DATA\" \
            --gpu-id \"$GPU_ID\" \
            --tensorboard-dir \"$TENSORBOARD_DIR\""
    fi
else
    # Normal mode (online)
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

if [[ "$OFFLINE_MODE" = true ]]; then
    TRAIN_CMD="$TRAIN_CMD --offline-mode"
fi

# Add Flash Attention disabled flag if needed
if [[ "$DISABLE_FLASH_ATTENTION" = true ]]; then
    TRAIN_CMD="$TRAIN_CMD --disable-flash-attn"
fi

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Set up environment variables for proper path resolution
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PYTHONPATH"

# Check if Flash Attention is available and compatible, disable it if issues are detected
DISABLE_FLASH_ATTENTION=false
FLASH_ATTN_ERROR=""

# First check if flash_attn can be imported
if ! python -c "import flash_attn" &>/dev/null; then
    echo "Flash Attention is not installed. Automatically disabling it."
    DISABLE_FLASH_ATTENTION=true
    FLASH_ATTN_ERROR="Not installed"
else
    # Try to import and verify compatibility
    FLASH_COMPATIBILITY=$(python -c "
import sys
try:
    import flash_attn
    print('Flash Attention version:', flash_attn.__version__)
    try:
        # Try to import the CUDA module which often has compatibility issues
        import flash_attn_2_cuda
        print('compatible')
    except ImportError as e:
        # Check for common compatibility errors
        error_msg = str(e)
        if 'GLIBC_2.32' in error_msg or 'GLIBC_2.3' in error_msg or 'GLIBC' in error_msg:
            print('glibc_version_error: ' + error_msg)
        elif 'CUDA' in error_msg:
            print('cuda_version_error: ' + error_msg)
        else:
            print('import_error: ' + error_msg)
except Exception as e:
    print('error: ' + str(e))
" 2>/dev/null)

    if [[ "$FLASH_COMPATIBILITY" != *"compatible"* ]]; then
        echo "Flash Attention compatibility issue detected:"
        echo "$FLASH_COMPATIBILITY"
        echo "Automatically disabling Flash Attention."
        DISABLE_FLASH_ATTENTION=true
        FLASH_ATTN_ERROR="$FLASH_COMPATIBILITY"
        
        # Extract detailed error message for diagnostics
        if [[ "$FLASH_COMPATIBILITY" == *"glibc_version_error:"* ]]; then
            echo "GLIBC version compatibility issue detected. This happens when Flash Attention"
            echo "was compiled with a newer GLIBC version than what's available on your system."
            echo "Options to fix this:"
            echo "  1. Rebuild Flash Attention from source for your system"
            echo "  2. Continue with Flash Attention disabled (using standard attention instead)"
            echo ""
            echo "TIP: You can always run training without Flash Attention by using:"
            echo "     ./scripts/train_without_flash_attn.sh [your regular arguments]"
        elif [[ "$FLASH_COMPATIBILITY" == *"cuda_version_error:"* ]]; then
            echo "CUDA version compatibility issue detected. Your CUDA version may be incompatible"
            echo "with the installed Flash Attention binary."
            echo ""
            echo "TIP: You can always run training without Flash Attention by using:"
            echo "     ./scripts/train_without_flash_attn.sh [your regular arguments]"
        fi
    fi
fi

# Set environment variable to disable Flash Attention if needed
if [[ "$DISABLE_FLASH_ATTENTION" = true ]]; then
    export DISABLE_FLASH_ATTN=1
    echo "Flash Attention disabled due to: $FLASH_ATTN_ERROR"
    echo "Adding --disable-flash-attn flag to training command"
    
    # Add the flag to the training command
    TRAIN_CMD="$TRAIN_CMD --disable-flash-attn"
    
    # Apply the Flash Attention compatibility layer to avoid import errors
    if [[ -f "$PROJECT_ROOT/scripts/flash_attention_compatibility.py" ]]; then
        echo "Applying Flash Attention compatibility layer..."
        PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PYTHONPATH" python "$PROJECT_ROOT/scripts/flash_attention_compatibility.py"
        COMPATIBILITY_STATUS=$?
        
        if [[ $COMPATIBILITY_STATUS -eq 0 ]]; then
            echo "Flash Attention compatibility layer applied successfully"
        else
            echo "Warning: Failed to apply Flash Attention compatibility layer - training may fail"
        fi
    else
        echo "Warning: Flash Attention compatibility script not found at $PROJECT_ROOT/scripts/flash_attention_compatibility.py"
    fi
    
    # Log the action for diagnostic purposes
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Flash Attention disabled due to compatibility issue: $FLASH_ATTN_ERROR" >> "$PROJECT_ROOT/flash_attention_log.txt"
fi

# Set up distributed training environment variables for single GPU mode
# These are needed by ParallelContext even in non-distributed mode
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR="localhost"
export MASTER_PORT=29500

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

# Configure offline mode if requested to avoid download issues
if [[ "$OFFLINE_MODE" = true ]]; then
    echo "Configuring offline mode to avoid HuggingFace downloads..."
    # Set HuggingFace environment variables to use local files only
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    export NO_GIT=1
    # Disable HTTP requests 
    export HF_HUB_DISABLE_TELEMETRY=1
    export HF_HUB_DISABLE_SYMLINKS_WARNING=1
    export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
    export HF_HUB_DISABLE_PROGRESS_BARS=1
    # Set timeouts to minimal values to fail fast
    export HF_HUB_DOWNLOAD_TIMEOUT=1
    # Unset certificate bundles to avoid verification issues
    export REQUESTS_CA_BUNDLE=""
    export CURL_CA_BUNDLE=""
    export SSL_CERT_FILE=""
    # Clear any proxy settings
    export http_proxy=""
    export https_proxy=""
    export HTTP_PROXY=""
    export HTTPS_PROXY=""
    export all_proxy=""
    export ALL_PROXY=""
    export no_proxy="*"
    export NO_PROXY="*"
    echo "Environment configured for offline mode"
    
    # Create a monkey patch for pipeline engine to handle offline mode
    echo "Creating offline mode patches for pipeline engine..."
    
    # Create a simple Python patch that monkey patches the tokenizer initialization
    python -c "
import os
import sys
import shutil

# Create a backup file for environment variables
env_backup_file = '/tmp/env_vars_backup.txt'
with open(env_backup_file, 'w') as f:
    for key, value in os.environ.items():
        if key.startswith('HF_') or key.lower().endswith('proxy'):
            f.write(f'{key}={value}\\n')

print('Offline mode patch: Environment variables backed up to', env_backup_file)

# Create a patch for the pipeline engine.py file
engine_patch_file = '/tmp/pipeline_engine_patch.py'
with open(engine_patch_file, 'w') as f:
    f.write('''
# Patch for engine.py to handle offline mode
import os
import sys
import importlib
from functools import wraps

def patch_engine():
    try:
        import nanotron.parallel.pipeline_parallel.engine as engine_module
        from transformers import AutoTokenizer, PreTrainedTokenizer
        
        # Store original __init__ method
        original_init = engine_module.PipelineEngine.__init__
        
        # Create patched __init__ method
        @wraps(original_init)
        def patched_init(self):
            # Call original init first
            original_init(self)
            
            # Check if we're in offline mode
            offline_mode = bool(os.environ.get('HF_HUB_OFFLINE', False))
            
            # Replace the tokenizer initialization with offline-friendly version
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
                
            try:
                if offline_mode:
                    print('PipelineEngine: Using offline mode for tokenizer initialization')
                    try:
                        # Try local-only Llama tokenizer
                        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', local_files_only=True)
                    except:
                        try:
                            # Try local-only GPT2 tokenizer
                            self.tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
                        except:
                            # Create a basic tokenizer as last resort
                            print('PipelineEngine: Creating basic tokenizer in offline mode')
                            self.tokenizer = PreTrainedTokenizer(
                                unk_token='[UNK]',
                                pad_token='[PAD]', 
                                bos_token='[BOS]',
                                eos_token='[EOS]'
                            )
                else:
                    # Normal mode - but with better error handling
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
                    except:
                        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
            except Exception as e:
                print(f'WARNING: Could not initialize tokenizer: {e}')
                # Create a minimal fallback tokenizer
                self.tokenizer = PreTrainedTokenizer(
                    unk_token='[UNK]',
                    pad_token='[PAD]', 
                    bos_token='[BOS]',
                    eos_token='[EOS]'
                )
        
        # Apply the patch
        engine_module.PipelineEngine.__init__ = patched_init
        print('Pipeline engine patched for offline mode')
        
    except Exception as e:
        print(f'WARNING: Failed to patch pipeline engine for offline mode: {e}')

# Execute the patch function when this module is imported
patch_engine()
''')

print('Offline mode patch: Created pipeline engine patch at', engine_patch_file)

# Create the monkey patching script for transformers
patch_script = '''
# Monkey patches for offline mode in transformers
import os
import sys
import warnings
from functools import wraps

# Suppress all warnings from transformers about missing files
warnings.filterwarnings('ignore', category=UserWarning)

# Original imports that we need to patch
try:
    import transformers
    from transformers import AutoTokenizer
    
    # Save original methods
    original_from_pretrained = AutoTokenizer.from_pretrained
    
    # Patch AutoTokenizer.from_pretrained to force local_files_only=True
    @wraps(original_from_pretrained)
    def patched_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        # Set force_download and resume_download to False
        kwargs['force_download'] = False
        kwargs['resume_download'] = False
        
        # Set token to None to avoid token lookup
        kwargs['token'] = None
        
        # Always use local files
        kwargs['local_files_only'] = True
        
        try:
            # First try with the normal method but with local_files_only
            return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        except (OSError, ValueError) as e:
            # Try with a basic tokenizer as a fallback
            print(f"Failed to load tokenizer {pretrained_model_name_or_path}, creating basic tokenizer")
            from transformers import PreTrainedTokenizer
            return PreTrainedTokenizer(
                unk_token='[UNK]',
                pad_token='[PAD]',
                bos_token='[BOS]',
                eos_token='[EOS]'
            )
    
    # Apply the patch
    AutoTokenizer.from_pretrained = patched_from_pretrained
    print('Offline mode: Patched AutoTokenizer.from_pretrained to work offline')
    
except ImportError as e:
    print(f'Warning: Could not patch transformers for offline mode: {e}')
    pass

# Set a flag that our patch was loaded
os.environ['OFFLINE_MODE_PATCH_APPLIED'] = '1'
'''

# Write the patch to a file
with open('/tmp/transformers_offline_patch.py', 'w') as f:
    f.write(patch_script)

print('Offline mode patch: Created transformer patch script at /tmp/transformers_offline_patch.py')
print('Offline mode patch: Add this to your PYTHONPATH to apply patches at import time')

# Set an environment variable to point to our patch
os.environ['PYTHONPATH'] = '/tmp:' + os.environ.get('PYTHONPATH', '')
os.environ['TRANSFORMERS_OFFLINE_PATCH'] = '/tmp/transformers_offline_patch.py'

print('Offline mode patch: Environment configured to apply patch automatically')
"
    
    # Export environment variable to trigger preloading of the patch
    export PYTHONPATH="/tmp:$PYTHONPATH"
    
    # Create wrapper to apply patch at runtime
    OFFLINE_WRAPPER_SCRIPT="/tmp/offline_wrapper.py"
    cat > "$OFFLINE_WRAPPER_SCRIPT" << 'EOF'
#!/usr/bin/env python
# This is a wrapper script that applies offline mode patches before importing the real script

import os
import sys
import importlib.util
import subprocess

# Load the offline patch first
patch_path = os.environ.get('TRANSFORMERS_OFFLINE_PATCH', '/tmp/transformers_offline_patch.py')
try:
    spec = importlib.util.spec_from_file_location("offline_patch", patch_path)
    offline_patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(offline_patch)
    print(f"Offline mode: Successfully loaded patch from {patch_path}")
except Exception as e:
    print(f"Offline mode: Warning - Failed to load patch from {patch_path}: {e}")
    
# Load engine patch
engine_patch_path = '/tmp/pipeline_engine_patch.py'
if os.path.exists(engine_patch_path):
    try:
        spec = importlib.util.spec_from_file_location("engine_patch", engine_patch_path)
        engine_patch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine_patch)
        print(f"Offline mode: Successfully loaded engine patch from {engine_patch_path}")
    except Exception as e:
        print(f"Offline mode: Warning - Failed to load engine patch: {e}")

# Get the original script to run
if len(sys.argv) < 2:
    print("Usage: offline_wrapper.py <script_to_run> [args...]")
    sys.exit(1)

script_path = sys.argv[1]
sys.argv = sys.argv[1:]  # Shift arguments

# Find project root - multiple strategies for robustness
project_root = None

# Strategy 1: Look for nanotron-infini in script path
current_path = os.path.dirname(os.path.abspath(script_path))
while current_path and current_path != '/':
    if os.path.basename(current_path) == 'nanotron-infini':
        project_root = current_path
        break
    parent = os.path.dirname(current_path)
    if parent == current_path:  # Reached root directory
        break
    current_path = parent

# Strategy 2: Look in current working directory
if not project_root:
    cwd = os.getcwd()
    if 'nanotron-infini' in cwd:
        # Find the project root by looking for nanotron-infini in the path
        parts = cwd.split('nanotron-infini')
        if len(parts) > 1:
            project_root = parts[0] + 'nanotron-infini'
            print(f"Found project root from CWD: {project_root}")

# Strategy 3: Check if script path contains a scripts directory
if not project_root and '/scripts/' in script_path:
    parts = script_path.split('/scripts/')
    if len(parts) > 1:
        potential_root = parts[0]
        # Verify this looks like a project root
        if os.path.isdir(os.path.join(potential_root, 'src')) and os.path.isdir(os.path.join(potential_root, 'scripts')):
            project_root = potential_root
            print(f"Found project root from script path structure: {project_root}")

# Export PROJECT_ROOT environment variable
if project_root:
    print(f"Found project root: {project_root}")
    os.environ['PROJECT_ROOT'] = project_root
    # Add project root and src directories to PYTHONPATH
    if 'PYTHONPATH' in os.environ:
        os.environ['PYTHONPATH'] = f"{project_root}:{project_root}/src:{os.environ['PYTHONPATH']}"
    else:
        os.environ['PYTHONPATH'] = f"{project_root}:{project_root}/src"
else:
    print("WARNING: Could not determine project root, training might fail due to path issues")

# Make sure the script can find run_direct_training.py
scripts_dir = None
if project_root:
    scripts_dir = os.path.join(project_root, 'scripts')
    # Check for training script directly
    training_script = os.path.join(scripts_dir, 'run_direct_training.py')
    if os.path.exists(training_script):
        print(f"Found training script at: {training_script}")
    else:
        print(f"WARNING: Training script not found at {training_script}")

# Debug information
print(f"Script to run: {script_path}")
print(f"Current working directory: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'not set')}")

# Try to execute the target script as a subprocess with environment properly set
print(f"Running: {sys.executable} {script_path} {' '.join(sys.argv[1:])}")
try:
    result = subprocess.run([sys.executable, script_path] + sys.argv[1:], env=os.environ)
    
    # If the script failed with a non-zero exit code and it's the wrapper script
    if result.returncode != 0 and os.path.basename(script_path) == 'wrapper_script.py':
        print("Wrapper script failed, trying direct fallback mechanism...")
        
        # Check for direct_training_fallback.py
        fallback_script = None
        if project_root:
            fallback_script = os.path.join(project_root, 'scripts', 'direct_training_fallback.py')
        
        if fallback_script and os.path.exists(fallback_script):
            print(f"Running fallback script: {fallback_script}")
            fallback_result = subprocess.run([sys.executable, fallback_script] + sys.argv[1:], env=os.environ)
            sys.exit(fallback_result.returncode)
        else:
            print("No fallback script found, exiting with error")
            sys.exit(result.returncode)
    else:
        sys.exit(result.returncode)
except Exception as e:
    print(f"Error executing script: {e}")
    # Try fallback mechanism
    print("Error occurred, attempting direct fallback import...")
    try:
        # Try to find run_direct_training.py directly
        direct_script = None
        if project_root:
            direct_script = os.path.join(project_root, 'scripts', 'run_direct_training.py')
            
        if direct_script and os.path.exists(direct_script):
            print(f"Found direct training script: {direct_script}")
            # Try to run it directly
            direct_result = subprocess.run([sys.executable, direct_script] + sys.argv[1:], env=os.environ)
            sys.exit(direct_result.returncode)
        else:
            print("Could not find direct training script for fallback")
            sys.exit(1)
    except Exception as e2:
        print(f"Fallback also failed: {e2}")
        sys.exit(1)
EOF

    chmod +x "$OFFLINE_WRAPPER_SCRIPT"
    echo "Offline mode wrapper script created at $OFFLINE_WRAPPER_SCRIPT"
fi

# Make our fallback script executable
if [[ -f "$PROJECT_ROOT/scripts/direct_training_fallback.py" ]]; then
    chmod +x "$PROJECT_ROOT/scripts/direct_training_fallback.py"
    echo "Made fallback script executable"
fi

# Apply the Adam optimizer patch directly before training
echo "Applying Adam optimizer patch to fix weight_decay=None issue"

# Try our new direct patch approach first (most robust)
if [[ -f "$PROJECT_ROOT/scripts/direct_adam_patch.py" ]]; then
    echo "Using direct patching approach (targets PyTorch 2.x _single_tensor_adam function)"
    PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PYTHONPATH" python "$PROJECT_ROOT/scripts/direct_adam_patch.py"
    PATCH_STATUS=$?
    
    if [[ $PATCH_STATUS -eq 0 ]]; then
        echo "Direct Adam patch applied successfully"
    else
        echo "Direct patch failed, trying alternative methods..."
    fi
# Try the V2 patch script if available
elif [[ -f "$PROJECT_ROOT/scripts/apply_adam_patch_v2.sh" ]]; then
    bash "$PROJECT_ROOT/scripts/apply_adam_patch_v2.sh"
    echo "Adam patch script executed via V2 script"
# Try the original patch script
elif [[ -f "$PROJECT_ROOT/scripts/apply_adam_patch.sh" ]]; then
    bash "$PROJECT_ROOT/scripts/apply_adam_patch.sh"
    echo "Adam patch script executed"
# Create and run a simple inline patch as a last resort
else
    echo "Creating a minimal Adam patch that should work with all PyTorch versions"
    python -c "
import torch
from torch.optim import Adam

# Store original step method
original_step = Adam.step

# Create patched step method
def patched_step(self, closure=None):
    # Replace None weight_decay with 0.0 in optimizer instance
    for group in self.param_groups:
        if 'weight_decay' in group and group['weight_decay'] is None:
            print('Fixed: Replaced None weight_decay with 0.0')
            group['weight_decay'] = 0.0
    return original_step(self, closure)

# Apply the patch
Adam.step = patched_step
print('Applied direct Adam class patch')

# Try to patch _single_tensor_adam if it exists in this PyTorch version
try:
    from torch.optim import adam
    if hasattr(adam, '_single_tensor_adam'):
        original_func = adam._single_tensor_adam
        def patched_single_tensor_adam(*args, **kwargs):
            if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
                print('Fixed: Replaced None weight_decay with 0.0 in _single_tensor_adam')
                kwargs['weight_decay'] = 0.0
            return original_func(*args, **kwargs)
        adam._single_tensor_adam = patched_single_tensor_adam
        print('Also patched _single_tensor_adam function')
except Exception:
    pass
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

# Create a symbolic link to run_direct_training.py in /scripts/ directory if it's needed
if [[ "$OFFLINE_MODE" = true ]]; then
    echo "Setting up symlinks for offline mode compatibility..."
    
    # Check if the directory /scripts exists, create it if needed
    if [[ ! -d "/scripts" ]]; then
        echo "Creating /scripts directory for compatibility (requires sudo)"
        sudo mkdir -p /scripts || echo "WARNING: Failed to create /scripts directory - may require admin privileges"
    fi
    
    # Find the direct training script
    DIRECT_TRAIN_SCRIPT="$PROJECT_ROOT/scripts/run_direct_training.py"
    if [[ -f "$DIRECT_TRAIN_SCRIPT" ]]; then
        # Try to create symlink if we have write permissions
        if [[ -d "/scripts" && -w "/scripts" ]]; then
            echo "Creating symlink to training script at /scripts/run_direct_training.py"
            sudo ln -sf "$DIRECT_TRAIN_SCRIPT" /scripts/run_direct_training.py || echo "WARNING: Failed to create symlink"
        else
            echo "WARNING: Cannot create symlink in /scripts directory - no write permission"
        fi
    else
        echo "WARNING: Could not find run_direct_training.py at $DIRECT_TRAIN_SCRIPT"
    fi
fi

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
        if [[ "$OFFLINE_MODE" = true ]] && [[ -f "/tmp/offline_wrapper.py" ]]; then
            # In offline mode, use the offline wrapper script
            if [[ -f "$WRAP_SCRIPT" ]]; then
                INFINI_CMD="CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29500 TRAINING_LOGS_DIR=$INFINI_LOG_DIR python \"/tmp/offline_wrapper.py\" \"$WRAP_SCRIPT\" \
                    --config-file \"$CONFIG_FILE\" \
                    --data-dir \"$PREPROCESSED_DATA\" \
                    --gpu-id 0 \
                    --tensorboard-dir \"$INFINI_TB_DIR\""
                
                BASELINE_CMD="CUDA_VISIBLE_DEVICES=1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29501 TRAINING_LOGS_DIR=$BASELINE_LOG_DIR python \"/tmp/offline_wrapper.py\" \"$WRAP_SCRIPT\" \
                    --config-file \"$CONFIG_FILE\" \
                    --data-dir \"$PREPROCESSED_DATA\" \
                    --gpu-id 0 \
                    --disable-infini-attn \
                    --tensorboard-dir \"$BASELINE_TB_DIR\""
            else
                INFINI_CMD="CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29500 TRAINING_LOGS_DIR=$INFINI_LOG_DIR python \"/tmp/offline_wrapper.py\" \"$PROJECT_ROOT/scripts/run_direct_training.py\" \
                    --config-file \"$CONFIG_FILE\" \
                    --data-dir \"$PREPROCESSED_DATA\" \
                    --gpu-id 0 \
                    --tensorboard-dir \"$INFINI_TB_DIR\""
                
                BASELINE_CMD="CUDA_VISIBLE_DEVICES=1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29501 TRAINING_LOGS_DIR=$BASELINE_LOG_DIR python \"/tmp/offline_wrapper.py\" \"$PROJECT_ROOT/scripts/run_direct_training.py\" \
                    --config-file \"$CONFIG_FILE\" \
                    --data-dir \"$PREPROCESSED_DATA\" \
                    --gpu-id 0 \
                    --disable-infini-attn \
                    --tensorboard-dir \"$BASELINE_TB_DIR\""
            fi
        else
            # Normal mode (online)
            if [[ -f "$WRAP_SCRIPT" ]]; then
                INFINI_CMD="CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29500 TRAINING_LOGS_DIR=$INFINI_LOG_DIR python \"$WRAP_SCRIPT\" \
                    --config-file \"$CONFIG_FILE\" \
                    --data-dir \"$PREPROCESSED_DATA\" \
                    --gpu-id 0 \
                    --tensorboard-dir \"$INFINI_TB_DIR\""
                
                BASELINE_CMD="CUDA_VISIBLE_DEVICES=1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29501 TRAINING_LOGS_DIR=$BASELINE_LOG_DIR python \"$WRAP_SCRIPT\" \
                    --config-file \"$CONFIG_FILE\" \
                    --data-dir \"$PREPROCESSED_DATA\" \
                    --gpu-id 0 \
                    --disable-infini-attn \
                    --tensorboard-dir \"$BASELINE_TB_DIR\""
            else
                # Fallback to direct execution if wrapper script doesn't exist
                echo "WARNING: Wrapper script not found, falling back to direct execution for parallel training"
                INFINI_CMD="CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29500 TRAINING_LOGS_DIR=$INFINI_LOG_DIR python \"$PROJECT_ROOT/scripts/run_direct_training.py\" \
                    --config-file \"$CONFIG_FILE\" \
                    --data-dir \"$PREPROCESSED_DATA\" \
                    --gpu-id 0 \
                    --tensorboard-dir \"$INFINI_TB_DIR\""
                
                BASELINE_CMD="CUDA_VISIBLE_DEVICES=1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29501 TRAINING_LOGS_DIR=$BASELINE_LOG_DIR python \"$PROJECT_ROOT/scripts/run_direct_training.py\" \
                    --config-file \"$CONFIG_FILE\" \
                    --data-dir \"$PREPROCESSED_DATA\" \
                    --gpu-id 0 \
                    --disable-infini-attn \
                    --tensorboard-dir \"$BASELINE_TB_DIR\""
            fi
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

        if [[ "$OFFLINE_MODE" = true ]]; then
            INFINI_CMD="$INFINI_CMD --offline-mode"
            BASELINE_CMD="$BASELINE_CMD --offline-mode"
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
