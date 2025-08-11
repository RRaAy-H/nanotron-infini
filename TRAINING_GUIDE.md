# Complete Training Guide: Llama 200M with Infini-Attention

This guide provides step-by-step instructions to train a Llama 200M model with Infini-Attention using HuggingFace datasets with **on-the-fly tokenization** - no pre-processing needed!

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Dataset Configuration](#dataset-configuration)
3. [Configuration Setup](#configuration-setup)
4. [Training Execution](#training-execution)
5. [Monitoring and Evaluation](#monitoring-and-evaluation)
6. [Troubleshooting](#troubleshooting)

## Overview: Simplified Training Pipeline

**Key Insight**: This guide uses the built-in dataloader which tokenizes data on-the-fly during training:
- ✅ **No pre-tokenization needed**: Uses `src/nanotron/dataloader.py` for automatic tokenization
- ✅ **Direct from HuggingFace**: Load any text dataset directly
- ✅ **Proven 200M config**: `examples/infinite-context-length/configs/exp53/exp53_200m_infini_llama2_*.yaml`
- ✅ **Standard training script**: `run_train.py`

## Environment Setup

### Step 1: Create Python Environment

```bash
# Create conda environment
conda create -n nanotron-infini python=3.10
conda activate nanotron-infini

# Alternative: using virtualenv
# python3.10 -m venv nanotron-infini
# source nanotron-infini/bin/activate
```

### Step 2: Install CUDA and PyTorch

```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

### Step 3: Install Nanotron Dependencies

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd nanotron-infini

# Install core dependencies
pip install -e .

# Install additional requirements
pip install -r requirements.txt

# Install specific packages for data processing
pip install datasets transformers accelerate
pip install flash-attn --no-build-isolation  # Flash attention
pip install einops  # For tensor operations
pip install wandb  # For experiment tracking (optional)
# Note: datatrove is NOT needed when using on-the-fly tokenization
```

### Step 4: Verify Installation

```bash
# Test basic imports
python -c "
import torch
import nanotron
import datasets
import transformers
print('All imports successful!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
"
```

## Dataset Configuration

**Note**: With on-the-fly tokenization, you can use any HuggingFace dataset directly without pre-processing!

### Step 1: Verify Dataset Access

```bash
# Test that you can access the dataset you want to use
# Example with FineWeb dataset:
python -c "
from datasets import load_dataset
try:
    # Try loading a small sample first
    dataset = load_dataset('HuggingFaceFW/fineweb', '10BT', split='train[:100]')
    print('✅ FineWeb dataset is accessible')
    print(f'Sample columns: {dataset.column_names}')
    print(f'Text column: \"text\" or \"content\"')
    print(f'Sample text: {dataset[0][\"text\"][:200]}...')
except Exception as e:
    print(f'❌ Error accessing dataset: {e}')
    print('Try a different dataset like: stas/openwebtext-10k')
"

# Alternative: Use a smaller dataset for testing
python -c "
from datasets import load_dataset
# Smaller test dataset (faster for initial testing)
dataset = load_dataset('HuggingFaceH4/testing_alpaca_small', split='train[:100]')
print('✅ Test dataset loaded')
print(f'Columns: {dataset.column_names}')
print(f'Sample size: {len(dataset)} examples')
"
```

### Step 2: Choose Your Dataset

You can use any of these datasets without pre-tokenization:

```python
# Popular options (no pre-processing needed!):

# 1. Small test dataset (quick testing)
dataset_name = "HuggingFaceH4/testing_alpaca_small"
text_column = "completion"

# 2. OpenWebText (medium size, good quality)  
dataset_name = "stas/openwebtext-10k"
text_column = "text"

# 3. FineWeb (large, high quality) - FROM HUGGINGFACE
dataset_name = "HuggingFaceFW/fineweb"
dataset_config = "10BT"  # or "100BT" for larger
text_column = "text"

# 4. LOCAL FineWeb dataset (already downloaded)
# If you have local parquet files like:
# data1/dataset/HuggingFaceFW/fineweb/fineweb-train-00000-of-00102.parquet
local_dataset_path = "data1/dataset/HuggingFaceFW/fineweb/"
text_column = "text"

# 5. The Pile (diverse content)
dataset_name = "EleutherAI/the_pile_deduplicated"
dataset_config = "all"
text_column = "text"

# 6. Custom dataset from HuggingFace
dataset_name = "your-username/your-dataset"
text_column = "text"  # or whatever column has the text
```

### Step 3: Test Your Local Dataset (If Using Local Files)

```bash
# Test that your local FineWeb dataset can be loaded with both approaches
python -c "
from datasets import load_dataset
import os
import glob

# Update this path to your local dataset
local_path = 'data1/dataset/HuggingFaceFW/fineweb/'

# Check files exist
parquet_files = [f for f in os.listdir(local_path) if f.endswith('.parquet')]
print(f'Found {len(parquet_files)} parquet files in {local_path}')

print('\\n=== Testing Approach 1: Direct Path ===')
try:
    # Approach 1: Use directory path directly
    dataset = load_dataset(local_path, split='train[:100]')
    print('✅ Approach 1 SUCCESS: Direct path loading works!')
    print(f'Columns: {dataset.column_names}')
    print(f'Sample text: {dataset[0][\"text\"][:200]}...')
except Exception as e:
    print(f'❌ Approach 1 failed: {e}')

print('\\n=== Testing Approach 2: Parquet with data_files ===')
try:
    # Approach 2: Use parquet loader with data_files
    parquet_files_full = glob.glob(os.path.join(local_path, '*.parquet'))
    dataset = load_dataset('parquet', data_files=parquet_files_full, split='train[:100]')
    print('✅ Approach 2 SUCCESS: Parquet with data_files works!')
    print(f'Columns: {dataset.column_names}')
    print(f'Sample text: {dataset[0][\"text\"][:200]}...')
except Exception as e:
    print(f'❌ Approach 2 failed: {e}')

print('\\n=== Testing Approach 3: Parquet with data_dir ===')
try:
    # Approach 3: Use parquet loader with data_dir
    dataset = load_dataset('parquet', data_dir=local_path, split='train[:100]')
    print('✅ Approach 3 SUCCESS: Parquet with data_dir works!')
    print(f'Columns: {dataset.column_names}')
    print(f'Sample text: {dataset[0][\"text\"][:200]}...')
except Exception as e:
    print(f'❌ Approach 3 failed: {e}')
"
```

## Configuration Setup

**Key Insight**: Use the existing proven 200M configuration instead of creating new ones!

### Step 1: Copy Existing 200M Configuration

```bash
# List available 200M configurations
ls examples/infinite-context-length/configs/exp*/exp*200m*

# Copy the best performing 200M config (exp53 has good hyperparameters)
cp examples/infinite-context-length/configs/exp53/exp53_200m_infini_llama2_256_ctx_length_and_64_segment_length_and_2m_bs_and_hard_sigmoid_act_and_zeros_init_and_global_lr_0.0000375_and_balance_factor_lr_0.00015.yaml \
   my_fineweb_200m_config.yaml

# View the copied config to understand the structure
head -50 my_fineweb_200m_config.yaml
```

### Step 2: Update Configuration for On-the-Fly Tokenization

Edit `my_fineweb_200m_config.yaml` to use HuggingFace datasets directly:

```yaml
# UPDATE THE DATA SECTION in my_fineweb_200m_config.yaml:
# This enables on-the-fly tokenization from HuggingFace datasets

# Add data_stages section (for on-the-fly tokenization)
data_stages:
  - name: "Training Stage"
    start_training_step: 1
    data:
      dataset:
        hf_dataset_or_datasets: "HuggingFaceH4/testing_alpaca_small"  # Change to your dataset
        hf_dataset_config_name: null  # Set if needed (e.g., "10BT" for FineWeb)
        hf_dataset_splits: "train"
        text_column_name: "completion"  # Change based on your dataset
        dataset_processing_num_proc_per_process: 4
        dataset_overwrite_cache: false
      num_loading_workers: 4
      seed: 42

# Also add tokenizer configuration
tokenizer:
  tokenizer_name_or_path: "gpt2"  # or "lvwerra/the-tokenizer-v1" for better tokenizer
  tokenizer_max_length: null
  tokenizer_revision: null

# Also update experiment tracking:
general:
  project: llama_200m_infini_fineweb
  run: fineweb_experiment
  seed: 42

checkpoints:
  checkpoints_path: ./checkpoints/llama_200m_fineweb  # UPDATE: new checkpoint path
```

### Step 3: Adjust for Your GPU Setup

#### For Single GPU:
```bash
# Edit parallelism section in my_fineweb_200m_config.yaml
```
```yaml
parallelism:
  dp: 1
  tp: 1 
  pp: 1

tokens:
  micro_batch_size: 4  # Reduce if you get OOM errors
```

#### For Multiple GPUs (same node):
```bash
# For 4 GPUs, edit parallelism section:
```
```yaml
parallelism:
  dp: 4  # Number of GPUs you have
  tp: 1  # Keep 1 for 200M model
  pp: 1

tokens:
  micro_batch_size: 8  # Can increase with more GPUs
```

### Step 4: Configuration Examples

#### Option A: Local FineWeb Dataset (For Your Case)

Since the current nanotron codebase doesn't support the `data_dir` parameter directly, we have two approaches:

**Approach 1: Direct Path (Recommended)**
```yaml
# Example: my_local_fineweb_200m_config.yaml
general:
  project: llama_200m_infini_fineweb
  run: local_fineweb_experiment
  seed: 42

# Data configuration for LOCAL FineWeb dataset using direct path
data_stages:
  - name: "Training Stage"
    start_training_step: 1
    data:
      dataset:
        hf_dataset_or_datasets: "data1/dataset/HuggingFaceFW/fineweb/"  # Direct path to parquet files
        hf_dataset_config_name: null
        hf_dataset_splits: "train"
        text_column_name: "text"  # FineWeb uses "text" column
        dataset_processing_num_proc_per_process: 4
        dataset_overwrite_cache: false
      num_loading_workers: 4
      seed: 42

tokenizer:
  tokenizer_name_or_path: "gpt2"  # Standard GPT-2 tokenizer
  
tokens:
  sequence_length: 256  # Adjust based on your needs
  train_steps: 10000
  micro_batch_size: 8
  batch_accumulation_per_replica: 1

# Keep the rest of the model config from exp53
```

**Approach 2: Parquet Loader (Alternative)**
```yaml
# Alternative if approach 1 doesn't work
data_stages:
  - name: "Training Stage"
    start_training_step: 1
    data:
      dataset:
        hf_dataset_or_datasets: "parquet"  # Use parquet loader
        hf_dataset_config_name: null
        hf_dataset_splits: "train"
        text_column_name: "text"
        dataset_processing_num_proc_per_process: 4
        dataset_overwrite_cache: false
        # Note: This approach might need code modification to pass data_files parameter
      num_loading_workers: 4
      seed: 42
```

#### Option B: HuggingFace FineWeb (Online)

```yaml
# Example: my_fineweb_200m_config.yaml
general:
  project: llama_200m_infini_fineweb
  run: fineweb_onthefly_experiment
  seed: 42

# Data configuration for on-the-fly tokenization from HuggingFace
data_stages:
  - name: "Training Stage"
    start_training_step: 1
    data:
      dataset:
        hf_dataset_or_datasets: "HuggingFaceFW/fineweb"  # FineWeb dataset
        hf_dataset_config_name: "10BT"  # 10 billion tokens version
        hf_dataset_splits: "train"
        text_column_name: "text"  # FineWeb uses "text" column
        dataset_processing_num_proc_per_process: 4
        dataset_overwrite_cache: false
      num_loading_workers: 4
      seed: 42

tokenizer:
  tokenizer_name_or_path: "gpt2"  # Standard GPT-2 tokenizer
  
tokens:
  sequence_length: 256  # Adjust based on your needs
  train_steps: 10000
  micro_batch_size: 8
  batch_accumulation_per_replica: 1

# Keep the rest of the model config from exp53
```

### Step 5: Verify Configuration

```bash
# Test that your config loads correctly
python -c "
from nanotron.config import get_config_from_file
config = get_config_from_file('my_local_fineweb_200m_config.yaml')  # Use your config filename
print('✅ Configuration loaded successfully!')
print(f'Model size: {config.model.model_config.hidden_size}')
print(f'Layers: {config.model.model_config.num_hidden_layers}')
print(f'Segment length: {config.infini_attention.segment_length}')
print(f'Dataset: {config.data_stages[0].data.dataset.hf_dataset_or_datasets}')
print(f'Text column: {config.data_stages[0].data.dataset.text_column_name}')
print(f'Tokenizer: {config.tokenizer.tokenizer_name_or_path}')
"

# Test that the dataset can be loaded with your configuration
python -c "
from nanotron.config import get_config_from_file
from nanotron.dataloader import get_datasets

config = get_config_from_file('my_local_fineweb_200m_config.yaml')
dataset_config = config.data_stages[0].data.dataset

print('Testing dataset loading with your configuration...')
try:
    raw_dataset = get_datasets(
        hf_dataset_or_datasets=dataset_config.hf_dataset_or_datasets,
        hf_dataset_config_name=dataset_config.hf_dataset_config_name,
        splits=dataset_config.hf_dataset_splits,
    )['train']
    
    print(f'✅ Dataset loaded successfully!')
    print(f'Dataset size: {len(raw_dataset)} samples')
    print(f'Columns: {raw_dataset.column_names}')
    print(f'Sample text: {raw_dataset[0][dataset_config.text_column_name][:200]}...')
except Exception as e:
    print(f'❌ Error loading dataset: {e}')
    print('Try the alternative approaches from Step 3')
"
```

## Training Execution

### Step 1: Prepare for Training

```bash
# Create necessary directories
mkdir -p checkpoints/llama_200m_fineweb
mkdir -p logs

# Set memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Step 2: Single GPU Training

```bash
# Simple single GPU training
python run_train.py --config-file my_fineweb_200m_config.yaml

# With specific GPU selection
CUDA_VISIBLE_DEVICES=0 python run_train.py --config-file my_fineweb_200m_config.yaml
```

### Step 3: Multi-GPU Training (Single Node)

```bash
# For 2 GPUs
torchrun --nproc_per_node=2 run_train.py --config-file my_fineweb_200m_config.yaml

# For 4 GPUs  
torchrun --nproc_per_node=4 run_train.py --config-file my_fineweb_200m_config.yaml

# For 8 GPUs
torchrun --nproc_per_node=8 run_train.py --config-file my_fineweb_200m_config.yaml
```

### Step 4: Quick Training Script

Create `train_fineweb_200m.sh`:

```bash
#!/bin/bash

# Simple training script with on-the-fly tokenization
set -e

CONFIG_FILE="my_fineweb_200m_config.yaml"
NUM_GPUS=${1:-1}

echo "🚀 Starting Llama 200M + Infini-Attention training with $NUM_GPUS GPU(s)"
echo "📦 Using on-the-fly tokenization - no pre-processing needed!"

# Verify config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: $CONFIG_FILE not found!"
    echo "Run: cp examples/infinite-context-length/configs/exp53/exp53_200m_*.yaml my_fineweb_200m_config.yaml"
    echo "Then add data_stages section for your HuggingFace dataset"
    exit 1
fi

# Create directories
mkdir -p checkpoints/llama_200m_fineweb logs

# Show dataset being used
echo "📊 Dataset configuration:"
grep -A 3 "hf_dataset_or_datasets" "$CONFIG_FILE" || echo "Add data_stages section!"

# Train
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Single GPU training..."
    python run_train.py --config-file "$CONFIG_FILE"
else
    echo "Multi-GPU training with $NUM_GPUS GPUs..."
    torchrun --nproc_per_node="$NUM_GPUS" run_train.py --config-file "$CONFIG_FILE"
fi

echo "✅ Training completed!"
```

```bash
chmod +x train_fineweb_200m.sh

# Usage:
./train_fineweb_200m.sh 1    # Single GPU
./train_fineweb_200m.sh 4    # 4 GPUs
```

## Monitoring and Evaluation

### Step 1: Monitor Training Progress

```bash
# Monitor training in real-time
tail -f logs/*.log

# Check GPU usage
watch -n 1 nvidia-smi

# View checkpoints being created
ls -la checkpoints/llama_200m_fineweb/

# Check training step progress
grep -i "train step" logs/*.log | tail -5
```

### Step 2: Use Existing Evaluation Scripts

```bash
# The repository has evaluation scripts ready to use:
ls examples/infinite-context-length/run_evals.py
ls examples/infinite-context-length/scripts/run_*_evals.sh

# Basic model evaluation (after some training)
python examples/infinite-context-length/run_evals.py \
    --config-file my_fineweb_200m_config.yaml \
    --checkpoint-path checkpoints/llama_200m_fineweb/latest

# Run passkey evaluation (tests long context ability)
python examples/infinite-context-length/scripts/run_passkey_eval.py \
    --config-file my_fineweb_200m_config.yaml \
    --checkpoint-path checkpoints/llama_200m_fineweb/latest \
    --context-length 4096
```

### Step 3: Quick Health Check

```bash
# Check training progress script
cat > check_training.py << 'EOF'
import os
import re
from pathlib import Path

# Check if training is running
log_files = list(Path("logs").glob("*.log"))
if not log_files:
    print("❌ No log files found. Is training started?")
    exit()

latest_log = max(log_files, key=os.path.getmtime)
print(f"📋 Checking latest log: {latest_log}")

with open(latest_log, 'r') as f:
    lines = f.readlines()

# Extract training metrics
losses = []
steps = []
for line in lines[-50:]:  # Check last 50 lines
    if 'train_step' in line.lower() and 'loss' in line.lower():
        step_match = re.search(r'train_step.*?(\d+)', line)
        loss_match = re.search(r'loss.*?([\d.]+)', line)
        if step_match and loss_match:
            steps.append(int(step_match.group(1)))
            losses.append(float(loss_match.group(1)))

if losses:
    print(f"✅ Training is active!")
    print(f"📊 Latest step: {steps[-1] if steps else 'Unknown'}")
    print(f"📉 Latest loss: {losses[-1]:.4f}")
    if len(losses) > 1:
        print(f"📈 Loss trend: {losses[-1] - losses[0]:.4f} (lower is better)")
else:
    print("⚠️  No recent training metrics found")

# Check checkpoints
checkpoint_dir = Path("checkpoints/llama_200m_fineweb")
if checkpoint_dir.exists():
    checkpoints = list(checkpoint_dir.glob("step_*"))
    print(f"💾 Checkpoints saved: {len(checkpoints)}")
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        print(f"📁 Latest checkpoint: {latest_checkpoint.name}")
EOF

python check_training.py
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM) Errors

```bash
# Reduce batch size in config
tokens:
  micro_batch_size: 2  # Reduce from 8

# Enable gradient checkpointing
model:
  ddp_bucket_cap_mb: 10  # Reduce from 25

# Use mixed precision more aggressively
model:
  dtype: float16  # Instead of bfloat16
```

#### 2. Data Loading Issues

**For HuggingFace Datasets:**
```bash
# Check if dataset can be loaded
python -c "
from datasets import load_dataset
try:
    # Replace with your dataset
    dataset = load_dataset('HuggingFaceH4/testing_alpaca_small', split='train[:10]')
    print(f'✅ Dataset loaded: {len(dataset)} samples')
    print(f'Columns: {dataset.column_names}')
except Exception as e:
    print(f'❌ Error: {e}')
"
```

**For Local FineWeb Dataset:**
```bash
# Check if local dataset path exists and files are accessible
python -c "
import os
import glob
from datasets import load_dataset

local_path = 'data1/dataset/HuggingFaceFW/fineweb/'  # Update with your path

# Check directory exists
if not os.path.exists(local_path):
    print(f'❌ Directory does not exist: {local_path}')
    exit(1)

# Check parquet files
parquet_files = glob.glob(os.path.join(local_path, '*.parquet'))
print(f'Found {len(parquet_files)} parquet files')

if len(parquet_files) == 0:
    print('❌ No parquet files found!')
    exit(1)

# Test loading
try:
    dataset = load_dataset(local_path, split='train[:10]')
    print(f'✅ Local dataset loaded: {len(dataset)} samples')
    print(f'Columns: {dataset.column_names}')
    print(f'Text sample: {dataset[0][\"text\"][:100]}...')
except Exception as e:
    print(f'❌ Error loading local dataset: {e}')
    print('Try alternative loading methods:')
    print('1. load_dataset(\"parquet\", data_files=parquet_files)')
    print('2. Check file permissions')
    print('3. Verify parquet files are not corrupted')
"

# Check tokenizer
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
text = 'Hello world'
tokens = tokenizer(text)
print(f'✅ Tokenizer works: {tokens}')
"
```

#### 3. CUDA Compatibility Issues

```bash
# Check CUDA version compatibility
python -c "
import torch
print(f'PyTorch CUDA version: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
"

# Reinstall PyTorch if needed
pip uninstall torch torchvision torchaudio
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

#### 4. Distributed Training Issues

```bash
# Check port availability for multi-GPU
netstat -tulpn | grep :29500

# Use different port if needed
torchrun --master_port=29501 --nproc_per_node=4 \
    run_train.py --config-file config_200m_infini.yaml
```

### Performance Optimization Tips

1. **Memory Optimization**:
   ```yaml
   # In config file
   optimizer:
     accumulate_grad_in_fp32: false  # Save memory
   model:
     dtype: bfloat16  # Use mixed precision
   ```

2. **Data Loading**:
   ```yaml
   data:
     num_loading_workers: 8  # Increase for faster data loading
   ```

3. **Checkpointing**:
   ```yaml
   checkpoints:
     checkpoint_interval: 1000  # Adjust frequency
   ```

### Expected Training Times (Estimates)

- **Single RTX 4090 (24GB)**: ~8-12 hours for 20K steps (200M model)
- **4x RTX 4090**: ~2-3 hours for 20K steps  
- **Single A100 (80GB)**: ~6-10 hours for 20K steps
- **Memory usage**: ~6-10GB for 200M model with reasonable batch sizes

### Quick Commands Summary

```bash
# 1. NO TOKENIZATION NEEDED! Skip straight to config setup

# 2. Copy proven config
cp examples/infinite-context-length/configs/exp53/exp53_200m_infini_llama2_256_ctx_length_and_64_segment_length_and_2m_bs_and_hard_sigmoid_act_and_zeros_init_and_global_lr_0.0000375_and_balance_factor_lr_0.00015.yaml \
   my_local_fineweb_200m_config.yaml

# 3. Edit config to add data_stages section for your LOCAL dataset
# For local FineWeb parquet files, add:
#   data_stages:
#     - name: "Training Stage"
#       data:
#         dataset:
#           hf_dataset_or_datasets: "data1/dataset/HuggingFaceFW/fineweb/"  # Your local path
#           text_column_name: "text"
#   tokenizer:
#     tokenizer_name_or_path: "gpt2"

# 4. Test your local dataset first
python -c "
from datasets import load_dataset
dataset = load_dataset('data1/dataset/HuggingFaceFW/fineweb/', split='train[:10]')
print(f'✅ Found {len(dataset)} samples, columns: {dataset.column_names}')
"

# 5. Train! (Tokenization happens automatically from local files)
python run_train.py --config-file my_local_fineweb_200m_config.yaml  # Single GPU
torchrun --nproc_per_node=4 run_train.py --config-file my_local_fineweb_200m_config.yaml  # Multi-GPU

# 6. Monitor
python check_training.py
```

**That's it!** This simplified approach uses on-the-fly tokenization, eliminating the need for pre-processing while leveraging the existing, proven infrastructure. The configurations have already been tested and optimized for 200M models with infini-attention.

## Benefits of On-the-Fly Tokenization

✅ **No pre-processing**: Start training immediately  
✅ **Flexible datasets**: Switch datasets without re-tokenizing  
✅ **Less storage**: No need to store tokenized files  
✅ **Easier experimentation**: Change tokenizers on the fly  
✅ **Automatic caching**: HuggingFace datasets handle caching  

The only trade-off is slightly slower first epoch as tokenization happens, but subsequent epochs use cached data.