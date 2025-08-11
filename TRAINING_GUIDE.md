# Complete Training Guide: Llama 200M with Infini-Attention

This guide provides step-by-step instructions to train a Llama 200M model with Infini-Attention using the **already downloaded** FineWeb 10B dataset and **existing proven configurations**.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Dataset Preparation](#dataset-preparation)
3. [Configuration Setup](#configuration-setup)
4. [Training Execution](#training-execution)
5. [Monitoring and Evaluation](#monitoring-and-evaluation)
6. [Troubleshooting](#troubleshooting)

## Overview: Using Existing Infrastructure

**Key Insight**: This codebase already contains everything needed! We'll use:
- ✅ **Existing tokenization script**: `examples/infinite-context-length/data/exp34/tokenize_finetunine_data_to_s3.py`
- ✅ **Proven 200M config**: `examples/infinite-context-length/configs/exp53/exp53_200m_infini_llama2_*.yaml`
- ✅ **Standard training script**: `run_train.py`
- ✅ **Your downloaded FineWeb dataset** (no re-downloading needed)

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
pip install datatrove  # For tokenization pipeline
pip install flash-attn --no-build-isolation  # Flash attention
pip install einops  # For tensor operations
pip install wandb  # For experiment tracking (optional)
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

## Dataset Preparation

**Note**: Since you already have FineWeb 10B downloaded, we'll just tokenize it using the existing script.

### Step 1: Locate Your Downloaded FineWeb Dataset

```bash
# Verify you have the FineWeb dataset downloaded locally
# (Update the path to where your dataset is stored)
ls -la /path/to/your/downloaded/fineweb/

# OR if it's accessible via HuggingFace cache:
python -c "
from datasets import load_dataset
try:
    dataset = load_dataset('HuggingFaceFW/fineweb', '10B', split='train[:100]')
    print('✅ FineWeb 10B dataset is accessible')
    print(f'Sample keys: {dataset.column_names}')
    print(f'Dataset size: {len(dataset)} samples (showing first 100)')
except Exception as e:
    print(f'❌ Error accessing dataset: {e}')
"
```

### Step 2: Use Existing Tokenization Script

```bash
# Use the proven tokenization script from the repository
python examples/infinite-context-length/data/exp34/tokenize_finetunine_data_to_s3.py \
    --dataset "HuggingFaceFW/fineweb" \
    --output_path "./data/fineweb_tokenized" \
    --tokenizer "lvwerra/the-tokenizer-v1" \
    --tasks 100 \
    --split "train[:1000000]"  # Adjust size as needed (1M samples for testing)

# For full dataset (if you have enough disk space and time):
# --split "train"  # Use entire dataset
```

```bash
# Check tokenization output
ls -la data/fineweb_tokenized/

# Verify the expected directory structure
find data/fineweb_tokenized -name "*.ds" | head -5

# Check file sizes
python -c "
import os
from pathlib import Path

# Check both possible output locations
possible_dirs = [
    './data/fineweb_tokenized/merged-dataset',
    './data/fineweb_tokenized/tokenized-v1/standard'
]

for data_dir in possible_dirs:
    if os.path.exists(data_dir):
        files = list(Path(data_dir).glob('*.ds'))
        print(f'✅ Found {len(files)} .ds files in {data_dir}')
        for f in files[:3]:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f'  {f.name}: {size_mb:.1f} MB')
        break
else:
    print('❌ No .ds files found. Check tokenization output.')
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

### Step 2: Update Only Data Paths

Edit `my_fineweb_200m_config.yaml` and change **only** the data section:

```bash
# Edit the config file to point to your tokenized FineWeb data
# Find the data section and update it:
```

```yaml
# UPDATE ONLY THIS SECTION in my_fineweb_200m_config.yaml:
data:
  dataset:
    dataloader_type: single
    dataset_max_tokens: null
    dataset_weights: [1.0]  # 100% FineWeb (instead of multiple datasets)
    datasets:
    - filename_pattern: "*.ds"  
      folder: "./data/fineweb_tokenized/merged-dataset/"  # UPDATE: your tokenized data path
      skip_tokens: 0
    pad_samples_to_global_batch_size: false
    skip_in_stream: true
  num_loading_workers: 4
  seed: 42

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

### Step 4: Verify Configuration

```bash
# Test that your config loads correctly
python -c "
from nanotron.config import get_config_from_file
config = get_config_from_file('my_fineweb_200m_config.yaml')
print('✅ Configuration loaded successfully!')
print(f'Model size: {config.model.model_config.hidden_size}')
print(f'Layers: {config.model.model_config.num_hidden_layers}')
print(f'Segment length: {config.infini_attention.segment_length}')
print(f'Data folder: {config.data.dataset.datasets[0].folder}')
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

# Simple training script using existing infrastructure
set -e

CONFIG_FILE="my_fineweb_200m_config.yaml"
NUM_GPUS=${1:-1}

echo "🚀 Starting Llama 200M + Infini-Attention training with $NUM_GPUS GPU(s)"

# Verify config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: $CONFIG_FILE not found!"
    echo "Run: cp examples/infinite-context-length/configs/exp53/exp53_200m_*.yaml my_fineweb_200m_config.yaml"
    exit 1
fi

# Create directories
mkdir -p checkpoints/llama_200m_fineweb logs

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

```bash
# Check data path
ls -la data/fineweb_tokenized/tokenized-v1/standard/

# Verify data format
python -c "
from datasets import load_from_disk
import os
data_dir = 'data/fineweb_tokenized/tokenized-v1/standard/'
files = [f for f in os.listdir(data_dir) if f.endswith('.ds')]
print(f'Found files: {files}')
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
# 1. Tokenize your FineWeb data (one-time setup)
python examples/infinite-context-length/data/exp34/tokenize_finetunine_data_to_s3.py \
    --dataset "HuggingFaceFW/fineweb" \
    --output_path "./data/fineweb_tokenized" \
    --tokenizer "lvwerra/the-tokenizer-v1" \
    --tasks 100 --split "train[:1000000]"

# 2. Copy proven config
cp examples/infinite-context-length/configs/exp53/exp53_200m_infini_llama2_256_ctx_length_and_64_segment_length_and_2m_bs_and_hard_sigmoid_act_and_zeros_init_and_global_lr_0.0000375_and_balance_factor_lr_0.00015.yaml \
   my_fineweb_200m_config.yaml

# 3. Edit data path in config
# Change folder: "./data/fineweb_tokenized/merged-dataset/"

# 4. Train!
python run_train.py --config-file my_fineweb_200m_config.yaml  # Single GPU
torchrun --nproc_per_node=4 run_train.py --config-file my_fineweb_200m_config.yaml  # Multi-GPU

# 5. Monitor
python check_training.py
```

**That's it!** This approach leverages the existing, proven infrastructure instead of recreating everything from scratch. The configurations have already been tested and optimized for 200M models with infini-attention.