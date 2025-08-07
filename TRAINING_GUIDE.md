# Training Llama with Infini Attention Model

This guide explains how to train the Llama model with Infini-Attention using the provided dataset.

## Setup Requirements

The nanotron-infini package has dependencies that require specific versions of Python and CUDA libraries. Based on the error messages, we need to:

1. Use Python 3.10 (recommended) instead of Python 3.13
2. Have CUDA installed for flash-attention

## Step 1: Create a Conda Environment with Python 3.10

```bash
conda create -n infi-llama python=3.10
conda activate infi-llama
```

## Step 2: Install nanotron-infini and Dependencies

```bash
cd /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini
pip install -e .
pip install datasets transformers huggingface_hub pyarrow pandas
```

## Step 3: Prepare Your Dataset

The dataset is in parquet format and needs to be processed for the training:

```bash
python prepare_data.py
```

## Step 4: Configure Infini-Attention

Before running the training, we need to properly set up the Infini-Attention configuration:

```python
from dataclasses import dataclass, field
import sys
sys.path.append('./src')
from nanotron import constants

@dataclass
class InfiniAttentionConfig:
    segment_length: int = 64
    turn_on_memory: bool = True
    balance_init_type: str = 'zeros'
    balance_act_type: str = 'orig_sigmoid'

@dataclass
class Config:
    infini_attention: InfiniAttentionConfig = field(default_factory=InfiniAttentionConfig)

# Set up the configuration
constants.CONFIG = Config()
```

## Step 5: Start Training

For training without CUDA, you can use the CPU-only version:

```bash
export CUDA_VISIBLE_DEVICES=""  # Force CPU mode
python run_train.py --config-file custom_infini_config.yaml
```

If you have CUDA available:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc_per_node=1 run_train.py --config-file custom_infini_config.yaml
```

## Configuration Details

The custom_infini_config.yaml file contains:

- Model configuration (hidden size, layers, etc.)
- Infini-attention parameters (segment length, memory settings)
- Training parameters (batch size, sequence length, etc.)
- Dataset location and processing options

## Troubleshooting

1. **Python Version Issues**: The code appears to be designed for Python 3.10, not Python 3.13.
   
2. **CUDA Issues**: If you're getting CUDA-related errors but don't have a GPU, edit the config to remove CUDA dependencies.

3. **flash-attn Issues**: The flash-attention library requires CUDA. If you're on CPU-only, you'll need to modify the code to not use flash-attention.

## Alternative Approach

If you continue facing issues with the direct installation, you could:

1. Use Docker with a pre-configured PyTorch environment
2. Use Google Colab or other cloud services with GPU support
3. Modify the code to work without flash-attention (for CPU-only training)
