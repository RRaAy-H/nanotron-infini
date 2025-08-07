#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/train_infini_llama.sh

cd /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini

# Install the package in development mode if not already installed
pip install -e .

# Install necessary dependencies
pip install flash-attn>=2.5.0 datasets transformers huggingface_hub pyarrow pandas

# Prepare the dataset
echo "Preparing dataset..."
python prepare_data.py

# Configure constants
python -c "
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
    balance_factor_lr: float = 0.001  # Added missing required parameter
    logging: bool = False  # Added missing required parameter
    logging_interval: int = 100  # Added missing required parameter
    log_grad: bool = False  # Added missing required parameter 
    log_segment_acts: bool = False  # Added missing required parameter

@dataclass
class Config:
    infini_attention: InfiniAttentionConfig = field(default_factory=InfiniAttentionConfig)

# Set up the configuration
constants.CONFIG = Config()
print('Infini attention constants configured successfully!')
print(f'Segment length: {constants.CONFIG.infini_attention.segment_length}')
print(f'Turn on memory: {constants.CONFIG.infini_attention.turn_on_memory}')
print(f'Balance init type: {constants.CONFIG.infini_attention.balance_init_type}')
print(f'Balance act type: {constants.CONFIG.infini_attention.balance_act_type}')
print(f'Balance factor lr: {constants.CONFIG.infini_attention.balance_factor_lr}')
print(f'Logging: {constants.CONFIG.infini_attention.logging}')
print(f'Logging interval: {constants.CONFIG.infini_attention.logging_interval}')
print(f'Log grad: {constants.CONFIG.infini_attention.log_grad}')
print(f'Log segment acts: {constants.CONFIG.infini_attention.log_segment_acts}')
"

# Set environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Run training
torchrun --nproc_per_node=1 run_train.py --config-file custom_infini_config.yaml
