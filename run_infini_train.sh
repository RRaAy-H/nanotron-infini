#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/run_infini_train.sh
#
# This script runs the Infini-Llama training with proper Python path setup
# and handles importing issues.

# Change to project root directory
cd "$(dirname "$0")"
ROOT_DIR="$(pwd)"

# Set Python path to include project root and src directory
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:$PYTHONPATH"

# Set up the configuration with the right Infini-Attention parameters
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

# Run the DistributedTrainer directly from the Python script
echo "Running Infini-Llama training with DistributedTrainer..."
python ./scripts/flexible_training_workflow.sh "$@"
