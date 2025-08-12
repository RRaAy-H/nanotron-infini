#!/bin/bash

# Single GPU training script for test config with fineweb 1% data (avoid port conflict)
export MASTER_ADDR=localhost
export MASTER_PORT=29501
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

echo "Starting single GPU test training with fineweb 1% data:"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "RANK=$RANK"
echo "LOCAL_RANK=$LOCAL_RANK"
echo ""

python run_train.py --config-file fineweb_local_200m_infini_test_config.yaml