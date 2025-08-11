#!/bin/bash

# Single GPU training script that sets up proper distributed environment
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

echo "Starting single GPU training with distributed environment:"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "RANK=$RANK"
echo "LOCAL_RANK=$LOCAL_RANK"
echo ""

python run_train.py --config-file fineweb_local_200m_infini_config.yaml