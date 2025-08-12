#!/bin/bash

export MASTER_ADDR=localhost
export MASTER_PORT=29503
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Starting 4-GPU distributed training with fineweb data (Standard Attention):"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT" 
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    run_train.py --config-file baseline_config.yaml