#!/bin/bash

# Multi-GPU training script for 4x RTX 4090 with optimized config
# Uses GPUs 0,1,2,3 with data parallelism for maximum memory utilization

export MASTER_ADDR=localhost
export MASTER_PORT=29502
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Starting 4-GPU distributed training with fineweb data:"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT" 
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Using 4x RTX 4090 (24GB each) with optimized batch sizes"
echo ""

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    run_train.py --config-file fineweb_local_200m_infini_4gpu_config.yaml