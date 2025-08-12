#!/bin/bash

echo "=========================================="
echo "Starting Tiny FineWeb Model Test"
echo "=========================================="
echo "This will:"
echo "- Use the same 200M parameter model from FineWeb config"
echo "- Train for only 20 steps"
echo "- Use a tiny test dataset instead of FineWeb"
echo "- Test the complete training pipeline with larger model"
echo ""

# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "Environment setup:"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "RANK=$RANK"
echo "LOCAL_RANK=$LOCAL_RANK"
echo "CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS"
echo ""

echo "Starting training..."
torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    run_train.py \
    --config-file debug_tiny_fineweb_config.yaml

echo ""
echo "=========================================="
echo "Tiny FineWeb model test completed!"
echo "=========================================="