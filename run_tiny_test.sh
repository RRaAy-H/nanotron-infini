#!/bin/bash

# Tiny Dataset Test Script
# This script runs a minimal training test with a very small dataset and model
# to verify the entire training pipeline works correctly.

echo "=========================================="
echo "Starting Tiny Dataset Training Test"
echo "=========================================="
echo "This will:"
echo "- Use a very small model (64 hidden size, 2 layers)"
echo "- Train for only 20 steps"
echo "- Use a tiny test dataset"
echo "- Test the complete training pipeline"
echo ""

# Set up environment for single GPU
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# Set CUDA connections (important for distributed operations)
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
python run_train.py --config-file debug_tiny_test_config.yaml

echo ""
echo "=========================================="
echo "Tiny test completed!"
echo "=========================================="