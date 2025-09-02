#!/bin/bash

depth_percent=${1:-0}

# Print out the depth percent for debugging/logging
echo "Running evaluation with depth percent: $depth_percent"

# Run the torchrun command with the calculated depth_percent
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=1 /data1/infini-attn/baseline/nanotron-infini/examples/infinite-context-length/run_evals.py \
    --ckpt-path /data1/infini-attn/baseline/nanotron-infini/checkpoints/fineweb_4gpu_200m_standard_test/30000 \
    --context_length 4096 \
    --depth_percent $depth_percent \
    --num_shots 0 \
    --num_digits 3 \
    --dp 1 \
    --pp 1 \
    --tp 1
