#!/bin/bash

depth_percent=${1:-90}

# Print out the depth percent for debugging/logging
echo "Running evaluation with depth percent: $depth_percent"

# Run the torchrun command with the calculated depth_percent
CUDA_VISIBLE_DEVICES=6 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=1 examples/infinite-context-length/run_evals.py \
    --ckpt-path /data1/infini-attn/infini-llama/nanotron-infini/checkpoints/fineweb_4gpu_300m_infini/15000 \
    --context_length 16384 \
    --depth_percent $depth_percent \
    --num_shots 3 \
    --num_digits 3 \
    --dp 1 \
    --pp 1 \
    --tp 1