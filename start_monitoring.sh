#!/bin/bash

# Quick launcher for monitoring passkey fine-tuning progress
set -e

echo "=========================================================="
echo "STARTING PASSKEY FINE-TUNING MONITOR"
echo "=========================================================="

CHECKPOINT_DIR="${1:-./checkpoints/passkey_finetune_300m_optimized}"
INTERVAL="${2:-15}"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Creating checkpoint directory: $CHECKPOINT_DIR"
    mkdir -p "$CHECKPOINT_DIR"
fi

echo "Monitoring directory: $CHECKPOINT_DIR"
echo "Check interval: $INTERVAL minutes"
echo ""
echo "This will monitor for new checkpoints and test passkey retrieval..."
echo "Press Ctrl+C to stop monitoring"
echo ""

# Start monitoring
MASTER_ADDR=localhost MASTER_PORT=29500 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
python monitor_passkey_finetuning.py \
    --training-dir "$CHECKPOINT_DIR" \
    --interval "$INTERVAL"
