#!/bin/bash
# This script prepares a checkpoint for finetuning by removing the training metadata
# This allows us to load just the model weights without the training state

set -e

CHECKPOINT_PATH="${1:-./checkpoints/fineweb_4gpu_300m_infini/30000}"
OUTPUT_PATH="${2:-./checkpoints/fineweb_4gpu_300m_infini_weights_only}"

echo "=========================================================="
echo "PREPARING CHECKPOINT FOR FINETUNING"
echo "=========================================================="
echo "Source checkpoint: $CHECKPOINT_PATH"
echo "Output checkpoint: $OUTPUT_PATH"
echo ""

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

# Copy all checkpoint files except metadata
echo "Copying checkpoint files..."
cp -r "$CHECKPOINT_PATH"/* "$OUTPUT_PATH/" 2>/dev/null || true

# Remove the metadata file that contains training state
echo "Removing training metadata..."
rm -f "$OUTPUT_PATH/metadata.json"

echo "Resetting step count in config.yaml..."
if [ -f "$OUTPUT_PATH/config.yaml" ]; then
    # Reset step and consumed_train_samples to null
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' 's/step: [0-9]*/step: null/g' "$OUTPUT_PATH/config.yaml"
        sed -i '' 's/consumed_train_samples: [0-9]*/consumed_train_samples: null/g' "$OUTPUT_PATH/config.yaml"
    else
        # Linux
        sed -i 's/step: [0-9]*/step: null/g' "$OUTPUT_PATH/config.yaml"
        sed -i 's/consumed_train_samples: [0-9]*/consumed_train_samples: null/g' "$OUTPUT_PATH/config.yaml"
    fi
fi

# Also remove optimizer and lr_scheduler states to save space
echo "Removing optimizer and lr_scheduler states (keeping only model weights)..."
rm -rf "$OUTPUT_PATH/optimizer" 2>/dev/null || true
rm -f "$OUTPUT_PATH/lr_scheduler.pt" 2>/dev/null || true

echo ""
echo "=========================================================="
echo "CHECKPOINT PREPARED SUCCESSFULLY!"
echo "=========================================================="
echo "The checkpoint at $OUTPUT_PATH now contains only model weights."