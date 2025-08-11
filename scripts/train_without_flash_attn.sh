#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/train_without_flash_attn.sh
#
# Wrapper script to run Infini-Llama training without Flash Attention
# This script sets the necessary environment variables and flags
# to disable Flash Attention before running the training workflow.
#
# Use this script to avoid GLIBC_2.32 not found errors and other
# Flash Attention compatibility issues that can occur on systems
# with older GLIBC versions or incompatible CUDA installations.

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Disable Flash Attention by setting the environment variable
export DISABLE_FLASH_ATTN=1

# Print banner
echo "======================================================"
echo "Running Infini-Llama training without Flash Attention"
echo "======================================================"
echo "This script will disable Flash Attention and use the standard"
echo "attention implementation for training."
echo ""
echo "This avoids common issues such as:"
echo " - GLIBC_2.32 not found errors"
echo " - CUDA version compatibility problems"
echo " - Other Flash Attention import or runtime errors"
echo ""
echo "Note: Training may be slower but will use less GPU memory"
echo "      and have better compatibility."
echo ""
echo "Command: $SCRIPT_DIR/flexible_training_workflow.sh $@ --disable-flash-attn"
echo "======================================================"

# Execute the flexible training workflow with all passed arguments plus disable-flash-attn
exec "$SCRIPT_DIR/flexible_training_workflow.sh" "$@" --disable-flash-attn
