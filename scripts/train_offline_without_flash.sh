#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/train_offline_without_flash.sh
#
# Wrapper script to run Infini-Llama training in offline mode and without Flash Attention
# This script sets the necessary environment variables and flags
# to enable offline mode and disable Flash Attention before running the training workflow.
#
# Use this script when you have network connectivity issues and Flash Attention compatibility issues.

# Set the offline mode flag and disable flash attention flag
FLAGS="--offline-mode --disable-flash-attn"

# Pass all arguments to the flexible_training_workflow.sh script
# and add our flags
"$(dirname "$0")/flexible_training_workflow.sh" $FLAGS "$@"
