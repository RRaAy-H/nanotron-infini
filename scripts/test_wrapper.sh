#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/test_wrapper.sh

# Test the wrapper script with a simple Python script
# This helps verify that our Adam optimizer patches work properly
# and that the wrapper script correctly executes Python scripts

set -e  # Exit on error

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Set up Python path
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PYTHONPATH"

# Set up environment variables similar to the training workflow
export CUDA_VISIBLE_DEVICES=0
export TRAINING_LOGS_DIR="$PROJECT_ROOT/training_logs/test_$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$TRAINING_LOGS_DIR"

echo "==== Testing Wrapper Script ===="
echo "Project root: $PROJECT_ROOT"
echo "Python path: $PYTHONPATH"
echo "Training logs dir: $TRAINING_LOGS_DIR"

# First, run the test script directly
echo -e "\n1. Running test script directly:"
python "$PROJECT_ROOT/scripts/test_wrapper.py" --arg1 value1 --arg2 value2

# Now, run the test script through the wrapper
echo -e "\n2. Running test script through wrapper:"

# Temporarily modify the wrapper script to use our test script
ORIGINAL_SCRIPT="$PROJECT_ROOT/scripts/wrapper_script.py"
BACKUP_SCRIPT="$PROJECT_ROOT/scripts/wrapper_script.py.bak"

# Backup the original wrapper script
cp "$ORIGINAL_SCRIPT" "$BACKUP_SCRIPT"

# Modify the wrapper script to use our test script
sed -i.bak "s|scripts/run_direct_training.py|scripts/test_wrapper.py|g" "$ORIGINAL_SCRIPT"

# Run the wrapper script
python "$PROJECT_ROOT/scripts/wrapper_script.py" --arg1 value1 --arg2 value2

# Restore the original wrapper script
mv "$BACKUP_SCRIPT" "$ORIGINAL_SCRIPT"

echo -e "\nWrapper script test completed!"
