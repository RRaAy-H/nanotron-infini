#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/apply_adam_patch.sh
# 
# This script creates and applies a patch to fix the weight_decay=None issue in PyTorch's Adam optimizer.
# Run this script before training or include it in your training workflow.

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Create a temporary patch script
PATCH_SCRIPT="$PROJECT_ROOT/scripts/temp_patch_adam.py"
cat > "$PATCH_SCRIPT" << 'EOF'
import torch.optim.adam

# Store original function
original_adam = torch.optim.adam.adam

def patched_adam(*args, **kwargs):
    # Check if weight_decay is None in kwargs and replace with 0.0
    if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
        print("Patched: Replaced None weight_decay with 0.0")
        kwargs['weight_decay'] = 0.0
    
    # Handle positional args for weight_decay (usually 4th arg)
    if len(args) >= 4 and args[3] is None:
        print("Patched: Replaced None weight_decay in positional args with 0.0")
        args = list(args)
        args[3] = 0.0
        args = tuple(args)
    
    # Call original function
    return original_adam(*args, **kwargs)

# Replace the original function with our patched version
torch.optim.adam.adam = patched_adam
print("Adam optimizer patch applied successfully")

# Test the patch
try:
    import torch
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=None)
    x = torch.randn(1, 10)
    y = model(x)
    loss = (y - torch.randn(1, 1)).pow(2).mean()
    loss.backward()
    optimizer.step()
    print("Test successful: Adam optimizer works with weight_decay=None")
except Exception as e:
    print(f"Test failed: {e}")
EOF

# Make the script executable
chmod +x "$PATCH_SCRIPT"

# Run the patch script
echo "Applying Adam optimizer patch..."
python "$PATCH_SCRIPT"
PATCH_STATUS=$?

# Check if the patch was applied successfully
if [ $PATCH_STATUS -eq 0 ]; then
    echo "Patch applied successfully."
    echo "You can now run your training script."
else
    echo "Patch application failed with status $PATCH_STATUS."
    echo "Check the error message above."
fi
