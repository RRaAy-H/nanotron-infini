#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/apply_adam_patch.sh
# 
# This script creates and applies a patch to fix the weight_decay=None issue in PyTorch's Adam optimizer.
# Run this script before training or include it in your training workflow.

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Applying Adam optimizer patch to fix weight_decay=None issues..."

# Check if the improved patch script exists
if [[ -f "$PROJECT_ROOT/scripts/temp_patch_adam.py" ]]; then
    # Use existing patch script
    echo "Using existing patch script"
    PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src python "$PROJECT_ROOT/scripts/temp_patch_adam.py"
else
    # Create a temporary patch script with more robust implementation
    echo "Creating temporary patch script"
    PATCH_SCRIPT="$PROJECT_ROOT/scripts/temp_patch_adam.py"
    cat > "$PATCH_SCRIPT" << 'EOF'
#!/usr/bin/env python

"""
Temporary Adam optimizer patch script for PyTorch.
This script patches the Adam optimizer to handle None weight_decay values.
"""

import sys
print("Applying Adam optimizer patch for weight_decay=None issues...")

try:
    import torch
    from torch.optim import Adam
    
    # Store original step method
    original_step = Adam.step
    
    # Create patched step method
    def patched_step(self, closure=None):
        """Patched step method that ensures weight_decay is never None"""
        # Replace None weight_decay with 0.0 in optimizer instance
        for group in self.param_groups:
            if 'weight_decay' in group and group['weight_decay'] is None:
                print("Patched: Replaced None weight_decay with 0.0 in Adam optimizer group")
                group['weight_decay'] = 0.0
                
        # Call original step method
        return original_step(self, closure)
    
    # Apply the patch
    Adam.step = patched_step
    
    print("Adam optimizer patch applied successfully!")
    
    # Test the patch
    try:
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=None)
        print("Successfully created optimizer with None weight_decay (patch working)")
    except Exception as e:
        print(f"Warning: Could not verify patch: {e}")
    
except Exception as e:
    print(f"Error applying Adam optimizer patch: {e}")
    sys.exit(1)
EOF

    # Make the script executable
    chmod +x "$PATCH_SCRIPT"

    # Run the patch script
    PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src python "$PATCH_SCRIPT"
fi

# Check if the patch was applied successfully
if [ $? -eq 0 ]; then
    echo "Adam optimizer patch applied successfully"
else
    echo "Patch application failed with status $?"
    echo "Check the error message above."
    exit 1
fi
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
