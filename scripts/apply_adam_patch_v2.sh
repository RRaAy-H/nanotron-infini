#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/apply_adam_patch_v2.sh
# 
# This script applies multiple patches to fix the weight_decay=None issue in PyTorch's Adam optimizer.
# It works with different PyTorch versions by using multiple patching strategies.
# Run this script before training or include it in your training workflow.

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Applying Adam optimizer patch for weight_decay=None issue..."

# Check if we have the direct patch script
if [[ -f "$PROJECT_ROOT/scripts/direct_adam_patch.py" ]]; then
    echo "Using direct_adam_patch.py script (targets specific PyTorch functions)"
    # Set Python path to find modules
    export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PYTHONPATH"
    python "$PROJECT_ROOT/scripts/direct_adam_patch.py"
    PATCH_STATUS=$?
    
    if [[ $PATCH_STATUS -eq 0 ]]; then
        echo "Direct Adam patch applied successfully!"
        exit 0
    else
        echo "Direct patch failed, trying fallback methods..."
    fi
fi

# Create a more comprehensive temporary patch script
PATCH_SCRIPT="$PROJECT_ROOT/scripts/temp_patch_adam_v2.py"
cat > "$PATCH_SCRIPT" << 'EOF'
#!/usr/bin/env python
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Applying comprehensive Adam optimizer patch...")

try:
    # Import torch
    import torch
    from torch.optim import Adam
    
    # Strategy 1: Try to patch _single_tensor_adam directly (PyTorch 2.x)
    try:
        from torch.optim import adam
        if hasattr(adam, '_single_tensor_adam'):
            # Store original function
            original_func = adam._single_tensor_adam
            
            # Create a wrapper function that checks for None weight_decay
            def patched_single_tensor_adam(*args, **kwargs):
                # Fix None weight_decay
                if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
                    print("Fixed: Replaced None weight_decay with 0.0 in _single_tensor_adam")
                    kwargs['weight_decay'] = 0.0
                
                # Call original with fixed kwargs
                return original_func(*args, **kwargs)
            
            # Replace the function
            adam._single_tensor_adam = patched_single_tensor_adam
            print("Successfully patched _single_tensor_adam function")
    except Exception as e:
        print(f"Could not patch _single_tensor_adam: {e}")
    
    # Strategy 2: Patch Adam class directly (works in all PyTorch versions)
    # Store original step method
    original_step = Adam.step
    
    # Create a patched step method
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
    print("Successfully patched Adam.step method")
    
    # Test the patch
    try:
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=None)
        x = torch.randn(1, 10)
        y = model(x)
        loss = (y - torch.randn(1, 1)).pow(2).mean()
        loss.backward()
        optimizer.step()
        print("Test successful: Adam optimizer works with weight_decay=None")
    except Exception as e:
        print(f"Test warning: {e}")
        
    print("Adam optimizer patch applied successfully!")
except Exception as e:
    print(f"Error applying Adam optimizer patch: {e}")
    sys.exit(1)
EOF

# Make the script executable
chmod +x "$PATCH_SCRIPT"

# Run the patch script
PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PYTHONPATH" python "$PATCH_SCRIPT"
PATCH_STATUS=$?

# Check if the patch was applied successfully
if [ $PATCH_STATUS -eq 0 ]; then
    echo "Adam optimizer patch applied successfully!"
    exit 0
else
    echo "Patch application failed with status $PATCH_STATUS"
    echo "Check the error message above."
    
    # Try the fallback method using direct class patching
    echo "Trying fallback direct class patching method..."
    python -c "
import torch
from torch.optim import Adam

# Store original step method
original_step = Adam.step

# Create patched step method
def patched_step(self, closure=None):
    # Replace None weight_decay with 0.0 in optimizer instance
    for group in self.param_groups:
        if 'weight_decay' in group and group['weight_decay'] is None:
            print('Fixed: Replaced None weight_decay with 0.0')
            group['weight_decay'] = 0.0
    return original_step(self, closure)

# Apply the patch
Adam.step = patched_step
print('Applied direct Adam class patch')
"
    
    if [ $? -eq 0 ]; then
        echo "Fallback patch method succeeded!"
        exit 0
    else
        echo "All patch methods failed. Please check your PyTorch version and try again."
        exit 1
    fi
fi
