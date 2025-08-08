# Weight Decay Fix for Adam Optimizer

This document provides information about the fix for the `unsupported operand type(s) for *: 'float' and 'NoneType'` error in the Adam optimizer.

## Problem Description

The error occurs when the Adam optimizer attempts to apply weight decay to parameters while `weight_decay` is `None`. Specifically, the error happens in this line in PyTorch's Adam optimizer implementation:

```python
param.mul(1 - lr * weight_decay)
```

If `weight_decay` is `None`, then `lr * weight_decay` fails with a `TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'`.

### Root Cause Analysis

The issue happens because:

1. The Adam optimizer can receive `weight_decay=None` in either:
   - Named parameters (kwargs): `optimizer = Adam(model.parameters(), lr=0.001, weight_decay=None)`
   - Positional parameters (args): `optimizer = Adam(model.parameters(), 0.001, None)`

2. PyTorch's Adam implementation doesn't validate that weight_decay is a numeric value before performing operations with it.

3. In the optimization loop, it attempts to perform the `param.mul(1 - lr * weight_decay)` operation, which fails when weight_decay is None.

4. In PyTorch 2.x versions, this happens in the `_single_tensor_adam` function, specifically at the line:
   ```python
   param.mul_(1 - lr * weight_decay)  # Fails when weight_decay is None
   ```

5. This can happen when:
   - Config files explicitly set weight_decay to null/None
   - No weight_decay is specified and the optimizer uses None as default
   - Parameter settings are improperly passed between modules
   - Using PyTorch 2.x with older patching approaches that don't target the `_single_tensor_adam` function

## Solution

We implemented multiple fixes to ensure weight decay always has a proper numeric value:

1. **Dedicated Module at `src/nanotron/optim/fix_weight_decay.py`**: A proper module that can be imported in any project file:

```python
import logging
import torch

logger = logging.getLogger(__name__)

def patch_adam_optimizer():
    """
    Patch PyTorch's Adam optimizer to handle None weight_decay values.
    This prevents 'unsupported operand type(s) for *: 'float' and 'NoneType'' errors.
    """
    if not hasattr(torch.optim.adam, 'adam'):
        logger.warning("Cannot patch Adam optimizer: 'adam' function not found in torch.optim.adam")
        return False

    # Store original function
    original_adam = torch.optim.adam.adam

    def patched_adam(*args, **kwargs):
        """Patched adam function that handles None weight_decay."""
        # Check if weight_decay is None in kwargs and replace with 0.0
        if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
            logger.info("Fixed: Replaced None weight_decay with 0.0 in Adam optimizer")
            kwargs['weight_decay'] = 0.0
        
        # Handle positional args for weight_decay (usually 4th arg)
        if len(args) >= 4 and args[3] is None:
            logger.info("Fixed: Replaced None weight_decay in positional args with 0.0")
            args = list(args)
            args[3] = 0.0
            args = tuple(args)
        
        # Call original function
        return original_adam(*args, **kwargs)

    # Replace the original function with our patched version
    torch.optim.adam.adam = patched_adam
    logger.info("Successfully patched Adam optimizer to handle None weight_decay values")
    return True
```

2. **Simple Patch Script (`scripts/adam_optimizer_patch.py`)**: A minimalist script that can be imported to apply the patch:

```python
import torch

# Store original function if it exists
if hasattr(torch.optim.adam, 'adam'):
    original_adam = torch.optim.adam.adam
    
    def patched_adam(*args, **kwargs):
        # Check if weight_decay is None in kwargs and replace with 0.0
        if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
            print("Fixed: Replaced None weight_decay with 0.0 in Adam optimizer")
            kwargs['weight_decay'] = 0.0
        
        # Handle positional args for weight_decay (usually 4th arg)
        if len(args) >= 4 and args[3] is None:
            print("Fixed: Replaced None weight_decay in positional args with 0.0")
            args = list(args)
            args[3] = 0.0
            args = tuple(args)
        
        # Call original function
        return original_adam(*args, **kwargs)

    # Replace the original function with our patched version
    torch.optim.adam.adam = patched_adam
    print("Successfully patched Adam optimizer to handle None weight_decay values")
```

2. **Config File Validation**: Added code to the `flexible_training_workflow.sh` script to ensure the config file always has a valid `weight_decay` value:

```bash
# Check if weight_decay is missing or None in the config and add it
if ! grep -q "weight_decay:" "$CONFIG_TEMP" || grep -q "weight_decay: *null" "$CONFIG_TEMP"; then
    # Add or replace weight_decay with default value 0.01
    sed -i.bak '/optimizer:/,/zero_stage:/ s/\(weight_decay: *\)null/\10.01/' "$CONFIG_TEMP"
    if ! grep -q "weight_decay:" "$CONFIG_TEMP"; then
        # If weight_decay is completely missing, add it before zero_stage
        sed -i.bak '/optimizer:/,/zero_stage:/ s/\(zero_stage:.*\)/weight_decay: 0.01\n  \1/' "$CONFIG_TEMP"
    fi
    echo "Added default weight_decay: 0.01 to config to avoid NoneType errors in optimizer"
fi
```

3. **Shell Script for Applying the Patch (`scripts/apply_adam_patch.sh`)**: A convenient shell script to apply the patch:

```bash
#!/bin/bash
# Script to apply the Adam optimizer patch to fix weight_decay=None issues

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_SCRIPT="$PROJECT_ROOT/scripts/fix_adam_none_issue.py"

echo "Applying Adam optimizer patch from $PATCH_SCRIPT"

if [ ! -f "$PATCH_SCRIPT" ]; then
    echo "Error: Patch script not found at $PATCH_SCRIPT"
    exit 1
fi

# Apply the patch using Python
PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src python "$PATCH_SCRIPT"

if [ $? -eq 0 ]; then
    echo "Adam optimizer patch applied successfully"
else
    echo "Failed to apply Adam optimizer patch"
    exit 1
fi

echo "Weight decay issues in Adam optimizer should now be fixed"
```

4. **Direct Patch in Training Workflow**: Added direct patching in `flexible_training_workflow.sh`:

```bash
# Apply the Adam optimizer patch directly before training
echo "Applying Adam optimizer patch to fix weight_decay=None issue"
if [[ -f "$PROJECT_ROOT/scripts/apply_adam_patch.sh" ]]; then
    bash "$PROJECT_ROOT/scripts/apply_adam_patch.sh"
    echo "Adam patch script executed"
else
    echo "Creating temporary Adam patch"
    python -c "
import torch.optim.adam
original_adam = torch.optim.adam.adam
def patched_adam(*args, **kwargs):
    if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
        kwargs['weight_decay'] = 0.0
    if len(args) >= 4 and args[3] is None:
        args = list(args)
        args[3] = 0.0
        args = tuple(args)
    return original_adam(*args, **kwargs)
torch.optim.adam.adam = patched_adam
print('Applied temporary Adam patch')
"
fi
```

5. **Pre-import Script**: Created a `preimport.py` script that applies the patch before any other code runs:

```python
# This is imported before any other imports in the training script
import preimport  # This applies the patches automatically
```

6. **Wrapper Script Technique**: Created a dedicated wrapper script (`wrapper_script.py`) that:
   - Imports our patches first
   - Then runs the actual training script with all arguments passed through
   
   **Important Note:** The wrapper script must be saved as a file with a fixed path, not a temporary file. Using a temporary file can cause Python to fail with a "__main__ module" error because Python tries to interpret the script filename as a module name.

## Usage

### Automatic Fix with Flexible Training Workflow

The fix is automatically applied when running the `flexible_training_workflow.sh` script. The script:

1. Ensures the config file has a valid weight_decay value 
2. Uses a dedicated wrapper script (`scripts/wrapper_script.py`) that applies our patches
3. Applies the Adam patch directly via the `apply_adam_patch.sh` script or inline Python code
4. Runs the training script through this wrapper

No additional steps are required as the fix is now fully integrated into the workflow.

### Manual Fix Options

If you're running training manually or creating your own scripts, you can apply the fix in several ways:

#### Option 1: Import the Dedicated Module

```python
# At the top of your training script
from nanotron.optim.fix_weight_decay import patch_adam_optimizer

# At the beginning of your main function
patch_adam_optimizer()

# Then continue with your training code
```

#### Option 2: Run the Patch Script Before Training

```bash
# Apply the patch using the shell script
bash scripts/apply_adam_patch.sh

# Or run the Python patch directly
python scripts/fix_adam_none_issue.py

# Then run your training
python run_direct_training.py --config-file your_config.yaml
```

#### Option 3: Import the Simple Patch

```python
# At the top of your script
import sys
sys.path.append('scripts')
import adam_optimizer_patch

# Then continue with your training code
```

#### Option 4: Modify Your Config Files

Ensure your config files always specify a numeric value for weight_decay:

```yaml
optimizer:
  name: "adam" 
  lr: 0.001
  weight_decay: 0.0  # Use 0.0 instead of None/null if you don't want weight decay
```

## Troubleshooting

If you encounter issues with the wrapper script, you can test it using:

```bash
# Test the wrapper with a simple script
./scripts/test_wrapper.sh
```

Common issues:
- **`__main__ module` error**: This can happen if:
  - The wrapper script is created as a temporary file without a proper path
  - The wrapper script path is passed with extra quotes in the command
  - Solution: Use the permanent `wrapper_script.py` and ensure it's referenced correctly in commands (without extra quotes)
- **Import errors**: Ensure that the Python path includes the project root and script directories.
- **Permission denied**: Make sure the wrapper script has execute permissions with `chmod +x scripts/wrapper_script.py`.

## Note for Developers

When working with optimizer parameters, always ensure numerical parameters have default values or proper type checking to avoid NoneType errors.
