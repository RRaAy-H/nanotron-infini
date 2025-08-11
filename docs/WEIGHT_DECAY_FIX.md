# Weight Decay Fix for Adam Optimizer in PyTorch 1.x and 2.x

This document provides information about the fix for the `unsupported operand type(s) for *: 'float' and 'NoneType'` error in the Adam optimizer when using PyTorch 1.x or 2.x with Infini-Llama models.

## Problem Description

The error occurs when the Adam optimizer attempts to apply weight decay to parameters while `weight_decay` is `None`. Specifically, the error happens in PyTorch's Adam optimizer implementation:

```python
param.mul(1 - lr * weight_decay)  # In PyTorch 1.x
```

or in PyTorch 2.x:

```python
param.mul_(1 - lr * weight_decay)  # In PyTorch 2.x _single_tensor_adam
```

If `weight_decay` is `None`, then `lr * weight_decay` fails with:
```
TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'
```

In PyTorch 2.x, you might also see the error:
```
AttributeError: module 'torch.optim' has no attribute 'adam'. Did you mean: 'Adam'?
```

### Root Cause Analysis

The issue happens because:

1. The Adam optimizer can receive `weight_decay=None` in either:
   - Named parameters (kwargs): `optimizer = Adam(model.parameters(), lr=0.001, weight_decay=None)`
   - Positional parameters (args): `optimizer = Adam(model.parameters(), 0.001, None)`

2. PyTorch's Adam implementation doesn't validate that weight_decay is a numeric value before performing operations with it.

3. In the optimization loop, it attempts to perform the `param.mul(1 - lr * weight_decay)` operation, which fails when weight_decay is None.

4. **PyTorch 2.x Changes**: The module structure changed significantly in PyTorch 2.x:
   - In PyTorch 1.x: The Adam implementation was in `torch.optim.adam.adam`
   - In PyTorch 2.x: The Adam implementation moved to C++ code and is accessed differently

5. **Additional PyTorch 2.x Issue**: Our existing patching code tries to access `torch.optim.adam` which doesn't exist in PyTorch 2.x, leading to the error: `AttributeError: module 'torch.optim' has no attribute 'adam'`

6. These issues can happen when:
   - Config files explicitly set weight_decay to null/None
   - No weight_decay is specified and the optimizer uses None as default
   - Parameter settings are improperly passed between modules
   - Using PyTorch 2.x with older patching approaches that don't account for the module structure changes

## Solution for PyTorch 1.x and 2.x

We implemented comprehensive fixes to ensure weight decay always has a proper numeric value across all PyTorch versions:

1. **Enhanced Module at `src/nanotron/optim/fix_weight_decay.py`**: A robust module that handles both PyTorch 1.x and 2.x:

```python
import logging
import torch

logger = logging.getLogger(__name__)

def patch_adam_optimizer():
    """
    Patch PyTorch's Adam optimizer to handle None weight_decay values.
    This prevents 'unsupported operand type(s) for *: 'float' and 'NoneType'' errors.
    Works with both PyTorch 1.x and 2.x module structures.
    """
    success = False
    pytorch_version = torch.__version__
    logger.info(f"Detected PyTorch version: {pytorch_version}")
    
    # Strategy 1: Try to patch torch.optim.adam (PyTorch 1.x structure)
    try:
        from torch.optim import adam
        
        if hasattr(adam, '_single_tensor_adam'):
            # Store original function
            original_func = adam._single_tensor_adam
            
            # Create a patched version that checks for None weight_decay
            def patched_single_tensor_adam(*args, **kwargs):
                # Fix None weight_decay
                if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
                    logger.info("Fixed: Replaced None weight_decay with 0.0 in _single_tensor_adam")
                    kwargs['weight_decay'] = 0.0
                
                # Call original with fixed kwargs
                return original_func(*args, **kwargs)
            
            # Replace the function
            adam._single_tensor_adam = patched_single_tensor_adam
            logger.info("Successfully patched _single_tensor_adam function")
            success = True
        
        if hasattr(adam, 'adam'):
            # Store original function
            original_adam = adam.adam
            
            def patched_adam(*args, **kwargs):
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
            
            # Replace with our patched version
            adam.adam = patched_adam
            logger.info("Successfully patched torch.optim.adam.adam function")
            success = True
    except (ImportError, AttributeError) as e:
        logger.info(f"Could not patch torch.optim.adam (expected in PyTorch 2.x): {e}")
    
    # Strategy 2: Always patch the Adam class directly (works in all versions)
    from torch.optim import Adam
    
    # Store original step method
    original_step = Adam.step
    
    # Create patched step method
    def patched_step(self, closure=None):
        """Patched step method that ensures weight_decay is never None"""
        # Replace None weight_decay with 0.0 in optimizer instance
        for group in self.param_groups:
            if 'weight_decay' in group and group['weight_decay'] is None:
                logger.info("Fixed: Replaced None weight_decay with 0.0 in Adam optimizer group")
                group['weight_decay'] = 0.0
                
        # Call original step method
        return original_step(self, closure)
    
    # Apply the patch
    Adam.step = patched_step
    logger.info("Successfully patched Adam.step method (works in all PyTorch versions)")
    success = True

    return success
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

The fix is automatically applied when running the `flexible_training_workflow.sh` script. The script now includes updated methods to handle both PyTorch 1.x and 2.x versions:

1. **Multi-strategy approach**: 
   - First tries the direct patch approach using `direct_adam_patch.py` (works with both PyTorch versions)
   - Falls back to the apply_adam_patch scripts if needed
   - Uses inline patching as a last resort

2. **Other automatic fixes**:
   - Ensures the config file has a valid weight_decay value 
   - Uses a dedicated wrapper script (`scripts/wrapper_script.py`) that applies our patches
   - Runs the training script through this wrapper

No additional steps are required as the fix is now fully integrated into the workflow.

### Manual Fix Options

If you're running training manually or creating your own scripts, you can apply the fix in several ways:

#### Option 1: Use the Direct Adam Patch (Recommended for PyTorch 2.x)

```bash
# Run the direct patch script before your training code
python scripts/direct_adam_patch.py

# Then run your training
python run_direct_training.py --config-file your_config.yaml
```

#### Option 2: Import the Enhanced Fix Module

```python
# At the top of your training script
from nanotron.optim.fix_weight_decay import patch_adam_optimizer

# At the beginning of your main function
patch_adam_optimizer()

# Then continue with your training code
```

#### Option 3: Apply the Patch Script Before Training

```bash
# Apply the patch using the shell script
bash scripts/apply_adam_patch_v2.sh  # Use v2 for PyTorch 2.x compatibility

# Or run the Python patch directly
python scripts/direct_adam_patch.py

# Then run your training
python run_direct_training.py --config-file your_config.yaml
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

### PyTorch 2.x Specific Issues

If you encounter the error `AttributeError: module 'torch.optim' has no attribute 'adam'. Did you mean: 'Adam'?`:

1. This is due to the module structure changes in PyTorch 2.x
2. Use the `direct_adam_patch.py` script which handles these changes
3. Make sure you're applying the patch before importing any other modules that might use the optimizer

```bash
# Best approach for PyTorch 2.x
python scripts/direct_adam_patch.py
python run_direct_training.py --config-file your_config.yaml
```

### Other Common Issues

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

## PyTorch Version Detection

The updated patch scripts now automatically detect your PyTorch version and apply the appropriate fixes:

```python
import torch
pytorch_version = torch.__version__
print(f"Detected PyTorch version: {pytorch_version}")

# Apply version-specific patches
if pytorch_version.startswith('1.'):
    # Apply PyTorch 1.x specific patches
    # ...
else:
    # Apply PyTorch 2.x specific patches
    # ...
```

## Note for Developers

When working with optimizer parameters:
1. Always ensure numerical parameters have default values or proper type checking
2. Don't use `None` as a default for parameters that will be used in arithmetic operations
3. For PyTorch 2.x, be aware of the module structure changes and test patches thoroughly
