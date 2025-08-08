# Flexible Training Workflow Guide

This guide explains how to use the `flexible_training_workflow.sh` script for training Infini-Llama models.

## Prerequisites

Before running the training script, ensure:

1. The Python environment is properly set up
2. The wrapper script exists at `scripts/wrapper_script.py`
3. You have either raw data to preprocess or already preprocessed data

## Basic Usage

```bash
./scripts/flexible_training_workflow.sh --preprocessed-data <PATH_TO_PREPROCESSED_DATA> --config-file <PATH_TO_CONFIG_FILE>
```

## Common Use Cases

### Training with Raw Data (Will Preprocess First)

```bash
./scripts/flexible_training_workflow.sh \
  --raw-data /path/to/raw/data \
  --config-file scripts/config/tiny_test_config.yaml \
  --output-dir my_preprocessed_data
```

### Training with Already Preprocessed Data

```bash
./scripts/flexible_training_workflow.sh \
  --preprocessed-data preprocessed_data/preprocessed_YYYYMMDD_HHMMSS \
  --config-file scripts/config/tiny_test_config.yaml
```

### Training the Baseline Model (Without Infini-Attention)

```bash
./scripts/flexible_training_workflow.sh \
  --preprocessed-data preprocessed_data/preprocessed_YYYYMMDD_HHMMSS \
  --config-file scripts/config/tiny_test_config.yaml \
  --disable-infini-attn
```

### Specifying a GPU

```bash
./scripts/flexible_training_workflow.sh \
  --preprocessed-data preprocessed_data/preprocessed_YYYYMMDD_HHMMSS \
  --config-file scripts/config/tiny_test_config.yaml \
  --gpu 1
```

### Running Both Models in Parallel (Requires 2+ GPUs)

```bash
./scripts/flexible_training_workflow.sh \
  --preprocessed-data preprocessed_data/preprocessed_YYYYMMDD_HHMMSS \
  --config-file scripts/config/tiny_test_config.yaml \
  --run-both-models
```

## Additional Options

- `--tensorboard-dir <PATH>`: Specify TensorBoard logs directory
- `--verbose`: Enable detailed logging
- `--no-gpu-dataloader`: Disable GPU-accelerated dataloader
- `--force-preprocess`: Force preprocessing even if data exists
- `--enable-fused-adam`: Enable fused Adam optimizer (disabled by default)

## Troubleshooting

### Adam Optimizer Weight Decay Error

If you encounter the following error during training:
```
TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'
```

This error occurs in PyTorch's Adam optimizer implementation when `weight_decay` is set to `None`. The error happens during the operation `param.mul(1 - lr * weight_decay)` in the optimizer step. Here are several ways to fix it:

#### Solution 1: Edit your config file directly

Make sure your config file has a valid `weight_decay` value (explicitly set to a number instead of None/null):
```yaml
optimizer:
  name: "adam"
  lr: 0.001
  weight_decay: 0.01  # Set this to a float value (0.0 if you don't want weight decay)
```

#### Solution 2: Use the patched wrapper script

The flexible training workflow automatically applies a patch via the wrapper script that handles the None weight_decay issue:
```bash
# The script should be using this wrapper by default
ls -l scripts/wrapper_script.py

# If you need to run training directly without the flexible_training_workflow.sh script:
python scripts/wrapper_script.py --config-file <YOUR_CONFIG> --data-dir <YOUR_DATA>
```

#### Solution 3: Apply manual patch before training

We have multiple pre-built patch scripts to fix the weight decay issue:

1. Using the dedicated patch script:
```bash
# Run the Adam patch script directly
bash scripts/apply_adam_patch.sh
```

2. Using the Python import method:
```bash
# Import the patch module directly at the start of your training script
python -c "import sys; sys.path.append('scripts'); import adam_optimizer_patch"

# Then run your training
python run_direct_training.py --config-file <YOUR_CONFIG> --data-dir <YOUR_DATA>
```

For more detailed information about this issue and its fixes, see the comprehensive guide at [docs/WEIGHT_DECAY_FIX.md](WEIGHT_DECAY_FIX.md).

3. Create a custom patch script if needed:
```bash
# Create a simple patch script
cat > scripts/fix_weight_decay_custom.py << EOF
#!/usr/bin/env python
import torch

# Store original function
if hasattr(torch.optim.adam, 'adam'):
    original_adam = torch.optim.adam.adam
    
    def patched_adam(*args, **kwargs):
        if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
            print("Fixing None weight_decay in Adam optimizer")
            kwargs['weight_decay'] = 0.0
        
        # Also handle positional args (weight_decay is typically the 4th arg)
        if len(args) >= 4 and args[3] is None:
            args = list(args)
            args[3] = 0.0
            args = tuple(args)
            
        return original_adam(*args, **kwargs)
    
    # Apply the patch
    torch.optim.adam.adam = patched_adam
    print("Adam optimizer patched successfully")
EOF

# Apply the patch before running training
PYTHONPATH=$PWD:$PWD/src python scripts/fix_weight_decay_custom.py
```

#### Solution 4: Use the proper nanotron module

We've added a proper module to fix this issue permanently in your codebase:

```python
# In your training script, add this import at the top
from nanotron.optim.fix_weight_decay import patch_adam_optimizer

# Call the patch function before creating your model/optimizer
patch_adam_optimizer()
```

### Other Common Issues and Troubleshooting

#### `__main__ module` Error

If you see an error like: `RuntimeError: __main__ module not found` or `No module named '__main__'`, this is often related to how the wrapper script is being executed:

```
Traceback (most recent call last):
  File "/tmp/wrapper_script_xyz.py", line XXX, in <module>
    runpy.run_path(script_path, run_name="__main__")
  File "/usr/lib/python3.X/runpy.py", line XXX, in run_path
    return _run_module_code(code, init_globals, run_name, path_name)
RuntimeError: __main__ module not found in script
```

**Solutions:**
1. Ensure the wrapper script is saved as a permanent file, not a temporary file
2. Check that quotes are handled correctly in commands (avoid extra quotes)
3. Ensure the script has proper permissions

#### Weight Decay Error Despite Patching

If you still see the weight decay error despite applying the patches:

1. **Check if the patch is actually being applied**:
   ```bash
   # Add this temporarily to your script
   python -c "import torch.optim.adam; print('Current Adam implementation:', torch.optim.adam.adam)"
   ```

2. **Verify that multiple Python processes aren't conflicting**:
   - Sometimes distributed training can launch separate processes that don't inherit the patch
   - Apply the patch in the entry point of each process

3. **Check for any custom optimizer implementations**:
   - Custom optimizers might need their own patches

#### Python Path Issues

```bash
# Check your current Python path
echo $PYTHONPATH

# Set it correctly if needed
export PYTHONPATH="$PWD:$PWD/src:$PYTHONPATH"

# Or use the -m flag to ensure proper module resolution
python -m scripts.wrapper_script --config-file config.yaml
```

#### Permission Issues

Make sure all scripts have execute permissions:

```bash
chmod +x scripts/wrapper_script.py
chmod +x scripts/flexible_training_workflow.sh
chmod +x scripts/apply_adam_patch.sh
```

#### Debugging Your Setup

For advanced debugging:

```bash
# Run with verbose output
./scripts/flexible_training_workflow.sh --verbose --preprocessed-data data/processed --config-file config.yaml

# Test only the wrapper without training
python scripts/wrapper_script.py --test

# Validate that the Adam patch works
python -c "
import torch
import sys
sys.path.append('scripts')
import adam_optimizer_patch
# Create optimizer with None weight_decay to test patch
model = torch.nn.Linear(10, 10)
opt = torch.optim.Adam(model.parameters(), weight_decay=None)
print('Successfully created optimizer with None weight_decay (patch working)')
"
```

## Monitoring Training Progress

You can monitor your training progress using TensorBoard:

```bash
tensorboard --logdir tensorboard_logs/infini_YYYYMMDD_HHMMSS
```

Or compare multiple runs:

```bash
tensorboard --logdir_spec=infini:tensorboard_logs/infini_YYYYMMDD_HHMMSS,baseline:tensorboard_logs/baseline_YYYYMMDD_HHMMSS
```
