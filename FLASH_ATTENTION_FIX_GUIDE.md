# Fixing Flash Attention Errors in Infini-Llama Training

This document provides a comprehensive guide to fix the common Flash Attention errors encountered during Infini-Llama training.

## Common Error

The most common error you're likely to see is:

```
ImportError: /path/to/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN2at4_ops5zeros4callEN3c108ArrayRefINS2_6SymIntEEENS2_8optionalINS2_10ScalarTypeEEENS6_INS2_6LayoutEEENS6_INS2_6DeviceEEENS6_IbEE
```

This error occurs due to a version mismatch between PyTorch and Flash Attention. The compiled Flash Attention library is looking for symbols from a different version of PyTorch than what is installed in your environment.

## Solution Overview

We've created several solutions to address this issue:

1. A patched script (`fix_and_train.sh`) that handles all the necessary setup
2. Enhanced code in `train_gpu_with_tensorboard.py` to detect Flash Attention issues early
3. An improved patching mechanism in `patch_flash_attention.py`

## Fix 1: Using the Comprehensive Fix Script

The easiest solution is to use the new `fix_and_train.sh` script:

```bash
cd /home/data/daal_insight/fiery/Infini-attention/nanotron-infini
chmod +x fix_and_train.sh
./fix_and_train.sh custom_infini_config_gpu.yaml
```

This script:
- Sets all necessary environment variables
- Disables Flash Attention
- Applies the patch to handle Flash Attention failures
- Sets up TensorBoard automatically
- Provides detailed logging

## Fix 2: Manually Running with Flash Attention Disabled

If you prefer to run the script manually:

```bash
cd /home/data/daal_insight/fiery/Infini-attention/nanotron-infini
export DISABLE_FLASH_ATTN=1
python patch_flash_attention.py
python train_gpu_with_tensorboard.py --config-file custom_infini_config_gpu.yaml --disable-flash-attn
```

## Fix 3: Using an Existing Script

You can also use the existing `train_infini_gpu_no_flash.sh` script:

```bash
cd /home/data/daal_insight/fiery/Infini-attention/nanotron-infini
chmod +x train_infini_gpu_no_flash.sh
./train_infini_gpu_no_flash.sh
```

## Technical Details of the Fix

The fix involves several components:

1. **Environment Variable Control**: Setting `DISABLE_FLASH_ATTN=1` instructs the code to skip Flash Attention.

2. **Code Patching**: The `patch_flash_attention.py` script modifies the Llama model code to:
   - Wrap Flash Attention imports in try-except blocks
   - Provide fallback implementations for standard attention
   - Add early detection of Flash Attention issues

3. **Import Checking**: The training script now checks if Flash Attention can be imported before starting training.

## Understanding the Performance Impact

When Flash Attention is disabled:

- Training will be slower (standard attention is less optimized)
- Memory usage will be higher
- You may need to reduce batch size for larger models

However, the model quality and training results will not be affected.

## Checking if the Fix Works

You can verify that the fix is working by:

1. Looking for this message in the logs:
   ```
   Flash Attention not available. Using standard attention implementation.
   ```

2. Checking if training progresses past the model initialization stage.

3. Confirming that no Flash Attention import errors appear in the logs.

## Permanent Solution Options

For a more permanent solution, you could:

1. Reinstall PyTorch and Flash Attention with compatible versions
2. Build Flash Attention from source against your specific PyTorch version
3. Continue using the standard attention implementation with the provided fixes
