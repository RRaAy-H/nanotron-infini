# Flash Attention Compatibility Fix Guide

This guide explains how to resolve compatibility issues between PyTorch and Flash Attention in the Infini-Llama training pipeline.

## The Problem

The error `ImportError: undefined symbol: _ZN2at4_ops5zeros4callEN3c108ArrayRefINS2_6SymIntEEENS2_8optionalINS2_10ScalarTypeEEENS6_INS2_6LayoutEEENS6_INS2_6DeviceEEENS6_IbEE` occurs when there's a mismatch between the PyTorch version and the Flash Attention library version.

This typically happens because:
1. PyTorch was upgraded, but Flash Attention wasn't rebuilt against the new PyTorch version
2. The installed versions of PyTorch and Flash Attention are incompatible
3. The CUDA versions used to build PyTorch and Flash Attention don't match

## The Solution

We've created a script that automatically fixes these compatibility issues by:

1. Installing a compatible version of PyTorch (2.0.1) with CUDA support
2. Installing a compatible version of Flash Attention (2.3.0)
3. Verifying that the installation works correctly

## How to Use the Fix

### Automatic Fix Script

Run the provided script to fix the compatibility issues:

```bash
# Make the script executable
chmod +x fix_flash_attention.py

# Run the fix script
python fix_flash_attention.py
```

After running this script, your environment should have compatible versions of PyTorch and Flash Attention.

### Fix and Train in One Step

We've also provided a script that combines the fix and training steps:

```bash
# Make the script executable
chmod +x fix_and_run_with_flash_attention.sh

# Run the fix and training script
./fix_and_run_with_flash_attention.sh [config_file]
```

This script will:
1. Fix the Flash Attention compatibility issues
2. Run the training with Flash Attention enabled

### Manual Fix

If you prefer to fix the issue manually, you can:

1. Install a compatible version of PyTorch with CUDA support:
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
   ```

2. Install a compatible version of Flash Attention:
   ```bash
   pip install flash-attn==2.3.0
   ```

3. Verify that the installation works:
   ```bash
   python -c "import torch; import flash_attn; print(f'PyTorch: {torch.__version__}, Flash Attention: {flash_attn.__version__}')"
   ```

## Troubleshooting

If you continue to experience issues:

1. **Check CUDA installation**: Make sure CUDA is properly installed on your system
   ```bash
   nvcc --version
   ```

2. **Check PyTorch CUDA support**: Verify that PyTorch was installed with CUDA support
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

3. **Check for conflicts**: Look for conflicting package versions
   ```bash
   pip list | grep torch
   pip list | grep flash
   ```

4. **Try a different CUDA version**: If you're still having issues, you can try a different CUDA version:
   ```bash
   python fix_flash_attention.py --cuda-version 11.7
   ```
   Available options are 11.7, 11.8, and 12.1.

5. **Use the verification mode**: You can verify your current installation without making changes:
   ```bash
   python fix_flash_attention.py --verify
   ```

## Reverting to Standard Attention

If you still can't get Flash Attention to work and need to use standard attention:

1. Run the training script with Flash Attention disabled:
   ```bash
   DISABLE_FLASH_ATTN=1 ./fix_and_train.sh
   ```

## Additional Resources

- [Flash Attention GitHub repository](https://github.com/Dao-AILab/flash-attention)
- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)
- [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
