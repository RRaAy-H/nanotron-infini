# Flash Attention Troubleshooting Guide

This guide helps you troubleshoot Flash Attention compatibility issues in the Infini-Llama training workflow, particularly the common "GLIBC_2.32 not found" error.

## What is Flash Attention?

Flash Attention is a more efficient attention implementation that significantly speeds up transformer models like Llama. However, it requires specific CUDA and GLIBC versions to work properly, which can cause compatibility issues.

## Common Flash Attention Issues

1. **GLIBC_2.32 not found error**: This happens when Flash Attention was compiled with a newer GLIBC version than what's available on your system.
2. **CUDA compatibility errors**: Flash Attention requires specific CUDA versions to work properly.
3. **Import errors**: Various Python import errors when trying to use Flash Attention.

## How Our Workflow Handles Flash Attention Issues

The training workflow has been enhanced to automatically detect Flash Attention compatibility issues:

1. When you run any training workflow script (including `flexible_training_workflow.sh` and `parquet_training_workflow.sh`), it automatically:
   - Uses our new `flash_attention_compatibility.py` script to check system compatibility
   - Detects your system's GLIBC version and compares it with Flash Attention requirements
   - Checks if Flash Attention is installed and can be properly imported
   - Tests if Flash Attention CUDA modules can be loaded

2. If any compatibility issues are detected, the workflow:
   - Automatically disables Flash Attention with no manual intervention needed
   - Sets appropriate environment variables
   - Creates mock modules to ensure imports don't fail
   - Falls back to standard attention implementation
   - Continues training without interruption

## Solutions for Flash Attention Issues

### Option 1: Automatic Detection and Handling (Recommended)

Our workflow now features intelligent auto-detection of Flash Attention compatibility issues:

- All training scripts now include the `--auto-detect-flash-attn` flag by default
- The system automatically checks if your GLIBC version is compatible (requires 2.32+)
- If incompatible, Flash Attention is automatically disabled and training continues with standard attention
- No manual intervention is needed - just run your training script as normal

For example:
```bash
# With parquet data:
./scripts/parquet_training_workflow.sh --parquet-data /path/to/parquet/files --config-file custom_infini_config_gpu.yaml

# With standard data:
./scripts/flexible_training_workflow.sh --config-file scripts/config/tiny_test_config.yaml --preprocessed-data tiny_test_data/preprocessed_20240808_123456
```

### Option 2: Manually Disable Flash Attention

If you explicitly want to disable Flash Attention regardless of compatibility:

```bash
./scripts/flexible_training_workflow.sh --config-file scripts/config/tiny_test_config.yaml --preprocessed-data tiny_test_data/preprocessed_20240808_123456 --disable-flash-attn
```

### Option 3: Rebuild Flash Attention from Source

For better performance, you can rebuild Flash Attention from source to match your system's GLIBC version:

```bash
# Use our provided script
./scripts/utils/install_flash_attention_from_source.sh
```

Or manually:

```bash
pip uninstall -y flash-attn
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-binary flash-attn
```

### Option 4: Check Flash Attention Compatibility

Use our diagnostic tool to check Flash Attention compatibility:

```bash
python scripts/utils/verify_flash_attention.py
```

## Additional Notes

- Training without Flash Attention will be slower but still works correctly
- If you're in a container environment, consider rebuilding Flash Attention inside the container
- On older systems with GLIBC < 2.31, you may need to upgrade your system or use a newer container image

## Still Having Issues?

If you're still experiencing issues with Flash Attention:

1. Run the verification script for detailed diagnostics: `python scripts/utils/verify_flash_attention.py`
2. Check your system's GLIBC version: `ldd --version`
3. Make sure you're using a compatible CUDA version
4. Try training with the `--disable-flash-attn` flag
