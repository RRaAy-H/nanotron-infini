# Flash Attention Usage Guide

This guide explains how to control Flash Attention usage in the Infini-Llama training workflow.

## Training Without Flash Attention

### Method 1: Use the Command-Line Flag

To explicitly disable Flash Attention and use the standard attention implementation:

```bash
./scripts/flexible_training_workflow.sh --config-file scripts/config/tiny_test_config.yaml \
    --preprocessed-data your_preprocessed_data_dir \
    --disable-flash-attn
```

This will:
- Explicitly disable Flash Attention
- Use the standard attention implementation instead
- Block Flash Attention imports in Python
- Set appropriate environment variables

### Method 2: Automatic Detection (Default Behavior)

The training workflow automatically detects Flash Attention compatibility issues (such as GLIBC version mismatches). If an issue is detected, Flash Attention will be automatically disabled:

```bash
./scripts/flexible_training_workflow.sh --config-file scripts/config/tiny_test_config.yaml \
    --preprocessed-data your_preprocessed_data_dir
```

### When Running Both Models (Multi-GPU)

If using multiple GPUs with the `--run-both-models` flag, you can disable Flash Attention for both models:

```bash
./scripts/flexible_training_workflow.sh --config-file scripts/config/tiny_test_config.yaml \
    --preprocessed-data your_preprocessed_data_dir \
    --run-both-models \
    --disable-flash-attn
```

## How it Works

When Flash Attention is disabled:
1. The environment variable `DISABLE_FLASH_ATTN=1` is set
2. The `--disable-flash-attn` flag is passed to the training script
3. A monkey patch blocks Flash Attention imports in Python
4. The training uses the standard attention implementation

Training will proceed normally but may be slightly slower than with Flash Attention enabled.

## Checking Flash Attention Status

During training startup, you'll see messages indicating whether Flash Attention is enabled or disabled:

```
# When Flash Attention is disabled:
Flash Attention is disabled - will use standard attention implementation
```

Any Flash Attention compatibility issues will be logged to `flash_attention_log.txt` in the project root directory.
