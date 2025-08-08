# Infini-Llama Flexible Training Workflow

This document summarizes the fixes and improvements made to the Infini-Llama training workflow.

## Key Improvements

1. **Fixed Adam Optimizer Issues**
   - Disabled fused Adam optimizer by default to avoid type errors during training
   - Added `--enable-fused-adam` flag to optionally enable it if your environment supports it
   - Fixed NoneType error in weight decay handling (added default value 0.01)
   - Implemented robust weight_decay default handling in the optimizer initialization code

2. **Fixed Flash Attention Warnings**
   - Created `fix_flash_attention_warnings.py` script to automatically patch Flash Attention library
   - Added fallback warning suppression using `PYTHONWARNINGS="ignore::FutureWarning"`

3. **Fixed Tensor Reshape/View Issues**
   - Replaced `.view()` with `.reshape()` in tensor operations to handle non-contiguous tensors
   - This prevents the "view size is not compatible with input tensor's size and stride" error

4. **Fixed Logging Path Issues**
   - Made log directory configurable via environment variables instead of hardcoded paths
   - Added `TRAINING_LOGS_DIR` environment variable to specify log location

5. **Added Parallel Training Support**
   - Added ability to train both Infini-Attention and baseline models in parallel
   - Added `--run-both-models` flag to enable parallel training on multiple GPUs

6. **Improved Error Handling and Messaging**
   - Added more descriptive error messages and status reporting
   - Added proper cleanup and process tracking for parallel training

## Usage Examples

### Basic Training

```bash
# Train with Infini-Attention enabled (default)
./scripts/flexible_training_workflow.sh --preprocessed-data path/to/data --config-file scripts/config/tiny_test_config.yaml

# Train baseline model (Infini-Attention disabled)
./scripts/flexible_training_workflow.sh --preprocessed-data path/to/data --config-file scripts/config/tiny_test_config.yaml --disable-infini-attn
```

### Advanced Usage

```bash
# Run both models in parallel (requires 2+ GPUs)
./scripts/flexible_training_workflow.sh --preprocessed-data path/to/data --config-file scripts/config/tiny_test_config.yaml --run-both-models

# Preprocess raw data and then train
./scripts/flexible_training_workflow.sh --raw-data path/to/raw/data --config-file scripts/config/tiny_test_config.yaml

# Enable fused Adam optimizer (if your environment supports it)
./scripts/flexible_training_workflow.sh --preprocessed-data path/to/data --config-file scripts/config/tiny_test_config.yaml --enable-fused-adam
```

## Monitoring Training Progress

You can monitor the training progress using TensorBoard:

```bash
tensorboard --logdir tensorboard_logs/
```

When comparing multiple models:

```bash
tensorboard --logdir_spec=infini:path/to/infini_logs,baseline:path/to/baseline_logs
```

## Common Issues and Solutions

1. **Fused Adam Optimizer Errors**
   - If you see errors related to `fused_adamw()`, make sure to keep `--enable-fused-adam` flag disabled
   
2. **Flash Attention Warnings**
   - The script attempts to automatically fix Flash Attention warnings
   - If warnings persist, they are harmless and suppressed in the output

3. **Memory Issues**
   - If you encounter out-of-memory errors, try reducing batch size in the config file
   - Use a smaller model configuration for testing
