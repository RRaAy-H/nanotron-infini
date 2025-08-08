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

If you encounter issues with the training script, try these troubleshooting steps:

1. **Check if the wrapper script exists**: Make sure `scripts/wrapper_script.py` is present and executable.
   ```bash
   ls -l scripts/wrapper_script.py
   chmod +x scripts/wrapper_script.py
   ```

2. **Verify Python path**: The script should correctly set the Python path to include the project root and src directory.
   ```bash
   echo $PYTHONPATH
   export PYTHONPATH="$PWD:$PWD/src:$PYTHONPATH"
   ```

3. **Check for permission issues**: Make sure all scripts and directories have the correct permissions.
   ```bash
   chmod -R 755 scripts/
   chmod -R 755 training_logs/
   ```

4. **Look for missing Python dependencies**: Ensure all required packages are installed.

5. **Check wrapper script logs**: If the wrapper script is failing, look for error messages in the output.

## Monitoring Training Progress

You can monitor your training progress using TensorBoard:

```bash
tensorboard --logdir tensorboard_logs/infini_YYYYMMDD_HHMMSS
```

Or compare multiple runs:

```bash
tensorboard --logdir_spec=infini:tensorboard_logs/infini_YYYYMMDD_HHMMSS,baseline:tensorboard_logs/baseline_YYYYMMDD_HHMMSS
```
