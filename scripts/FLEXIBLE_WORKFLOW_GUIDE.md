# Flexible Training Workflow Guide for Infini-Llama

This guide provides detailed instructions for training Infini-Llama models using the flexible training workflow script. This workflow offers improved configuration options, better error handling, and support for various training scenarios.

## Prerequisites

Before running the training workflow, ensure you have:

1. Set up your Python environment with all required dependencies
2. CUDA-compatible GPU(s) for training
3. Flash Attention installed (recommended for performance)
4. Preprocessed data or raw data ready for preprocessing

### Installing Flash Attention

Flash Attention is required for optimal performance. If you encounter the error `ModuleNotFoundError: No module named 'flash_attn'`, install it with:

```bash
pip install flash-attn --no-build-isolation
```

If you're using an older CUDA version or encounter compatibility issues, you can install from source:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install .
```

## Basic Usage

The flexible training workflow can be used in several ways depending on your needs:

### Option 1: Training with Pre-Processed Data

If you already have preprocessed data:

```bash
./scripts/flexible_training_workflow.sh \
  --preprocessed-data path/to/preprocessed/data \
  --config-file scripts/config/tiny_test_config.yaml
```

### Option 2: Processing Raw Data and Training

If you have raw data that needs preprocessing:

```bash
./scripts/flexible_training_workflow.sh \
  --raw-data path/to/raw/data \
  --config-file scripts/config/tiny_test_config.yaml \
  --output-dir path/for/preprocessed/output
```

### Option 3: Training the Baseline Model (Without Infini-Attention)

To train the baseline model without Infini-Attention:

```bash
./scripts/flexible_training_workflow.sh \
  --preprocessed-data path/to/preprocessed/data \
  --config-file scripts/config/tiny_test_config.yaml \
  --disable-infini-attn
```

### Option 4: Running Both Models in Parallel

To train both Infini-Attention and baseline models in parallel (requires 2+ GPUs):

```bash
./scripts/flexible_training_workflow.sh \
  --preprocessed-data path/to/preprocessed/data \
  --config-file scripts/config/tiny_test_config.yaml \
  --run-both-models
```

## Common Command-Line Options

| Option                | Description                                         | Default                          |
|----------------------|-----------------------------------------------------|----------------------------------|
| `--raw-data`         | Path to raw data for preprocessing                   | (Required if no preprocessed data) |
| `--preprocessed-data` | Path to already preprocessed data                   | (Required if no raw data)         |
| `--config-file`      | Path to configuration YAML file                     | scripts/config/tiny_test_config.yaml |
| `--output-dir`       | Directory to save preprocessed data                 | preprocessed_data                |
| `--disable-infini-attn` | Disable Infini-Attention (run baseline model)      | false                            |
| `--gpu`              | GPU ID to use                                       | 0                                |
| `--run-both-models`  | Run both models in parallel (requires 2+ GPUs)      | false                            |
| `--tensorboard-dir`  | Directory for TensorBoard logs                      | tensorboard_logs/[type]_[timestamp] |
| `--no-gpu-dataloader` | Disable GPU-accelerated dataloader                  | false                            |
| `--force-preprocess` | Force preprocessing even if data exists             | false                            |
| `--verbose`          | Enable verbose logging                              | false                            |
| `--offline-mode`     | Run in offline mode (no downloads from HuggingFace) | false                            |
| `--enable-fused-adam` | Enable fused Adam optimizer                         | false                            |

## Distributed Training Environment

The flexible workflow automatically sets up the required distributed training environment variables:

```
RANK=0
WORLD_SIZE=1
LOCAL_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29500
```

These are needed for single-GPU training too, as the underlying framework uses PyTorch's distributed capabilities.

## Advanced Usage

### GPU Selection and Multi-GPU Training

To specify which GPU to use:

```bash
./scripts/flexible_training_workflow.sh \
  --preprocessed-data path/to/preprocessed/data \
  --config-file scripts/config/tiny_test_config.yaml \
  --gpu 1
```

### Offline Training Mode

If you need to train without internet access or HuggingFace downloads:

```bash
./scripts/flexible_training_workflow.sh \
  --preprocessed-data path/to/preprocessed/data \
  --config-file scripts/config/tiny_test_config.yaml \
  --offline-mode
```

### Forcing Preprocessing

To force reprocessing of data even if preprocessed data exists:

```bash
./scripts/flexible_training_workflow.sh \
  --raw-data path/to/raw/data \
  --preprocessed-data path/to/existing/preprocessed/data \
  --config-file scripts/config/tiny_test_config.yaml \
  --force-preprocess
```

## Monitoring Training Progress

You can monitor the training progress using TensorBoard:

```bash
tensorboard --logdir tensorboard_logs/
```

When comparing multiple models (e.g., when using `--run-both-models`):

```bash
tensorboard --logdir_spec=infini:tensorboard_logs/infini_[timestamp],baseline:tensorboard_logs/baseline_[timestamp]
```

## Troubleshooting

### Flash Attention Errors

If you see errors related to Flash Attention:

1. Install Flash Attention as described in the Prerequisites section:
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. If you see `ModuleNotFoundError: No module named 'flash_attn'`, the workflow will automatically disable Flash Attention, but you can also explicitly disable it:
   ```bash
   ./scripts/flexible_training_workflow.sh \
     --preprocessed-data path/to/preprocessed/data \
     --config-file scripts/config/tiny_test_config.yaml \
     --disable-flash-attn
   ```

3. If you encounter warnings from Flash Attention about memory layouts, these are typically harmless and are automatically suppressed.

### Adam Optimizer Errors

If you encounter optimizer-related errors like `unsupported operand type(s) for *: 'float' and 'NoneType'`:

1. The workflow automatically applies several patches to fix the Adam optimizer:
   - Applies a direct patch to the Adam optimizer class
   - Modifies the YAML config to ensure weight_decay is not None
   - Sets default weight_decay to 0.01 if missing

2. If you still encounter issues, you can try disabling the fused Adam implementation:
   ```bash
   ./scripts/flexible_training_workflow.sh \
     --preprocessed-data path/to/preprocessed/data \
     --config-file scripts/config/tiny_test_config.yaml \
     --disable-infini-attn
   ```

3. To debug optimizer issues, check your YAML config file to ensure `weight_decay` is explicitly set to a float value (e.g., 0.01 or 0.0).

### Environment and Path Issues

If you encounter Python module import errors:

1. Make sure your Python environment is properly activated with all dependencies installed
2. Verify the PYTHONPATH includes the project root and src directories:
   ```bash
   export PYTHONPATH="/path/to/nanotron-infini:/path/to/nanotron-infini/src:$PYTHONPATH"
   ```

3. Check file permissions for all scripts:
   ```bash
   chmod +x scripts/flexible_training_workflow.sh
   chmod +x scripts/wrapper_script.py
   ```

4. Run with `--verbose` flag for detailed logging that can help identify import issues

### Distributed Training Issues

If using distributed training and encountering errors about missing environment variables:

1. The script automatically sets up required distributed environment variables:
   ```
   RANK=0
   WORLD_SIZE=1
   LOCAL_RANK=0
   MASTER_ADDR=localhost
   MASTER_PORT=29500
   ```

2. If running manual distributed training with `torchrun`, make sure these variables are set correctly

3. For multi-node distributed training, you'll need to set the correct master address and update the world size

### CUDA and GPU Issues

If encountering GPU-related errors:

1. Ensure CUDA is properly installed and compatible with your PyTorch version
2. Check GPU availability with `nvidia-smi`
3. Try specifying a specific GPU if you have multiple:
   ```bash
   ./scripts/flexible_training_workflow.sh \
     --preprocessed-data path/to/preprocessed/data \
     --gpu 0 \
     --config-file scripts/config/tiny_test_config.yaml
   ```
4. Set `CUDA_VISIBLE_DEVICES` manually if needed:
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1
   ```

## Example Complete Workflow

Here's a complete example that processes raw data and trains the model:

```bash
# Make sure script is executable
chmod +x scripts/flexible_training_workflow.sh

# Run the flexible workflow with raw data
./scripts/flexible_training_workflow.sh \
  --raw-data /path/to/raw/data \
  --config-file scripts/config/custom_infini_config_gpu.yaml \
  --output-dir my_processed_data \
  --gpu 0 \
  --tensorboard-dir my_tensorboard_logs \
  --verbose
```

For further information about the Infini-Llama model or detailed configuration options, refer to the project documentation.

## Key Workflow Scripts

The flexible training workflow relies on the following essential scripts:

### Core Scripts
- **flexible_training_workflow.sh**: Main entry point for the workflow that handles both preprocessing and training
- **wrapper_script.py**: Applies necessary patches and sets up the environment before calling the training script
- **run_direct_training.py**: Handles the training process using the DistributedTrainer

### Support Scripts
- **preprocessing/preprocess_data_fixed.py**: Processes raw data into format suitable for training
- **training/train_infini_llama.py**: Implementation of the training process
- **direct_adam_patch.py**: Fixes Adam optimizer weight_decay issues

### Configuration Files
- **config/tiny_test_config.yaml**: Default minimal configuration for testing
- **config/custom_infini_config_gpu.yaml**: Configuration optimized for GPU training
- **config/custom_infini_config_cpu.yaml**: Configuration for CPU training

When troubleshooting or modifying the workflow, focus on these key scripts as they form the backbone of the training process.
