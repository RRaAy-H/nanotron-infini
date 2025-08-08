# Infini-Llama Training Guide

This guide describes the complete workflow for training Infini-Llama models with separated preprocessing and training stages.

## Benefits of Separating Preprocessing and Training

1. **Robustness**: If training crashes, you don't need to redo the time-consuming preprocessing step
2. **Reusability**: Preprocessed data can be reused across multiple training runs
3. **Resource Optimization**: You can allocate different hardware resources for preprocessing vs training
4. **Debugging**: Easier to debug issues in preprocessing and training separately

## Directory Structure

```
nanotron-infini/
├── scripts/
│   ├── config/                  # Configuration YAML files
│   ├── preprocessing/
│   │   └── preprocess_data.py   # Data preprocessing script
│   ├── training/
│   │   └── train_infini_llama.py # Training script using preprocessed data
│   └── run_infini_llama_workflow.sh # Complete workflow script
├── src/
│   └── nanotron/
│       └── gpu_dataloader.py    # GPU-accelerated data processing
```

## Step-by-Step Workflow

### 1. Preprocessing Stage

This stage tokenizes the raw text data, chunks it into training examples, and saves the result to disk.

```bash
python scripts/preprocessing/preprocess_data.py \
  --config-file scripts/config/custom_infini_config_gpu.yaml \
  --output-dir preprocessed_data \
  --gpu-id 0  # Use GPU acceleration when available
```

**Key Options:**
- `--config-file`: Path to your configuration YAML file
- `--output-dir`: Directory to save preprocessed data
- `--gpu-id`: GPU ID to use for acceleration (or `--no-gpu` to disable)
- `--batch-size`: Batch size for GPU processing (default: 2048)

The script will create a timestamped directory inside the output directory, containing:
- Preprocessed dataset files
- Metadata about the preprocessing
- A reference to the most recent preprocessing run

### 2. Training Stage

This stage loads the preprocessed data and trains the model.

```bash
python scripts/training/train_infini_llama.py \
  --config-file scripts/config/custom_infini_config_gpu.yaml \
  --data-dir preprocessed_data/preprocessed_20231025_134522 \
  --use-gpu-dataloader \
  --tensorboard-dir tensorboard_logs
```

**Key Options:**
- `--config-file`: Path to your configuration YAML file
- `--data-dir`: Directory containing preprocessed data
- `--use-gpu-dataloader`: Enable GPU-accelerated data loading
- `--disable-flash-attn`: Disable Flash Attention (if needed)
- `--tensorboard-dir`: Directory for TensorBoard logs

### 3. Using the Workflow Script

For convenience, you can use the provided workflow script that handles both stages:

```bash
bash scripts/run_infini_llama_workflow.sh \
  --config scripts/config/custom_infini_config_gpu.yaml \
  --data-dir preprocessed_data \
  --gpu 0
```

**Key Options:**
- `--config`, `-c`: Configuration file
- `--data-dir`, `-d`: Directory for preprocessed data
- `--tensorboard-dir`, `-t`: Directory for TensorBoard logs
- `--gpu`, `-g`: GPU device number to use
- `--disable-flash-attn`: Disable Flash Attention
- `--skip-preprocessing`: Skip the preprocessing stage (use if already done)
- `--distributed`: Enable distributed training
- `--num-nodes`: Number of nodes for distributed training

## Advanced Configuration

### GPU-Accelerated Data Processing

The workflow supports GPU-accelerated data processing, which significantly speeds up:
- Text tokenization and chunking during preprocessing
- Data loading and collation during training

### Flash Attention Support

Flash Attention is enabled by default when available, providing:
- Reduced memory usage
- Faster attention computation
- Better training throughput

To check if Flash Attention is available:
```bash
python -c "import flash_attn; print('Flash Attention is available!')"
```

### Distributed Training

For multi-GPU training, use the distributed option with the workflow script:

```bash
bash scripts/run_infini_llama_workflow.sh \
  --config scripts/config/custom_infini_config_gpu.yaml \
  --data-dir preprocessed_data \
  --distributed \
  --num-nodes 1
```

Or manually with torchrun:

```bash
torchrun --nproc_per_node=8 scripts/training/train_infini_llama.py \
  --config-file scripts/config/custom_infini_config_gpu.yaml \
  --data-dir preprocessed_data/preprocessed_20231025_134522
```

## Troubleshooting

### Preprocessing Issues

1. **Out of Memory**: Reduce batch size with `--batch-size`
2. **Slow Processing**: Ensure GPU acceleration is enabled
3. **Invalid Data**: Check the data source configuration

### Training Issues

1. **Model Loading Errors**: Check model configuration and paths
2. **Flash Attention Errors**: Use `--disable-flash-attn` to disable it
3. **CUDA Out of Memory**: Reduce batch size in the config file
4. **Data Loading Errors**: Ensure the preprocessed data directory is correct

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Preprocess the data (this may take a while)
python scripts/preprocessing/preprocess_data.py \
  --config-file scripts/config/custom_infini_config_gpu.yaml \
  --output-dir preprocessed_data \
  --gpu-id 0

# 2. Train the model with the preprocessed data
python scripts/training/train_infini_llama.py \
  --config-file scripts/config/custom_infini_config_gpu.yaml \
  --data-dir preprocessed_data/preprocessed_20231025_134522 \
  --use-gpu-dataloader \
  --tensorboard-dir tensorboard_logs

# Or use the combined workflow script
bash scripts/run_infini_llama_workflow.sh \
  --config scripts/config/custom_infini_config_gpu.yaml \
  --data-dir preprocessed_data \
  --gpu 0
```
