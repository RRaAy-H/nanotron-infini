# Training Infini-Llama with Parquet Data

Follow these steps to train your Infini-Llama model using the parquet data files.

## Step 1: Process the Parquet Data

First, process your parquet data files to prepare them for training:

```bash
cd /home/data/daal_insight/fiery/Infini-attention/nanotron-infini
python scripts/parquet_data_loader.py --data-dir /home/data/daal_insight/fiery/Infini-attention/nanotron-infini/data --output-dir processed_data
```

This will create a timestamped directory in `processed_data/` containing your preprocessed dataset.

## Step 2: Run Training Using the Direct Training Script

After preprocessing is complete, use the direct training script which avoids import issues:

```bash
cd /home/data/daal_insight/fiery/Infini-attention/nanotron-infini
python scripts/run_direct_training.py \
  --config-file custom_infini_config_gpu.yaml \
  --data-dir processed_data/preprocessed_YYYYMMDD_HHMMSS \
  --gpu-id 0
```

Replace `preprocessed_YYYYMMDD_HHMMSS` with the actual timestamp of your preprocessed directory.

## Step 3: Or Use the Parquet Training Workflow (Alternative)

Alternatively, you can use the parquet training workflow which handles both preprocessing and training:

```bash
cd /home/data/daal_insight/fiery/Infini-attention/nanotron-infini
./scripts/parquet_training_workflow.sh \
  --parquet-data /home/data/daal_insight/fiery/Infini-attention/nanotron-infini/data \
  --config-file custom_infini_config_gpu.yaml
```

## Important Notes

- The error `cannot import name 'LlamaForCausalLM' from 'nanotron.models.llama'` occurs because the train_infini_llama.py script is trying to import a class that doesn't exist in your version of the code.
- The `run_direct_training.py` script doesn't have this issue as it uses the model classes defined in your configuration file.
- The `parquet_training_workflow.sh` script is designed to handle all the complexity for you, including using the correct scripts.

## Additional Options

You can customize the training with these flags:

- `--disable-infini-attn`: Train without Infini-Attention (baseline model)
- `--use-gpu-dataloader`: Use GPU acceleration for data loading
- `--verbose`: Enable verbose logging
- `--disable-flash-attn`: Disable Flash Attention (for compatibility issues)
- `--tensorboard-dir PATH`: Specify a directory for TensorBoard logs

## Troubleshooting

If you encounter other errors during training:

1. Check that your environment is properly set up:
   ```bash
   echo $PYTHONPATH
   ```
   Ensure it includes the project root and src directories.

2. If you have CUDA compatibility issues:
   ```bash
   ./scripts/parquet_training_workflow.sh --parquet-data /path/to/data --config-file your_config.yaml --disable-flash-attn
   ```
