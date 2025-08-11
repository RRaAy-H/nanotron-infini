# Training with Parquet Data Files

This guide explains how to use parquet files for training Infini-Llama models.

## Overview

Parquet files are an efficient columnar storage format that can be used to store preprocessed text data for training large language models. This repository now includes support for loading parquet files directly and using them to train Infini-Llama models.

## Required Files

The necessary scripts have been added to support parquet files:

1. `scripts/parquet_data_loader.py` - Python script that processes parquet files and prepares them for training
2. `scripts/parquet_training_workflow.sh` - A dedicated script to run the entire workflow from parquet files to trained model

## How to Use

### Option 1: Using the Parquet Training Workflow

This is the simplest approach that handles the entire process:

```bash
cd /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini
./scripts/parquet_training_workflow.sh --parquet-data /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/data --config-file custom_infini_config_gpu.yaml
```

Additional options:
- `--tokenizer` - Specify the tokenizer name or path (default: meta-llama/Llama-2-7b-hf)
- `--max-seq-length` - Set the maximum sequence length for tokenization (default: 2048)
- `--disable-infini-attn` - Train without Infini-Attention (baseline model)
- `--gpu-id` - Specify which GPU to use (default: 0)

### Option 2: Step-by-Step Approach

If you prefer more control over the process, you can run the steps separately:

1. **Process the parquet files:**
```bash
cd /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini
python scripts/parquet_data_loader.py --data-dir /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/data --output-dir processed_data
```

2. **Train using the processed data:**
```bash
cd /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini
python scripts/training/train_infini_llama.py --config-file custom_infini_config_gpu.yaml --data-dir processed_data/preprocessed_YYYYMMDD_HHMMSS
```

## Parquet File Format Requirements

The parquet files should contain a column with text data. By default, the script looks for a column named `text`. If your data uses a different column name, the script will attempt to use common alternatives like `content`, `document`, or `sentence`.

## Example Parquet Data Structure

Ideally, your parquet files should have this structure:

```
Column: text
Values: ["This is a text example for training...", "Here is another example...", ...]
```

Or if using a different column name:

```
Column: content
Values: ["This is a text example for training...", "Here is another example...", ...]
```

## Flash Attention Compatibility

The parquet training workflow now includes automatic detection and handling of Flash Attention compatibility issues. If you're experiencing the GLIBC_2.32 error or other Flash Attention compatibility issues, the workflow will:

1. Automatically detect the incompatibility
2. Disable Flash Attention
3. Continue training with standard attention implementation

You don't need to do anything special - this happens automatically when you run the `parquet_training_workflow.sh` script.

## Troubleshooting

If you encounter issues:

1. Make sure your parquet files are valid and can be opened with tools like pandas
2. Check that the parquet files contain text data in a recognizable column
3. Ensure your GPU has enough memory for the configuration specified in your config file
4. Try reducing the batch size or sequence length in your config file if you experience out-of-memory errors
5. If you have Flash Attention errors, check the logs to see if the auto-detection system handled it properly
6. For detailed information about Flash Attention compatibility, see `docs/FLASH_ATTENTION_TROUBLESHOOTING.md`
