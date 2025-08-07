# Training Llama with Infi-Attention

This document provides step-by-step instructions for training the Llama model with Infi-Attention using your custom dataset.

## Files Created for Training

1. **`prepare_data.py`**: Script to process your parquet dataset for training
2. **`custom_infini_config.yaml`**: Configuration file for GPU training
3. **`custom_infini_config_cpu.yaml`**: Configuration file for CPU training (smaller model)
4. **`run_train_cpu.py`**: Modified training script compatible with CPU
5. **`train_infini_llama_cpu.sh`**: Shell script to run the CPU training
6. **`TRAINING_GUIDE.md`**: Detailed guide with troubleshooting tips

## Steps to Train the Model

### Option 1: Using a Conda Environment (Recommended)

1. Create a conda environment with Python 3.10:

```bash
conda create -y -n infi-llama python=3.10
conda activate infi-llama
```

2. Install dependencies:

```bash
cd /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini
pip install -e .
pip install datasets transformers huggingface_hub pyarrow pandas
```

3. Prepare the dataset:

```bash
python prepare_data.py
```

4. Make the training script executable:

```bash
chmod +x train_infini_llama_cpu.sh
```

5. Run the training script:

```bash
./train_infini_llama_cpu.sh
```

### Option 2: Direct Execution

If you prefer not to use conda, you can try running the scripts directly:

1. Prepare the dataset:

```bash
python prepare_data.py
```

2. Run the CPU training script:

```bash
export CUDA_VISIBLE_DEVICES=""
python run_train_cpu.py --config-file custom_infini_config_cpu.yaml
```

## Model Configuration

The Infi-Attention model is configured with:
- Segment length: 64
- Memory enabled: True
- Balance initialization: zeros
- Balance activation: orig_sigmoid

## Modifying Model Size

For faster training or to accommodate resource limitations, you can modify:
- `hidden_size`: Controls the dimensionality of hidden layers
- `num_hidden_layers`: Controls the depth of the model
- `num_attention_heads`: Controls the number of attention heads
- `intermediate_size`: Controls the size of feed-forward network

## Using GPU (If Available)

If you have a GPU with CUDA support, you can:
1. Install CUDA toolkit
2. Set the CUDA_HOME environment variable
3. Install flash-attention: `pip install flash-attn>=2.5.0`
4. Use the original configuration file: `custom_infini_config.yaml`

## Troubleshooting

If you encounter errors:

1. **Python Version**: Ensure you're using Python 3.10, not 3.13
2. **Package Installation**: Check that nanotron is installed with `pip install -e .`
3. **Memory Issues**: Reduce model size in the configuration file
4. **Dataset Issues**: Verify the dataset is properly processed in `/data/processed`

For more details, see TRAINING_GUIDE.md.
