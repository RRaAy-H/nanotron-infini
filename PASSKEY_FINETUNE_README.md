# Passkey Retrieval Finetuning for 300M Infini-Attention Model

This guide explains how to finetune your trained 300M Infini-Attention model on the passkey retrieval task using 10K token sequences.

## Overview

The passkey retrieval task trains the model to find and output a hidden numeric "pass key" embedded within a long sequence of distractor text. This tests the model's ability to maintain and retrieve information across long contexts using the Infini-Attention mechanism.

### Task Format
```
Input: "There is an important info hidden inside a lot of irrelevant text... 
        The grass is green. The sky is blue... 
        The pass key is 9054. Remember it. 9054 is the pass key...
        The grass is green. The sky is blue...
        What is the pass key? The pass key is"

Output: "9054"
```

## Quick Start

### Run the Finetuning

```bash
# Basic usage (assumes checkpoint at ./checkpoints/fineweb_4gpu_300m_infini/30000)
./run_passkey_finetune_300m.sh

# Or specify your checkpoint path
./run_passkey_finetune_300m.sh ./path/to/your/checkpoint 2000 42

# Arguments:
# - checkpoint_path: Path to your trained 300M model checkpoint (e.g., step 30000)
# - num_examples: Number of training examples to generate (default: 2000)
# - seed: Random seed for reproducibility (default: 42)
```

## Detailed Steps

### 1. Generate the Dataset Manually (Optional)

The main script does this automatically, but you can also generate the dataset separately:

```bash
python3 generate_passkey_finetune_data.py \
    --tokenizer_path lvwerra/the-tokenizer-v1 \
    --num_examples 2000 \
    --target_length 10240 \
    --save_path ./passkey_finetune_data_10k \
    --seed 42
```

This creates:
- `passkey_finetune_data_10k/` - Dataset directory
- `passkey_finetune_data_10k.parquet` - Parquet file for easy loading

### 2. Configure the Finetuning

Edit `passkey_finetune_300m_config.yaml` if needed:

```yaml
# Key settings to adjust:
resume_checkpoint_path: ./checkpoints/fineweb_4gpu_300m_infini/30000  # Your checkpoint
sequence_length: 10240  # 10K tokens  
train_steps: 500  # Number of finetuning steps
learning_rate: 0.00005  # Lower LR for finetuning
micro_batch_size: 1  # Adjust based on GPU memory
dp: 4  # Number of GPUs for data parallelism
```

### 3. Run Training

```bash
# The script will:
# 1. Generate the passkey dataset
# 2. Update config with your checkpoint path
# 3. Run the finetuning for 500 steps

./run_passkey_finetune_300m.sh ./checkpoints/fineweb_4gpu_300m_infini/30000
```

## Training Details

### Hyperparameters (from config)
- **Sequence Length**: 10,240 tokens (~10K)
- **Training Steps**: 500
- **Learning Rate**: 5e-5 (with cosine decay)
- **Warmup Steps**: 50 (10% of total steps)
- **Min Learning Rate**: 5e-6 (after decay)
- **Batch Size**: 1 per GPU (due to long sequences)
- **Gradient Accumulation**: 1 (no accumulation)
- **Optimizer**: AdamW (β1=0.9, β2=0.95, eps=1e-8)
- **Weight Decay**: 0.01
- **Gradient Clipping**: 1.0
- **Precision**: bfloat16
- **Accumulate Gradients in FP32**: Yes

### Infini-Attention Settings (from config)
- **Segment Length**: 1024 tokens (DO NOT CHANGE - must match pretraining)
- **Memory**: Enabled (`turn_on_memory: true`)
- **Balance Factor LR**: 0.01
- **Balance Factor Weight Decay**: 0.0 (no decay)
- **Activation**: Hard sigmoid (`balance_act_type`)
- **Initialization**: Zeros (`balance_init_type`)
- **Logging Interval**: Every 100 steps
- **Max Position Embeddings**: 8192 (matches pretraining)

### Dataset Characteristics
- **Total Examples**: 2000 (configurable)
- **Passkey Positions**: 0%, 25%, 50%, 75%, 100% depth
- **Passkey Format**: 4-digit numbers (1000-9999)
- **Distractor Text**: Repeated phrases about grass, sky, sun

## Monitoring Training

Training logs will show:
- **Loss per iteration**: Logged every 10 steps (`iteration_step_info_interval: 10`)
- **Learning rate schedule**: Cosine decay from 5e-5 to 5e-6
- **Memory usage statistics**: Logged every 100 steps (Infini-attention `logging_interval`)
- **Checkpoints**: Saved every 100 steps at steps 100, 200, 300, 400, 500

## After Training

### Evaluate the Model

```bash
# Use the existing evaluation script
./examples/infinite-context-length/scripts/run_passkey_eval_300m.sh \
    ./checkpoints/passkey_finetune_300m/500 \
    10240 \
    25
```

### Generate with the Model

```bash
torchrun --nproc_per_node=4 run_generate.py \
    --ckpt-path ./checkpoints/passkey_finetune_300m/500 \
    --dp 4 --tp 1 --pp 1
```

## Expected Results

After 500 finetuning steps, the model should achieve:
- **High accuracy** (>90%) on passkey retrieval at various depths
- **Consistent performance** across different passkey positions
- **Ability to handle 10K token sequences** effectively

## Troubleshooting

### Out of Memory (OOM)
- Reduce `micro_batch_size` to 1 (already set by default)
- Reduce `sequence_length` if needed (e.g., 8192)
- Use fewer GPUs and adjust `dp` accordingly

### Checkpoint Not Found
- Verify the checkpoint path exists
- Check that the checkpoint contains all necessary files:
  - `model_state_dict.pt`
  - `optimizer_state_dict.pt`
  - `training_metadata.json`

### Dataset Generation Issues
- Ensure `datasets` and `transformers` are installed
- Check that the tokenizer can be loaded
- Verify sufficient disk space for the dataset

### Training Doesn't Start
- Check CUDA availability: `nvidia-smi`
- Verify PyTorch installation: `python3 -c "import torch; print(torch.cuda.is_available())"`
- Check for port conflicts if using multiple training runs

## Advanced Configuration

### Using Different Context Lengths

To train with different sequence lengths:

1. Generate dataset with desired length:
```bash
python3 generate_passkey_finetune_data.py --target_length 8192
```

### Custom Passkey Formats

Modify `generate_passkey_finetune_data.py` to:
- Use different passkey lengths (change `random.randint(1000, 9999)`)
- Add more depth percentages
- Use different distractor text patterns
