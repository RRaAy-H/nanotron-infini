# Infini-Attention Model Evaluation Guide

This guide explains how to evaluate trained Infini-Attention models using the `run_evals.py` script, which performs **needle-in-haystack** evaluation to test long-context understanding capabilities.

## Overview

The evaluation script tests the model's ability to retrieve specific information (the "needle") from long context sequences (the "haystack") at various depths. This is crucial for validating that the Infini-Attention mechanism is working correctly for long-context tasks.

## Prerequisites

- Trained model checkpoint with `config.yaml`
- CUDA-capable GPU(s)
- Python environment with nanotron-infini installed

## Basic Command Structure

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc_per_node=<num_gpus> examples/infinite-context-length/run_evals.py \
    --ckpt-path <checkpoint_path> \
    --context_length <length> \
    --depth_percent <percent> \
    --num_shots <shots> \
    --num_digits <digits> \
    [optional arguments...]
```

## Required Arguments

### `--ckpt-path <path>`
**Description**: Path to the trained model checkpoint directory  
**Requirements**: Must contain `config.yaml` and model weights  
**Examples**:
```bash
--ckpt-path ./checkpoints/my_model/final
--ckpt-path /path/to/experiment/checkpoints/step_1000
--ckpt-path ./outputs/fineweb_200m_infini/200
```

### `--context_length <integer>`
**Description**: Total context length for the evaluation sequence  
**Purpose**: Tests the model's ability to handle long contexts  
**Examples**:
```bash
--context_length 4096    # 4K context
--context_length 8192    # 8K context  
--context_length 16384   # 16K context
--context_length 32768   # 32K context
```
**Note**: Should be ≤ your model's `max_position_embeddings`

### `--depth_percent <integer>`
**Description**: Percentage depth where the "needle" information is placed in the "haystack"  
**Range**: 0-100  
**Purpose**: Tests retrieval at different positions in long sequences  
**Examples**:
```bash
--depth_percent 1     # Near beginning (1% depth)
--depth_percent 25    # First quarter (25% depth)
--depth_percent 50    # Middle (50% depth)
--depth_percent 75    # Third quarter (75% depth)  
--depth_percent 99    # Near end (99% depth)
```

### `--num_shots <integer>`
**Description**: Number of evaluation examples/trials to run  
**Purpose**: More shots provide better statistical reliability  
**Examples**:
```bash
--num_shots 1     # Single trial (quick test)
--num_shots 5     # Few trials (development)
--num_shots 10    # Standard evaluation
--num_shots 50    # Thorough evaluation
```

### `--num_digits <integer>`
**Description**: Number of digits in the needle information  
**Purpose**: Controls the difficulty of the retrieval task  
**Examples**:
```bash
--num_digits 3    # Easy: 3-digit numbers (123)
--num_digits 5    # Medium: 5-digit numbers (12345)  
--num_digits 7    # Hard: 7-digit numbers (1234567)
--num_digits 10   # Very hard: 10-digit numbers
```

## Optional Arguments

### `--dp <integer>`, `--pp <integer>`, `--tp <integer>`
**Description**: Data/Pipeline/Tensor parallelism settings  
**Default**: Uses values from checkpoint config  
**Examples**:
```bash
--tp 1        # Single GPU tensor parallel
--tp 4        # 4-way tensor parallel
--pp 2        # 2-stage pipeline parallel
--dp 2        # 2-way data parallel
```

### `--max-new-tokens <integer>`
**Description**: Maximum tokens to generate in response  
**Default**: 30  
**Examples**:
```bash
--max-new-tokens 10   # Short answers
--max-new-tokens 30   # Default
--max-new-tokens 100  # Longer responses
```

## Example Usage

### 1. Quick Development Test
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc_per_node=1 examples/infinite-context-length/run_evals.py \
    --ckpt-path ./checkpoints/debug_model/50 \
    --context_length 2048 \
    --depth_percent 50 \
    --num_shots 1 \
    --num_digits 3
```

### 2. Standard Evaluation
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc_per_node=1 examples/infinite-context-length/run_evals.py \
    --ckpt-path ./checkpoints/fineweb_200m_infini/200 \
    --context_length 8192 \
    --depth_percent 25 \
    --num_shots 10 \
    --num_digits 5
```

### 3. Multi-GPU Evaluation
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc_per_node=4 examples/infinite-context-length/run_evals.py \
    --ckpt-path ./checkpoints/llama_8b_infini/1000 \
    --context_length 16384 \
    --depth_percent 75 \
    --num_shots 20 \
    --num_digits 7 \
    --tp 4
```

### 4. Comprehensive Long-Context Test
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc_per_node=2 examples/infinite-context-length/run_evals.py \
    --ckpt-path ./checkpoints/production_model/final \
    --context_length 32768 \
    --depth_percent 90 \
    --num_shots 50 \
    --num_digits 8 \
    --tp 2 \
    --max-new-tokens 50
```

## Evaluation Scenarios

### Testing Different Depths
Run multiple evaluations at different depth percentages:
```bash
# Test retrieval at beginning
torchrun --nproc_per_node=1 examples/infinite-context-length/run_evals.py \
    --ckpt-path ./checkpoints/model/100 --context_length 8192 --depth_percent 1 --num_shots 10 --num_digits 5

# Test retrieval at middle  
torchrun --nproc_per_node=1 examples/infinite-context-length/run_evals.py \
    --ckpt-path ./checkpoints/model/100 --context_length 8192 --depth_percent 50 --num_shots 10 --num_digits 5

# Test retrieval at end
torchrun --nproc_per_node=1 examples/infinite-context-length/run_evals.py \
    --ckpt-path ./checkpoints/model/100 --context_length 8192 --depth_percent 99 --num_shots 10 --num_digits 5
```

### Testing Different Context Lengths
```bash
# Short context baseline
torchrun --nproc_per_node=1 examples/infinite-context-length/run_evals.py \
    --ckpt-path ./checkpoints/model/100 --context_length 2048 --depth_percent 50 --num_shots 10 --num_digits 5

# Medium context  
torchrun --nproc_per_node=1 examples/infinite-context-length/run_evals.py \
    --ckpt-path ./checkpoints/model/100 --context_length 8192 --depth_percent 50 --num_shots 10 --num_digits 5

# Long context
torchrun --nproc_per_node=1 examples/infinite-context-length/run_evals.py \
    --ckpt-path ./checkpoints/model/100 --context_length 16384 --depth_percent 50 --num_shots 10 --num_digits 5
```

## Understanding the Output

The script will output:
- **Input**: The full context with embedded needle information
- **Generation**: The model's response attempting to retrieve the needle
- **Success/Failure**: Whether the model correctly identified the needle

Example output:
```
08/12/2025 18:41:23 [INFO|DP=0|PP=0|TP=0]: input: The needle information is 42857. [long haystack text...]
08/12/2025 18:41:23 [INFO|DP=0|PP=0|TP=0]: generation: 42857
--------------------------------------------------
```

## Performance Expectations

### Good Infini-Attention Performance:
- **High accuracy** (>90%) at all depth percentages
- **Consistent performance** across different context lengths
- **Stable retrieval** even at 99% depth

### Poor Performance Indicators:
- **Declining accuracy** with increasing depth
- **Failure at long contexts** that worked during training
- **Random/hallucinated responses** instead of correct needle retrieval

## Troubleshooting

### Common Issues:

1. **AttributeError: 'NoneType' object has no attribute 'infini_attention'**
   - **Solution**: Ensure the script includes `constants.CONFIG = config` after loading config

2. **CUDA out of memory**
   - **Solutions**: 
     - Reduce `--context_length`
     - Increase `--tp` for tensor parallelism
     - Reduce `--num_shots`

3. **Checkpoint not found**
   - **Check**: Verify checkpoint path exists and contains `config.yaml`
   - **Check**: Ensure model weights are present in the directory

4. **Context length too large**
   - **Limit**: Use `--context_length` ≤ model's `max_position_embeddings`
   - **Check**: Review model config for position embedding limits

## Best Practices

1. **Start Small**: Begin with short contexts and few shots for debugging
2. **Systematic Testing**: Test multiple depth percentages and context lengths
3. **Statistical Validity**: Use sufficient `--num_shots` for reliable results
4. **Resource Management**: Use appropriate parallelism settings for your hardware
5. **Comparative Analysis**: Compare results across different model checkpoints

This evaluation framework helps validate that your Infini-Attention implementation correctly extends the model's effective context length while maintaining retrieval accuracy throughout long sequences.