# Nanotron-Baseline

A large language model pre-training and fine-tuning framework with standard attention implementation for baseline comparisons.

## Overview

A distributed training framework that provides standard attention mechanisms for training large language models. This project serves as a baseline implementation for comparing against attention variants like Infini-attention, enabling fair performance comparisons and model evaluation.

## Key Features

- **Standard Attention Implementation**: Traditional scaled dot-product attention for reliable baseline results
- **Distributed Training**: Multi-GPU and multi-node training support with tensor, pipeline, and data parallelism  
- **Model Support**: LLaMA model family with standard attention mechanisms
- **Flexible Configuration**: YAML-based configuration system for different training scenarios
- **Performance Benchmarking**: Optimized baseline implementation for fair comparisons with attention variants
- **Memory Efficient**: Standard memory management without additional attention modifications

## Quick Start

### Training

#### Single GPU Training
```bash
export CUDA_VISIBLE_DEVICES=0
bash run_single_gpu.sh
```

#### Multi-GPU Training (4 GPUs)
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
bash run_baseline_4gpu.sh
```

#### Custom Training Command
```bash
torchrun --nproc_per_node=4 run_train.py --config-file baseline_config.yaml
```

### Generation
```bash
python run_generate.py --checkpoint-path /path/to/checkpoint
```

### Evaluation

Model evaluation can be performed using the lm-evaluation-harness repository.

For standard context evaluation:
```bash
python examples/infinite-context-length/scripts/run_evals.py --config baseline_config.yaml
```

## Configuration

The project includes configuration files optimized for baseline training scenarios:

- `baseline_config.yaml`: Standard attention training config for 4-GPU setup
- `fineweb_local_200m_baseline_config.yaml`: 200M parameter baseline model configuration
- Various GPU configurations from single GPU to multi-node setups

### Key Configuration Parameters

```yaml
# Disable Infini-attention for baseline
infini_attention:
  turn_on_memory: false  # This ensures standard attention is used

# Standard model configuration
model:
  model_config:
    hidden_size: 1024
    num_hidden_layers: 6
    num_attention_heads: 8
    max_position_embeddings: 512
```

## Project Structure

- `src/nanotron/`: Core framework implementation with standard attention
- `baseline_config.yaml`: Main configuration file for baseline training
- `run_baseline_4gpu.sh`: Multi-GPU training script
- `run_single_gpu.sh`: Single GPU training script
- `examples/`: Training examples and evaluation scripts

## Training Data

The framework supports various data formats:
- **FineWeb**: High-quality web text dataset
- **Parquet files**: Custom dataset format
- **HuggingFace datasets**: Direct integration with HF ecosystem

## Performance Characteristics

This baseline implementation provides:
- Standard O(n²) attention complexity
- Reliable memory usage patterns
- Consistent training dynamics
- Established convergence behavior

Perfect for:
- Establishing performance baselines
- Comparing attention mechanisms
- Standard LLM pre-training tasks
- Research benchmarking

## License

Licensed under the Apache License, Version 2.0.