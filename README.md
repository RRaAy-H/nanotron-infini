# Nanotron-Infini

A large language model pre-training and fine-tuning framework with **Infini-attention** implementation.

## Overview

A distributed training framework that incorporates Infini-attention mechanisms, enabling efficient processing of extremely long sequences. The project provides distributed training capabilities for large language models with extended context windows.

## Key Features

- **Infini-attention Implementation**: Enables "infinite-length" context processing with memory-efficient attention mechanisms
- **Distributed Training**: Multi-GPU and multi-node training support with tensor, pipeline, and data parallelism  
- **Model Support**: LLaMA model family with Infini-attention modifications
- **Flexible Configuration**: YAML-based configuration system for different training scenarios
- **Memory Optimization**: Balance factor optimization for managing memory states in long contexts

## Quick Start

### Training
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc_per_node=8 run_train.py --config-file fineweb_local_300m_infini_4gpu_config.yaml
```

### Generation
```bash
python run_generate.py --checkpoint-path /path/to/checkpoint
```

### Evaluation
- Model evaluation can be performed using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) repository.

- For long context needle-in-a-haystack evaluation (up to 32k):
```bash
bash examples/infinite-context-length/scripts/run_evals.sh [depth_percent]
```

## Configuration

The project includes various configuration files for different training scenarios:
- `fineweb_local_*_infini_*gpu_config.yaml`: Infini-attention training configs
- `passkey_finetune_*_optimized_infini_config.yaml`: Fine-tuning for long context tasks

## Project Structure

- `src/nanotron/`: Core framework implementation
- `examples/infinite-context-length/`: Infini-attention specific examples and needle-in-a-haystack evaluations
- `scripts/`: Analysis and utility scripts for balance factors and memory content

## License

Licensed under the Apache License, Version 2.0.
