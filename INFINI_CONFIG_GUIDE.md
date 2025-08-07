# Infini-Attention Configuration Guide

This document provides information about configuring the Infini-Attention mechanism for Llama models.

## Required Configuration Parameters

When configuring Infini-Attention, make sure to include all the following required parameters:

```yaml
infini_attention:
  segment_length: 64                # Length of segments for attention computation
  turn_on_memory: true              # Whether to enable memory in attention
  balance_init_type: zeros          # How to initialize balance factors
  balance_act_type: orig_sigmoid    # Activation type for balance factors
  balance_factor_lr: 0.001          # Learning rate for balance factors
  logging: false                    # Enable detailed logging of attention mechanism
  logging_interval: 100             # How frequently to log attention details
  log_grad: false                   # Whether to log gradients
  log_segment_acts: false           # Whether to log segment activations
```

## Python Configuration

If you're setting up Infini-Attention in Python code, make sure your class includes all required fields:

```python
@dataclass
class InfiniAttentionConfig:
    segment_length: int = 64
    turn_on_memory: bool = True
    balance_init_type: str = "zeros"
    balance_act_type: str = "orig_sigmoid"
    balance_factor_lr: float = 0.001
    logging: bool = False
    logging_interval: int = 100
    log_grad: bool = False
    log_segment_acts: bool = False
```

## Common Issues

1. **Missing Parameters**: If you see an error about missing fields in `InfiniAttentionConfig`, make sure all the parameters above are defined in your config.

2. **Configuration Mismatch**: Ensure that your YAML configuration and Python class definitions have matching parameters.

3. **Type Errors**: Make sure you're using the correct types for each parameter (boolean values should be `true`/`false` in YAML, `True`/`False` in Python).

## Customizing Parameters

- **segment_length**: Longer segments may capture more context but use more memory
- **balance_factor_lr**: Higher values can lead to faster adaptation but potential instability
- **logging**: Enable only during development/debugging as it can slow down training

## Verifying Your Configuration

You can verify your configuration by running:

```bash
python verify_config.py --config-file your_config.yaml
```
