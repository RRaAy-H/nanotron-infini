# Llama 200M Model Architecture: Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Model Configuration Overview](#model-configuration-overview)
3. [Parameter Definitions & Their Roles](#parameter-definitions--their-roles)
4. [200M Parameter Count Calculation](#200m-parameter-count-calculation)
5. [Architecture Components Deep Dive](#architecture-components-deep-dive)
6. [Code Implementation Analysis](#code-implementation-analysis)
7. [Practical Implications](#practical-implications)

## Introduction

This document provides a comprehensive analysis of the Llama 200M parameter model architecture implemented in the nanotron-infini codebase. We'll explore both the theoretical foundations and practical implementation details, making it accessible to both LLM professionals and deep learning practitioners new to language model training.

### Model Location in Codebase
- **Main Architecture**: `src/nanotron/models/llama.py`
- **Configuration**: `src/nanotron/config/models_config.py`
- **Example 200M Config**: `examples/infinite-context-length/configs/exp33/exp33_200m_infini_llama2_256_ctx_length_and_64_segment_length_and_2m_bs.yaml`

## Model Configuration Overview

The 200M Llama model uses the following core configuration:

```yaml
model_config:
    hidden_size: 1024           # Core embedding dimension
    intermediate_size: 4096     # FFN intermediate dimension  
    num_hidden_layers: 6        # Number of transformer blocks
    num_attention_heads: 8      # Parallel attention heads
    num_key_value_heads: 8      # Key-value heads (GQA support)
    vocab_size: 49152          # Vocabulary size
    max_position_embeddings: 256 # Maximum sequence length
    tie_word_embeddings: false  # Separate input/output embeddings
```

## Parameter Definitions & Their Roles

### 1. `hidden_size: 1024`

**Professional Interpretation**: The fundamental dimensionality of the model's representation space. All token embeddings, attention computations, and layer outputs operate in this 1024-dimensional space.

**Deep Learning Background**: Think of this as the "width" of your neural network. Each token gets represented as a 1024-dimensional vector throughout the model. This is analogous to the hidden units in a traditional feedforward network, but here it's the consistent representation dimension across all operations.

**Code Reference** (`src/nanotron/models/llama.py:398`):
```python
self.d_model = config.hidden_size  # Used throughout attention computations
```

**Impact**: Larger `hidden_size` = more model capacity but quadratically more parameters in attention layers.

### 2. `intermediate_size: 4096`

**Professional Interpretation**: The expansion factor in the feedforward network (FFN) within each transformer block. Following the standard 4x expansion ratio (4096/1024 = 4x).

**Deep Learning Background**: In each transformer layer, after attention, tokens pass through a 2-layer MLP that first expands from 1024 to 4096 dimensions (allowing more complex feature interactions), then contracts back to 1024. This is similar to having a bottleneck layer but in reverse - expand then contract.

**Code Reference** (`src/nanotron/models/llama.py:245-246`):
```python
gate_up_contiguous_chunks = (
    config.intermediate_size,  # shape of gate_linear
    config.intermediate_size,  # shape of up_linear  
)
```

**Impact**: Controls the expressiveness of the FFN. Larger values capture more complex patterns but significantly increase parameter count.

### 3. `num_hidden_layers: 6`

**Professional Interpretation**: The depth of the transformer stack. Each layer contains self-attention + FFN + residual connections + layer normalization.

**Deep Learning Background**: This is like stacking 6 identical neural network blocks on top of each other. Each layer can learn increasingly abstract representations - early layers might learn syntax, later layers might learn semantics. 6 layers is relatively shallow for modern standards but appropriate for a 200M parameter model.

**Code Reference** (`src/nanotron/models/llama.py:1144-1162`):
```python
self.decoder = nn.ModuleList([
    PipelineBlock(
        module_builder=LlamaDecoderLayer,
        # ... module configuration
    )
    for layer_idx in range(config.num_hidden_layers)  # Creates 6 layers
])
```

### 4. `num_attention_heads: 8`

**Professional Interpretation**: Number of parallel attention mechanisms. Each head learns different types of relationships between tokens (e.g., syntactic vs semantic dependencies).

**Deep Learning Background**: Instead of one large attention mechanism, we split the 1024-dimensional space into 8 heads of 128 dimensions each (1024/8 = 128). Each head independently computes attention, then results are concatenated. This is like having 8 different "attention experts" that can focus on different aspects of the input.

**Code Reference** (`src/nanotron/models/llama.py:393-397`):
```python
self.n_local_q_heads = config.num_attention_heads // tp_pg.size()
# ...
self.d_qk = config.hidden_size // config.num_attention_heads  # 1024/8 = 128 per head
```

### 5. `num_key_value_heads: 8`

**Professional Interpretation**: Number of key-value heads for Grouped Query Attention (GQA). When equal to `num_attention_heads`, it's standard multi-head attention. When fewer, it implements GQA for efficiency.

**Deep Learning Background**: In standard attention, each of the 8 query heads has its own key and value heads. GQA allows multiple query heads to share the same key-value heads, reducing memory usage during inference while maintaining most of the performance.

**Code Reference** (`src/nanotron/models/llama.py:376-391`):
```python
assert (
    config.num_attention_heads % config.num_key_value_heads == 0
), f"Number of attention heads ({config.num_attention_heads}) must be divisible by number of key/value heads ({config.num_key_value_heads})."
```

### 6. `vocab_size: 49152`

**Professional Interpretation**: The size of the model's vocabulary - the number of unique tokens the model can process. This is typically set to be divisible by powers of 2 for computational efficiency.

**Deep Learning Background**: Think of this as the model's "dictionary size". Each unique word, subword, or character piece gets assigned a number from 0 to 49,151. The model learns an embedding vector for each of these tokens. Larger vocabularies can represent text more efficiently but require more parameters.

**Code Reference** (`src/nanotron/models/llama.py:1084-1090`):
```python
self.token_embedding = TensorParallelEmbedding(
    num_embeddings=config.vocab_size,  # 49152 embeddings
    embedding_dim=config.hidden_size,  # each is 1024-dimensional
    # ...
)
```

### 7. `max_position_embeddings: 256`

**Professional Interpretation**: Maximum sequence length the model can handle. The model learns position-specific information up to this length using Rotary Position Embeddings (RoPE).

**Deep Learning Background**: Unlike CNNs where position is implicit, transformers need explicit position information. This parameter sets the maximum sequence length. 256 is quite short by modern standards - typically used for experimental or efficient training scenarios.

**Code Reference** (`src/nanotron/models/llama.py:428-434`):
```python
if config.rope_interleaved:
    self.rotary_embedding = RotaryEmbedding(
        dim=self.d_qk, 
        end=config.max_position_embeddings,  # 256
        theta=config.rope_theta
    )
```

### 8. `tie_word_embeddings: false`

**Professional Interpretation**: Whether input token embeddings and output language modeling head share the same weight matrix. When false, they are separate parameters.

**Deep Learning Background**: The model has two embedding matrices: one that converts token IDs to vectors (input), and one that converts final hidden states back to vocabulary probabilities (output). These can share the same weights (tied) to save parameters, or be separate (untied) for more flexibility.

**Code Reference** (`src/nanotron/models/llama.py:1441-1446`):
```python
def get_embeddings_lm_head_tied_names(self):
    """Get the names of the tied embeddings and lm_head weights"""
    if self.config.tie_word_embeddings is True:
        return ["model.token_position_embeddings.pp_block.token_embedding.weight", 
                "model.lm_head.pp_block.weight"]
    else:
        return []  # Empty list when untied
```

## 200M Parameter Count Calculation

Let's break down exactly how we reach ~201.3M parameters:

### 1. Token Embeddings
```python
# Code: src/nanotron/models/llama.py:1084-1090
parameters = vocab_size × hidden_size
parameters = 49,152 × 1,024 = 50,331,648
```

**Explanation**: Each of the 49,152 vocabulary tokens needs a 1024-dimensional embedding vector.

### 2. Per-Layer Parameters (6 layers total)

#### Attention Components

**QKV Projections** (`src/nanotron/models/llama.py:416-424`):
```python
# Query, Key, Value projections
qkv_parameters = hidden_size × (num_attention_heads × d_qk + 2 × num_key_value_heads × d_qk)
qkv_parameters = 1,024 × (8 × 128 + 2 × 8 × 128)
qkv_parameters = 1,024 × (1,024 + 2,048) = 1,024 × 3,072 = 3,145,728
```

**Attention Output Projection** (`src/nanotron/models/llama.py:450-457`):
```python
o_proj_parameters = (num_attention_heads × d_qk) × hidden_size
o_proj_parameters = 1,024 × 1,024 = 1,048,576
```

#### MLP Components (`src/nanotron/models/llama.py:248-264`)

**Gate & Up Projections**:
```python
gate_up_parameters = hidden_size × (2 × intermediate_size)
gate_up_parameters = 1,024 × (2 × 4,096) = 1,024 × 8,192 = 8,388,608
```

**Down Projection**:
```python
down_parameters = intermediate_size × hidden_size
down_parameters = 4,096 × 1,024 = 4,194,304
```

#### Layer Normalization
```python
# Input LayerNorm + Post-Attention LayerNorm
layernorm_parameters = hidden_size × 2 = 1,024 × 2 = 2,048
```

**Total Per Layer**:
```
Attention: 3,145,728 + 1,048,576 = 4,194,304
MLP: 8,388,608 + 4,194,304 = 12,582,912
LayerNorm: 2,048
─────────────────────────
Per Layer Total: 16,779,264
```

**All 6 Layers**: 16,779,264 × 6 = 100,675,584

### 3. Final Components

**Final Layer Norm**: 1,024 parameters

**Language Model Head** (since `tie_word_embeddings: false`):
```python
lm_head_parameters = hidden_size × vocab_size = 1,024 × 49,152 = 50,331,648
```

### Final Calculation
```
Token Embeddings:      50,331,648
Transformer Layers:   100,675,584  
Final Layer Norm:           1,024
LM Head:              50,331,648
──────────────────────────────────
Total:               201,339,904 ≈ 201.3M parameters
```

## Architecture Components Deep Dive

### Multi-Head Attention Implementation

The attention mechanism (`CausalSelfAttention` class) implements several sophisticated features:

```python
# src/nanotron/models/llama.py:361-987
class CausalSelfAttention(nn.Module, AttachableStore):
    def __init__(self, config: LlamaConfig, parallel_config, tp_pg, layer_idx):
        # Tensor parallel considerations
        self.n_local_q_heads = config.num_attention_heads // tp_pg.size()
        self.n_local_kv_heads = config.num_key_value_heads // tp_pg.size()
        
        # Support for Grouped Query Attention
        self.n_repeats = config.num_attention_heads // config.num_key_value_heads
        self.is_gqa = config.num_attention_heads != config.num_key_value_heads
```

**Key Features**:
1. **Flash Attention Integration**: Uses `flash_attn_varlen_func` for memory-efficient attention
2. **Rotary Position Embeddings**: Implements RoPE for better position understanding
3. **KV Caching**: Supports efficient generation with key-value caching
4. **Tensor Parallelism**: Built-in support for distributed training

### MLP Implementation

```python
# src/nanotron/models/llama.py:229-271
class MLP(nn.Module):
    def __init__(self, config, parallel_config, tp_pg):
        # SwiGLU activation: Split into gate and up projections
        self.gate_up_proj = TensorParallelColumnLinear(
            config.hidden_size,
            2 * config.intermediate_size,  # Gate + Up combined
            # ...
        )
        self.down_proj = TensorParallelRowLinear(
            config.intermediate_size,
            config.hidden_size,
            # ...
        )
        self.split_silu_mul = GLUActivation(config.hidden_act)  # SiLU activation
```

**Architecture Choice**: Uses SwiGLU activation (SiLU + Gating), which has been shown to outperform traditional ReLU in language models.

### Infini-Attention Extensions

The model includes experimental Infini-Attention for handling longer sequences:

```python
# src/nanotron/models/llama.py:471-522
self.segment_length = constants.CONFIG.infini_attention.segment_length  # 64
# Balance factors for local vs global attention
self.balance_factors = create_sharded_parameter_from_config(
    parameter=balance_factors,
    pg=tp_pg,
    # ...
)
```

This allows the model to process sequences longer than `max_position_embeddings` by:
1. Splitting sequences into segments
2. Maintaining global memory across segments
3. Balancing local (segment) vs global (memory) attention

## Code Implementation Analysis

### Distributed Training Support

The codebase is built with distributed training as a first-class citizen:

```python
# Pipeline Parallelism
self.token_position_embeddings = PipelineBlock(
    p2p=self.p2p,
    module_builder=Embedding,
    # ...
)

# Tensor Parallelism
self.qkv_proj = TensorParallelColumnLinear(
    self.d_model,
    config.num_attention_heads * self.d_qk + 2 * config.num_key_value_heads * self.d_qk,
    pg=tp_pg,  # Process group for tensor parallelism
    # ...
)
```

### Parameter Initialization

The model supports multiple initialization strategies:

```python
# src/nanotron/models/llama.py:1376-1439
def init_model_randomly(self, config: Config):
    init_method = config.model.init_method
    if isinstance(init_method, RandomInit):
        parametrizator_cls = StandardParametrizator
    elif isinstance(init_method, SpectralMupInit):
        parametrizator_cls = SpectralMupParametrizator
    # ...
```

This includes support for μP (Maximal Update Parameterization) for better scaling behavior.

## Practical Implications

### Model Size vs Performance Trade-offs

**200M Parameter Model Characteristics**:
- **Strengths**: Fast training/inference, low memory footprint, good for experimentation
- **Limitations**: Limited knowledge capacity, may struggle with complex reasoning
- **Sweet Spot**: Proof-of-concept, domain-specific applications, edge deployment

### Memory Requirements

**Training Memory Estimation**:
```
Parameters: 201.3M × 4 bytes (fp32) = ~805MB
Gradients: 201.3M × 4 bytes = ~805MB  
Optimizer States (AdamW): 201.3M × 8 bytes = ~1.6GB
Activations: Varies with batch size and sequence length
Total: ~3.2GB + activations
```

**Inference Memory** (with optimizations):
```
Parameters: 201.3M × 2 bytes (fp16) = ~403MB
KV Cache: Varies with sequence length
Total: ~403MB + KV cache
```

### Computational Complexity

The model's FLOPs scale as:
- **Attention**: O(sequence_length²) per layer
- **FFN**: O(sequence_length × hidden_size × intermediate_size) per layer
- **Total**: Dominated by FFN for short sequences, attention for long sequences

### Configuration Choices Rationale

1. **6 layers**: Balances capacity with efficiency for 200M parameters
2. **8 heads**: Provides good attention diversity without excessive overhead
3. **4x FFN expansion**: Standard ratio that works well empirically
4. **Untied embeddings**: Provides more flexibility at the cost of parameters
5. **Short context (256)**: Enables efficient training and experimentation

This architecture represents a well-balanced design for a compact language model, suitable for research, experimentation, and resource-constrained applications while maintaining the core architectural innovations of modern transformer models.