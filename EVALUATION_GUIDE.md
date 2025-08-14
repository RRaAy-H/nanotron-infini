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

---

# Passkey Retrieval Benchmark

## Overview

The Passkey Retrieval Benchmark is a specialized evaluation that tests your model's ability to find and retrieve specific information ("passkeys" - usually numbers) from long contexts at various depths. This is particularly important for Infini-Attention models as it validates that the memory mechanism works correctly across segment boundaries.

### What Makes This Different

Unlike the basic needle-in-haystack evaluation above, the passkey benchmark:
- **Uses pre-built standardized datasets** from HuggingFace
- **Tests all depths systematically** (0-100% in 5% increments) 
- **Provides comprehensive analysis** specific to Infini-Attention performance
- **No data generation required** - datasets contain complete prompts

## Quick Start for Your 300M Model

Based on your `fineweb_local_300m_infini_4gpu_config.yaml` configuration:

### 1. Basic Evaluation (1K Context)

```bash
# Run basic passkey evaluation
./examples/infinite-context-length/scripts/run_passkey_eval_300m.sh
```

This will:
- Use your checkpoint at `./checkpoints/fineweb_4gpu_300m_infini/30000`
- Test 1024 token context (within single segment)
- Evaluate 50 samples per depth (0-100%)
- Save results with timestamp

### 2. Custom Checkpoint and Context

```bash
# Specify checkpoint and context length
./examples/infinite-context-length/scripts/run_passkey_eval_300m.sh \
    ./checkpoints/fineweb_4gpu_300m_infini/25000 \
    2048
```

### 3. Full Benchmark (Multiple Context Lengths)

```bash
# Test across 1K, 2K, 4K, and 8K contexts
./examples/infinite-context-length/scripts/run_full_passkey_benchmark_300m.sh
```

This comprehensive benchmark will:
- Test 1024, 2048, 4096, and 8192 token contexts
- Show how performance scales with segment count
- Generate a complete performance report

## Direct Commands (Without Scripts)

If you prefer direct control:

### Basic 1K Context Evaluation
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc_per_node=4 \
    examples/infinite-context-length/scripts/run_passkey_eval.py \
    --ckpt-path ./checkpoints/fineweb_4gpu_300m_infini/30000 \
    --save_path ./results/passkey_test \
    --eval_dataset_path nanotron/llama3-1024-passkey-retrieval-eval \
    --num_shots 0 \
    --num_digits 0 \
    --seed 42 \
    --num_samples 50 \
    --max-new-tokens 15 \
    --dp 4 \
    --tp 1 \
    --pp 1
```

### 16K Context Evaluation
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc_per_node=4 \
    examples/infinite-context-length/scripts/run_passkey_eval.py \
    --ckpt-path ./checkpoints/fineweb_4gpu_300m_infini/30000 \
    --save_path ./results/passkey_16k \
    --eval_dataset_path nanotron/llama3-16k-passkey-retrieval-eval \
    --num_shots 0 \
    --num_digits 0 \
    --seed 42 \
    --num_samples 25 \
    --max-new-tokens 15 \
    --dp 4 \
    --tp 1 \
    --pp 1
```

## Available Pre-built Datasets

### Using Local Datasets (Recommended for Offline Use)

If you have downloaded the datasets locally, place them in your project root:

```bash
# Expected directory structure for parquet files:
./llama3-1024-passkey-retrieval-eval/
└── train-00000-of-00001.parquet          # 1K dataset

./llama3-16k-passkey-retrieval-eval/  
└── train-00000-of-00001.parquet          # 16K dataset
```

The scripts will automatically detect and use local parquet datasets, avoiding network downloads.

### HuggingFace Datasets (Online)

| Dataset | Context Length | Samples | Best For |
|---------|---------------|---------|----------|
| `nanotron/llama3-1024-passkey-retrieval-eval` | 1K tokens | 12,600 | Single segment testing |
| `nanotron/llama3-16k-passkey-retrieval-eval` | 16K tokens | Available | Multi-segment testing |

These datasets contain:
- ✅ Complete prompts with embedded passkeys
- ✅ Passkeys placed at various depths (0-100%)
- ✅ Different shot counts (0, 1, 2, 3 examples)
- ✅ Various passkey lengths (1-4 digits)

### Dataset Structure Example
```python
{
  'prompt': 'There is a pass key hidden inside a lot of irrelevant text...[HAYSTACK]...What is the pass key? The pass key is ',
  'answer': 1234,
  'context_length': 1024,
  'depth_percent': 50,
  'num_shots': 0,
  'num_digits': 4
}
```

## Analyzing Results

### Automatic Analysis

Results are automatically analyzed when using the wrapper scripts:

```bash
# Analyze specific results directory
python examples/infinite-context-length/scripts/analyze_passkey_results.py ./results/passkey_300m_20250814_143022/
```

### Expected Output

```
======================================================================
PASSKEY RETRIEVAL ANALYSIS - 300M INFINI-ATTENTION MODEL
======================================================================
Model: 300M Infini-Attention (segment_length=1024)
Context Length: 1024 tokens
Total Samples: 1050
Expected Segments: 1
======================================================================
Depth   0%: ██████████████████████████████  95.23% (20/21)
Depth   5%: ███████████████████████████████  97.62% (41/42)
Depth  10%: ██████████████████████████████   92.86% (39/42)
...
Depth 100%: ████████████████████████████     88.10% (37/42)

----------------------------------------------------------------------
SUMMARY STATISTICS:
Average Accuracy: 91.45%
Std Deviation: 4.23%
Min Accuracy: 85.71% (at depth 85%)
Max Accuracy: 97.62% (at depth 5%)

----------------------------------------------------------------------
PERFORMANCE ASSESSMENT:
✓ VERY_GOOD: Strong performance within segment
Consistency: CONSISTENT (std dev: 4.23%)

Detailed summary saved to: ./results/passkey_300m_20250814_143022/summary.json
```

## Understanding Performance for Your 300M Infini Model

### Context Length vs Segments

Your model uses `segment_length: 1024`, so:

| Context | Segments | Expected Performance |
|---------|----------|---------------------|
| 1024 tokens | 1 segment | >95% accuracy (within segment) |
| 2048 tokens | 2 segments | >85% accuracy (cross-segment) |
| 4096 tokens | 4 segments | >80% accuracy (longer dependencies) |
| 8192 tokens | 8 segments | >75% accuracy (full training context) |

### Performance Categories

#### ✅ Excellent (>90% average)
- Infini-Attention memory working perfectly
- Strong cross-segment information flow
- Ready for production long-context tasks

#### ✅ Good (80-90% average)  
- Infini-Attention working well
- Minor degradation expected with more segments
- Suitable for most applications

#### ⚠️ Fair (70-80% average)
- Infini-Attention partially working
- May need hyperparameter tuning
- Check balance_factor_lr or segment_length

#### ❌ Poor (<70% average)
- Potential issues with Infini-Attention implementation
- Check training logs and model config
- Consider retraining with different settings

### Troubleshooting Poor Performance

If your model shows poor passkey performance:

1. **Check Infini-Attention Settings**:
   ```yaml
   infini_attention:
     segment_length: 1024
     balance_factor_lr: 0.01    # Try 0.001 or 0.1
     balance_act_type: hard_sigmoid  # Try 'tanh' or 'sigmoid'
   ```

2. **Verify Training Context**: Your model was trained with 8192 tokens, so it should handle contexts up to that length.

3. **Check Tokenizer**: Make sure evaluation uses the same tokenizer (`lvwerra/the-tokenizer-v1`).

## Advanced Usage

### Custom Analysis Options

```bash
# Quiet mode (minimal output)
python examples/infinite-context-length/scripts/analyze_passkey_results.py ./results/ --quiet

# JSON output only
python examples/infinite-context-length/scripts/analyze_passkey_results.py ./results/ --json-only
```

### Comparing Multiple Runs

```bash
# Analyze multiple result directories
for dir in ./results/passkey_*; do
    echo "=== $dir ==="
    python examples/infinite-context-length/scripts/analyze_passkey_results.py "$dir" --quiet
done
```

## Best Practices

1. **Start Small**: Begin with 1K context to verify basic functionality
2. **Progressive Testing**: Test 1K → 2K → 4K → 8K to see scaling behavior  
3. **Multiple Seeds**: Run with different seeds to ensure consistent results
4. **Baseline Comparison**: Compare against model without Infini-Attention
5. **Monitor Memory**: Check GPU memory usage during evaluation

---

# Long-Context Evaluation (Testing True Infini-Attention Scaling)

## Overview

While the pre-built datasets are limited to 16K tokens, Infini-Attention should theoretically handle much longer contexts (32K, 64K, 128K+ tokens) through its memory mechanism. This section shows how to test your model's true long-context capabilities.

## Why Test Beyond Training Context?

Your model was trained with 8192 tokens, but Infini-Attention should generalize to longer sequences:
- **32K tokens = 32 segments**: Tests memory across many boundaries
- **64K tokens = 64 segments**: Tests very long-range dependencies  
- **128K+ tokens**: Tests theoretical limits of the memory mechanism

## Long-Context Testing

### 1. Quick Long-Context Test (32K tokens)

```bash
# Test 32K tokens (32 segments) - 4x your training context
./examples/infinite-context-length/scripts/run_long_context_passkey_eval.sh \
    ./checkpoints/fineweb_4gpu_300m_infini/30000 \
    32768 \
    25
```

**What this tests:**
- Cross-segment information flow over 32 segments
- Memory mechanism stability with long sequences
- Performance degradation vs segment count

### 2. Very Long Context Test (64K tokens)

```bash
# Test 64K tokens (64 segments) - 8x your training context
./examples/infinite-context-length/scripts/run_long_context_passkey_eval.sh \
    ./checkpoints/fineweb_4gpu_300m_infini/30000 \
    65536 \
    15
```

### 3. Extreme Context Test (128K tokens)

```bash
# Test 128K tokens (128 segments) - 16x your training context
./examples/infinite-context-length/scripts/run_long_context_passkey_eval.sh \
    ./checkpoints/fineweb_4gpu_300m_infini/30000 \
    131072 \
    10
```

## Understanding Long-Context Results

### Expected Performance Patterns

| Context Length | Segments | Good Infini-Attention | Poor Implementation |
|---------------|----------|----------------------|-------------------|
| 8K (training) | 8 segments | >85% accuracy | >80% accuracy |
| 32K tokens | 32 segments | >80% accuracy | <60% accuracy |
| 64K tokens | 64 segments | >75% accuracy | <40% accuracy |
| 128K tokens | 128 segments | >70% accuracy | <20% accuracy |

### Performance Indicators

#### ✅ **Excellent Long-Context Performance**
- Accuracy stays >75% even at 128K tokens
- Consistent performance across all depths (0-100%)
- Minimal degradation with increased segment count
- **Conclusion**: Infini-Attention memory mechanism working perfectly

#### ✅ **Good Long-Context Performance**  
- Accuracy >70% at 64K tokens, >65% at 128K
- Some degradation but still functional
- Better performance at early depths vs late depths
- **Conclusion**: Infini-Attention working well with minor optimization potential

#### ⚠️ **Fair Long-Context Performance**
- Accuracy drops significantly beyond training context
- Performance varies widely by depth
- Still better than standard attention (which would fail completely)
- **Conclusion**: Infini-Attention partially working, needs tuning

#### ❌ **Poor Long-Context Performance**
- Accuracy drops to random levels (25-30%) beyond 16K tokens
- No consistent pattern across depths
- Similar to what you'd expect from standard attention
- **Conclusion**: Infini-Attention mechanism not working properly

## How the Long-Context Evaluation Works

The `run_long_context_passkey_eval.sh` script:

1. **Generates Custom Data**: Uses the original `generate_data.py` with your tokenizer
2. **Creates Haystack Content**: Automatically generates diverse text content
3. **Tests Multiple Depths**: Evaluates passkey retrieval at 0%, 25%, 50%, 75%, 100% depths
4. **Handles Full Pipeline**: Data generation → Evaluation → Analysis

### What Makes This Different from Pre-built Datasets

- **Unlimited Context Length**: Test 32K, 64K, 128K, or even longer contexts
- **Custom Content**: Fresh text content that wasn't in training data
- **True Scaling Test**: See how Infini-Attention performs with many segments

## Troubleshooting Long-Context Issues

### If Long-Context Performance is Poor

1. **Check Memory Usage**: Long contexts need significant GPU memory
   ```bash
   # Monitor GPU memory during evaluation
   watch -n 1 nvidia-smi
   ```

2. **Reduce Batch Size**: May need smaller batches for very long contexts
   - The script uses batch_size=1 by default for long contexts

3. **Check Balance Factor**: Long contexts may need different balance_factor_lr
   ```yaml
   infini_attention:
     balance_factor_lr: 0.001  # Try smaller values for long contexts
   ```

4. **Verify Segment Boundaries**: Ensure segments align properly
   - 32K context = exactly 32 segments of 1024 tokens each
   - Misaligned segments can hurt performance

### Expected Resource Requirements

| Context Length | GPU Memory | Evaluation Time | Recommended Samples |
|---------------|------------|----------------|-------------------|
| 32K tokens | ~8-12 GB | ~10-20 min | 25 samples |
| 64K tokens | ~16-24 GB | ~20-40 min | 15 samples |
| 128K tokens | ~32-48 GB | ~40-80 min | 10 samples |

## Advanced Long-Context Testing

### Custom Context Lengths

```bash
# Test specific context length
./examples/infinite-context-length/scripts/run_long_context_passkey_eval.sh \
    <checkpoint> \
    <context_length> \
    <num_samples>

# Examples:
# 24K context (24 segments)
./examples/infinite-context-length/scripts/run_long_context_passkey_eval.sh \
    ./checkpoints/fineweb_4gpu_300m_infini/30000 24576 20

# 96K context (96 segments)  
./examples/infinite-context-length/scripts/run_long_context_passkey_eval.sh \
    ./checkpoints/fineweb_4gpu_300m_infini/30000 98304 8
```

### Comparing Context Lengths

```bash
# Test progression: 8K → 16K → 32K → 64K
CHECKPOINT="./checkpoints/fineweb_4gpu_300m_infini/30000"

echo "Testing context length scaling:"
for CONTEXT in 8192 16384 32768 65536; do
    echo "Testing ${CONTEXT} tokens..."
    ./examples/infinite-context-length/scripts/run_long_context_passkey_eval.sh \
        $CHECKPOINT $CONTEXT 10
done
```

This long-context evaluation is **critical** for validating that Infini-Attention truly extends your model's capabilities beyond the training context length.