# Infini-Attention Memory Testing Guide

## Overview

This guide provides comprehensive testing tools to verify whether your Infini-Attention model actually uses memory during generation/inference. The suite includes automated scripts, visualizations, and detailed analysis to definitively answer: **"Is my model using the memory mechanism?"**

## Quick Start (5 minutes)

### One-Command Verification

```bash
# Test your checkpoint for memory usage
python scripts/test_memory_comprehensive.py --checkpoint ./checkpoints/fineweb_4gpu_300m_infini/30000

# Expected output: PASS/FAIL with confidence score
```

### Quick Visual Check

```bash
# Analyze balance factors (memory vs attention preferences)
python scripts/analyze_balance_factors.py --checkpoint ./checkpoints/fineweb_4gpu_300m_infini/30000
```

**What to Look For:**
- ✅ **Memory Working**: Balance factors vary across layers (0.1-0.9 range)
- ❌ **Memory Not Working**: All balance factors near 0.0 or 1.0

## Understanding Infini-Attention Memory

### What is the Memory Mechanism?

Infini-Attention extends context length by:
1. **Local Attention**: Standard attention within segments (1024 tokens)
2. **Memory Storage**: Compress segment information into memory tensors
3. **Memory Retrieval**: Access compressed information from previous segments
4. **Balance Factor**: Learned gating between local attention and memory retrieval

### Key Indicators of Working Memory

| Indicator | Working Memory | Broken Memory |
|-----------|----------------|---------------|
| Balance Factors | Varied (0.1-0.9) | Uniform (all ~0 or ~1) |
| Cross-Segment Performance | >80% accuracy | <60% accuracy |
| Memory Tensor Norms | Increasing with context | Near zero |
| Long Context Performance | Stable to 32K+ tokens | Drops after 1024 tokens |

## Comprehensive Testing Workflow (30 minutes)

### Phase 1: Balance Factor Analysis

```bash
# Check if model learned to balance memory vs attention
python scripts/analyze_balance_factors.py \
    --checkpoint ./checkpoints/fineweb_4gpu_300m_infini/30000 \
    --output-dir ./memory_analysis/balance_factors
```

**Interpretation:**
- **Healthy Distribution**: Balance factors spread across 0-1 range
- **Layer Variation**: Different layers prefer different balance points
- **Head Diversity**: Different attention heads show different memory preferences

### Phase 2: Memory Usage During Inference

```bash
# Monitor actual memory retrieval during generation
python scripts/debug_memory_usage.py \
    --checkpoint ./checkpoints/fineweb_4gpu_300m_infini/30000 \
    --context-lengths 1024,2048,4096 \
    --output-dir ./memory_analysis/usage_patterns
```

**What This Tests:**
- Memory retrieval activation at segment boundaries (1024, 2048, 3072 tokens)
- Memory tensor content and evolution
- Cross-segment information flow

### Phase 3: Performance Comparison

```bash
# Compare memory-enabled vs memory-disabled performance
python scripts/compare_memory_vs_no_memory.py \
    --checkpoint ./checkpoints/fineweb_4gpu_300m_infini/30000 \
    --test-contexts 2048,4096,8192 \
    --output-dir ./memory_analysis/comparison
```

**Expected Results:**
- **With Memory**: High accuracy across all context lengths
- **Without Memory**: Performance drops significantly beyond 1024 tokens

### Phase 4: Cross-Segment Information Flow

```bash
# Test if memory retains meaningful information across segments
python scripts/memory_content_analysis.py \
    --checkpoint ./checkpoints/fineweb_4gpu_300m_infini/30000 \
    --passkey-depths 0,25,50,75,100 \
    --output-dir ./memory_analysis/content_analysis
```

**Key Metrics:**
- **Information Retention**: Can early information be retrieved from memory?
- **Memory Decay**: How does information degrade across segments?
- **Content Quality**: Is retrieved information semantically meaningful?

## Advanced Analysis (2 hours)

### Progressive Context Length Testing

```bash
# Test memory scaling across increasing context lengths
python scripts/progressive_context_test.py \
    --checkpoint ./checkpoints/fineweb_4gpu_300m_infini/30000 \
    --max-context 16384 \
    --step-size 1024 \
    --samples-per-length 20
```

### Memory State Visualization

```bash
# Launch interactive dashboard for real-time memory monitoring
python scripts/memory_dashboard.py \
    --checkpoint ./checkpoints/fineweb_4gpu_300m_infini/30000 \
    --port 8080
```

Visit `http://localhost:8080` to see:
- Real-time memory state heatmaps
- Balance factor evolution during inference
- Cross-layer memory flow visualization

## Test Results Interpretation

### Balance Factor Analysis

```python
# Sample output interpretation
{
    "balance_factor_stats": {
        "mean": 0.35,           # Average across all layers/heads
        "std": 0.28,            # Variation (higher = more diverse)
        "min": 0.01,            # Minimum balance factor
        "max": 0.89,            # Maximum balance factor
        "layers_preferring_memory": 12,  # Layers with balance > 0.5
        "layers_preferring_attention": 20  # Layers with balance < 0.5
    },
    "interpretation": "HEALTHY: Good balance between memory and attention",
    "confidence": 0.85
}
```

### Memory Usage Patterns

```python
# Sample memory monitoring output
{
    "segment_0": {"memory_norm": 0.0, "retrieval_norm": 0.0},      # No memory yet
    "segment_1": {"memory_norm": 2.43, "retrieval_norm": 0.0},     # Memory stored
    "segment_2": {"memory_norm": 3.12, "retrieval_norm": 1.87},    # Memory retrieved!
    "segment_3": {"memory_norm": 3.89, "retrieval_norm": 2.31},    # Active memory use
    "conclusion": "Memory mechanism is ACTIVE",
    "cross_segment_info_flow": True
}
```

### Performance Comparison

```python
# Sample A/B test results
{
    "with_memory": {
        "1024_tokens": 0.95,    # High accuracy within segment
        "2048_tokens": 0.91,    # Still high with memory
        "4096_tokens": 0.87,    # Memory maintains performance
        "8192_tokens": 0.82     # Long context still good
    },
    "without_memory": {
        "1024_tokens": 0.94,    # Similar baseline
        "2048_tokens": 0.67,    # Sharp drop without memory
        "4096_tokens": 0.45,    # Poor long context
        "8192_tokens": 0.32     # Random performance
    },
    "memory_impact": "SIGNIFICANT (p < 0.001)",
    "effect_size": 0.78        # Large effect size
}
```

## Common Issues and Troubleshooting

### Issue 1: All Balance Factors Near 0

**Symptoms**: Balance factors are 0.01-0.05 across all layers
**Cause**: Model learned to ignore memory mechanism
**Solutions**:
- Check if model was trained with `turn_on_memory: true`
- Verify `balance_factor_lr` was > 0 during training
- Consider retraining with higher balance factor learning rate

### Issue 2: Memory Tensors Always Zero

**Symptoms**: Memory norms remain 0.0 during inference
**Cause**: Memory update mechanism not working
**Solutions**:
- Check if `segment_length` matches training configuration
- Verify checkpoint contains memory-related parameters
- Ensure model architecture includes memory components

### Issue 3: No Cross-Segment Performance Gain

**Symptoms**: Performance identical with/without memory
**Cause**: Memory not providing useful information
**Solutions**:
- Test with longer contexts (4K+ tokens)
- Use more challenging cross-segment tasks
- Check if memory contains meaningful information

### Issue 4: Script Errors

**Common Fixes**:
```bash
# Missing dependencies
pip install torch transformers datasets safetensors plotly

# CUDA out of memory
export CUDA_VISIBLE_DEVICES=0
python script.py --batch-size 1

# Checkpoint path issues
python script.py --checkpoint /full/absolute/path/to/checkpoint
```

## Statistical Interpretation Guide

### Confidence Levels

- **High Confidence (>0.8)**: Clear evidence for/against memory usage
- **Medium Confidence (0.5-0.8)**: Likely working but with some uncertainty  
- **Low Confidence (<0.5)**: Inconclusive results, need more testing

### Effect Sizes

- **Large Effect (>0.8)**: Memory has major impact on performance
- **Medium Effect (0.5-0.8)**: Memory provides meaningful improvement
- **Small Effect (<0.5)**: Memory impact unclear or minimal

### P-Values

- **p < 0.001**: Extremely significant difference
- **p < 0.01**: Highly significant difference
- **p < 0.05**: Significant difference
- **p > 0.05**: No significant difference

## Performance Benchmarks

### Expected Performance Ranges

| Context Length | Memory Working | Memory Broken |
|---------------|----------------|---------------|
| 1024 tokens   | 90-95%        | 90-95%       |
| 2048 tokens   | 85-92%        | 60-75%       |
| 4096 tokens   | 80-90%        | 40-60%       |
| 8192 tokens   | 75-85%        | 25-45%       |
| 16384 tokens  | 70-80%        | 20-35%       |

### Memory Usage Signatures

**Healthy Memory Pattern**:
```
Segment 0: memory_norm=0.00, retrieval=0.00  (baseline)
Segment 1: memory_norm=2.1,  retrieval=0.00  (storing)
Segment 2: memory_norm=2.8,  retrieval=1.4   (retrieving)
Segment 3: memory_norm=3.2,  retrieval=1.9   (active use)
```

**Broken Memory Pattern**:
```
Segment 0: memory_norm=0.00, retrieval=0.00
Segment 1: memory_norm=0.01, retrieval=0.00  (not storing)
Segment 2: memory_norm=0.01, retrieval=0.00  (not retrieving)
Segment 3: memory_norm=0.01, retrieval=0.00  (inactive)
```

## Example Usage Scenarios

### Scenario 1: Verify Training Worked

```bash
# After training an infini-attention model
python scripts/test_memory_comprehensive.py \
    --checkpoint ./my_trained_model/final \
    --quick-test  # Fast verification
```

### Scenario 2: Compare Checkpoints

```bash
# Compare different training checkpoints
for step in 5000 10000 15000 20000; do
    python scripts/analyze_balance_factors.py \
        --checkpoint ./checkpoints/step_$step \
        --output-dir ./analysis/step_$step
done
```

### Scenario 3: Debug Poor Long-Context Performance

```bash
# Deep analysis of memory mechanism issues
python scripts/debug_memory_usage.py \
    --checkpoint ./problematic_model \
    --verbose \
    --save-memory-states \
    --context-lengths 1024,2048,4096,8192
```

### Scenario 4: Research Analysis

```bash
# Comprehensive research-grade analysis
python scripts/test_memory_comprehensive.py \
    --checkpoint ./research_model \
    --full-analysis \
    --statistical-tests \
    --save-all-outputs \
    --generate-report
```

## Output Files and Formats

### Generated Files

```
memory_analysis/
├── balance_factors/
│   ├── balance_factor_heatmap.png
│   ├── layer_wise_distribution.png
│   └── balance_factor_stats.json
├── usage_patterns/
│   ├── memory_evolution.png
│   ├── retrieval_patterns.png
│   └── usage_logs.json
├── comparison/
│   ├── performance_comparison.png
│   ├── statistical_results.json
│   └── significance_tests.json
└── comprehensive_report.html
```

### Report Contents

The HTML report includes:
- Executive summary with pass/fail determination
- Visual analysis with interpretable charts
- Statistical results with confidence intervals
- Recommendations for improvement
- Technical details and raw data

## Contributing and Extending

### Adding New Tests

1. Create new script in `scripts/` directory
2. Follow the common interface pattern:
   - `--checkpoint` parameter for model path
   - `--output-dir` for results
   - JSON output with standardized format
3. Update this README with usage instructions

### Common Utilities

All scripts use shared utilities in `scripts/utils/`:
- `checkpoint_loader.py`: Consistent model loading
- `memory_hooks.py`: Standard memory monitoring hooks
- `statistical_tests.py`: Statistical analysis functions
- `visualization.py`: Common plotting functions

## References and Theory

### Infini-Attention Paper
- Original paper: "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention"
- Key concepts: Compressive memory, balance factors, segment-wise attention

### Testing Methodology
- Balance factor analysis based on learned gating mechanisms
- Cross-segment information flow testing
- Statistical significance testing for performance differences
- Memory content analysis for semantic meaningfulness

### Implementation Details
- Memory storage: Linear transformation of key-value pairs
- Memory retrieval: Attention-like mechanism over compressed memory
- Balance learning: Trainable parameters governing memory vs attention preference

---

## Quick Reference Commands

```bash
# Quick verification (5 minutes)
python scripts/test_memory_comprehensive.py --checkpoint <path> --quick

# Visual analysis (10 minutes)  
python scripts/analyze_balance_factors.py --checkpoint <path>

# Deep debugging (30 minutes)
python scripts/debug_memory_usage.py --checkpoint <path> --verbose

# A/B comparison (20 minutes)
python scripts/compare_memory_vs_no_memory.py --checkpoint <path>

# Interactive dashboard
python scripts/memory_dashboard.py --checkpoint <path>
```

For questions or issues, consult the troubleshooting section above or examine the detailed logs generated by each script.