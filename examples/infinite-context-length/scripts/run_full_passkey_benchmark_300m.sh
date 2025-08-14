#!/bin/bash
# Full passkey benchmark for 300M Infini-Attention model

set -e  # Exit on any error

# Configuration
CHECKPOINT_PATH="${1:-./checkpoints/fineweb_4gpu_300m_infini/30000}"
NUM_SAMPLES="${2:-25}"  # Reduced per context to speed up full benchmark

# Create base results directory with timestamp
BASE_SAVE_DIR="./results/passkey_benchmark_300m_$(date +%Y%m%d_%H%M%S)"
mkdir -p $BASE_SAVE_DIR

# Validate checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint path does not exist: $CHECKPOINT_PATH"
    echo "Available checkpoints:"
    ls -la ./checkpoints/fineweb_4gpu_300m_infini/ 2>/dev/null || echo "  No checkpoints found"
    exit 1
fi

# Test different context lengths that 300M model can handle
# Start with smaller contexts and work up to the training context (8192)
CONTEXT_LENGTHS=(1024 2048 4096 8192)

echo "============================================================="
echo "FULL PASSKEY BENCHMARK FOR 300M INFINI-ATTENTION MODEL"
echo "============================================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Samples per context: $NUM_SAMPLES"
echo "Context lengths to test: ${CONTEXT_LENGTHS[@]}"
echo "Infini segment length: 1024 tokens"
echo "Expected segments per context:"
for CONTEXT in "${CONTEXT_LENGTHS[@]}"; do
    SEGMENTS=$((CONTEXT / 1024))
    echo "  ${CONTEXT} tokens = ${SEGMENTS} segment(s)"
done
echo "Results will be saved to: $BASE_SAVE_DIR"
echo "============================================================="

# Initialize summary file
SUMMARY_FILE="$BASE_SAVE_DIR/benchmark_summary.txt"
echo "PASSKEY BENCHMARK SUMMARY - 300M INFINI-ATTENTION MODEL" > $SUMMARY_FILE
echo "Checkpoint: $CHECKPOINT_PATH" >> $SUMMARY_FILE
echo "Date: $(date)" >> $SUMMARY_FILE
echo "=============================================================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Track overall progress
TOTAL_CONTEXTS=${#CONTEXT_LENGTHS[@]}
CURRENT_CONTEXT=0
FAILED_CONTEXTS=()
SUCCESSFUL_CONTEXTS=()

for CONTEXT in "${CONTEXT_LENGTHS[@]}"; do
    CURRENT_CONTEXT=$((CURRENT_CONTEXT + 1))
    
    echo ""
    echo "[$CURRENT_CONTEXT/$TOTAL_CONTEXTS] ==============================================="
    echo "TESTING CONTEXT LENGTH: $CONTEXT tokens"
    echo "Expected segments: $((CONTEXT / 1024))"
    echo "=================================================="
    
    # Create context-specific results directory
    SAVE_DIR="${BASE_SAVE_DIR}/context_${CONTEXT}"
    mkdir -p $SAVE_DIR
    
    # Check for local datasets first, fallback to HuggingFace
    LOCAL_1K_DATASET="./llama3-1024-passkey-retrieval-eval"
    LOCAL_16K_DATASET="./llama3-16k-passkey-retrieval-eval"
    
    if [ "$CONTEXT" -le 1024 ]; then
        if [ -d "$LOCAL_1K_DATASET" ]; then
            DATASET="$LOCAL_1K_DATASET"
            echo "Using local 1K dataset: $DATASET"
        else
            DATASET="nanotron/llama3-1024-passkey-retrieval-eval"
            echo "Using HuggingFace 1K dataset: $DATASET"
        fi
    elif [ "$CONTEXT" -le 16384 ]; then
        if [ -d "$LOCAL_16K_DATASET" ]; then
            DATASET="$LOCAL_16K_DATASET"
            echo "Using local 16K dataset: $DATASET"
        else
            DATASET="nanotron/llama3-16k-passkey-retrieval-eval"
            echo "Using HuggingFace 16K dataset: $DATASET"
        fi
    else
        echo "WARNING: Context length $CONTEXT exceeds available datasets"
        echo "Skipping context length $CONTEXT"
        FAILED_CONTEXTS+=($CONTEXT)
        continue
    fi
    
    # Set environment variables
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export CUDA_VISIBLE_DEVICES=4,5,6,7
    
    echo "Starting evaluation for $CONTEXT token context..."
    
    # Run evaluation with error handling
    if torchrun --nproc_per_node=4 \
        examples/infinite-context-length/scripts/run_passkey_eval.py \
        --ckpt-path $CHECKPOINT_PATH \
        --save_path $SAVE_DIR \
        --eval_dataset_path $DATASET \
        --num_shots 0 \
        --num_digits 0 \
        --seed 42 \
        --num_samples $NUM_SAMPLES \
        --max-new-tokens 15 \
        --dp 4 \
        --tp 1 \
        --pp 1; then
        
        echo "✓ Context $CONTEXT: COMPLETED SUCCESSFULLY"
        SUCCESSFUL_CONTEXTS+=($CONTEXT)
        
        # Count result files
        RESULT_FILES=$(ls $SAVE_DIR/*.pkl 2>/dev/null | wc -l)
        echo "Generated $RESULT_FILES result files"
        
        # Quick analysis if script exists
        if [ -f "examples/infinite-context-length/scripts/analyze_passkey_results.py" ] && [ $RESULT_FILES -gt 0 ]; then
            echo "Running quick analysis..."
            python examples/infinite-context-length/scripts/analyze_passkey_results.py $SAVE_DIR >> $SUMMARY_FILE 2>&1
            echo "" >> $SUMMARY_FILE
        fi
        
    else
        echo "✗ Context $CONTEXT: FAILED"
        FAILED_CONTEXTS+=($CONTEXT)
        echo "Error details logged to: $SAVE_DIR/error.log"
    fi
    
    echo "Context $CONTEXT evaluation finished."
done

# Final summary
echo ""
echo "============================================================="
echo "BENCHMARK COMPLETED!"
echo "============================================================="
echo "Successful contexts: ${SUCCESSFUL_CONTEXTS[@]:-None}"
echo "Failed contexts: ${FAILED_CONTEXTS[@]:-None}"

# Write final summary
{
    echo ""
    echo "FINAL SUMMARY:"
    echo "============================================================="
    echo "Successful contexts (${#SUCCESSFUL_CONTEXTS[@]}): ${SUCCESSFUL_CONTEXTS[@]:-None}"
    echo "Failed contexts (${#FAILED_CONTEXTS[@]}): ${FAILED_CONTEXTS[@]:-None}"
    echo ""
    echo "Analysis: With segment_length=1024, we expect:"
    echo "- 1024 tokens: 1 segment (best performance)"
    echo "- 2048 tokens: 2 segments (cross-segment retrieval)"
    echo "- 4096 tokens: 4 segments (longer dependencies)"
    echo "- 8192 tokens: 8 segments (full training context)"
    echo ""
    echo "Good Infini-Attention should show:"
    echo "- >95% accuracy within single segment (1024 tokens)"
    echo "- >85% accuracy across 2-4 segments"
    echo "- >75% accuracy at full training context (8192 tokens)"
} >> $SUMMARY_FILE

echo ""
echo "Complete results saved to: $BASE_SAVE_DIR"
echo "Summary saved to: $SUMMARY_FILE"

# Show quick results if available
if [ ${#SUCCESSFUL_CONTEXTS[@]} -gt 0 ]; then
    echo ""
    echo "Quick Results Preview:"
    echo "----------------------"
    for CONTEXT in "${SUCCESSFUL_CONTEXTS[@]}"; do
        RESULTS_DIR="$BASE_SAVE_DIR/context_$CONTEXT"
        if [ -f "$RESULTS_DIR/summary.json" ]; then
            AVG_ACC=$(python -c "import json; print(f\"{json.load(open('$RESULTS_DIR/summary.json'))['average_accuracy']:.1f}%\")" 2>/dev/null || echo "N/A")
            echo "$CONTEXT tokens: $AVG_ACC average accuracy"
        fi
    done
fi

if [ ${#FAILED_CONTEXTS[@]} -gt 0 ]; then
    echo ""
    echo "WARNING: Some contexts failed. Check individual error logs in:"
    for CONTEXT in "${FAILED_CONTEXTS[@]}"; do
        echo "  $BASE_SAVE_DIR/context_$CONTEXT/"
    done
    exit 1
else
    echo ""
    echo "All contexts completed successfully! 🎉"
fi