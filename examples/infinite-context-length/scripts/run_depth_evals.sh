#!/bin/bash
# USAGE: ./run_depth_evals.sh [checkpoint_path] [context_length] [gpu_id] [output_file]

CHECKPOINT_PATH="${1}"
CONTEXT_LENGTH="${2:-16384}"
GPU_ID="${3:-6}"
OUTPUT_FILE="${4:-depth_eval_results.json}"

# Check required parameters
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint path is required."
    echo "Usage: $0 <checkpoint_path> [context_length] [gpu_id] [output_file]"
    echo "Example: $0 /path/to/checkpoint 16384 6 results.json"
    exit 1
fi

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_PATH"
    exit 1
fi

# Check if torchrun is available
if ! command -v torchrun &> /dev/null; then
    echo "Error: torchrun not found. Please ensure PyTorch is installed with distributed training support."
    echo "You may need to activate your conda/virtual environment first."
    exit 1
fi

echo "=========================================================="
echo "DEPTH PERCENTAGE EVALUATION"
echo "=========================================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Context Length: $CONTEXT_LENGTH"
echo "GPU ID: $GPU_ID"
echo "Output File: $OUTPUT_FILE"
echo "=========================================================="

# Define depth percentages to test
depth_percentages=(0 25 50 75 95)

# Output JSON file
output_file="$OUTPUT_FILE"

# Initialize results array
echo "{" > $output_file
echo "  \"results\": [" >> $output_file

# Run evaluation for each depth percentage
for i in "${!depth_percentages[@]}"; do
    depth=${depth_percentages[$i]}
    echo "Running evaluation with depth percent: $depth"
    
    # Run the evaluation and capture output to a temporary file
    temp_output=$(mktemp)
    
    CUDA_VISIBLE_DEVICES=$GPU_ID CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=1 examples/infinite-context-length/run_evals.py \
        --ckpt-path "$CHECKPOINT_PATH" \
        --context_length $CONTEXT_LENGTH \
        --depth_percent $depth \
        --num_shots 0 \
        --num_digits 3 \
        --dp 1 \
        --pp 1 \
        --tp 1 > "$temp_output" 2>&1
    
    # Read the output
    result=$(cat "$temp_output")
    
    # Extract accuracy from output - try multiple patterns
    accuracy=$(echo "$result" | grep -E "(Accuracy|accuracy): [0-9]*\.?[0-9]*" | grep -oE "[0-9]+\.?[0-9]*" | tail -1)
    
    # Try other common accuracy formats
    if [ -z "$accuracy" ]; then
        accuracy=$(echo "$result" | grep -E "(acc|ACC): [0-9]*\.?[0-9]*" | grep -oE "[0-9]+\.?[0-9]*" | tail -1)
    fi
    
    # Try percentage format
    if [ -z "$accuracy" ]; then
        accuracy=$(echo "$result" | grep -E "[0-9]+\.?[0-9]*%" | grep -oE "[0-9]+\.?[0-9]*" | tail -1)
    fi
    
    # Save the full output for debugging
    echo "=== Full output for depth $depth ===" >> "debug_output_${depth}.log"
    cat "$temp_output" >> "debug_output_${depth}.log"
    echo "" >> "debug_output_${depth}.log"
    
    # Clean up temp file
    rm "$temp_output"
    
    # If accuracy not found, set to null
    if [ -z "$accuracy" ]; then
        accuracy_value="null"
        echo "Warning: Could not extract accuracy for depth $depth. Check debug_output_${depth}.log"
    else
        accuracy_value="$accuracy"
    fi
    
    # Add to JSON (with comma separator except for last item)
    echo "    {" >> $output_file
    echo "      \"depth_percent\": $depth," >> $output_file
    echo "      \"accuracy\": $accuracy_value" >> $output_file
    
    if [ $i -eq $((${#depth_percentages[@]} - 1)) ]; then
        echo "    }" >> $output_file
    else
        echo "    }," >> $output_file
    fi
    
    echo "Completed depth $depth with accuracy: $accuracy_value"
    echo "---"
done

# Close JSON structure
echo "  ]," >> $output_file
echo "  \"timestamp\": \"$(date -Iseconds)\"," >> $output_file
echo "  \"config\": {" >> $output_file
echo "    \"context_length\": $CONTEXT_LENGTH," >> $output_file
echo "    \"num_shots\": 3," >> $output_file
echo "    \"num_digits\": 3," >> $output_file
echo "    \"gpu_id\": $GPU_ID," >> $output_file
echo "    \"checkpoint_path\": \"$CHECKPOINT_PATH\"" >> $output_file
echo "  }" >> $output_file
echo "}" >> $output_file

echo ""
echo "=========================================================="
echo "EVALUATION COMPLETED"
echo "=========================================================="
echo "Results saved to: $output_file"
echo "Debug logs saved to: debug_output_*.log files"
echo "=========================================================="