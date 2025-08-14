#!/bin/bash
# Generate and evaluate long-context passkey data for testing Infini-Attention scaling
set -e

CHECKPOINT_PATH="${1:-./checkpoints/fineweb_4gpu_300m_infini/30000}"
CONTEXT_LENGTH="${2:-32768}"  # Default to 32K tokens to test cross-segment performance
NUM_SAMPLES="${3:-25}"

# Create results directory
SAVE_DIR="./results/long_context_${CONTEXT_LENGTH}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $SAVE_DIR

echo "=========================================================="
echo "LONG-CONTEXT PASSKEY EVALUATION FOR INFINI-ATTENTION"
echo "=========================================================="
echo "Context Length: $CONTEXT_LENGTH tokens"
echo "Expected Segments: $((CONTEXT_LENGTH / 1024))"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Results: $SAVE_DIR"
echo "=========================================================="

# Validate checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

# Create haystack text files (needed by generate_data.py)
HAYSTACK_DIR="./haystack_txt"
mkdir -p $HAYSTACK_DIR

# Generate diverse haystack content
cat > $HAYSTACK_DIR/content1.txt << 'EOF'
The history of ancient civilizations reveals fascinating insights into human development and social structures. Archaeological evidence suggests that complex societies emerged independently in various regions around the world. The development of agriculture was a crucial turning point that allowed humans to settle in permanent communities. These early settlements eventually grew into cities and empires that shaped the course of human history.

Climate change represents one of the most significant challenges facing our planet today. Rising global temperatures are causing ice caps to melt and sea levels to rise. Weather patterns are becoming more extreme, with hurricanes, droughts, and floods occurring with greater frequency and intensity. Scientists around the world are working to understand these changes and develop solutions to mitigate their impact.

Modern technology has transformed the way we communicate and interact with each other. The internet has connected people across the globe, enabling instant communication and access to information. Social media platforms have changed how we share experiences and maintain relationships. Mobile devices have put powerful computing capabilities in the hands of billions of people worldwide.

The study of quantum physics has revealed the strange and counterintuitive nature of reality at the smallest scales. Particles can exist in multiple states simultaneously, and the act of observation can affect the outcome of experiments. Quantum entanglement allows particles to be connected in ways that Einstein called "spooky action at a distance." These discoveries are leading to new technologies like quantum computers and quantum cryptography.
EOF

cat > $HAYSTACK_DIR/content2.txt << 'EOF'
Artificial intelligence is rapidly advancing and changing industries across the globe. Machine learning algorithms can now recognize images, understand speech, and make decisions with increasing accuracy. Deep learning networks modeled after the human brain are solving complex problems in fields ranging from medicine to transportation. As AI systems become more sophisticated, questions arise about their impact on employment and society.

Space exploration continues to push the boundaries of human knowledge and capability. Robotic missions have visited every planet in our solar system, providing detailed information about their composition and characteristics. The International Space Station serves as a laboratory for scientific research in microgravity. Private companies are now developing commercial spaceflight capabilities, opening new possibilities for space tourism and colonization.

The development of renewable energy sources is crucial for sustainable development and environmental protection. Solar panels and wind turbines are becoming more efficient and cost-effective. Battery technology is improving, making it easier to store energy from intermittent sources. Governments and corporations around the world are investing heavily in clean energy infrastructure to reduce carbon emissions.

Ocean conservation efforts are essential for maintaining marine ecosystems and biodiversity. Coral reefs are bleaching due to rising water temperatures and acidification. Plastic pollution is accumulating in massive garbage patches that threaten marine life. Overfishing is depleting fish stocks and disrupting food chains. International cooperation is needed to address these challenges and protect our oceans for future generations.
EOF

# Generate custom passkey data for the specific context length
echo "Generating passkey evaluation data for $CONTEXT_LENGTH tokens..."

# Test multiple depths to see cross-segment performance
DEPTHS=(0 25 50 75 100)
for DEPTH in "${DEPTHS[@]}"; do
    echo "Generating data for depth ${DEPTH}%..."
    
    python examples/infinite-context-length/generate_data.py \
        --context_length $CONTEXT_LENGTH \
        --depth_percent $DEPTH \
        --num_prompts $NUM_SAMPLES \
        --tokenizer_path "lvwerra/the-tokenizer-v1" \
        --id 1 \
        --haystack_dir $HAYSTACK_DIR \
        --save_path $SAVE_DIR \
        --is_exact_context_length "no" \
        --is_padding "no" \
        --is_eval "yes" \
        --num_shots 0 \
        --num_digits 4
    
    # Convert generated data to HuggingFace format and run evaluation
    DATASET_PATH="${SAVE_DIR}/needle_eval_data_and_${CONTEXT_LENGTH}_ctx_and_depth_${DEPTH}_and_id_1_and_num_shots_0"
    
    if [ -d "$DATASET_PATH" ]; then
        echo "Running evaluation for depth ${DEPTH}%..."
        
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        export CUDA_VISIBLE_DEVICES=4,5,6,7
        torchrun --nproc_per_node=4 \
            examples/infinite-context-length/scripts/run_passkey_eval.py \
            --ckpt-path $CHECKPOINT_PATH \
            --save_path "${SAVE_DIR}/results_depth_${DEPTH}" \
            --eval_dataset_path $DATASET_PATH \
            --num_shots 0 \
            --num_digits 0 \
            --seed 42 \
            --num_samples $NUM_SAMPLES \
            --max-new-tokens 15 \
            --dp 4 \
            --tp 1 \
            --pp 1
    else
        echo "WARNING: Dataset generation failed for depth ${DEPTH}%"
    fi
done

echo ""
echo "=========================================================="
echo "LONG-CONTEXT EVALUATION COMPLETED!"
echo "Context Length: $CONTEXT_LENGTH tokens ($((CONTEXT_LENGTH / 1024)) segments)"
echo "Results saved to: $SAVE_DIR"
echo "=========================================================="

# Run analysis if possible
if [ -f "examples/infinite-context-length/scripts/analyze_passkey_results.py" ]; then
    echo "Analyzing results..."
    for DEPTH in "${DEPTHS[@]}"; do
        RESULT_DIR="${SAVE_DIR}/results_depth_${DEPTH}"
        if [ -d "$RESULT_DIR" ] && [ "$(ls -A $RESULT_DIR/*.pkl 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "Analysis for depth ${DEPTH}%:"
            python examples/infinite-context-length/scripts/analyze_passkey_results.py $RESULT_DIR --quiet
            echo ""
        fi
    done
fi

echo "To test even longer contexts, run:"
echo "./examples/infinite-context-length/scripts/run_long_context_passkey_eval.sh $CHECKPOINT_PATH 65536 25  # 64K tokens"
echo "./examples/infinite-context-length/scripts/run_long_context_passkey_eval.sh $CHECKPOINT_PATH 131072 15  # 128K tokens"