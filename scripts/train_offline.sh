#!/bin/bash
# This script provides a convenient way to run training in offline mode
# It ensures the tokenizer initialization won't try to download from HuggingFace

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Display header
echo "========================================================="
echo "🔌 Running Infini-Llama training in offline mode"
echo "   No downloads from HuggingFace will be attempted"
echo "========================================================="

# Set environment variables for offline mode
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NO_GIT=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DOWNLOAD_TIMEOUT=1

# Unset proxy variables to prevent connectivity attempts
export http_proxy=""
export https_proxy=""
export HTTP_PROXY=""
export HTTPS_PROXY=""
export all_proxy=""
export ALL_PROXY=""
export no_proxy="*"
export NO_PROXY="*"

# Set PYTHONPATH to include our offline patch
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PYTHONPATH"

# Create a temporary offline wrapper
TEMP_DIR=$(mktemp -d)
WRAPPER_SCRIPT="$TEMP_DIR/offline_wrapper.py"

cat > "$WRAPPER_SCRIPT" << 'PYTHON_EOF'
#!/usr/bin/env python
import os
import sys
import importlib.util

# First import our tokenizer patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import scripts.tokenizer_offline_patch
    scripts.tokenizer_offline_patch.apply_patch()
    print("✅ Applied offline tokenizer patch")
except Exception as e:
    print(f"⚠️ Failed to apply tokenizer patch: {e}")

# Now import the actual workflow script
if len(sys.argv) > 1:
    script_path = sys.argv[1]
    sys.argv = sys.argv[1:]  # Shift arguments
    
    with open(script_path) as f:
        script_code = f.read()
    
    # Execute the script
    exec(compile(script_code, script_path, 'exec'))
else:
    print("Error: No script specified to run")
    sys.exit(1)
PYTHON_EOF

chmod +x "$WRAPPER_SCRIPT"

# Add the --offline-mode flag if it's not already in the arguments
if [[ "$*" != *"--offline-mode"* ]]; then
    ARGS="$* --offline-mode"
else
    ARGS="$*"
fi

# Run the flexible_training_workflow.sh script using our wrapper
python "$WRAPPER_SCRIPT" "$PROJECT_ROOT/scripts/flexible_training_workflow.sh" $ARGS
