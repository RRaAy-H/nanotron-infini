#!/bin/bash
# enable_offline_mode.sh - Script to enable offline mode for Infini-Llama training
# This script sets up the necessary environment variables and patches to allow
# training without internet connectivity

# Set paths - adjust the project root to point to the repository root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "==========================================="
echo "Enabling offline mode for Infini-Llama"
echo "==========================================="

# Set HuggingFace environment variables to use local files only
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NO_GIT=1

# Disable HTTP requests 
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

# Set timeouts to minimal values to fail fast
export HF_HUB_DOWNLOAD_TIMEOUT=1

# Unset certificate bundles to avoid verification issues
export REQUESTS_CA_BUNDLE=""
export CURL_CA_BUNDLE=""
export SSL_CERT_FILE=""

# Clear any proxy settings
export http_proxy=""
export https_proxy=""
export HTTP_PROXY=""
export HTTPS_PROXY=""
export all_proxy=""
export ALL_PROXY=""
export no_proxy="*"
export NO_PROXY="*"

echo "Environment variables set for offline mode"
echo ""

echo "To train with offline mode, run:"
echo "  ./scripts/flexible_training_workflow.sh --offline-mode [your other options]"
echo ""
echo "For example:"
echo "  ./scripts/flexible_training_workflow.sh --offline-mode --raw-data /path/to/data --config-file scripts/config/your_config.yaml"
echo ""
echo "The changes to engine.py have been made to ensure offline tokenizer loading works correctly."
echo "==========================================="

# Make the script executable
chmod +x scripts/enable_offline_mode.sh

# Create a simple README file for offline mode
cat > /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/OFFLINE_MODE_README.md << 'EOL'
# Running Infini-Llama in Offline Mode

This guide explains how to run Infini-Llama training in offline mode to avoid network connectivity issues.

## Quick Start

To enable offline mode and run training:

```bash
# First, source the offline mode environment variables (optional)
source ./scripts/enable_offline_mode.sh

# Then run your training with the --offline-mode flag
./scripts/flexible_training_workflow.sh --offline-mode --raw-data /path/to/data --config-file scripts/config/your_config.yaml
```

## What Offline Mode Does

When offline mode is enabled:

1. All HuggingFace downloads are prevented
2. Tokenizers load with `local_files_only=True`
3. Network connections to huggingface.co are blocked
4. Fallback tokenizers are used when needed

## Troubleshooting

If you encounter any issues:

1. Make sure you included the `--offline-mode` flag
2. Verify that the engine.py changes are working correctly
3. Check the logs for any tokenizer initialization errors

## Model Selection

In offline mode, you can only use models that are already cached locally. If a model is not found locally, the system will try to use fallback models or create a basic tokenizer.
EOL

echo "Created OFFLINE_MODE_README.md with instructions"
