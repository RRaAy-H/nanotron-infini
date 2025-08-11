# Network Connectivity Fix for Infini-Llama Training

This document explains how to use the offline mode feature to fix network connectivity issues during Infini-Llama training.

## Problem Description

When training Infini-Llama models, you might encounter the following error:

```
HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /api/models/gpt2/tree/main (Caused by ProxyError('Unable to connect to proxy', ConnectTimeoutError(...)))
```

This occurs because:
1. The training pipeline attempts to download tokenizer files from Hugging Face
2. Network connectivity or proxy issues prevent successful connection
3. The training fails during model initialization

## Solution: Offline Mode

We've implemented a comprehensive solution by:

1. Adding proper offline mode support to the pipeline engine
2. Creating helper scripts for easy offline mode configuration
3. Implementing fallback mechanisms for tokenizer initialization

## How to Use Offline Mode

### Option 1: Use the `--offline-mode` Flag

Simply add the `--offline-mode` flag to your training command:

```bash
./scripts/flexible_training_workflow.sh --offline-mode --raw-data /path/to/data --config-file scripts/config/your_config.yaml
```

This will automatically:
- Set necessary environment variables to prevent downloads
- Apply runtime patches to tokenizer loading
- Use locally cached models or fallbacks

### Option 2: Set Up the Environment Manually

1. Run the included helper script:

```bash
./scripts/enable_offline_mode.sh
```

2. Then run your training command with the offline mode flag:

```bash
./scripts/flexible_training_workflow.sh --offline-mode [your options]
```

## Testing Offline Mode

You can verify that offline mode is working correctly by running:

```bash
python ./scripts/test_offline_mode.py
```

This will test:
1. Loading tokenizers in offline mode
2. The fallback mechanism for unavailable models
3. Network blocking functionality

## Implementation Details

1. **Environment Variables**:
   - Sets `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`
   - Disables telemetry, token usage, and other network features
   - Clears all proxy settings to prevent connection attempts

2. **Tokenizer Loading**:
   - Modified `engine.py` to detect offline mode
   - Added proper `local_files_only=True` parameter to tokenizer loading
   - Implemented multi-level fallback mechanisms for tokenizers

3. **Error Handling**:
   - Added robust exception handling for tokenizer loading failures
   - Improved logging for better troubleshooting
   - Created proper fallback to basic tokenizer when needed

## Troubleshooting

If you still encounter issues:

1. Verify your environment has the transformers library installed
2. Check that you've included the `--offline-mode` flag
3. Run the test script to identify any issues with tokenizer loading
4. Check for any error messages about tokenizer initialization in the logs
