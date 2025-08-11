# Offline Mode for Infini-Llama Training

This document explains how to use the offline mode feature in the flexible training workflow for Infini-Llama models, which helps avoid issues with proxy connections and network access when training.

## Problem Description

When training Infini-Llama models, the training pipeline attempts to download models and tokenizers from the Hugging Face Hub. In environments with restricted network access or behind problematic proxy servers, this can lead to errors like:

```
HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /api/models/gpt2/tree/main/additional_chat_templates?recursive=False&expand=False (Caused by ProxyError('Unable to connect to proxy', ConnectTimeoutError(...)))
```

These errors typically occur when:
1. You're working in a corporate environment with strict firewalls
2. You're behind an unreliable or misconfigured proxy
3. You're in an environment with no internet access
4. The HuggingFace server is unreachable for any reason

The offline mode feature prevents these network access attempts and allows training to proceed using local resources only.

## Using Offline Mode

To enable offline mode, add the `--offline-mode` flag when running the training workflow:

```bash
./flexible_training_workflow.sh --offline-mode --raw-data /path/to/data --config-file scripts/config/your_config.yaml
```

### What Offline Mode Does

When offline mode is enabled:

1. Sets environment variables to prevent Hugging Face libraries from attempting network connections:
   - `HF_DATASETS_OFFLINE=1`
   - `TRANSFORMERS_OFFLINE=1`
   - `HF_HUB_OFFLINE=1`
   - `NO_GIT=1`
   - `HF_HUB_DISABLE_TELEMETRY=1`
   - `HF_HUB_DISABLE_SYMLINKS_WARNING=1`
   - `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`
   - `HF_HUB_DOWNLOAD_TIMEOUT=1` (minimal timeout to fail fast)

2. Clears any proxy settings that might be causing issues:
   - `http_proxy`, `https_proxy`
   - `HTTP_PROXY`, `HTTPS_PROXY`
   - `all_proxy`, `ALL_PROXY`
   - Sets `no_proxy="*"` to bypass all proxies

3. Creates and applies a comprehensive patching system:
   - A wrapper script that patches transformers before import
   - Monkey patches for `AutoTokenizer.from_pretrained()` method
   - Fallback implementations for tokenizers when local files aren't available
   - Patches for `nanotron.trainer` module to prevent network access

4. Provides graceful fallbacks if tokenizers can't be loaded:
   - Attempts to use locally cached tokenizers first
   - Creates basic fallback tokenizers when needed
   - Prevents training failures due to missing tokenizers

## Requirements

To use offline mode effectively, you should have:

1. Pre-downloaded model weights if needed
2. All necessary Python packages already installed
3. Cached tokenizers (from previous runs) if available

## Technical Implementation

The offline mode implements several layers of protection against network access:

1. **Environment Variable Configuration**: Sets various environment variables to prevent HuggingFace libraries from making network requests.

2. **Proxy Bypass**: Clears all proxy settings and configures the system to bypass any proxies.

3. **Python Import Wrapper**: Creates a wrapper script that loads patches before importing any modules.

4. **Monkey Patching**: Patches key methods like `AutoTokenizer.from_pretrained()` to force `local_files_only=True`.

5. **Fallback Mechanisms**: Provides fallback implementations for tokenizers when local files aren't available.

## Troubleshooting

If you encounter issues with offline mode:

1. **First run may fail**: If you've never downloaded the required models before, the first run may fail. In this case:
   - Run once with internet access to cache the required models
   - Then run in offline mode for subsequent training

2. **Missing tokenizers**: If you see errors about missing tokenizers:
   - The offline mode will create a minimal tokenizer that should work for training
   - Pre-run the script with internet access at least once to cache tokenizers

3. **Other network-related errors**: If you still see network-related errors:
   - Check if any environment variables in your system are overriding the offline settings
   - Try running with the additional flag `--verbose` to see more detailed logs
   - Look for logs that contain "Offline mode:" prefix to see what the patch is doing

4. **Proxy issues**: If proxy settings are being reapplied somewhere in your environment:
   - Manually unset them in your shell before running the script: `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`
   - Check your `.bashrc` or `.zshrc` files for automatic proxy settings
   - Use the `--verbose` flag to see if proxy settings are being reapplied

## Advanced Usage

For complex setups, you may need additional configuration:

```bash
# Run in offline mode with extra debugging information
./flexible_training_workflow.sh --offline-mode --verbose --raw-data /path/to/data --config-file scripts/config/your_config.yaml

# Run in offline mode with specific GPU
./flexible_training_workflow.sh --offline-mode --gpu 1 --raw-data /path/to/data --config-file scripts/config/your_config.yaml
```
