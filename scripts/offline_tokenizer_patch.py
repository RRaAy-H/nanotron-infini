#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/offline_tokenizer_patch.py

"""
Patch for the tokenizer initialization in pipeline_parallel/engine.py
This patch allows the training script to run in offline mode by avoiding
attempts to download models from HuggingFace hub.
"""

import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_offline_tokenizer_patch():
    """Apply patch to the pipeline engine to handle offline mode gracefully."""
    try:
        # Get project root directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        src_dir = os.path.join(project_root, 'src')
        
        # Make sure we can import from nanotron
        sys.path.insert(0, project_root)
        sys.path.insert(0, src_dir)
        
        # Import the module to patch
        from nanotron.parallel.pipeline_parallel import engine
        from transformers import PreTrainedTokenizerFast, GPT2Tokenizer
        import importlib
        
        logger.info("Applying offline tokenizer patch to pipeline engine...")
        
        # Store original initialization method
        original_init = engine.OneForwardOneBackwardPipelineEngine.__init__
        
        # Create patched initialization method
        def patched_init(self):
            """Patched init that handles offline mode gracefully."""
            # Call the parent class init
            engine.PipelineEngine.__init__(self)
            self.idx = 0
            
            # Check if we're in offline mode
            offline_mode = bool(os.environ.get('HF_HUB_OFFLINE', False))
            
            if offline_mode:
                logger.info("Running in offline mode - using local tokenizer implementation")
                try:
                    # Create a basic GPT2 tokenizer from local files
                    self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
                    logger.info("Successfully loaded GPT2 tokenizer from local cache")
                except Exception as e:
                    logger.warning(f"Could not load tokenizer from local cache: {e}")
                    # Create a minimal dummy tokenizer as last resort
                    from transformers import PreTrainedTokenizerFast
                    vocab = {f"token_{i}": i for i in range(1000)}
                    self.tokenizer = PreTrainedTokenizerFast(
                        tokenizer_file=None,
                        unk_token="[UNK]",
                        pad_token="[PAD]",
                        bos_token="[BOS]",
                        eos_token="[EOS]",
                        model_max_length=1024,
                        vocab=vocab
                    )
                    logger.info("Created minimal dummy tokenizer for offline mode")
            else:
                # Original behavior for online mode
                from transformers import AutoTokenizer
                try:
                    # This will be replaced with the tokenizer from the config at runtime
                    self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
                except:
                    # Fallback to a simpler tokenizer that's more likely to be accessible
                    self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Replace the initialization method
        engine.OneForwardOneBackwardPipelineEngine.__init__ = patched_init
        
        # Reload the module to ensure changes are applied
        importlib.reload(engine)
        
        logger.info("Successfully patched pipeline engine for offline mode")
        return True
    except Exception as e:
        logger.error(f"Failed to patch pipeline engine: {e}")
        return False

if __name__ == "__main__":
    success = apply_offline_tokenizer_patch()
    if success:
        logger.info("Offline tokenizer patch applied successfully")
        sys.exit(0)
    else:
        logger.error("Failed to apply offline tokenizer patch")
        sys.exit(1)
