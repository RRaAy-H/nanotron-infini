#!/usr/bin/env python
# This script patches the PipelineEngine initialization to always use offline mode for tokenizer loading

import os
import sys
import importlib.util
from functools import wraps
import warnings

# Suppress warnings from transformers about missing files
warnings.filterwarnings('ignore', category=UserWarning)

def apply_patch():
    """Apply offline mode patch to nanotron's tokenizer initialization"""
    try:
        # First check if nanotron is importable
        from nanotron.parallel.pipeline_parallel import engine
        
        # Get the original PipelineEngine.__init__ method
        original_init = engine.PipelineEngine.__init__
        
        @wraps(original_init)
        def patched_init(self, *args, **kwargs):
            # Call the original __init__ first
            original_init(self, *args, **kwargs)
            
            # Now patch the tokenizer initialization to always use offline mode
            from transformers import AutoTokenizer, PreTrainedTokenizer
            
            # Log that we're using the patched version
            print("Applying offline mode patch for tokenizer initialization")
            
            try:
                # Try loading with local_files_only=True
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Llama-2-7b-hf",
                    local_files_only=True
                )
                print("Successfully loaded Meta Llama tokenizer in offline mode")
            except Exception as e1:
                print(f"Failed to load Meta Llama tokenizer: {e1}")
                try:
                    # Try GPT-2 tokenizer as fallback
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        "gpt2",
                        local_files_only=True
                    )
                    print("Successfully loaded GPT-2 tokenizer in offline mode")
                except Exception as e2:
                    print(f"Failed to load GPT-2 tokenizer: {e2}")
                    # Last resort - create a basic tokenizer
                    print("Creating basic tokenizer as last resort")
                    self.tokenizer = PreTrainedTokenizer(
                        unk_token="[UNK]",
                        pad_token="[PAD]", 
                        bos_token="[BOS]",
                        eos_token="[EOS]"
                    )
        
        # Apply our patch
        engine.PipelineEngine.__init__ = patched_init
        print("✅ Successfully patched PipelineEngine for offline tokenizer loading")
        return True
        
    except ImportError as e:
        print(f"⚠️ Could not patch nanotron engine: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Error while applying tokenizer patch: {e}")
        return False

if __name__ == "__main__":
    # Set offline environment variables
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" 
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["NO_GIT"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    
    success = apply_patch()
    if success:
        print("Offline tokenizer patch ready to use")
        print("To use this patch, add this import before importing nanotron:")
        print("import tokenizer_offline_patch; tokenizer_offline_patch.apply_patch()")
    else:
        print("Failed to prepare offline tokenizer patch")
