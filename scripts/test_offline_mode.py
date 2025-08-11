#!/usr/bin/env python
# Test script for offline mode tokenizer loading
# This script verifies that the tokenizer can be loaded without internet access

import os
import sys

# Set offline mode environment variables
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1" 
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["NO_GIT"] = "1"

# Suppress warnings about missing files
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

print("Testing offline mode tokenizer loading...")
print("-" * 50)

# Attempt to load tokenizers with offline mode
try:
    from transformers import AutoTokenizer
    
    print("1. Testing basic tokenizer loading with offline mode...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        print("✅ Successfully loaded GPT-2 tokenizer in offline mode")
        print(f"   Vocabulary size: {len(tokenizer.get_vocab())}")
        print(f"   Model type: {tokenizer.__class__.__name__}")
    except Exception as e:
        print(f"❌ Failed to load GPT-2 tokenizer: {str(e)}")
        
    print("\n2. Testing Llama tokenizer loading with offline mode...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", local_files_only=True)
        print("✅ Successfully loaded Llama-2-7b tokenizer in offline mode")
        print(f"   Vocabulary size: {len(tokenizer.get_vocab())}")
        print(f"   Model type: {tokenizer.__class__.__name__}")
    except Exception as e:
        print(f"❌ Failed to load Llama-2-7b tokenizer: {str(e)}")
        print(f"   Error: {str(e)}")
        print("   This is expected if you haven't cached this model locally")
        
    print("\n3. Testing fallback mechanism...")
    try:
        # Simulate engine.py's fallback logic
        try:
            # Try to load a non-existent model to trigger fallback
            tokenizer = AutoTokenizer.from_pretrained("non-existent-model", local_files_only=True)
        except:
            # Fallback to GPT-2
            try:
                tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
                print("✅ Fallback to GPT-2 tokenizer successful")
            except:
                # Final fallback to basic tokenizer
                from transformers import PreTrainedTokenizer
                tokenizer = PreTrainedTokenizer(
                    unk_token="[UNK]",
                    pad_token="[PAD]",
                    bos_token="[BOS]",
                    eos_token="[EOS]"
                )
                print("✅ Fallback to basic tokenizer successful")
        
        print(f"   Final tokenizer type: {tokenizer.__class__.__name__}")
    except Exception as e:
        print(f"❌ Fallback mechanism failed: {str(e)}")
        
except ImportError:
    print("❌ Failed to import transformers library")
    print("   Please install it with: pip install transformers")
    sys.exit(1)

print("\n4. Testing network blocking...")
try:
    import socket
    original_getaddrinfo = socket.getaddrinfo
    
    def test_connection():
        try:
            # Try to connect to huggingface.co
            socket.getaddrinfo("huggingface.co", 443)
            return True
        except:
            return False
    
    # Check if connection works normally
    normal_connection = test_connection()
    
    # Define a blocking getaddrinfo function
    def blocked_getaddrinfo(*args, **kwargs):
        if "huggingface.co" in args[0]:
            raise socket.gaierror([-2, 'Name or service not known'])
        return original_getaddrinfo(*args, **kwargs)
    
    # Apply the patch
    socket.getaddrinfo = blocked_getaddrinfo
    
    # Check if connection is now blocked
    blocked_connection = not test_connection()
    
    # Restore original function
    socket.getaddrinfo = original_getaddrinfo
    
    if normal_connection:
        print("ℹ️ Normal network connectivity to huggingface.co is available")
    else:
        print("ℹ️ No network connectivity to huggingface.co detected")
        
    if blocked_connection:
        print("✅ Network blocking patch works correctly")
    else:
        print("❌ Network blocking patch failed")
        
except Exception as e:
    print(f"❌ Network test failed: {str(e)}")

print("\n" + "-" * 50)
print("Offline mode test complete")
print("""
To use offline mode in training:
1. Run the enable_offline_mode.sh script first: 
   ./scripts/enable_offline_mode.sh
   
2. Use the --offline-mode flag with flexible_training_workflow.sh:
   ./scripts/flexible_training_workflow.sh --offline-mode [other options]
""")
