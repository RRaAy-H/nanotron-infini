#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/preimport.py

"""
Pre-import script to apply patches to PyTorch and other libraries.
This script should be imported before any other imports in training scripts.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def patch_pytorch_adam():
    """Patch the PyTorch Adam optimizer to handle None weight_decay values."""
    try:
        import torch.optim.adam
        
        # Store original function
        original_adam = torch.optim.adam.adam
        
        def patched_adam(*args, **kwargs):
            """Patched adam function that handles None weight_decay."""
            # Check if weight_decay is None in kwargs and replace with 0.0
            if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
                logger.info("Replaced None weight_decay with 0.0 in Adam optimizer")
                kwargs['weight_decay'] = 0.0
            
            # Handle positional args for weight_decay (usually 4th arg)
            if len(args) >= 4 and args[3] is None:
                logger.info("Replaced None weight_decay in positional args with 0.0")
                args = list(args)
                args[3] = 0.0
                args = tuple(args)
            
            # Call original function
            return original_adam(*args, **kwargs)
        
        # Replace the original function with our patched version
        torch.optim.adam.adam = patched_adam
        
        # Also patch the specific functions for single and multi tensor operations
        if hasattr(torch.optim.adam, '_single_tensor_adam'):
            original_single_tensor = torch.optim.adam._single_tensor_adam
            
            def patched_single_tensor_adam(*args, **kwargs):
                """Patched _single_tensor_adam that handles None weight_decay."""
                if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
                    logger.info("Replaced None weight_decay with 0.0 in _single_tensor_adam")
                    kwargs['weight_decay'] = 0.0
                return original_single_tensor(*args, **kwargs)
            
            torch.optim.adam._single_tensor_adam = patched_single_tensor_adam
        
        logger.info("Successfully patched PyTorch Adam optimizer")
        return True
    except Exception as e:
        logger.error(f"Failed to patch PyTorch Adam optimizer: {e}")
        return False

# Immediately apply patches when this module is imported
patch_pytorch_adam()
logger.info("Pre-import patches applied")

# Add the fix to PYTHONPATH
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
logger.info(f"Added {project_root} and {os.path.join(project_root, 'src')} to Python path")
