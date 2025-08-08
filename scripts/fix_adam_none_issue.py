#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/fix_adam_none_issue.py

"""
This script fixes the 'unsupported operand type(s) for *: 'float' and 'NoneType'' error
in the PyTorch Adam optimizer.

The error occurs when weight_decay is None and the optimizer tries to perform:
param.mul(1 - lr * weight_decay)

This script should be run before training or imported at the beginning of your training script.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def patch_adam_optimizer():
    """
    Patch the PyTorch Adam optimizer to handle None weight_decay values.
    """
    try:
        import torch
        
        if not hasattr(torch.optim, 'adam'):
            logger.warning("torch.optim.adam module not found, cannot apply patch")
            return False
            
        # Store original function
        original_adam = torch.optim.adam.adam
        
        def patched_adam(*args, **kwargs):
            """
            A wrapper around the PyTorch Adam implementation that ensures weight_decay is never None.
            """
            # Check if weight_decay is None in kwargs and replace with 0.0
            if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
                logger.info("Replaced None weight_decay with 0.0 in Adam optimizer")
                kwargs['weight_decay'] = 0.0
            
            # Handle positional args for weight_decay (usually the 4th argument)
            if len(args) >= 4 and args[3] is None:
                logger.info("Replaced None weight_decay in positional args with 0.0")
                args = list(args)
                args[3] = 0.0
                args = tuple(args)
            
            # Call the original adam function
            return original_adam(*args, **kwargs)
        
        # Replace the original function with our patched version
        torch.optim.adam.adam = patched_adam
        
        # Also patch specific implementations if they exist
        if hasattr(torch.optim.adam, '_single_tensor_adam'):
            original_single = torch.optim.adam._single_tensor_adam
            
            def patched_single_tensor_adam(*args, **kwargs):
                if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
                    kwargs['weight_decay'] = 0.0
                return original_single(*args, **kwargs)
            
            torch.optim.adam._single_tensor_adam = patched_single_tensor_adam
        
        logger.info("Successfully patched PyTorch Adam optimizer")
        return True
    except ImportError:
        logger.error("Failed to import torch - is PyTorch installed?")
        return False
    except Exception as e:
        logger.error(f"Failed to patch PyTorch Adam optimizer: {e}")
        return False

def verify_patch():
    """
    Verify that the patch works by creating an optimizer with weight_decay=None
    """
    try:
        import torch
        
        # Create a simple model and optimizer
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=None)
        
        # Try to step the optimizer
        data = torch.randn(1, 10)
        target = torch.randn(1, 1)
        loss = torch.nn.functional.mse_loss(model(data), target)
        loss.backward()
        optimizer.step()
        
        logger.info("Verification successful: Adam optimizer works with weight_decay=None")
        return True
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Applying Adam optimizer patch...")
    success = patch_adam_optimizer()
    
    if success:
        logger.info("Verifying that the patch works...")
        verify_patch()
        
        logger.info("Patch applied successfully. You can now run your training script.")
        logger.info("To use this patch in your code, add the following at the top of your script:")
        logger.info("import fix_adam_none_issue")
        logger.info("fix_adam_none_issue.patch_adam_optimizer()")
    else:
        logger.error("Failed to apply patch. See error messages above.")
        sys.exit(1)
