"""
Fix for the 'unsupported operand type(s) for *: 'float' and 'NoneType'' error
in the PyTorch Adam optimizer.

This module can be imported directly to apply the fix:
    import nanotron.optim.fix_weight_decay

"""

import torch
import logging

# Configure logging
logger = logging.getLogger(__name__)

def _patch_adam_optimizer():
    """Patch the PyTorch Adam optimizer to handle None weight_decay values."""
    try:
        if not hasattr(torch.optim, 'adam'):
            logger.warning("torch.optim.adam module not found, cannot apply patch")
            return False

        # Store original function
        original_adam = torch.optim.adam.adam
        
        def patched_adam(*args, **kwargs):
            """Patched adam function that handles None weight_decay."""
            # Check if weight_decay is None in kwargs and replace with 0.0
            if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
                logger.info("Fixed: Replaced None weight_decay with 0.0 in Adam optimizer")
                kwargs['weight_decay'] = 0.0
            
            # Handle positional args for weight_decay (usually 4th arg)
            if len(args) >= 4 and args[3] is None:
                logger.info("Fixed: Replaced None weight_decay in positional args with 0.0")
                args = list(args)
                args[3] = 0.0
                args = tuple(args)
            
            # Call original function
            return original_adam(*args, **kwargs)
        
        # Replace with our patched version
        torch.optim.adam.adam = patched_adam
        
        # Also patch _single_tensor_adam if it exists
        if hasattr(torch.optim.adam, '_single_tensor_adam'):
            original_single = torch.optim.adam._single_tensor_adam
            
            def patched_single(*args, **kwargs):
                if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
                    kwargs['weight_decay'] = 0.0
                return original_single(*args, **kwargs)
            
            torch.optim.adam._single_tensor_adam = patched_single
        
        logger.info("Adam optimizer successfully patched to handle None weight_decay")
        return True
    except Exception as e:
        logger.error(f"Failed to patch PyTorch Adam optimizer: {e}")
        return False

# Apply patch when the module is imported
_patch_result = _patch_adam_optimizer()
if not _patch_result:
    logger.warning("Failed to apply Adam optimizer patch, weight_decay=None issues may occur")
