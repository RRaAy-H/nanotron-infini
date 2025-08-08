"""
Autopatching module for Adam optimizer weight decay issue.
This module automatically applies patches to the PyTorch Adam optimizer
to handle None weight_decay values.
"""

import logging
import torch
from torch.optim.adam import adam as torch_adam_func

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def patched_adam(*args, **kwargs):
    """
    A wrapper around the PyTorch Adam implementation that ensures weight_decay is never None.
    """
    # Check if weight_decay is None in kwargs and replace with 0.0
    if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
        logger.info("Replaced None weight_decay with 0.0 in Adam optimizer")
        kwargs['weight_decay'] = 0.0
    
    # If args are used (positional weight_decay), we need to handle that too
    # The weight_decay is usually the 4th positional argument in adam
    if len(args) >= 4 and args[3] is None:
        logger.info("Replaced None weight_decay in positional args with 0.0 in Adam optimizer")
        args = list(args)
        args[3] = 0.0
        args = tuple(args)
    
    # Call the original adam function
    return torch_adam_func(*args, **kwargs)

# Apply patches automatically when this module is imported
def _apply_patches():
    """Apply the monkey patch to PyTorch's Adam optimizer."""
    try:
        # Save reference to original function
        original_adam = torch.optim.adam.adam
        
        # Replace with our patched version
        torch.optim.adam.adam = patched_adam
        
        # Also patch the specific functions that handle single tensor operations
        if hasattr(torch.optim.adam, '_single_tensor_adam'):
            original_single_tensor = torch.optim.adam._single_tensor_adam
            
            # Define a patched version
            def patched_single_tensor_adam(params, grads, exp_avgs, exp_avg_sqs, 
                                         max_exp_avg_sqs, state_steps, *, 
                                         beta1, beta2, lr, weight_decay, eps, 
                                         maximize, capturable, differentiable, grad_scale=None, found_inf=None):
                # Ensure weight_decay is not None
                if weight_decay is None:
                    logger.info("Replaced None weight_decay with 0.0 in _single_tensor_adam")
                    weight_decay = 0.0
                
                return original_single_tensor(params, grads, exp_avgs, exp_avg_sqs, 
                                           max_exp_avg_sqs, state_steps, 
                                           beta1=beta1, beta2=beta2, lr=lr, 
                                           weight_decay=weight_decay, eps=eps, 
                                           maximize=maximize, capturable=capturable, 
                                           differentiable=differentiable, 
                                           grad_scale=grad_scale, found_inf=found_inf)
            
            # Apply the patch
            torch.optim.adam._single_tensor_adam = patched_single_tensor_adam
        
        # Also patch multi tensor version if it exists
        if hasattr(torch.optim.adam, '_multi_tensor_adam'):
            original_multi_tensor = torch.optim.adam._multi_tensor_adam
            
            # Define a patched version
            def patched_multi_tensor_adam(*args, **kwargs):
                # Ensure weight_decay is not None in kwargs
                if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
                    logger.info("Replaced None weight_decay with 0.0 in _multi_tensor_adam")
                    kwargs['weight_decay'] = 0.0
                
                return original_multi_tensor(*args, **kwargs)
            
            # Apply the patch
            torch.optim.adam._multi_tensor_adam = patched_multi_tensor_adam
        
        logger.info("Successfully patched PyTorch Adam optimizer")
        return True
    except Exception as e:
        logger.error(f"Failed to patch PyTorch Adam optimizer: {e}")
        return False

# Apply patches when this module is imported
_apply_patches()
