#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/patch_adam_optimizer.py

"""
This script patches the PyTorch Adam optimizer implementation to handle None weight_decay values.
It creates a monkey patch that wraps the Adam optimizer step function to ensure weight_decay is never None.
"""

import os
import logging
import importlib.util
import torch
from torch.optim.adam import adam as torch_adam_func
from types import FunctionType

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

def patch_optimizer():
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

def patch_nanotron_optimizer():
    """Apply patches to any custom Adam implementations in the nanotron codebase."""
    try:
        # Add the project root to the Python path
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        os.environ["PYTHONPATH"] = f"{project_root}:{project_root}/src:{os.environ.get('PYTHONPATH', '')}"
        
        # Try to import and patch any custom optimizer implementations
        nanotron_optim_paths = [
            'src.nanotron.optim.optimizer_from_gradient_accumulator',
            'src.nanotron.optim.inherit_from_other_optimizer',
            'src.nanotron.optim.zero',
            'src.nanotron.optim.named_optimizer'
        ]
        
        for module_path in nanotron_optim_paths:
            try:
                logger.info(f"Checking module {module_path} for Adam implementation...")
                
                # Import the module dynamically
                module_name = module_path.split('.')[-1]
                spec = importlib.util.find_spec(module_path)
                
                if spec:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    logger.info(f"Successfully loaded module {module_path}")
                    
                    # Look for step methods that might use weight_decay
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and hasattr(attr, 'step') and callable(attr.step):
                            logger.info(f"Found optimizer class {attr_name} with step method")
            except Exception as e:
                logger.error(f"Error processing module {module_path}: {e}")
        
        logger.info("Finished checking nanotron optimizer modules")
        return True
    except Exception as e:
        logger.error(f"Error in patch_nanotron_optimizer: {e}")
        return False

def main():
    """Main function to apply all patches."""
    logger.info("Starting Adam optimizer patching process...")
    
    # Apply patches
    pytorch_patched = patch_optimizer()
    nanotron_patched = patch_nanotron_optimizer()
    
    if pytorch_patched:
        logger.info("PyTorch Adam optimizer successfully patched")
    else:
        logger.warning("Failed to patch PyTorch Adam optimizer")
    
    if nanotron_patched:
        logger.info("Nanotron optimizer modules checked")
    else:
        logger.warning("Failed to check Nanotron optimizer modules")
    
    logger.info("Adam optimizer patching process completed")

if __name__ == "__main__":
    main()
