#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/direct_adam_patch.py

"""
Direct patch for the Adam optimizer's _single_tensor_adam function.
This patch targets the specific function that's causing the error in PyTorch 2.x versions.
"""

import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_direct_patch():
    """Apply direct patch to Adam optimizer implementation."""
    try:
        import torch
        import inspect
        from torch.optim import adam
        
        # Check if we have the _single_tensor_adam function
        if hasattr(adam, '_single_tensor_adam'):
            logger.info("Found _single_tensor_adam function, applying patch...")
            
            # Store original function
            original_func = adam._single_tensor_adam
            original_source = inspect.getsource(original_func)
            
            # Create a patched version that checks for None weight_decay
            def patched_single_tensor_adam(params,
                                          grads,
                                          exp_avgs,
                                          exp_avg_sqs,
                                          max_exp_avg_sqs,
                                          state_steps,
                                          *,
                                          beta1,
                                          beta2,
                                          lr,
                                          weight_decay,
                                          eps,
                                          amsgrad,
                                          maximize,
                                          capturable):
                # Fix None weight_decay
                if weight_decay is None:
                    logger.info("Fixed: Replaced None weight_decay with 0.0 in _single_tensor_adam")
                    weight_decay = 0.0
                
                # Call original with fixed weight_decay
                return original_func(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, 
                                    state_steps, beta1=beta1, beta2=beta2, lr=lr, 
                                    weight_decay=weight_decay, eps=eps, amsgrad=amsgrad,
                                    maximize=maximize, capturable=capturable)
            
            # Replace the function
            adam._single_tensor_adam = patched_single_tensor_adam
            logger.info("Successfully patched _single_tensor_adam function")
            
            # Also try to patch the adam function if it exists
            if hasattr(adam, 'adam'):
                original_adam = adam.adam
                
                def patched_adam(*args, **kwargs):
                    # Fix None weight_decay in kwargs
                    if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
                        logger.info("Fixed: Replaced None weight_decay with 0.0 in adam function kwargs")
                        kwargs['weight_decay'] = 0.0
                    
                    # Fix None weight_decay in positional args (typically 4th arg)
                    if len(args) >= 4 and args[3] is None:
                        logger.info("Fixed: Replaced None weight_decay in adam function positional args")
                        args = list(args)
                        args[3] = 0.0
                        args = tuple(args)
                    
                    # Call original function
                    return original_adam(*args, **kwargs)
                
                # Replace the function
                adam.adam = patched_adam
                logger.info("Also patched adam function")
            
            return True
        else:
            logger.warning("_single_tensor_adam function not found in torch.optim.adam")
            
            # Fall back to patching the Adam class directly
            logger.info("Falling back to patching Adam class directly...")
            from torch.optim import Adam
            
            # Store original step method
            original_step = Adam.step
            
            # Create a patched step method
            def patched_step(self, closure=None):
                # Fix None weight_decay in param_groups
                for group in self.param_groups:
                    if 'weight_decay' in group and group['weight_decay'] is None:
                        logger.info("Fixed: Replaced None weight_decay with 0.0 in Adam param_group")
                        group['weight_decay'] = 0.0
                
                # Call original step method
                return original_step(self, closure)
            
            # Replace the method
            Adam.step = patched_step
            logger.info("Successfully patched Adam.step method")
            
            return True
    except Exception as e:
        logger.error(f"Failed to patch Adam optimizer: {e}")
        return False

def test_patch():
    """Test that the patch works by creating an optimizer with weight_decay=None."""
    try:
        import torch
        
        # Create a simple model and optimizer
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=None)
        
        # Try using it with dummy data
        x = torch.randn(1, 10)
        y_pred = model(x)
        loss = (y_pred - torch.randn(1, 1)).pow(2).mean()
        loss.backward()
        optimizer.step()
        
        logger.info("Test successful: optimizer.step() worked with weight_decay=None")
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Applying direct patch to Adam optimizer...")
    success = apply_direct_patch()
    
    if success:
        logger.info("Testing patch...")
        test_success = test_patch()
        
        if test_success:
            logger.info("Adam optimizer patch verified and working correctly")
        else:
            logger.error("Patch applied but failed verification test")
            sys.exit(1)
    else:
        logger.error("Failed to apply patch")
        sys.exit(1)
