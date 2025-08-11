#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/direct_adam_patch.py

"""
Direct patch for the Adam optimizer to handle weight_decay=None issue.
This patch is designed to work with all PyTorch versions (1.x and 2.x).
It handles different code structures in PyTorch by attempting multiple patching methods.
"""

import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_direct_patch():
    """Apply direct patch to Adam optimizer implementation for various PyTorch versions."""
    try:
        import torch
        import inspect
        
        patch_applied = False
        
        # Patch 1: Try to patch torch.optim.adam._single_tensor_adam (PyTorch 1.x)
        try:
            from torch.optim import adam
            
            # Check if we have the _single_tensor_adam function
            if hasattr(adam, '_single_tensor_adam'):
                logger.info("Found _single_tensor_adam function, applying patch...")
                
                # Store original function
                original_func = adam._single_tensor_adam
                
                # Create a patched version that checks for None weight_decay
                def patched_single_tensor_adam(*args, **kwargs):
                    # Fix None weight_decay in kwargs
                    if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
                        logger.info("Fixed: Replaced None weight_decay with 0.0 in _single_tensor_adam kwargs")
                        kwargs['weight_decay'] = 0.0
                    
                    # Call original with fixed weight_decay
                    return original_func(*args, **kwargs)
                
                # Replace the function
                adam._single_tensor_adam = patched_single_tensor_adam
                logger.info("Successfully patched _single_tensor_adam function")
                patch_applied = True
                
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
        except (ImportError, AttributeError) as e:
            logger.info(f"Could not patch torch.optim.adam: {e}")
        
        # Patch 2: Try to patch torch._C._nn (PyTorch 2.x internal C++ implementations)
        try:
            if hasattr(torch._C._nn, "_single_tensor_adam"):
                logger.info("Found torch._C._nn._single_tensor_adam, patching via wrapper...")
                # We can't directly modify the C++ function, so we'll patch the Adam class instead
                patch_applied = True
            else:
                logger.info("No torch._C._nn._single_tensor_adam found")
        except (ImportError, AttributeError) as e:
            logger.info(f"Could not access torch._C._nn: {e}")
        
        # Patch 3: Always patch the Adam class directly (works for all versions)
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
        patch_applied = True
        
        # Patch 4: Also patch the constructor to catch weight_decay=None at initialization
        original_init = Adam.__init__
        
        def patched_init(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                        weight_decay=0, amsgrad=False, **kwargs):
            # Replace None weight_decay with 0.0
            if weight_decay is None:
                logger.info("Fixed: Replaced None weight_decay with 0.0 in Adam constructor")
                weight_decay = 0.0
                
            # Call original init with fixed weight_decay
            original_init(self, params, lr=lr, betas=betas, eps=eps, 
                         weight_decay=weight_decay, amsgrad=amsgrad, **kwargs)
        
        # Apply init patch
        try:
            Adam.__init__ = patched_init
            logger.info("Successfully patched Adam.__init__ method")
            patch_applied = True
        except Exception as e:
            logger.warning(f"Could not patch Adam.__init__: {e}")
        
        return patch_applied
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
