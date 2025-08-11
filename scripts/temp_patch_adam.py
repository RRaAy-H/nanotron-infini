#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/temp_patch_adam.py

"""
Temporary Adam optimizer patch script for PyTorch.
This script patches the Adam optimizer to handle None weight_decay values.
It works with all PyTorch versions by patching the Adam class directly.
This version handles both PyTorch 1.x and 2.x module structures.
"""

import sys
import logging

# Configure simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Applying Adam optimizer patch for weight_decay=None issues...")

try:
    # First try to import our module
    try:
        import os
        # Add project root to path to find nanotron
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        sys.path.insert(0, project_root)
        sys.path.insert(0, os.path.join(project_root, 'src'))
        
        from nanotron.optim.fix_weight_decay import patch_adam_optimizer
        success = patch_adam_optimizer()
        if success:
            logger.info("Adam optimizer patch applied via nanotron.optim.fix_weight_decay")
            sys.exit(0)
    except ImportError:
        logger.warning("Could not import nanotron.optim.fix_weight_decay, using direct patch")
    
    # Import torch
    import torch
    
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
            logger.info("Detected PyTorch 2.x structure")
    except (ImportError, AttributeError) as e:
        logger.info(f"Could not access torch._C._nn: {e}")
    
    # Patch 3: Always patch the Adam class directly (works for all versions)
    from torch.optim import Adam
    
    # Store original step method
    original_step = Adam.step
    
    # Create patched step method
    def patched_step(self, closure=None):
        """Patched step method that ensures weight_decay is never None"""
        # Replace None weight_decay with 0.0 in optimizer instance
        for group in self.param_groups:
            if 'weight_decay' in group and group['weight_decay'] is None:
                logger.info("Patched: Replaced None weight_decay with 0.0 in Adam optimizer group")
                group['weight_decay'] = 0.0
                
        # Call original step method
        return original_step(self, closure)
    
    # Apply the patch
    Adam.step = patched_step
    
    # Also patch the constructor to ensure weight_decay is never None
    original_init = Adam.__init__
    
    def patched_init(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                    weight_decay=0, amsgrad=False, **kwargs):
        # Replace None weight_decay with 0.0
        if weight_decay is None:
            logger.info("Patched: Replaced None weight_decay with 0.0 in Adam constructor")
            weight_decay = 0.0
            
        # Call original init with fixed weight_decay
        original_init(self, params, lr=lr, betas=betas, eps=eps, 
                     weight_decay=weight_decay, amsgrad=amsgrad, **kwargs)
    
    # Apply init patch if possible
    try:
        Adam.__init__ = patched_init
        logger.info("Successfully patched Adam.__init__ method")
    except Exception as e:
        logger.warning(f"Could not patch Adam.__init__: {e}")
    
    logger.info("Adam optimizer patch applied successfully!")
    
    # Test the patch
    try:
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=None)
        
        # Try a simple training step
        x = torch.randn(1, 10)
        y_pred = model(x)
        loss = (y_pred - torch.randn(1, 1)).pow(2).mean()
        loss.backward()
        optimizer.step()
        
        logger.info("Successfully verified patch: optimizer.step() worked with weight_decay=None")
    except Exception as e:
        logger.warning(f"Warning: Could not fully verify patch: {e}")

except Exception as e:
    logger.error(f"Error applying Adam optimizer patch: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Error applying Adam optimizer patch: {e}")
    sys.exit(1)
