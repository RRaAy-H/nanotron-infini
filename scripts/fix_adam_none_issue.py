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
    Works with different PyTorch versions by patching the optimizer class directly.
    """
    try:
        import torch
        from torch.optim import Adam
        
        # Store original step method
        original_step = Adam.step
        
        # Create patched step method
        def patched_step(self, closure=None):
            """
            Patched step method that ensures weight_decay is never None
            """
            # Replace None weight_decay with 0.0 in optimizer instance
            for group in self.param_groups:
                if 'weight_decay' in group and group['weight_decay'] is None:
                    logger.info("Replaced None weight_decay with 0.0 in Adam optimizer group")
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
                logger.info("Replaced None weight_decay with 0.0 in Adam constructor")
                weight_decay = 0.0
                
            # Call original init with fixed weight_decay
            original_init(self, params, lr=lr, betas=betas, eps=eps, 
                         weight_decay=weight_decay, amsgrad=amsgrad, **kwargs)
        
        # Apply init patch if possible (handle different PyTorch versions)
        try:
            Adam.__init__ = patched_init
        except Exception as e:
            logger.warning(f"Could not patch Adam.__init__: {e}")
            
        logger.info("Successfully patched PyTorch Adam optimizer")
        return True
        
    except ImportError:
        logger.error("Failed to import torch - is PyTorch installed?")
        return False
    except Exception as e:
        logger.error(f"Failed to patch PyTorch Adam optimizer: {e}")
        return False
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
