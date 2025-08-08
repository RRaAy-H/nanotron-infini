#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/temp_patch_adam.py

"""
Temporary Adam optimizer patch script for PyTorch.
This script patches the Adam optimizer to handle None weight_decay values.
It works with all PyTorch versions by patching the Adam class directly.
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
    
    # Direct patch if module import failed
    import torch
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
    except Exception as e:
        logger.warning(f"Could not patch Adam.__init__: {e}")
    
    logger.info("Adam optimizer patch applied successfully!")
    
    # Test the patch
    try:
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=None)
        logger.info("Successfully created optimizer with None weight_decay (patch working)")
    except Exception as e:
        logger.warning(f"Warning: Could not verify patch: {e}")
    
except Exception as e:
    logger.error(f"Error applying Adam optimizer patch: {e}")
    sys.exit(1)
