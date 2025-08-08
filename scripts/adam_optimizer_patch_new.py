#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/adam_optimizer_patch.py
"""
Direct Adam optimizer patch that can be imported into any script.
This module will patch the PyTorch Adam optimizer to handle None weight_decay values.

Usage:
    import adam_optimizer_patch  # Just importing applies the patch

To verify:
    python -c "import adam_optimizer_patch; adam_optimizer_patch.test_patch()"
"""

import os
import sys
import logging

# Configure simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_patch():
    """Apply the patch to the Adam optimizer class directly"""
    try:
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
                    print("[PATCH] Replaced None weight_decay with 0.0 in Adam optimizer group")
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
                print("[PATCH] Replaced None weight_decay with 0.0 in Adam constructor")
                weight_decay = 0.0
                
            # Call original init with fixed weight_decay
            original_init(self, params, lr=lr, betas=betas, eps=eps, 
                         weight_decay=weight_decay, amsgrad=amsgrad, **kwargs)
        
        # Apply init patch if possible
        try:
            Adam.__init__ = patched_init
        except Exception as e:
            print(f"[WARNING] Could not patch Adam.__init__: {e}")
        
        print("[PATCH] Adam optimizer successfully patched to handle None weight_decay")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to apply Adam optimizer patch: {e}")
        return False

def test_patch():
    """Test that the patch works by creating an optimizer with weight_decay=None"""
    try:
        import torch
        
        # Create a simple model and optimizer
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=None)
        
        # Try using it
        x = torch.randn(1, 10)
        y_pred = model(x)
        loss = (y_pred - torch.randn(1, 1)).pow(2).mean()
        loss.backward()
        optimizer.step()
        
        print("[TEST] Successfully used Adam optimizer with weight_decay=None")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to test Adam optimizer patch: {e}")
        return False

# Apply patch immediately when imported
try:
    apply_patch()
except Exception as e:
    print(f"[ERROR] Failed to apply Adam optimizer patch: {e}")
    # Don't raise - we want to avoid breaking imports
