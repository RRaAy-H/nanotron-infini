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

def apply_patch():
    """Apply the patch to torch.optim.adam.adam"""
    import torch.optim.adam
    
    # Store original function
    original_adam = torch.optim.adam.adam
    
    def patched_adam(*args, **kwargs):
        """Patched adam function that handles None weight_decay"""
        # Check if weight_decay is None in kwargs and replace with 0.0
        if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
            print("[PATCH] Replaced None weight_decay with 0.0 in Adam optimizer")
            kwargs['weight_decay'] = 0.0
        
        # Handle positional args for weight_decay (usually 4th arg)
        if len(args) >= 4 and args[3] is None:
            print("[PATCH] Replaced None weight_decay in positional args with 0.0")
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
    
    print("[PATCH] Adam optimizer successfully patched to handle None weight_decay")
    return True

def test_patch():
    """Test that the patch works by creating an optimizer with weight_decay=None"""
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

# Apply patch immediately when imported
try:
    apply_patch()
except Exception as e:
    print(f"[ERROR] Failed to apply Adam optimizer patch: {e}")
    # Don't raise - we want to avoid breaking imports
