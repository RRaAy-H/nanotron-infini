"""
Fix for the 'unsupported operand type(s) for *: 'float' and 'NoneType'' error
in the PyTorch Adam optimizer.

This module can be imported directly to apply the fix:
    import nanotron.optim.fix_weight_decay

Or call the patch function directly:
    from nanotron.optim.fix_weight_decay import patch_adam_optimizer
    patch_adam_optimizer()
    
The patch supports all PyTorch versions including 2.x where the module 
structure has changed significantly.
"""

import torch
import logging
import inspect
import sys

# Configure logging
logger = logging.getLogger(__name__)

def patch_adam_optimizer():
    """
    Patch the PyTorch Adam optimizer to handle None weight_decay values.
    Works with different PyTorch versions by attempting multiple patch strategies.
    
    For PyTorch 1.x: Patches torch.optim.adam._single_tensor_adam
    For PyTorch 2.x: Patches the Adam class directly and attempts to access
                    internal implementations if possible
    """
    try:
        success = False
        
        # Get PyTorch version to apply appropriate patches
        pytorch_version = torch.__version__
        logger.info(f"Detected PyTorch version: {pytorch_version}")
        
        # Strategy 1: Try to patch torch.optim.adam (PyTorch 1.x structure)
        try:
            # This will fail in PyTorch 2.x as the module structure changed
            from torch.optim import adam
            
            # Patch _single_tensor_adam if it exists
            if hasattr(adam, '_single_tensor_adam'):
                # Store original function
                original_func = adam._single_tensor_adam
                
                # Create a patched version that checks for None weight_decay
                def patched_single_tensor_adam(*args, **kwargs):
                    # Fix None weight_decay
                    if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
                        logger.info("Fixed: Replaced None weight_decay with 0.0 in _single_tensor_adam")
                        kwargs['weight_decay'] = 0.0
                    
                    # Call original with fixed kwargs
                    return original_func(*args, **kwargs)
                
                # Replace the function
                adam._single_tensor_adam = patched_single_tensor_adam
                logger.info("Successfully patched _single_tensor_adam function")
                success = True
            
            # Also patch the adam function if it exists
            if hasattr(adam, 'adam'):
                original_adam_func = adam.adam
                
                def patched_adam_func(*args, **kwargs):
                    # Fix None weight_decay in kwargs
                    if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
                        logger.info("Fixed: Replaced None weight_decay with 0.0 in adam function")
                        kwargs['weight_decay'] = 0.0
                    
                    # Fix None weight_decay in positional args (typically 4th arg)
                    if len(args) >= 4 and args[3] is None:
                        logger.info("Fixed: Replaced None weight_decay in adam function positional args")
                        args = list(args)
                        args[3] = 0.0
                        args = tuple(args)
                    
                    # Call original
                    return original_adam_func(*args, **kwargs)
                
                # Replace the function
                adam.adam = patched_adam_func
                logger.info("Successfully patched adam.adam function")
                success = True
        except (ImportError, AttributeError) as e:
            logger.info(f"Could not patch torch.optim.adam module (expected in PyTorch 2.x): {e}")
        
        # Strategy 2: Try to patch PyTorch 2.x internal implementations
        try:
            # Try to access the C++ implementation in PyTorch 2.x
            if hasattr(torch._C._nn, "_single_tensor_adam"):
                logger.info("Detected PyTorch 2.x C++ implementation of _single_tensor_adam")
                success = True
            
            # We can't directly modify the C++ function, so we'll rely on patching 
            # the Adam class below instead.
        except (AttributeError, ImportError) as e:
            logger.info(f"Could not access torch._C._nn (this is normal in some PyTorch versions): {e}")
        
        # Strategy 3: Patch Adam class directly (works in all PyTorch versions)
        # This is the most reliable method and works in both PyTorch 1.x and 2.x
        from torch.optim import Adam
        
        # Store original step method
        original_step = Adam.step
        
        # Create patched step method
        def patched_step(self, closure=None):
            """Patched step method that ensures weight_decay is never None"""
            # Replace None weight_decay with 0.0 in optimizer instance
            for group in self.param_groups:
                if 'weight_decay' in group and group['weight_decay'] is None:
                    logger.info("Fixed: Replaced None weight_decay with 0.0 in Adam optimizer group")
                    group['weight_decay'] = 0.0
                    
            # Call original step method
            return original_step(self, closure)
        
        # Apply the patch
        Adam.step = patched_step
        logger.info("Successfully patched Adam.step method")
        success = True
        
        # Also patch the constructor to ensure weight_decay is never None at initialization
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
        
        # Apply init patch if possible
        try:
            Adam.__init__ = patched_init
            logger.info("Successfully patched Adam.__init__ method")
        except Exception as e:
            logger.warning(f"Could not patch Adam.__init__: {e}")
        
        logger.info("Successfully patched Adam optimizer class")
        
        # Strategy 4: Try to patch torch.optim.adam directly if it exists as a module
        # This is for older PyTorch versions
        try:
            if hasattr(torch.optim, 'adam') and hasattr(torch.optim.adam, 'adam'):
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
                
                logger.info("Successfully patched torch.optim.adam.adam function")
        except Exception as e:
            logger.warning(f"Could not patch torch.optim.adam: {e}")
            
        # Successfully applied at least one patch strategy
        return True
    except Exception as e:
        logger.error(f"Failed to patch PyTorch Adam optimizer: {e}")
        return False

# Define a function to verify the patch worked
def verify_adam_patch():
    """Test that the patch works correctly by attempting a full training step."""
    try:
        import torch
        
        # Create a simple model and optimizer with None weight_decay
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=None)
        
        # Try a full forward/backward/step cycle
        x = torch.randn(1, 10)
        y_pred = model(x)
        loss = (y_pred - torch.randn(1, 1)).pow(2).mean()
        loss.backward()
        optimizer.step()
        
        logger.info("Adam optimizer patch verification successful")
        return True
    except Exception as e:
        logger.error(f"Adam patch verification failed: {e}")
        return False

# Apply patch when the module is imported
_patch_result = patch_adam_optimizer()
if _patch_result:
    logger.info("Adam optimizer patch applied via nanotron.optim.fix_weight_decay")
    
    # Verify the patch works as expected
    if verify_adam_patch():
        logger.info("Patch successfully verified with real optimizer usage")
    else:
        logger.warning("Patch applied but verification failed")
else:
    logger.warning("Failed to apply Adam optimizer patch, weight_decay=None issues may occur")
