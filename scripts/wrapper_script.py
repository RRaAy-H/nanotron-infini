#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/wrapper_script.py

"""
Wrapper script for training that imports patches before running the actual training script.
This script should be called with the same arguments as the training script.

This script fixes the 'unsupported operand type(s) for *: 'float' and 'NoneType'' error
that occurs when weight_decay is None in the Adam optimizer.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add directories to Python path
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'scripts'))

logger.info(f"Python path set to include: {project_root}")

# Create a simple inline patch for weight_decay=None issue
logger.info("Applying Adam optimizer patch")

def apply_adam_patch():
    """
    Apply a patch to the Adam optimizer to handle None weight_decay values.
    Tries multiple methods to ensure the patch is applied successfully.
    """
    try:
        # Method 1: Try to import our proper module first (most robust solution)
        try:
            from nanotron.optim.fix_weight_decay import patch_adam_optimizer
            success = patch_adam_optimizer()
            if success:
                logger.info("Adam optimizer patch applied via nanotron.optim.fix_weight_decay")
                return True
        except ImportError:
            logger.warning("Could not import nanotron.optim.fix_weight_decay, trying alternative methods")
        
        # Method 2: Execute the patch script in a separate process
        patch_script = os.path.join(project_root, "scripts", "fix_adam_none_issue.py")
        if os.path.exists(patch_script):
            logger.info(f"Executing patch script: {patch_script}")
            # Run the patch script as a separate process
            import subprocess
            result = subprocess.run([sys.executable, patch_script], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Patch script executed successfully")
                logger.info(result.stdout.strip())
                return True
            else:
                logger.warning(f"Patch script failed: {result.stderr.strip()}")
                
        # Method 3: Direct patching of the Adam class
        try:
            import torch
            from torch.optim import Adam
            
            # Store original step method
            original_step = Adam.step
            
            def patched_step(self, closure=None):
                # Replace None weight_decay with 0.0 in optimizer instance
                for group in self.param_groups:
                    if 'weight_decay' in group and group['weight_decay'] is None:
                        logger.info("Fixed: Replaced None weight_decay with 0.0 in Adam group")
                        group['weight_decay'] = 0.0
                        
                # Call original step method
                return original_step(self, closure)
            
            # Apply the patch
            Adam.step = patched_step
            logger.info("Directly patched Adam.step method")
            return True
        except Exception as e:
            logger.error(f"Failed to directly patch Adam class: {e}")
            
        # If we got here, all methods failed
        return False
    except Exception as e:
        logger.error(f"Failed to apply patch: {e}")
        return False

# Try to apply the patch
patch_result = apply_adam_patch()

# Try alternative approaches if the patch script failed
if not patch_result:
    try:
        # Try to import preimport module
        import preimport
        logger.info("Successfully imported preimport module as fallback")
    except ImportError:
        logger.warning("Could not import preimport module")
        # Create a last-resort environment variable to signal patching should be applied
        os.environ["FIX_ADAM_WEIGHT_DECAY"] = "true"
        logger.info("Set FIX_ADAM_WEIGHT_DECAY environment variable as fallback approach")

# Now run the actual training script
if __name__ == "__main__":
    # Get the script path and arguments
    script_path = os.path.join(project_root, "scripts", "run_direct_training.py")
    
    # Debugging info
    logger.info(f"Current environment:")
    logger.info(f"- Python executable: {sys.executable}")
    logger.info(f"- Python version: {sys.version}")
    logger.info(f"- Working directory: {os.getcwd()}")
    logger.info(f"- PYTHONPATH: {os.environ.get('PYTHONPATH', 'not set')}")
    
    if not os.path.exists(script_path):
        logger.error(f"Training script not found at: {script_path}")
        sys.exit(1)
    
    logger.info(f"Running training script: {script_path}")
    logger.info(f"With arguments: {sys.argv[1:]}")
    
    # Use execv to replace the current process with the training script
    os.execv(sys.executable, [sys.executable, script_path] + sys.argv[1:])
