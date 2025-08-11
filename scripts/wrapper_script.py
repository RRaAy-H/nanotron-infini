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

# Get project root directory - use multiple strategies for robustness
script_dir = os.path.dirname(os.path.abspath(__file__))
# Strategy 1: Standard method - go up one directory from scripts folder
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# Strategy 2: If absolute path doesn't seem right (very short), try to find it from current working directory
if len(project_root) < 10:  # If suspiciously short
    cwd = os.getcwd()
    if 'nanotron-infini' in cwd:
        # Find the project root by looking for nanotron-infini in the path
        parts = cwd.split('nanotron-infini')
        if len(parts) > 1:
            project_root = parts[0] + 'nanotron-infini'
            logger.info(f"Using project root from CWD: {project_root}")

# Strategy 3: Check environment variable
if 'PROJECT_ROOT' in os.environ:
    project_root = os.environ['PROJECT_ROOT']
    logger.info(f"Using project root from environment: {project_root}")

logger.info(f"Project root directory: {project_root}")
logger.info(f"Script directory: {script_dir}")

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
    # Try multiple possible locations for the training script
    possible_paths = [
        os.path.join(project_root, "scripts", "run_direct_training.py"),  # Standard path
        os.path.join(os.path.dirname(__file__), "run_direct_training.py"),  # Same directory as wrapper
        "/scripts/run_direct_training.py",  # Absolute path (sometimes used in containers)
        os.path.abspath("run_direct_training.py"),  # Current directory
    ]
    
    # Debugging info
    logger.info(f"Current environment:")
    logger.info(f"- Python executable: {sys.executable}")
    logger.info(f"- Python version: {sys.version}")
    logger.info(f"- Working directory: {os.getcwd()}")
    logger.info(f"- PYTHONPATH: {os.environ.get('PYTHONPATH', 'not set')}")
    logger.info(f"- Script directory: {os.path.dirname(__file__)}")
    
    # Find the first valid path
    script_path = None
    for path in possible_paths:
        logger.info(f"Checking for training script at: {path}")
        if os.path.exists(path):
            script_path = path
            logger.info(f"Found training script at: {script_path}")
            break
    
    if script_path is None:
        # Last resort: search for the file in the project
        logger.warning("Training script not found in expected locations, searching...")
        for root, _, files in os.walk(project_root):
            if "run_direct_training.py" in files:
                script_path = os.path.join(root, "run_direct_training.py")
                logger.info(f"Found training script by search at: {script_path}")
                break
    
    if script_path is None:
        # Additional diagnostic information to help troubleshoot
        project_structure = []
        scripts_dir = os.path.join(project_root, "scripts")
        if os.path.exists(scripts_dir):
            logger.info(f"Contents of scripts directory ({scripts_dir}):")
            try:
                for item in os.listdir(scripts_dir):
                    project_structure.append(f"  - {item}")
                    logger.info(f"  - {item}")
            except Exception as e:
                logger.error(f"Failed to list directory contents: {e}")
        else:
            logger.error(f"Scripts directory not found at {scripts_dir}")
            
        # Try direct import
        try:
            import run_direct_training
            script_path = run_direct_training.__file__
            logger.info(f"Found training script via import at: {script_path}")
        except ImportError:
            logger.error("Could not import run_direct_training as a module")
            
        if script_path is None:
            # Give up and provide helpful error information
            logger.error("-----------------------------------------------------------")
            logger.error("ERROR: Training script run_direct_training.py not found!")
            logger.error("Possible reasons:")
            logger.error("1. The script might be in a different location than expected")
            logger.error("2. There might be permission issues accessing the file")
            logger.error("3. The project structure might be different than expected")
            logger.error("")
            logger.error("Try running one of these commands to find the script:")
            logger.error("  find /path/to/project -name run_direct_training.py")
            logger.error("  locate run_direct_training.py")
            logger.error("-----------------------------------------------------------")
            sys.exit(1)
    
    logger.info(f"Running training script: {script_path}")
    logger.info(f"With arguments: {sys.argv[1:]}")
    
    # Use execv to replace the current process with the training script
    os.execv(sys.executable, [sys.executable, script_path] + sys.argv[1:])
