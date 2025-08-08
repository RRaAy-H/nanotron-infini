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

# Import our patches first - will apply the Adam optimizer patch
try:
    import preimport
    logger.info("Successfully imported pre-import patches")
except ImportError as e:
    logger.warning(f"Failed to import pre-import patches: {e}")
    logger.warning("Weight decay issues may occur during training")

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
