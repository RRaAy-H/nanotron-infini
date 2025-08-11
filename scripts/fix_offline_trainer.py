#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/fix_offline_trainer.py

"""
This script patches the trainer.py file to avoid network access when in offline mode.
The patch prevents attempts to download from HuggingFace Hub or other remote sources.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def patch_trainer_module():
    """
    Apply patches to the trainer.py module to avoid network access in offline mode.
    """
    try:
        # Get project root directory
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        src_dir = project_root / "src"
        
        # Ensure we can import from nanotron
        sys.path.insert(0, str(project_root))
        sys.path.insert(0, str(src_dir))
        
        # Import the trainer module
        from nanotron import trainer
        import importlib
        
        # Check if we're in offline mode
        offline_mode = bool(os.environ.get('HF_HUB_OFFLINE', False))
        if not offline_mode:
            logger.info("Not in offline mode, no patches needed")
            return True
            
        logger.info("Applying offline mode patches to trainer module")
        
        # Create patches for relevant functions
        if hasattr(trainer, "DistributedTrainer"):
            logger.info("Patching DistributedTrainer class")
            
            # Store original __init__
            original_init = trainer.DistributedTrainer.__init__
            
            def patched_init(self, config_file=None, config=None):
                """Patched __init__ method that sets offline flags before initialization"""
                logger.info("Initializing DistributedTrainer in offline mode")
                
                # Set offline flags
                os.environ['HF_DATASETS_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_HUB_OFFLINE'] = '1'
                
                # Call original init
                return original_init(self, config_file=config_file, config=config)
            
            # Apply the patch
            trainer.DistributedTrainer.__init__ = patched_init
            logger.info("Successfully patched DistributedTrainer.__init__")
        
        # Reload the module
        importlib.reload(trainer)
        logger.info("Trainer module reloaded with patches")
        
        return True
    except Exception as e:
        logger.error(f"Failed to patch trainer module: {e}")
        return False

if __name__ == "__main__":
    success = patch_trainer_module()
    if success:
        logger.info("Trainer module patched successfully for offline mode")
        sys.exit(0)
    else:
        logger.error("Failed to patch trainer module")
        sys.exit(1)
