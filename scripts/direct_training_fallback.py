#!/usr/bin/env python
# This is a fallback script that can be used if run_direct_training.py can't be found
# It simply imports and runs the run_direct_training.py file

import os
import sys
import importlib.util
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function that attempts to find and run the direct training script.
    """
    logger.info("Attempting to locate and run training script via fallback mechanism")
    
    # Try to find the run_direct_training.py file
    project_root = os.environ.get('PROJECT_ROOT')
    if not project_root:
        # Try to determine project root
        cwd = os.getcwd()
        if 'nanotron-infini' in cwd:
            parts = cwd.split('nanotron-infini')
            if len(parts) > 1:
                project_root = parts[0] + 'nanotron-infini'
    
    # List of possible locations for the training script
    search_paths = []
    if project_root:
        search_paths.extend([
            os.path.join(project_root, "scripts", "run_direct_training.py"),
            os.path.join(project_root, "run_direct_training.py"),
        ])
    
    # Add current directory and script directory
    search_paths.extend([
        os.path.join(os.path.dirname(__file__), "run_direct_training.py"),
        os.path.join(os.getcwd(), "scripts", "run_direct_training.py"),
        os.path.join(os.getcwd(), "run_direct_training.py"),
    ])
    
    # Try to find the script
    script_path = None
    for path in search_paths:
        if os.path.exists(path):
            script_path = path
            logger.info(f"Found training script at: {path}")
            break
    
    if not script_path:
        # Last resort: search for it
        search_dirs = []
        if project_root:
            search_dirs.append(project_root)
        search_dirs.append(os.getcwd())
        
        for search_dir in search_dirs:
            for root, _, files in os.walk(search_dir):
                if "run_direct_training.py" in files:
                    script_path = os.path.join(root, "run_direct_training.py")
                    logger.info(f"Found training script by search at: {script_path}")
                    break
            if script_path:
                break
    
    if not script_path:
        logger.error("Could not find run_direct_training.py anywhere!")
        sys.exit(1)
    
    # Execute the script
    logger.info(f"Executing training script: {script_path}")
    spec = importlib.util.spec_from_file_location("run_direct_training", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Call the main function if it exists
    if hasattr(module, "main") and callable(module.main):
        logger.info("Calling main() function in training script")
        module.main()
    else:
        logger.info("Script does not have a main() function, assuming it ran during import")
    
    logger.info("Training script execution completed")

if __name__ == "__main__":
    main()
