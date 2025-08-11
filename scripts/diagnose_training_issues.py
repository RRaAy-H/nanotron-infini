#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/diagnose_training_issues.py
# 
# Utility to diagnose and fix common issues with training in offline mode
# and with Flash Attention compatibility problems
#

import os
import sys
import importlib
import subprocess
from pathlib import Path

def check_environment():
    """Check the Python environment for common issues"""
    print("\n=== Environment Check ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check if important environment variables are set
    env_vars = [
        "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "DISABLE_FLASH_ATTN", 
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "USE_FLASH_ATTENTION",
        "RANK", "WORLD_SIZE", "LOCAL_RANK"
    ]
    
    for var in env_vars:
        print(f"{var}: {os.environ.get(var, 'not set')}")
    
    # Check if we're running in an offline environment
    offline_mode = bool(os.environ.get("HF_HUB_OFFLINE", False))
    print(f"Running in offline mode: {offline_mode}")
    
    # Check if Flash Attention is disabled
    flash_attn_disabled = bool(os.environ.get("DISABLE_FLASH_ATTN", False))
    print(f"Flash Attention disabled: {flash_attn_disabled}")
    
    return offline_mode, flash_attn_disabled

def find_project_root():
    """Try to find the project root directory"""
    print("\n=== Project Structure Check ===")
    
    # First check for environment variable
    if "PROJECT_ROOT" in os.environ:
        project_root = os.environ["PROJECT_ROOT"]
        print(f"Project root from environment variable: {project_root}")
        return project_root
    
    # Try to locate project root from current working directory
    cwd = os.getcwd()
    if "nanotron-infini" in cwd:
        # Find the project root by looking for nanotron-infini in the path
        parts = cwd.split("nanotron-infini")
        if len(parts) > 1:
            project_root = parts[0] + "nanotron-infini"
            print(f"Project root determined from CWD: {project_root}")
            return project_root
    
    # Try to find from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if "nanotron-infini" in script_dir:
        parts = script_dir.split("nanotron-infini")
        if len(parts) > 1:
            project_root = parts[0] + "nanotron-infini"
            print(f"Project root determined from script path: {project_root}")
            return project_root
    
    # If we get here, try to search upwards
    current_path = os.path.abspath(os.path.dirname(__file__))
    for _ in range(10):  # Limit depth to avoid infinite loop
        if os.path.basename(current_path) == "nanotron-infini":
            print(f"Project root found by directory search: {current_path}")
            return current_path
        parent = os.path.dirname(current_path)
        if parent == current_path:  # Reached root directory
            break
        current_path = parent
    
    print("Warning: Could not determine project root directory")
    return None

def check_files(project_root):
    """Check for the existence of critical files"""
    print("\n=== Critical Files Check ===")
    
    if not project_root:
        print("Cannot check files: project root not found")
        return False
    
    files_to_check = [
        "scripts/run_direct_training.py",
        "scripts/wrapper_script.py",
        "scripts/flexible_training_workflow.sh",
        "scripts/flash_attention_compatibility.py",
        "scripts/direct_training_fallback.py"
    ]
    
    all_files_present = True
    for file_path in files_to_check:
        full_path = os.path.join(project_root, file_path)
        exists = os.path.exists(full_path)
        print(f"{file_path}: {'✓' if exists else '✗'}")
        if not exists:
            all_files_present = False
    
    return all_files_present

def check_flash_attention():
    """Check Flash Attention availability and compatibility"""
    print("\n=== Flash Attention Check ===")
    
    try:
        import flash_attn
        print(f"Flash Attention is installed: version {flash_attn.__version__}")
        
        try:
            # Try to import the CUDA module which often has compatibility issues
            import flash_attn_2_cuda
            print("Flash Attention CUDA support: Compatible ✓")
            return True
        except ImportError as e:
            print(f"Flash Attention CUDA support: Incompatible ✗ - {e}")
            return False
            
    except ImportError:
        print("Flash Attention is not installed")
        return False

def check_dependencies():
    """Check for all required dependencies"""
    print("\n=== Dependencies Check ===")
    
    required_packages = [
        "torch", "transformers", "datasets", "nanotron", "huggingface_hub",
        "accelerate", "pydantic"
    ]
    
    for package in required_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "unknown version")
            print(f"{package}: ✓ ({version})")
        except ImportError:
            print(f"{package}: ✗ (not installed)")

def setup_environment(project_root):
    """Set up the environment for training"""
    print("\n=== Environment Setup ===")
    
    if not project_root:
        print("Cannot set up environment: project root not found")
        return False
    
    # Set environment variables
    os.environ["PYTHONPATH"] = f"{project_root}:{project_root}/src:{os.environ.get('PYTHONPATH', '')}"
    print(f"PYTHONPATH set to include project root and src directory")
    
    # Set PROJECT_ROOT environment variable
    os.environ["PROJECT_ROOT"] = project_root
    print(f"PROJECT_ROOT set to {project_root}")
    
    return True

def fix_wrapper_script(project_root):
    """Fix the wrapper script if needed"""
    print("\n=== Wrapper Script Check ===")
    
    if not project_root:
        print("Cannot fix wrapper script: project root not found")
        return False
    
    wrapper_script = os.path.join(project_root, "scripts/wrapper_script.py")
    if not os.path.exists(wrapper_script):
        print(f"Wrapper script not found at {wrapper_script}")
        return False
    
    # Make the wrapper script executable
    try:
        os.chmod(wrapper_script, 0o755)
        print("Made wrapper script executable")
    except Exception as e:
        print(f"Failed to make wrapper script executable: {e}")
    
    # Create a symlink to run_direct_training.py in /scripts if needed and if we have permissions
    direct_train_script = os.path.join(project_root, "scripts/run_direct_training.py")
    if os.path.exists(direct_train_script):
        try:
            if not os.path.exists("/scripts"):
                print("Creating /scripts directory (may require sudo)...")
                subprocess.run(["sudo", "mkdir", "-p", "/scripts"], check=False)
            
            if os.path.isdir("/scripts") and os.access("/scripts", os.W_OK):
                link_path = "/scripts/run_direct_training.py"
                # Remove existing link if it exists
                if os.path.exists(link_path) or os.path.islink(link_path):
                    os.unlink(link_path)
                # Create the symlink
                os.symlink(direct_train_script, link_path)
                print(f"Created symlink from {direct_train_script} to {link_path}")
            else:
                print("Cannot create symlink: /scripts directory not writable")
        except Exception as e:
            print(f"Failed to create symlink: {e}")
    
    return True

def fix_flash_attention_compatibility(project_root):
    """Apply Flash Attention compatibility fixes"""
    print("\n=== Flash Attention Compatibility Setup ===")
    
    if not project_root:
        print("Cannot set up Flash Attention compatibility: project root not found")
        return False
    
    compatibility_script = os.path.join(project_root, "scripts/flash_attention_compatibility.py")
    if not os.path.exists(compatibility_script):
        print(f"Compatibility script not found at {compatibility_script}")
        return False
    
    try:
        # Make script executable
        os.chmod(compatibility_script, 0o755)
        print("Made compatibility script executable")
        
        # Run the script
        print("Applying Flash Attention compatibility layer...")
        result = subprocess.run([sys.executable, compatibility_script], check=False, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            print("Successfully applied Flash Attention compatibility layer")
            print(result.stdout)
        else:
            print("Failed to apply Flash Attention compatibility layer:")
            print(result.stderr)
    except Exception as e:
        print(f"Error applying Flash Attention compatibility: {e}")
    
    return True

def test_imports():
    """Test important imports to verify environment setup"""
    print("\n=== Import Test ===")
    
    import_tests = [
        "import torch",
        "import transformers",
        "import datasets",
        "from pathlib import Path",
        "import sys",
        "import os",
        "import argparse",
    ]
    
    advanced_imports = [
        "from nanotron import constants",
        "from nanotron.trainer import DistributedTrainer",
        "from nanotron.dataloader import DataCollatorForCLM",
        "from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks",
    ]
    
    flash_attn_import = "import flash_attn"
    
    # First test basic imports
    for import_stmt in import_tests:
        try:
            exec(import_stmt)
            print(f"{import_stmt}: ✓")
        except ImportError as e:
            print(f"{import_stmt}: ✗ - {e}")
    
    # Test flash_attn import
    try:
        exec(flash_attn_import)
        print(f"{flash_attn_import}: ✓ (real module)")
    except ImportError:
        print(f"{flash_attn_import}: ✓ (mock module or not available)")
    
    # Test advanced imports
    print("\nAdvanced imports (may fail if not in correct environment):")
    for import_stmt in advanced_imports:
        try:
            exec(import_stmt)
            print(f"{import_stmt}: ✓")
        except ImportError as e:
            print(f"{import_stmt}: ✗ - {e}")
            print(f"  This is expected if nanotron is not in your PYTHONPATH")

def main():
    print("===== Infini-Llama Training Diagnostics =====")
    
    # Check environment
    offline_mode, flash_attn_disabled = check_environment()
    
    # Find project root
    project_root = find_project_root()
    
    # Check files
    files_ok = check_files(project_root)
    
    # Check Flash Attention
    flash_attn_compatible = check_flash_attention()
    
    # Check dependencies
    check_dependencies()
    
    # Set up environment
    env_ok = setup_environment(project_root)
    
    # Fix wrapper script
    wrapper_ok = fix_wrapper_script(project_root)
    
    # Fix Flash Attention compatibility if needed
    if flash_attn_disabled or not flash_attn_compatible:
        fix_flash_attention_compatibility(project_root)
    
    # Test imports
    test_imports()
    
    # Provide recommendations
    print("\n===== Recommendations =====")
    if not files_ok:
        print("- Some critical files are missing. Try reinstalling or fetching the missing files.")
    
    if not flash_attn_compatible:
        print("- Flash Attention has compatibility issues. Use --disable-flash-attn flag when training.")
        print("  Recommended command: ./scripts/train_offline_without_flash.sh <your other arguments>")
    
    if offline_mode:
        print("- You are running in offline mode. Make sure the tokenizer files are available locally.")
    
    if not env_ok or not wrapper_ok:
        print("- Environment setup issues detected. Run the script with sudo to fix permissions:")
        print(f"  sudo python {os.path.abspath(__file__)}")
    
    print("\nIf training still fails, try:")
    print("1. ./scripts/train_offline_without_flash.sh --preprocessed-data <your-data-dir> --config-file <your-config>")
    print("2. Use absolute paths for all arguments")
    print("3. Ensure the wrapper_script.py and run_direct_training.py files are executable")
    
    print("\n===== Done =====")

if __name__ == "__main__":
    main()
