#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/utils/verify_flash_attention.py

"""
Utility script to verify Flash Attention compatibility and diagnose issues.
This script checks if Flash Attention is installed and whether it can be imported
without errors. It will report any compatibility issues it finds, particularly
with GLIBC versions.
"""

import sys
import os
import importlib.util
from importlib.metadata import version, PackageNotFoundError
import subprocess


def check_glibc_version():
    """Check the system's GLIBC version."""
    try:
        # Get the GLIBC version using ldd
        result = subprocess.run(
            ["ldd", "--version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        # Parse the output to get the version
        output = result.stdout if result.returncode == 0 else result.stderr
        for line in output.splitlines():
            if "GLIBC" in line and "version" in line.lower():
                return line.strip()
        
        return "GLIBC version could not be determined from ldd output"
    except Exception as e:
        return f"Error getting GLIBC version: {e}"


def check_cuda_version():
    """Check the CUDA version if available."""
    try:
        import torch
        if torch.cuda.is_available():
            return f"CUDA Version: {torch.version.cuda}, CUDNN Version: {torch.backends.cudnn.version()}"
        else:
            return "CUDA is not available in PyTorch"
    except ImportError:
        try:
            result = subprocess.run(
                ["nvcc", "--version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            return result.stdout if result.returncode == 0 else "CUDA installed but version unknown"
        except:
            return "CUDA version could not be determined"


def check_flash_attn():
    """Check if Flash Attention is installed and can be imported."""
    flash_attn_spec = importlib.util.find_spec("flash_attn")
    
    if flash_attn_spec is None:
        return {
            "installed": False,
            "version": None,
            "error": "Flash Attention is not installed",
            "can_import": False,
            "cuda_module_importable": False
        }
    
    # Flash Attention is installed, check if it can be imported
    try:
        import flash_attn
        version = getattr(flash_attn, "__version__", "unknown")
        
        # Check if the CUDA module can be imported
        try:
            import flash_attn_2_cuda
            cuda_importable = True
            cuda_error = None
        except ImportError as e:
            cuda_importable = False
            cuda_error = str(e)
        
        return {
            "installed": True,
            "version": version,
            "error": None,
            "can_import": True,
            "cuda_module_importable": cuda_importable,
            "cuda_error": cuda_error
        }
    except ImportError as e:
        return {
            "installed": True,
            "version": "unknown",
            "error": str(e),
            "can_import": False,
            "cuda_module_importable": False
        }


def check_system_compatibility():
    """Print system compatibility information."""
    print("System Compatibility Information:")
    print("---------------------------------")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check PyTorch version
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"PyTorch CUDA version: {torch.version.cuda}")
            print(f"CUDNN version: {torch.backends.cudnn.version()}")
            print(f"GPU(s): {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not installed")
    
    # Check GLIBC version
    print(f"GLIBC information: {check_glibc_version()}")
    
    # Check CUDA version
    print(f"CUDA information: {check_cuda_version()}")
    

def main():
    """Main function to check Flash Attention compatibility."""
    print("Flash Attention Compatibility Check")
    print("==================================")
    
    # Check system compatibility
    check_system_compatibility()
    print()
    
    # Check Flash Attention
    flash_status = check_flash_attn()
    
    print("Flash Attention Status:")
    print("---------------------------------")
    print(f"Installed: {flash_status['installed']}")
    if flash_status["installed"]:
        print(f"Version: {flash_status['version']}")
        print(f"Can import main module: {flash_status['can_import']}")
        
        if not flash_status["can_import"]:
            print(f"Import error: {flash_status['error']}")
        else:
            print(f"CUDA module importable: {flash_status['cuda_module_importable']}")
            if not flash_status["cuda_module_importable"]:
                print(f"CUDA module error: {flash_status['cuda_error']}")
                
                # Provide hints based on error
                error_str = str(flash_status['cuda_error']).lower()
                if 'glibc' in error_str:
                    print("\nDiagnosis: GLIBC version incompatibility detected.")
                    print("This typically means Flash Attention was compiled with a newer GLIBC version")
                    print("than what is available on your system.")
                    print("\nPossible solutions:")
                    print("1. Rebuild Flash Attention from source for your system:")
                    print("   pip uninstall -y flash-attn")
                    print("   FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-binary flash-attn")
                    print("2. Use standard attention implementation instead (disable Flash Attention)")
                    print("   Add --disable-flash-attn flag when running the training script")
                elif 'cuda' in error_str:
                    print("\nDiagnosis: CUDA version incompatibility detected.")
                    print("This typically means Flash Attention was compiled with a different CUDA version")
                    print("than what is available on your system.")
    else:
        print("Flash Attention is not installed in this environment.")
    
    # Final compatibility verdict
    print("\nCompatibility Verdict:")
    if flash_status["installed"] and flash_status["can_import"] and flash_status["cuda_module_importable"]:
        print("✅ Flash Attention is fully compatible with your system.")
    elif flash_status["installed"] and flash_status["can_import"]:
        print("❌ Flash Attention is installed but the CUDA module has compatibility issues.")
        print("   Training will fail unless you use the --disable-flash-attn flag.")
    elif flash_status["installed"]:
        print("❌ Flash Attention is installed but cannot be imported.")
        print("   This environment will automatically disable Flash Attention during training.")
    else:
        print("⚠️ Flash Attention is not installed.")
        print("   Training will use standard attention implementation.")


if __name__ == "__main__":
    main()
