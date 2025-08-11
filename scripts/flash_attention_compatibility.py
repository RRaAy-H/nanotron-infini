#!/usr/bin/env python
"""
Flash Attention Compatibility Checker

This script provides functionality to check if the current system is compatible with Flash Attention,
particularly focusing on GLIBC version requirements.
"""

import os
import sys
import ctypes
import re
from pathlib import Path

def get_glibc_version():
    """
    Get the GLIBC version installed on the system.
    
    Returns:
        tuple: A tuple containing (major, minor) version numbers, or None if GLIBC not found.
    """
    try:
        # Load the C standard library
        process_namespace = ctypes.CDLL(None)
        
        # Get the GNU C Library version string
        if hasattr(process_namespace, 'gnu_get_libc_version'):
            gnu_get_libc_version = process_namespace.gnu_get_libc_version
            gnu_get_libc_version.restype = ctypes.c_char_p
            version_str = gnu_get_libc_version().decode('utf-8')
            
            # Parse version string (e.g., "2.31")
            match = re.match(r'(\d+)\.(\d+)', version_str)
            if match:
                return (int(match.group(1)), int(match.group(2)))
        
        return None
    except Exception as e:
        print(f"Error getting GLIBC version: {e}")
        return None

def check_flash_attention_compatibility():
    """
    Check if the system is compatible with Flash Attention based on GLIBC version.
    
    Flash Attention 2.0+ typically requires GLIBC 2.32 or higher.
    
    Returns:
        bool: True if compatible, False if not.
    """
    glibc_version = get_glibc_version()
    
    if glibc_version is None:
        # Could not determine GLIBC version, assume incompatible for safety
        print("Could not determine GLIBC version, assuming Flash Attention incompatibility")
        return False
    
    major, minor = glibc_version
    
    # Flash Attention 2.0+ typically requires GLIBC 2.32 or higher
    if major < 2 or (major == 2 and minor < 32):
        print(f"GLIBC version {major}.{minor} is below 2.32, Flash Attention may not be compatible")
        return False
    
    print(f"GLIBC version {major}.{minor} should be compatible with Flash Attention")
    return True

def check_flash_attention_import():
    """
    Try to import flash_attn to check if the module is available and functional.
    
    Returns:
        bool: True if import successful, False if not.
    """
    try:
        import flash_attn
        print(f"Successfully imported flash_attn version {flash_attn.__version__}")
        return True
    except ImportError:
        print("flash_attn module not found")
        return False
    except Exception as e:
        print(f"Error importing flash_attn: {e}")
        return False

def is_flash_attention_compatible():
    """
    Comprehensive check for Flash Attention compatibility.
    
    Returns:
        bool: True if Flash Attention is likely to work, False otherwise.
    """
    # First check GLIBC version
    glibc_compatible = check_flash_attention_compatibility()
    
    if not glibc_compatible:
        return False
    
    # Then try importing the module
    import_successful = check_flash_attention_import()
    
    return import_successful

def disable_flash_attention():
    """
    Configure environment variables to disable Flash Attention usage.
    """
    os.environ["DISABLE_FLASH_ATTN"] = "1"
    os.environ["USE_FLASH_ATTENTION"] = "0"
    print("Flash Attention has been disabled via environment variables")

if __name__ == "__main__":
    if is_flash_attention_compatible():
        print("System appears to be compatible with Flash Attention")
        sys.exit(0)
    else:
        print("System is not compatible with Flash Attention")
        disable_flash_attention()
        sys.exit(1)
