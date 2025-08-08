#!/usr/bin/env python

import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

print("Python path:")
for p in sys.path:
    print(f"  {p}")

# Try to import preimport
try:
    import preimport
    print("\nSuccessfully imported preimport module!")
except ImportError as e:
    print(f"\nFailed to import preimport: {e}")

# Print environment info
print("\nEnvironment information:")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
