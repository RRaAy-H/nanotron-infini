#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/test_wrapper.py

"""
Test script for the wrapper_script.py
This script simulates being launched by the wrapper and will print out
the environment and arguments it receives.
"""

import os
import sys
import torch

def main():
    print("=" * 50)
    print("Test Wrapper Script")
    print("=" * 50)
    
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Command line arguments: {sys.argv}")
    
    print("\nPython path:")
    for p in sys.path:
        print(f"  {p}")
    
    print("\nTesting PyTorch Adam:")
    try:
        # Create a simple model and optimizer for testing
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=None)
        print("  ✓ Created optimizer with weight_decay=None successfully")
        print(f"  ✓ Optimizer state: {optimizer.state_dict()}")
        
        # Try to step the optimizer
        loss = torch.nn.functional.mse_loss(model(torch.randn(1, 10)), torch.randn(1, 2))
        loss.backward()
        optimizer.step()
        print("  ✓ Optimizer step successful!")
    except Exception as e:
        print(f"  ✗ Error testing optimizer: {e}")
    
    print("\nEnvironment variables:")
    for key, value in sorted(os.environ.items()):
        if key in ['PYTHONPATH', 'PATH', 'CUDA_VISIBLE_DEVICES', 'TRAINING_LOGS_DIR']:
            print(f"  {key}={value}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
