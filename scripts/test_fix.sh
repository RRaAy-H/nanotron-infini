#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/test_fix.sh

# Simple test script to verify the weight decay fix
# This script tests both with and without the fix to show the difference

set -e  # Exit on error

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Set up Python path
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PYTHONPATH"

# Create a simple test script
TEST_SCRIPT="$PROJECT_ROOT/scripts/test_adam_fix.py"
cat > "$TEST_SCRIPT" << EOF
#!/usr/bin/env python
import torch

def test_weight_decay_none():
    """Test creating an Adam optimizer with weight_decay=None"""
    try:
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=None)
        
        # Try a step to trigger the weight decay calculation
        data = torch.randn(1, 10)
        target = torch.randn(1, 1)
        loss = torch.nn.functional.mse_loss(model(data), target)
        loss.backward()
        optimizer.step()
        
        print("SUCCESS: Adam optimizer with weight_decay=None works")
        return True
    except Exception as e:
        print(f"ERROR: Adam optimizer with weight_decay=None failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Adam optimizer with weight_decay=None...")
    test_weight_decay_none()
EOF

chmod +x "$TEST_SCRIPT"

# 1. Test without fix
echo -e "\n=== Testing without fix ==="
python "$TEST_SCRIPT"

# 2. Test with fix through wrapper script
echo -e "\n=== Testing with fix through wrapper script ==="
WRAP_SCRIPT="$PROJECT_ROOT/scripts/wrapper_script.py"
chmod +x "$WRAP_SCRIPT"
python "$WRAP_SCRIPT" "$TEST_SCRIPT"

# 3. Test with fix through direct import
echo -e "\n=== Testing with fix through direct import ==="
cat > "$PROJECT_ROOT/scripts/test_with_import.py" << EOF
#!/usr/bin/env python
import sys
import os
# Import the fix first
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))
try:
    import preimport
    print("Successfully imported preimport")
except ImportError:
    print("Failed to import preimport")

# Now run the test
sys.path.insert(0, os.path.dirname(__file__))
from test_adam_fix import test_weight_decay_none
test_weight_decay_none()
EOF

chmod +x "$PROJECT_ROOT/scripts/test_with_import.py"
python "$PROJECT_ROOT/scripts/test_with_import.py"

echo -e "\nAll tests completed!"
