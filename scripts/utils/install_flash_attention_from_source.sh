#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/utils/install_flash_attention_from_source.sh

# Script to compile and install Flash Attention from source
# This resolves GLIBC compatibility issues by building with the system's GLIBC

echo "====================================================================="
echo "Flash Attention Source Installation"
echo "This script will uninstall any existing Flash Attention installation"
echo "and compile a new one from source using your system's GLIBC version."
echo "====================================================================="

# Check if we are in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
  echo "WARNING: It doesn't appear that you are in a virtual environment."
  echo "It's recommended to run this in the same environment you use for training."
  read -p "Continue anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation aborted."
    exit 1
  fi
fi

# Check for required build tools
echo "Checking for required build tools..."
MISSING_TOOLS=0

# Check for compiler
if ! command -v gcc &> /dev/null || ! command -v g++ &> /dev/null; then
  echo "❌ C/C++ compiler (gcc/g++) not found"
  MISSING_TOOLS=1
fi

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
  echo "❌ CUDA toolkit (nvcc) not found"
  MISSING_TOOLS=1
else
  echo "✅ Found CUDA: $(nvcc --version | head -n1)"
fi

# Check for pip
if ! command -v pip &> /dev/null; then
  echo "❌ pip not found"
  MISSING_TOOLS=1
fi

if [ $MISSING_TOOLS -eq 1 ]; then
  echo "Please install the missing tools before continuing."
  exit 1
fi

# Check for development packages
echo "Checking PyTorch installation..."
if ! python -c "import torch; print(f'PyTorch {torch.__version__} installed with CUDA: {torch.cuda.is_available()}')" &> /dev/null; then
  echo "❌ PyTorch not found or not properly installed"
  echo "Please install PyTorch with CUDA support first."
  exit 1
else
  python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed with CUDA: {torch.cuda.is_available()}')"
fi

# Display GLIBC version
echo "System GLIBC version:"
ldd --version 2>&1 | head -n1 || echo "Could not determine GLIBC version"

# Uninstall existing Flash Attention
echo -e "\n====================================================================="
echo "Uninstalling existing Flash Attention installation..."
pip uninstall -y flash-attn

# Set environment variables for compilation
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
export MAX_JOBS=4  # Limit parallel compilation jobs to avoid memory issues

echo -e "\n====================================================================="
echo "Installing Flash Attention from source..."
echo "This may take several minutes..."

# Install from source with build flag
pip install flash-attn --no-binary flash-attn

# Check if installation was successful
if python -c "import flash_attn; import flash_attn_2_cuda; print('Flash Attention installation successful!')" &> /dev/null; then
  echo -e "\n====================================================================="
  echo "✅ Flash Attention successfully installed from source!"
  echo "You should now be able to use Flash Attention without GLIBC compatibility issues."
else
  echo -e "\n====================================================================="
  echo "❌ Flash Attention installation failed or has compatibility issues."
  echo "You may need to use the --disable-flash-attn flag when training."
fi

# Run the verification script if available
if [ -f "$(dirname "$0")/verify_flash_attention.py" ]; then
  echo -e "\n====================================================================="
  echo "Running Flash Attention verification..."
  python "$(dirname "$0")/verify_flash_attention.py"
fi

echo -e "\n====================================================================="
echo "Process complete!"
