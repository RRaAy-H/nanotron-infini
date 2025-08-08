#!/bin/bash
# setup_infini_env.sh
# Script to set up the Python environment for Infini-Llama

set -e  # Exit on error

CONDA_ENV_NAME="infini_llama"

echo "======================================================================"
echo "Setting up Python environment for Infini-Llama"
echo "======================================================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "conda is not installed or not in PATH. Please install conda first."
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "${CONDA_ENV_NAME}"; then
    echo "Environment ${CONDA_ENV_NAME} already exists."
    echo "Activating environment..."
    echo "Run: conda activate ${CONDA_ENV_NAME}"
else
    echo "Creating new conda environment with Python 3.10..."
    conda create -y -n "${CONDA_ENV_NAME}" python=3.10
    echo "Environment created."
    echo "Activating environment..."
    echo "Run: conda activate ${CONDA_ENV_NAME}"
fi

echo ""
echo "After activating the environment, install required packages with:"
echo "pip install -e ."
echo "pip install torch packaging dacite pyyaml numpy tqdm"
echo ""
echo "Then you can run the training scripts with:"
echo "./run_both_models.sh --config-file scripts/config/tiny_test_config.yaml --data-dir tiny_test_data/preprocessed_*"
