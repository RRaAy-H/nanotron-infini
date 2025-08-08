#!/bin/bash
# This script runs just the preprocessing step on the tiny dataset to validate the configuration

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration and output paths
CONFIG_FILE="${SCRIPT_DIR}/config/tiny_test_config.yaml"
OUTPUT_DIR="${ROOT_DIR}/tiny_test_data"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "======================================================================"
echo "Running preprocessing only on tiny dataset"
echo "======================================================================"
echo "Configuration: ${CONFIG_FILE}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

# Run the fixed preprocessing script instead
echo "Executing fixed preprocessing script..."
python "${SCRIPT_DIR}/preprocessing/preprocess_data_fixed.py" \
  --config-file "$CONFIG_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --gpu-id 0

# Check if preprocessing was successful
if [ $? -eq 0 ]; then
  echo ""
  echo "======================================================================"
  echo "Preprocessing completed successfully!"
  echo "You can now run training using the preprocessed data in: ${OUTPUT_DIR}"
  echo "======================================================================"
else
  echo ""
  echo "======================================================================"
  echo "Preprocessing failed. Check the error messages above."
  echo "======================================================================"
  exit 1
fi
