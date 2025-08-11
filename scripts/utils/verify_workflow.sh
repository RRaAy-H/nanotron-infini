#!/bin/bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/utils/verify_workflow.sh

# This script verifies all components of the Infini-Llama training workflow
# It checks that all required files exist and prints their status

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================"
echo "Infini-Llama Training Workflow Verification"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# Define required files
declare -A REQUIRED_FILES=(
    ["scripts/flexible_training_workflow.sh"]="Main workflow script"
    ["scripts/wrapper_script.py"]="Wrapper script that applies patches"
    ["scripts/run_direct_training.py"]="Direct training script"
    ["scripts/preprocessing/preprocess_data_fixed.py"]="Data preprocessing script"
    ["scripts/training/train_infini_llama.py"]="Training implementation"
    ["scripts/direct_adam_patch.py"]="Adam optimizer patch"
    ["scripts/fix_flash_attention_warnings.py"]="Flash attention warnings fix"
    ["scripts/fix_adam_none_issue.py"]="Adam optimizer None issue fix"
    ["scripts/config/tiny_test_config.yaml"]="Default test configuration"
)

# Check each required file
echo "Checking required files:"
missing=0
for file in "${!REQUIRED_FILES[@]}"; do
    description="${REQUIRED_FILES[$file]}"
    if [[ -f "$PROJECT_ROOT/$file" ]]; then
        echo "✅ $file - $description"
    else
        echo "❌ $file - $description (MISSING)"
        missing=$((missing + 1))
    fi
done

if [[ $missing -gt 0 ]]; then
    echo ""
    echo "WARNING: $missing required files are missing!"
    echo "Please create these files to ensure the workflow functions correctly."
else
    echo ""
    echo "All required files are present."
fi

echo ""

# Check for potentially unused files
echo "Checking for potentially unused files:"
declare -A UNUSED_FILES=(
    ["scripts/run_infini_llama_workflow.sh"]="Old workflow script, replaced by flexible_training_workflow.sh"
    ["scripts/preprocessing/preprocess_data.py"]="Old preprocessing script, replaced by preprocess_data_fixed.py"
)

for file in "${!UNUSED_FILES[@]}"; do
    description="${UNUSED_FILES[$file]}"
    if [[ -f "$PROJECT_ROOT/$file" ]]; then
        echo "⚠️  $file - $description (can be archived)"
    else
        echo "👌 $file - $description (already removed/not present)"
    fi
done

echo ""
echo "To archive unused files, run:"
echo "python scripts/utils/archive_unused_scripts.py"
echo ""
echo "To use the workflow, run:"
echo "./scripts/flexible_training_workflow.sh --help"
echo ""
echo "========================================"
