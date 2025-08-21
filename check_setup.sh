#!/bin/bash

# Quick setup verification script
echo "=========================================================="
echo "SETUP VERIFICATION FOR ULTRA-OPTIMIZED PASSKEY FINE-TUNING"
echo "=========================================================="

CHECKPOINT_PATH="${1:-./checkpoints/fineweb_4gpu_300m_infini/30000}"
CONFIG_FILE="passkey_finetune_300m_simple_config.yaml"

echo "Checking required files and directories..."
echo ""

# Check current directory
echo "Current directory: $(pwd)"
echo ""

# Check training script
if [ -f "run_train.py" ]; then
    echo "✓ Training script found: run_train.py"
else
    echo "❌ Training script NOT found: run_train.py"
    echo "   Make sure you're in the correct nanotron directory"
fi

# Check config file
if [ -f "$CONFIG_FILE" ]; then
    echo "✓ Configuration file found: $CONFIG_FILE"
    # Show key optimizations
    echo "   Key optimizations:"
    grep -E "(segment_length|train_steps|learning_rate)" "$CONFIG_FILE" | head -3 | sed 's/^/     /'
else
    echo "❌ Configuration file NOT found: $CONFIG_FILE"
fi

# Check checkpoint
if [ -d "$CHECKPOINT_PATH" ]; then
    echo "✓ Base checkpoint found: $CHECKPOINT_PATH"
else
    echo "❌ Base checkpoint NOT found: $CHECKPOINT_PATH"
    echo "   Available checkpoints:"
    ls -la ./checkpoints/ 2>/dev/null || echo "   No checkpoints directory found"
fi

# Check training data
TRAINING_DATA_DIR="/data1/infini-attn/infini-llama/nanotron-infini/finetuning"
if [ -d "$TRAINING_DATA_DIR" ]; then
    echo "✓ Training data found: $TRAINING_DATA_DIR"
    echo "   Files in training data:"
    ls -la "$TRAINING_DATA_DIR"/*.parquet 2>/dev/null | head -4
else
    echo "❌ Training data NOT found: $TRAINING_DATA_DIR"
fi

# Check monitoring script
if [ -f "monitor_passkey_finetuning.py" ]; then
    echo "✓ Monitoring script found: monitor_passkey_finetuning.py"
else
    echo "❌ Monitoring script NOT found: monitor_passkey_finetuning.py"
fi

# Check balance factor fix
if [ -f "apply_balance_fix_standalone.py" ]; then
    echo "✓ Balance factor fix found: apply_balance_fix_standalone.py"
else
    echo "❌ Balance factor fix NOT found: apply_balance_fix_standalone.py"
fi

echo ""
echo "=========================================================="

# Summary
ERRORS=0
[ ! -f "run_train.py" ] && ((ERRORS++))
[ ! -f "$CONFIG_FILE" ] && ((ERRORS++))
[ ! -d "$CHECKPOINT_PATH" ] && ((ERRORS++))
[ ! -d "$TRAINING_DATA_DIR" ] && ((ERRORS++))

if [ $ERRORS -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED - Ready for ultra-optimized fine-tuning!"
    echo ""
    echo "🚀 ULTRA-OPTIMIZATIONS ACTIVE:"
    echo "   • segment_length = 64 (32x more memory operations)"
    echo "   • train_steps = 30400 (extended training)"  
    echo "   • balance_factor_lr = 0.05 (aggressive learning)"
    echo ""
    echo "To start fine-tuning, run:"
    echo "   ./run_passkey_finetune_300m.sh $CHECKPOINT_PATH"
    echo ""
    echo "To monitor progress, run in another terminal:"
    echo "   ./start_monitoring.sh"
else
    echo "❌ $ERRORS ERROR(S) FOUND - Please fix before running"
fi

echo "=========================================================="