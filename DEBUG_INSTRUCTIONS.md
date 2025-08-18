# Dataloader Error Debug Instructions

This directory contains comprehensive debugging tools to diagnose the `TypeError: 'NoneType' object is not an iterator` error.

## Quick Start

Run the main debug suite:
```bash
python3 run_debug_tests.py --config-file passkey_finetune_300m_simple_config.yaml
```

## Individual Debug Scripts

### 1. `reproduce_error.py` - Error Reproduction
Reproduces the exact error scenario with detailed analysis.
```bash
python3 reproduce_error.py --config-file passkey_finetune_300m_simple_config.yaml
```

### 2. `debug_dataloader_comprehensive.py` - Comprehensive Analysis  
Provides detailed debugging with multiple modes.
```bash
# Full analysis
python3 debug_dataloader_comprehensive.py --config-file passkey_finetune_300m_simple_config.yaml --mode all

# Just reproduce the error
python3 debug_dataloader_comprehensive.py --config-file passkey_finetune_300m_simple_config.yaml --mode reproduce

# Analyze dataloader state transitions
python3 debug_dataloader_comprehensive.py --config-file passkey_finetune_300m_simple_config.yaml --mode analyze
```

### 3. `inspect_dataloader_state.py` - State Inspection
Inspects dataloader state at various points in the training process.
```bash
python3 inspect_dataloader_state.py --config-file passkey_finetune_300m_simple_config.yaml
```

### 4. `patch_and_test.py` - Live Debugging
Patches trainer.py with debug prints and runs a test.
```bash
# Add debug patches and test
python3 patch_and_test.py --config-file passkey_finetune_300m_simple_config.yaml

# Restore original file
python3 patch_and_test.py --restore
```

### 5. `debug_train.py` - Debug Training Script
Drop-in replacement for run_train.py with extensive logging.
```bash
python3 debug_train.py --config-file passkey_finetune_300m_simple_config.yaml
```

### 6. `test_real_scenario.py` - Real Scenario Testing
Tests with controlled real scenario setup.
```bash
# Test with fake checkpoint
python3 test_real_scenario.py

# Test with real checkpoint  
python3 test_real_scenario.py /path/to/real/checkpoint

# Use debug test config
python3 test_real_scenario.py --test-config

# Clean up test files
python3 test_real_scenario.py --cleanup
```

## Test Configurations

### `debug_test_config.yaml`
Minimal configuration designed to reproduce the issue quickly:
- Very small model (64 hidden size, 2 layers)
- Single GPU
- Dummy data (no file dependencies)
- Resume from step 30000
- Data stage starts at step 1

## Understanding the Error

The error occurs because:

1. **Original Issue**: The training script resumes from step 30000+ (checkpoint)
2. **Stage Configuration**: Data stage starts at step 1
3. **Matching Logic**: Code looks for `stage.start_training_step == iteration_step`
4. **Problem**: No stage starts at step 30001, so dataloader becomes None
5. **Error**: `train_batches = (next(dataloader) for _ in range(...))` fails with TypeError

## Expected Debug Output

When you run these scripts, look for:

- **Error Reproduction**: Scripts should reproduce the TypeError
- **State Analysis**: Current dataloader should be None at step 30001+
- **Stage Matching**: No stages match iteration_step 30001
- **Root Cause**: Gap between resume step (30000+) and stage start step (1)

## Analyzing Results

1. **Run the debug suite**: `python3 run_debug_tests.py --config-file passkey_finetune_300m_simple_config.yaml`
2. **Check the generated report**: Look for `debug_report_YYYYMMDD_HHMMSS.txt`
3. **Look for TypeError**: Confirm the error is reproduced
4. **Review state transitions**: See where dataloader becomes None
5. **Send logs back**: Include all output and generated files

## Files Generated During Testing

- `debug_report_*.txt` - Comprehensive test report
- `dataloader_debug.json` - Detailed debug log
- `debug_real_scenario.yaml` - Test configuration
- `fake_checkpoint_30000/` - Fake checkpoint for testing
- `fake_passkey_data.parquet` - Fake dataset for testing

## Cleanup

To clean up all test files:
```bash
python3 test_real_scenario.py --cleanup
rm -f debug_*.txt debug_*.json debug_*.yaml
rm -rf fake_checkpoint_* debug_*_checkpoints
```

## Next Steps

1. Run these tests on your remote server with the real environment
2. Send back all generated logs and reports
3. Include the full output from the debug scripts
4. The analysis will help identify the exact fix needed