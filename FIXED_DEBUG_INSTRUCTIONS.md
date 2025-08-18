# Fixed Debug Instructions for Distributed Environment

The previous debug scripts failed because they didn't account for the distributed training environment. Here are the corrected instructions.

## Quick Test to Reproduce Error

Run this command to reproduce the exact error with debugging:

```bash
torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29401 test_actual_error.py
```

This will:
1. Create a minimal config that simulates resuming from step 30000
2. Use dummy data (no file dependencies)
3. Add detailed debug logging to show exactly where the error occurs
4. Reproduce the `TypeError: 'NoneType' object is not an iterator`

## Alternative: Debug the Original Script

If you want to debug the original passkey script:

```bash
# First, backup and patch trainer.py with debug info
python3 patch_and_test.py --config-file passkey_finetune_300m_simple_config.yaml

# Then run the original script (this will show debug output)
./run_passkey_finetune_300m.sh ./checkpoints/fineweb_4gpu_300m_infini/30000

# Restore original file when done
python3 patch_and_test.py --restore
```

## Minimal Test (Faster)

For a very quick test with minimal resources:

```bash
torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29401 debug_with_torchrun.py --config-file minimal_debug_config.yaml
```

## What the Debug Output Will Show

When you run these, look for:

1. **Stage Selection Logic**:
   ```
   [RANK0] iteration_step: 30001
   [RANK0] Stage 0 'Passkey Finetune': start=1, current=30001, matches=False
   [RANK0] *** CRITICAL: No matching stage found! dataloader is None ***
   ```

2. **Training Step Error**:
   ```
   [RANK0] === training_step ===
   [RANK0] dataloader is None in training_step!
   [RANK0] About to execute: train_batches = (next(dataloader) for _ in range(8))
   [RANK0] This will cause: TypeError: 'NoneType' object is not an iterator
   ```

3. **The Actual Error**:
   ```
   TypeError: 'NoneType' object is not an iterator
   ```

## Understanding the Root Cause

The debug output will confirm:

1. **Resume Step**: Training resumes from step 30000, so next step is 30001
2. **Stage Start**: Data stage starts at step 1
3. **Logic Failure**: Code looks for `stage.start_training_step == 30001`
4. **No Match**: No stage starts at 30001, so dataloader becomes None
5. **Error**: `next(None)` in generator expression fails

## Proposed Fix

Based on the debug output, the fix is to change the stage selection logic in `src/nanotron/trainer.py` around line 403:

```python
# Instead of exact match:
if stage.start_training_step == self.iteration_step:

# Use this logic:
if stage.start_training_step <= self.iteration_step:
    current_stage = stage  # Keep updating to find the latest applicable stage
```

## Running the Tests

1. **Quick reproduction**: `torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29401 test_actual_error.py`

2. **Send back the output** - the debug output will show exactly where and why the error occurs

3. **Apply the fix** once we confirm the root cause from your logs

The key is that these scripts work with your distributed environment and will show the exact moment when `current_dataloader` becomes None.