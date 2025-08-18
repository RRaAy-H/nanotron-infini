# Debug Tools for Dataloader Error

This directory contains minimal, focused debug tools to diagnose and fix the `TypeError: 'NoneType' object is not an iterator` error.

## Quick Start

To reproduce the error with detailed debugging:

```bash
torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29401 test_actual_error.py
```

## Files

### Core Debug Tools
- **`test_actual_error.py`** - Main test that reproduces the exact error with detailed logging
- **`debug_with_torchrun.py`** - Alternative debug script that works with existing configs
- **`minimal_debug_config.yaml`** - Lightweight config for fast testing

### Documentation
- **`FIXED_DEBUG_INSTRUCTIONS.md`** - Complete usage instructions and analysis

## The Error

When resuming training from step 30000:
1. Next iteration step is 30001
2. Data stage starts at step 1
3. Code looks for `stage.start_training_step == 30001` (exact match)
4. No stage found → dataloader becomes None
5. `next(None)` fails with TypeError

## Expected Debug Output

```
[RANK0] iteration_step: 30001
[RANK0] Stage 0 'Test Resume Stage': start=1, current=30001, matches=False
[RANK0] *** CRITICAL: No matching stage found! dataloader is None ***
[RANK0] This will cause: TypeError: 'NoneType' object is not an iterator
TypeError: 'NoneType' object is not an iterator
```

## Usage

Run the test, capture the output, and send it back for analysis. The debug output will confirm the root cause and guide the fix implementation.

## Cleanup

To remove debug files:
```bash
rm test_actual_error.py debug_with_torchrun.py minimal_debug_config.yaml README_DEBUG.md FIXED_DEBUG_INSTRUCTIONS.md
```