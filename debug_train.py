#!/usr/bin/env python3
"""
Debug version of run_train.py with extensive logging to diagnose the dataloader issue.
Run this instead of run_train.py to get detailed information about what's happening.

Usage: python debug_train.py --config-file passkey_finetune_300m_simple_config.yaml
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_log(message, **kwargs):
    """Enhanced logging with extra context"""
    context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"DEBUG: {message} | {context}")

def inspect_dataloader(dataloader, name="dataloader"):
    """Inspect dataloader properties"""
    debug_log(f"Inspecting {name}", 
              type=type(dataloader).__name__,
              is_none=dataloader is None,
              is_callable=callable(dataloader) if dataloader is not None else False)
    
    if hasattr(dataloader, '__iter__'):
        debug_log(f"{name} is iterable")
    if hasattr(dataloader, '__next__'):
        debug_log(f"{name} has __next__ method")
    
    return dataloader

def debug_get_dataloader(trainer):
    """Debug version of get_dataloader with extensive logging"""
    from run_train import get_dataloader_from_data_stage
    from nanotron.config import DatasetStageArgs
    from typing import cast
    
    debug_log("Starting get_dataloader", 
              has_data_stages=hasattr(trainer.config, 'data_stages'),
              data_stages_count=len(trainer.config.data_stages) if hasattr(trainer.config, 'data_stages') else 0)
    
    if not hasattr(trainer.config, 'data_stages') or trainer.config.data_stages is None:
        debug_log("No data_stages found, using simple dataloader")
        # Handle single dataloader case
        return None
    
    sorted_stages = sorted(trainer.config.data_stages, key=lambda stage: stage.start_training_step)
    debug_log("Sorted stages", stages=[f"{s.name}@{s.start_training_step}" for s in sorted_stages])
    
    dataloaders = {}
    for idx, stage in enumerate(sorted_stages):
        stage = cast(DatasetStageArgs, stage)
        debug_log(f"Processing stage {idx}", 
                  stage_name=stage.name,
                  start_step=stage.start_training_step,
                  is_first_stage=idx==0)
        
        if idx == 0:
            # Create actual dataloader for first stage
            debug_log(f"Creating actual dataloader for {stage.name}")
            try:
                dataloader = get_dataloader_from_data_stage(trainer, stage.data)
                inspect_dataloader(dataloader, f"stage_{idx}_dataloader")
            except Exception as e:
                debug_log(f"Error creating dataloader for {stage.name}", error=str(e))
                raise
        else:
            # Create lambda for lazy initialization
            debug_log(f"Creating lazy dataloader for {stage.name}")
            dataloader = lambda stage=stage: get_dataloader_from_data_stage(trainer, stage.data)
            inspect_dataloader(dataloader, f"stage_{idx}_lambda")
        
        dataloaders[stage.name] = dataloader
        debug_log(f"Added dataloader for {stage.name}", total_dataloaders=len(dataloaders))
    
    debug_log("Final dataloaders", dataloader_keys=list(dataloaders.keys()))
    return dataloaders

def debug_trainer_state(trainer):
    """Debug trainer state before training"""
    debug_log("Trainer state before training",
              iteration_step=trainer.iteration_step,
              start_iteration_step=trainer.start_iteration_step,
              current_dataloader_type=type(trainer.current_dataloader).__name__ if trainer.current_dataloader else None,
              current_dataloader_is_none=trainer.current_dataloader is None)

def debug_update_dataloader(trainer, dataloaders):
    """Debug version of _update_dataloader_based_on_training_stages"""
    debug_log("Before _update_dataloader_based_on_training_stages",
              iteration_step=trainer.iteration_step,
              current_dataloader_is_none=trainer.current_dataloader is None,
              dataloaders_keys=list(dataloaders.keys()) if isinstance(dataloaders, dict) else type(dataloaders))
    
    # Call the original method
    trainer._update_dataloader_based_on_training_stages(dataloaders)
    
    debug_log("After _update_dataloader_based_on_training_stages",
              current_dataloader_is_none=trainer.current_dataloader is None,
              current_dataloader_type=type(trainer.current_dataloader).__name__ if trainer.current_dataloader else None)

def main():
    # Import here to get better error messages
    try:
        from nanotron.trainer import DistributedTrainer
        from run_train import get_dataloader
        debug_log("Successfully imported required modules")
    except ImportError as e:
        debug_log("Import error", error=str(e))
        raise
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    args = parser.parse_args()
    
    debug_log("Starting debug training", config_file=args.config_file)
    
    try:
        # Load trainer
        debug_log("Creating DistributedTrainer")
        trainer = DistributedTrainer(args.config_file)
        debug_log("DistributedTrainer created successfully")
        
        debug_trainer_state(trainer)
        
        # Get dataloader with debugging
        debug_log("Getting dataloader")
        dataloader = debug_get_dataloader(trainer)
        debug_log("Dataloader creation completed")
        
        inspect_dataloader(dataloader, "main_dataloader")
        
        # Debug the first few training steps
        debug_log("Starting training loop debug")
        
        # Manually call the dataloader update method with debugging
        debug_update_dataloader(trainer, dataloader)
        
        # Try to simulate the exact failure point
        debug_log("Simulating training_step call")
        try:
            # This is where the error occurs
            if trainer.current_dataloader is None:
                debug_log("CRITICAL: current_dataloader is None - this will cause the error!")
                debug_log("Dataloader state dump",
                          dataloaders_type=type(dataloader),
                          dataloaders_keys=list(dataloader.keys()) if isinstance(dataloader, dict) else "not_dict",
                          data_stages=[f"{s.name}@{s.start_training_step}" for s in trainer.config.data_stages])
                raise RuntimeError("current_dataloader is None - this reproduces the original error")
            else:
                debug_log("SUCCESS: current_dataloader is properly set", 
                          type=type(trainer.current_dataloader).__name__)
                # Try to get one batch to verify it works
                try:
                    train_batches = (next(trainer.current_dataloader) for _ in range(trainer.n_micro_batches_per_batch))
                    debug_log("Successfully created train_batches generator")
                    # Try to get first batch
                    first_batch = next(train_batches)
                    debug_log("Successfully got first batch", batch_keys=list(first_batch.keys()) if isinstance(first_batch, dict) else "not_dict")
                except Exception as batch_error:
                    debug_log("Error getting batch", error=str(batch_error))
                    
        except Exception as training_error:
            debug_log("Training step simulation failed", error=str(training_error))
            raise
        
    except Exception as e:
        debug_log("Main execution failed", error=str(e), error_type=type(e).__name__)
        raise

if __name__ == "__main__":
    main()