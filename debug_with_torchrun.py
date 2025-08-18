#!/usr/bin/env python3
"""
Debug script that works with torchrun distributed environment.
This script adds extensive debugging to the actual training process.

Usage: 
  torchrun --nproc_per_node=4 debug_with_torchrun.py --config-file passkey_finetune_300m_simple_config.yaml
"""

import argparse
import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - RANK%(process)d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_log(message, **kwargs):
    """Enhanced logging with extra context"""
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"RANK{rank} DEBUG: {message} | {context}")

def patch_trainer_with_debug():
    """Patch trainer methods with debug logging"""
    from nanotron.trainer import DistributedTrainer
    
    # Store original methods
    original_update_dataloader = DistributedTrainer._update_dataloader_based_on_training_stages
    original_training_step = DistributedTrainer.training_step
    
    def debug_update_dataloader(self, dataloaders):
        """Debug version of _update_dataloader_based_on_training_stages"""
        debug_log("_update_dataloader_based_on_training_stages called",
                  iteration_step=self.iteration_step,
                  current_dataloader_is_none=self.current_dataloader is None,
                  dataloaders_type=type(dataloaders).__name__)
        
        if hasattr(self.config, 'data_stages') and self.config.data_stages:
            debug_log("Data stages info",
                      stages=[(s.name, s.start_training_step) for s in self.config.data_stages])
        
        # Call original method
        result = original_update_dataloader(self, dataloaders)
        
        debug_log("After _update_dataloader_based_on_training_stages",
                  current_dataloader_is_none=self.current_dataloader is None,
                  current_dataloader_type=type(self.current_dataloader).__name__ if self.current_dataloader else None)
        
        return result
    
    def debug_training_step(self, dataloader):
        """Debug version of training_step"""
        debug_log("training_step called",
                  dataloader_is_none=dataloader is None,
                  dataloader_type=type(dataloader).__name__ if dataloader else None,
                  iteration_step=self.iteration_step,
                  n_micro_batches=self.n_micro_batches_per_batch)
        
        if dataloader is None:
            debug_log("CRITICAL ERROR: dataloader is None in training_step")
            debug_log("This will cause TypeError: 'NoneType' object is not an iterator")
            
            # Print detailed debug info
            debug_log("Debug info",
                      current_dataloader_is_none=self.current_dataloader is None,
                      iteration_step=self.iteration_step)
            
            if hasattr(self.config, 'data_stages'):
                for i, stage in enumerate(self.config.data_stages):
                    matches = stage.start_training_step == self.iteration_step
                    debug_log(f"Stage {i} matching",
                              stage_name=stage.name,
                              start_step=stage.start_training_step,
                              current_step=self.iteration_step,
                              matches=matches)
            
            # Still call original to trigger the actual error
            debug_log("About to trigger TypeError...")
        
        return original_training_step(self, dataloader)
    
    # Apply patches
    DistributedTrainer._update_dataloader_based_on_training_stages = debug_update_dataloader
    DistributedTrainer.training_step = debug_training_step
    
    debug_log("Debug patches applied to DistributedTrainer")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()
    
    debug_log("Starting debug training", config_file=args.config_file)
    
    try:
        # Apply debug patches
        patch_trainer_with_debug()
        
        # Import after patching
        from nanotron.trainer import DistributedTrainer
        from run_train import get_dataloader
        
        debug_log("Creating DistributedTrainer")
        trainer = DistributedTrainer(args.config_file)
        
        debug_log("Trainer created successfully",
                  start_iteration_step=trainer.start_iteration_step,
                  current_iteration_step=trainer.iteration_step,
                  current_dataloader_is_none=trainer.current_dataloader is None)
        
        debug_log("Creating dataloaders")
        dataloader = get_dataloader(trainer)
        
        debug_log("Dataloaders created",
                  type=type(dataloader).__name__,
                  is_dict=isinstance(dataloader, dict))
        
        if isinstance(dataloader, dict):
            for name, dl in dataloader.items():
                debug_log(f"Dataloader '{name}'",
                          type=type(dl).__name__,
                          is_callable=callable(dl))
        
        debug_log("Starting training - this should trigger the error")
        
        # This should trigger the dataloader issue
        trainer.train(dataloader)
        
    except Exception as e:
        debug_log("Training failed with error",
                  error=str(e),
                  error_type=type(e).__name__)
        
        # Check if this is the expected error
        if "TypeError" in str(e) and "'NoneType' object is not an iterator" in str(e):
            debug_log("SUCCESS: Reproduced the expected TypeError!")
        
        # Re-raise to see full traceback
        raise

if __name__ == "__main__":
    main()