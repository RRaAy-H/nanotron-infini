#!/usr/bin/env python3
"""
Comprehensive debugging script for dataloader issues.
This script provides multiple debugging modes and extensive logging.

Usage:
  python debug_dataloader_comprehensive.py --config-file passkey_finetune_300m_simple_config.yaml --mode all
  python debug_dataloader_comprehensive.py --config-file passkey_finetune_300m_simple_config.yaml --mode reproduce
  python debug_dataloader_comprehensive.py --config-file passkey_finetune_300m_simple_config.yaml --mode analyze
"""

import argparse
import sys
import os
import json
import yaml
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class DataloaderDebugger:
    def __init__(self, config_file):
        self.config_file = config_file
        self.trainer = None
        self.dataloaders = None
        self.debug_log = []
        
    def log(self, message, **kwargs):
        """Log debug information"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = {
            "timestamp": timestamp,
            "message": message,
            "data": kwargs
        }
        self.debug_log.append(log_entry)
        
        # Also print to console
        context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        print(f"[{timestamp}] {message} | {context}")
    
    def save_debug_log(self, filename="debug_log.json"):
        """Save debug log to file"""
        with open(filename, 'w') as f:
            json.dump(self.debug_log, f, indent=2, default=str)
        print(f"\nDebug log saved to: {filename}")
    
    def load_config_info(self):
        """Load and analyze configuration"""
        self.log("Loading configuration", config_file=self.config_file)
        
        try:
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            general_step = config_data.get('general', {}).get('step')
            train_steps = config_data.get('tokens', {}).get('train_steps')
            data_stages = config_data.get('data_stages', [])
            
            self.log("Config loaded", 
                     general_step=general_step,
                     train_steps=train_steps,
                     num_data_stages=len(data_stages))
            
            for i, stage in enumerate(data_stages):
                self.log(f"Data stage {i}",
                         name=stage.get('name'),
                         start_step=stage.get('start_training_step'))
            
            return config_data
            
        except Exception as e:
            self.log("Config loading failed", error=str(e))
            raise
    
    def create_trainer(self):
        """Create trainer with debugging"""
        self.log("Creating DistributedTrainer")
        
        try:
            from nanotron.trainer import DistributedTrainer
            
            self.trainer = DistributedTrainer(self.config_file)
            
            self.log("Trainer created",
                     start_iteration_step=self.trainer.start_iteration_step,
                     current_iteration_step=self.trainer.iteration_step,
                     current_dataloader_is_none=self.trainer.current_dataloader is None,
                     n_micro_batches_per_batch=self.trainer.n_micro_batches_per_batch)
            
            return True
            
        except Exception as e:
            self.log("Trainer creation failed", error=str(e), error_type=type(e).__name__)
            import traceback
            self.log("Traceback", traceback=traceback.format_exc())
            return False
    
    def create_dataloaders(self):
        """Create dataloaders with debugging"""
        self.log("Creating dataloaders")
        
        try:
            from run_train import get_dataloader
            
            self.dataloaders = get_dataloader(self.trainer)
            
            self.log("Dataloaders created",
                     type=type(self.dataloaders).__name__,
                     is_dict=isinstance(self.dataloaders, dict))
            
            if isinstance(self.dataloaders, dict):
                for name, dl in self.dataloaders.items():
                    self.log(f"Dataloader '{name}'",
                             type=type(dl).__name__,
                             is_callable=callable(dl),
                             is_none=dl is None)
            
            return True
            
        except Exception as e:
            self.log("Dataloader creation failed", error=str(e), error_type=type(e).__name__)
            import traceback
            self.log("Traceback", traceback=traceback.format_exc())
            return False
    
    def test_dataloader_update_at_step(self, step):
        """Test dataloader update at specific step"""
        self.log(f"Testing dataloader update at step {step}")
        
        # Save original state
        original_step = self.trainer.iteration_step
        original_current_dataloader = self.trainer.current_dataloader
        
        try:
            # Set the step
            self.trainer.iteration_step = step
            
            self.log(f"Before update at step {step}",
                     iteration_step=self.trainer.iteration_step,
                     current_dataloader_is_none=self.trainer.current_dataloader is None)
            
            # Call the update method
            self.trainer._update_dataloader_based_on_training_stages(self.dataloaders)
            
            self.log(f"After update at step {step}",
                     current_dataloader_is_none=self.trainer.current_dataloader is None,
                     current_dataloader_type=type(self.trainer.current_dataloader).__name__ if self.trainer.current_dataloader else None)
            
            # Test if this would cause the error
            if self.trainer.current_dataloader is None:
                self.log(f"CRITICAL at step {step}: current_dataloader is None - would cause TypeError")
                return False
            else:
                self.log(f"SUCCESS at step {step}: current_dataloader is properly set")
                return True
                
        except Exception as e:
            self.log(f"Error during update at step {step}", error=str(e))
            return False
        finally:
            # Restore original state
            self.trainer.iteration_step = original_step
            self.trainer.current_dataloader = original_current_dataloader
    
    def reproduce_exact_error(self):
        """Reproduce the exact TypeError that's occurring"""
        self.log("Reproducing exact TypeError")
        
        # Set up the exact scenario from the traceback
        self.trainer.iteration_step = 30001  # From the error
        
        # Update dataloader (this should result in None)
        self.trainer._update_dataloader_based_on_training_stages(self.dataloaders)
        
        self.log("State before error line",
                 iteration_step=self.trainer.iteration_step,
                 current_dataloader_is_none=self.trainer.current_dataloader is None,
                 n_micro_batches_per_batch=self.trainer.n_micro_batches_per_batch)
        
        # This is the exact line that fails in trainer.py:574
        try:
            self.log("Executing: train_batches = (next(dataloader) for _ in range(self.n_micro_batches_per_batch))")
            train_batches = (next(self.trainer.current_dataloader) for _ in range(self.trainer.n_micro_batches_per_batch))
            self.log("UNEXPECTED: No error occurred")
            return False
        except TypeError as e:
            self.log("REPRODUCED TypeError", error=str(e))
            return True
        except Exception as e:
            self.log("Different error occurred", error=str(e), error_type=type(e).__name__)
            return False
    
    def analyze_dataloader_state_transitions(self):
        """Analyze how dataloader state changes during training"""
        self.log("Analyzing dataloader state transitions")
        
        steps_to_test = [1, 2, 100, 1000, 29999, 30000, 30001, 30002, 30500]
        
        for step in steps_to_test:
            success = self.test_dataloader_update_at_step(step)
            if not success:
                self.log(f"FAILURE POINT IDENTIFIED", step=step)
    
    def run_mode_reproduce(self):
        """Mode: Just reproduce the error"""
        self.log("=== MODE: REPRODUCE ===")
        config_data = self.load_config_info()
        
        if not self.create_trainer():
            return False
        
        if not self.create_dataloaders():
            return False
        
        return self.reproduce_exact_error()
    
    def run_mode_analyze(self):
        """Mode: Analyze the issue comprehensively"""
        self.log("=== MODE: ANALYZE ===")
        config_data = self.load_config_info()
        
        if not self.create_trainer():
            return False
        
        if not self.create_dataloaders():
            return False
        
        self.analyze_dataloader_state_transitions()
        return True
    
    def run_mode_all(self):
        """Mode: Complete comprehensive debugging"""
        self.log("=== MODE: ALL ===")
        
        config_data = self.load_config_info()
        
        if not self.create_trainer():
            return False
        
        if not self.create_dataloaders():
            return False
        
        self.analyze_dataloader_state_transitions()
        error_reproduced = self.reproduce_exact_error()
        
        self.log("Final summary",
                 error_reproduced=error_reproduced,
                 total_log_entries=len(self.debug_log))
        
        return error_reproduced

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML config file")
    parser.add_argument("--mode", type=str, choices=['reproduce', 'analyze', 'all'], default='all', 
                       help="Debug mode: reproduce (just reproduce error), analyze (comprehensive analysis), all (everything)")
    parser.add_argument("--save-log", type=str, default="dataloader_debug.json", help="File to save debug log")
    args = parser.parse_args()
    
    debugger = DataloaderDebugger(args.config_file)
    
    print("COMPREHENSIVE DATALOADER DEBUGGING")
    print("=" * 80)
    print(f"Config: {args.config_file}")
    print(f"Mode: {args.mode}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:2]}")
    print("=" * 80)
    
    try:
        if args.mode == 'reproduce':
            success = debugger.run_mode_reproduce()
        elif args.mode == 'analyze':
            success = debugger.run_mode_analyze()
        else:  # 'all'
            success = debugger.run_mode_all()
        
        debugger.save_debug_log(args.save_log)
        
        print("\n" + "=" * 80)
        if success:
            print("✓ Debugging completed successfully")
        else:
            print("✗ Issues detected during debugging")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        debugger.save_debug_log(args.save_log)

if __name__ == "__main__":
    main()