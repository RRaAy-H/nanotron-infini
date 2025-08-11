#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/run_direct_training.py

"""
Direct training script for Infini-Llama that uses DistributedTrainer directly.
This script avoids the import issues with the other training scripts.
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# Add project root and src directories to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Set up argument parsing
parser = argparse.ArgumentParser(description="Train Infini-Llama model using DistributedTrainer")
parser.add_argument("--config-file", type=str, required=True, help="Path to the config file")
parser.add_argument("--data-dir", type=str, required=True, help="Path to preprocessed data directory")
parser.add_argument("--disable-infini-attn", action="store_true", help="Disable Infini-Attention for baseline model")
parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use (default: 0)")
parser.add_argument("--tensorboard-dir", type=str, default=None, help="Directory for TensorBoard logs")
parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
parser.add_argument("--use-gpu-dataloader", action="store_true", help="Use GPU-accelerated dataloader")
parser.add_argument("--offline-mode", action="store_true", help="Run in offline mode (no downloads from HuggingFace)")
parser.add_argument("--disable-flash-attn", action="store_true", help="Disable Flash Attention (for compatibility issues)")
parser.add_argument("--auto-detect-flash-attn", action="store_true", help="Automatically detect and handle Flash Attention compatibility")
args = parser.parse_args()

# Configure the environment
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# Configure logging level
if args.verbose:
    os.environ["NANOTRON_LOG_LEVEL"] = "debug"
    print("Verbose logging enabled")

# Configure offline mode if requested
if args.offline_mode:
    print("Configuring offline mode to avoid HuggingFace downloads...")
    # Set HuggingFace environment variables to use local files only
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["NO_GIT"] = "1"
    # Disable HTTP requests 
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    
# Check Flash Attention compatibility if auto-detection is enabled
need_to_disable_flash_attn = args.disable_flash_attn or os.environ.get("DISABLE_FLASH_ATTN") == "1"

if args.auto_detect_flash_attn:
    try:
        # Try to use our flash_attention_compatibility module
        sys.path.insert(0, os.path.join(project_root, "scripts"))
        from flash_attention_compatibility import is_flash_attention_compatible
        
        if not is_flash_attention_compatible():
            print("Automatically detected Flash Attention incompatibility - will disable it")
            need_to_disable_flash_attn = True
        else:
            print("Flash Attention compatibility check passed - will use Flash Attention")
    except ImportError:
        print("Could not import flash_attention_compatibility module, skipping auto-detection")
        # Try a simple check for GLIBC version
        try:
            import ctypes
            import re
            
            # Try to get GLIBC version
            process_namespace = ctypes.CDLL(None)
            if hasattr(process_namespace, 'gnu_get_libc_version'):
                gnu_get_libc_version = process_namespace.gnu_get_libc_version
                gnu_get_libc_version.restype = ctypes.c_char_p
                version_str = gnu_get_libc_version().decode('utf-8')
                
                # Check if version is below 2.32 (Flash Attention typically needs 2.32+)
                match = re.match(r'(\d+)\.(\d+)', version_str)
                if match:
                    major, minor = int(match.group(1)), int(match.group(2))
                    if major < 2 or (major == 2 and minor < 32):
                        print(f"Detected GLIBC {version_str}, which is below 2.32 - will disable Flash Attention")
                        need_to_disable_flash_attn = True
                    else:
                        print(f"Detected GLIBC {version_str}, which should be compatible with Flash Attention")
        except Exception as e:
            print(f"Error during GLIBC version check: {e}")

# Configure Flash Attention disabling if needed
if need_to_disable_flash_attn:
    print("Flash Attention is disabled - will use standard attention implementation")
    
    # Set environment variable for consistent behavior across all modules
    os.environ["DISABLE_FLASH_ATTN"] = "1"
    # Also set this environment variable for transformers library compatibility
    os.environ["USE_FLASH_ATTENTION"] = "0"
    
    # Create mock flash attention modules instead of blocking imports completely
    import types
    
    # Create a mock flash_attn module with dummy implementations
    mock_flash_attn = types.ModuleType('flash_attn')
    sys.modules['flash_attn'] = mock_flash_attn
    
    # Create layers submodule
    mock_layers = types.ModuleType('flash_attn.layers')
    sys.modules['flash_attn.layers'] = mock_layers
    mock_flash_attn.layers = mock_layers
    
    # Create rotary submodule
    mock_rotary = types.ModuleType('flash_attn.layers.rotary')
    sys.modules['flash_attn.layers.rotary'] = mock_rotary
    mock_layers.rotary = mock_rotary
    
    # Create ops submodule
    mock_ops = types.ModuleType('flash_attn.ops')
    sys.modules['flash_attn.ops'] = mock_ops
    mock_flash_attn.ops = mock_ops
    
    # Add dummy version attribute
    mock_flash_attn.__version__ = "0.0.0-disabled"
    
    # Create dummy RotaryEmbedding class
    class DummyRotaryEmbedding:
        def __init__(self, *args, **kwargs):
            print("Using dummy RotaryEmbedding (Flash Attention disabled)")
            
        def __call__(self, *args, **kwargs):
            raise NotImplementedError("Flash Attention is disabled")
    
    # Add the dummy class to the mock module
    mock_rotary.RotaryEmbedding = DummyRotaryEmbedding
    
    print("Mock Flash Attention modules created for compatibility")
    
    # Try to prevent flash_attn imports in modules that might use it
    try:
        import transformers.models.llama.modeling_llama
        # Monkey patch transformers to avoid flash attention usage
        if hasattr(transformers.models.llama.modeling_llama, "_is_flash_attn_available"):
            transformers.models.llama.modeling_llama._is_flash_attn_available = lambda: False
            print("Successfully patched transformers to avoid Flash Attention usage")
    except (ImportError, AttributeError):
        pass
    # Set timeouts to minimal values to fail fast
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "1"
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["CURL_CA_BUNDLE"] = ""
    # Clear any proxy settings
    for proxy_var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        if proxy_var in os.environ:
            del os.environ[proxy_var]
    print("Environment configured for offline mode")

# Apply multiple methods to fix the Adam optimizer weight_decay=None issue
adam_patched = False

# Method 1: Try the fix_adam_none_issue module
try:
    sys.path.insert(0, os.path.join(project_root, "scripts"))
    from fix_adam_none_issue import patch_adam_optimizer
    if patch_adam_optimizer():
        print("Adam optimizer patch applied via fix_adam_none_issue module")
        adam_patched = True
except ImportError:
    print("Could not import fix_adam_none_issue module")

# Method 2: Try our project's built-in patch
if not adam_patched:
    try:
        from nanotron.optim import patch_adam
        print("Adam optimizer patch applied via nanotron.optim.patch_adam")
        adam_patched = True
    except ImportError:
        print("Warning: Could not import nanotron.optim.patch_adam")

# Method 3: Apply direct monkey patch if needed
if not adam_patched or os.environ.get("FIX_ADAM_WEIGHT_DECAY") == "true":
    try:
        import torch.optim.adam
        original_adam = torch.optim.adam.adam
        
        def patched_adam(*args, **kwargs):
            if 'weight_decay' in kwargs and kwargs['weight_decay'] is None:
                print("Direct patch: Replaced None weight_decay with 0.0")
                kwargs['weight_decay'] = 0.0
            if len(args) >= 4 and args[3] is None:
                args = list(args)
                args[3] = 0.0
                args = tuple(args)
            return original_adam(*args, **kwargs)
        
        torch.optim.adam.adam = patched_adam
        print("Applied direct Adam optimizer patch in training script")
        adam_patched = True
    except Exception as e:
        print(f"Warning: Failed to apply direct Adam optimizer patch: {e}")

# Import after setting environment variables
from dataclasses import dataclass, field

# Import SummaryWriter conditionally to avoid errors if not available
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

@dataclass
class InfiniAttentionConfig:
    segment_length: int = 64
    turn_on_memory: bool = True
    balance_init_type: str = 'zeros'
    balance_act_type: str = 'orig_sigmoid'
    balance_factor_lr: float = 0.001
    logging: bool = False
    logging_interval: int = 100
    log_grad: bool = False
    log_segment_acts: bool = False

@dataclass
class Config:
    infini_attention: InfiniAttentionConfig = field(default_factory=InfiniAttentionConfig)

# Set up the Infini-Attention configuration
from nanotron import constants
constants.CONFIG = Config()

# If disabling Infini-Attention, update configuration
if args.disable_infini_attn:
    print("Disabling Infini-Attention (running baseline model)")
    constants.CONFIG.infini_attention.turn_on_memory = False

# Import trainer
from nanotron.trainer import DistributedTrainer
from datasets import load_from_disk
from transformers import AutoTokenizer

def main():
    print(f"Creating trainer with config: {args.config_file}")
    # Create trainer
    trainer = DistributedTrainer(args.config_file)
    
    print(f"Loading dataset from {args.data_dir}")
    # Load dataset
    dataset_path = Path(args.data_dir) / "processed_dataset"
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)
    
    train_dataset = load_from_disk(str(dataset_path))
    
    # Load tokenizer
    tokenizer_path = Path(args.data_dir) / "tokenizer"
    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    else:
        # Try to get tokenizer from metadata or use default
        metadata_path = Path(args.data_dir) / "metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            tokenizer_name = metadata.get("tokenizer", "meta-llama/Llama-2-7b-hf")
        else:
            tokenizer_name = "meta-llama/Llama-2-7b-hf"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Set up data collator
    from nanotron.dataloader import DataCollatorForCLM
    from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
    
    # Get the correct input and output pipeline parallel ranks
    # These will be determined based on the trainer's model
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)
    
    # Initialize data collator with proper parameters
    data_collator = DataCollatorForCLM(
        sequence_length=trainer.config.tokens.sequence_length,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        parallel_context=trainer.parallel_context,
    )
    
    # Create DataLoader for training
    print(f"Creating DataLoader with batch size: {trainer.config.tokens.micro_batch_size}")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=trainer.config.tokens.micro_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    print(f"DataLoader created successfully with {len(train_dataset)} samples")
    
    # Set up TensorBoard if requested
    if args.tensorboard_dir:
        os.makedirs(args.tensorboard_dir, exist_ok=True)
        if SummaryWriter is not None:
            tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)
            print(f"TensorBoard logging enabled at {args.tensorboard_dir}")
        else:
            print("TensorBoard not available. Install torch.utils.tensorboard for logging.")
            tb_writer = None
    else:
        tb_writer = None
    
    # Train model
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Starting training on device: {device}")
    print(f"Model type: {'baseline' if args.disable_infini_attn else 'Infini-Attention'}")
    print(f"Config file: {args.config_file}")
    print(f"Dataset size: {len(train_dataset)} samples")
    print(f"Batch size: {trainer.config.tokens.micro_batch_size}")
    print(f"Sequence length: {trainer.config.tokens.sequence_length}")
    print(f"TensorBoard logging: {'Enabled' if args.tensorboard_dir else 'Disabled'}")
    print(f"Training logs directory: {os.environ.get('TRAINING_LOGS_DIR', 'training_logs')}")
    
    try:
        print("Calling trainer.train() with the dataloader...")
        print(f"Using training stage name: 'Training Stage' as defined in the config")
        # Pass the dataloader with the stage name matching the config and additional parameters as kwargs
        trainer.train(
            dataloader_or_dls={"Training Stage": train_loader}, 
            tokenizer=tokenizer,
            device=device
        )
        print("Training completed successfully!")
    
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        if tb_writer:
            tb_writer.close()
    
    # Save final model
    if trainer.is_main_process:
        model_type = "baseline" if args.disable_infini_attn else "infini"
        from datetime import datetime
        output_dir = Path("models") / f"{model_type}_llama_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving final model to {output_dir}")
        trainer.save_model(output_dir)

if __name__ == "__main__":
    main()
