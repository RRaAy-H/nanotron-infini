#!/usr/bin/env python3
"""
Permanent fix for balance factor loading in Infini-Attention models.

This module automatically loads balance factors correctly whenever a model is loaded,
fixing the parameter name mismatch between checkpoint storage and model expectations.
"""

import logging
from pathlib import Path
from typing import Optional
import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)


def auto_load_balance_factors(model, checkpoint_path: str, verbose: bool = False):
    """
    Automatically load balance factors from checkpoint after model loading.
    
    This function fixes the issue where balance factors are saved in checkpoint
    but not properly loaded during model initialization.
    
    Args:
        model: The loaded model (LlamaForTraining or similar)
        checkpoint_path: Path to the checkpoint directory
        verbose: Whether to print detailed loading information
    """
    
    checkpoint_path = Path(checkpoint_path)
    
    if verbose:
        logger.info("🔧 Auto-loading balance factors...")
    
    loaded_count = 0
    total_count = 0
    
    for layer_idx, layer in enumerate(model.model.decoder):
        total_count += 1
        
        # Access balance factors through pipeline block structure
        if hasattr(layer, 'pp_block') and hasattr(layer.pp_block, 'attn') and hasattr(layer.pp_block.attn, 'balance_factors'):
            bf_file = checkpoint_path / f"model/model/decoder/{layer_idx}/pp_block/attn/model_balance_factors_pp-rank-0-of-1_tp-rank-0-of-1.safetensors"
            
            if bf_file.exists():
                try:
                    with safe_open(str(bf_file), framework='pt', device='cpu') as f:
                        if 'data' in f.keys():
                            saved_bf = f.get_tensor('data')
                            
                            # Move to correct device and dtype
                            target_device = layer.pp_block.attn.balance_factors.device
                            target_dtype = layer.pp_block.attn.balance_factors.dtype
                            saved_bf = saved_bf.to(device=target_device, dtype=target_dtype)
                            
                            # Update model parameters
                            with torch.no_grad():
                                layer.pp_block.attn.balance_factors.data.copy_(saved_bf)
                            
                            loaded_count += 1
                            
                            if verbose:
                                logger.info(f"  ✅ Layer {layer_idx}: Loaded balance factors (mean={saved_bf.mean().item():.3f})")
                                
                except Exception as e:
                    if verbose:
                        logger.warning(f"  ❌ Layer {layer_idx}: Failed to load balance factors: {e}")
            else:
                if verbose:
                    logger.warning(f"  ❌ Layer {layer_idx}: Balance factor file not found")
    
    if loaded_count > 0:
        if verbose:
            logger.info(f"✅ Successfully loaded balance factors for {loaded_count}/{total_count} layers")
        
        # Verify the fix worked by checking the first layer
        layer0 = model.model.decoder[0]
        if hasattr(layer0, 'pp_block') and hasattr(layer0.pp_block, 'attn') and hasattr(layer0.pp_block.attn, 'balance_factors'):
            bf = layer0.pp_block.attn.balance_factors.data
            if bf.std().item() > 0.1:  # Non-zero variation indicates successful loading
                if verbose:
                    activated = layer0.pp_block.attn.balance_act_func(bf)
                    avg_memory_weight = activated.mean().item()
                    logger.info(f"🧠 Verification: Layer 0 using {avg_memory_weight*100:.1f}% memory (expected ~94%)")
                return True
    
    if verbose:
        logger.warning(f"❌ Balance factor loading failed: {loaded_count}/{total_count} layers loaded")
    return False


def is_balance_factor_loading_needed(model) -> bool:
    """
    Check if balance factor loading is needed.
    
    Returns True if balance factors exist but appear to be uninitialized (all zeros).
    """
    
    # Check first layer as representative
    layer0 = model.model.decoder[0]
    if hasattr(layer0, 'pp_block') and hasattr(layer0.pp_block, 'attn') and hasattr(layer0.pp_block.attn, 'balance_factors'):
        bf = layer0.pp_block.attn.balance_factors.data
        # If std is very low, balance factors are likely uninitialized
        return bf.std().item() < 0.1
    
    return False


def apply_balance_factor_fix_if_needed(model, checkpoint_path: str, verbose: bool = True):
    """
    Apply balance factor fix only if needed.
    
    This is the main function to call after model loading.
    """
    
    if is_balance_factor_loading_needed(model):
        if verbose:
            logger.info("🔍 Detected uninitialized balance factors, applying fix...")
        return auto_load_balance_factors(model, checkpoint_path, verbose=verbose)
    else:
        if verbose:
            logger.info("✅ Balance factors already properly loaded")
        return True
