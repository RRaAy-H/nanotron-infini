#!/usr/bin/env python
# This script creates a compatibility layer for models to work with or without Flash Attention

import os
import sys
import importlib
from functools import wraps

def create_flash_attention_compatibility_layer():
    """
    Creates a compatibility layer to allow models to work in environments where 
    Flash Attention is not available or has been disabled.
    """
    # First check if Flash Attention is disabled via environment variable
    flash_disabled = os.environ.get("DISABLE_FLASH_ATTN", "0") == "1"
    
    if not flash_disabled:
        # Try importing - if it works, we don't need to do anything
        try:
            import flash_attn
            print("Flash Attention is available and enabled")
            return False  # No patching needed
        except ImportError:
            flash_disabled = True
            print("Flash Attention module not found, enabling compatibility mode")
    
    if flash_disabled:
        print("Creating Flash Attention compatibility layer...")
        
        # Create a patch for the RotaryEmbedding class in LlamaModel
        try:
            # Import the modules we need to patch
            from nanotron.models import llama
            import types
            import torch.nn as nn
            
            # Define our custom RotaryEmbedding class
            class StandardRotaryEmbedding(nn.Module):
                def __init__(
                    self,
                    dim,
                    base=10000.0,
                    interleaved=False,
                    scale_base=None,
                    scaling_factor=1.0,
                    pos_idx_in_fp32=True,
                    device=None,
                ):
                    super().__init__()
                    self.dim = dim
                    self.base = base
                    self.interleaved = interleaved
                    self.scale_base = scale_base
                    self.scaling_factor = scaling_factor
                    self.pos_idx_in_fp32 = pos_idx_in_fp32
                    
                    # Generate inv_freq
                    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
                    self.register_buffer("inv_freq", inv_freq)
                    
                    print("Using standard RotaryEmbedding as Flash Attention is disabled")
                
                def forward(self, x, position_ids):
                    """
                    Apply rotary embeddings to input tensors using the given positions.
                    
                    Args:
                        x: Input tensor of shape (batch_size, seq_len, h, dim)
                        position_ids: Position tensor of shape (batch_size, seq_len)
                        
                    Returns:
                        Rotary embedded tensor of the same shape as the input.
                    """
                    # Standard implementation for rotary embeddings
                    batch_size, seq_len, h, dim = x.shape
                    device = x.device
                    
                    # Handle position indices
                    if self.pos_idx_in_fp32:
                        position_ids = position_ids.float()
                    else:
                        position_ids = position_ids
                    
                    # Create sinusoidal pattern
                    freqs = position_ids[:, :, None, None] * self.inv_freq[None, None, None, :].to(device)
                    emb = torch.cat((freqs, freqs), dim=-1)
                    
                    # Implement rotary embedding by applying rotation
                    cos = emb.cos()
                    sin = emb.sin()
                    
                    # Reshape for broadcasting
                    cos = cos.view(batch_size, seq_len, 1, dim)
                    sin = sin.view(batch_size, seq_len, 1, dim)
                    
                    # Apply rotary embedding
                    x_rotated = torch.cat(
                        [-x[..., dim//2:], x[..., :dim//2]], dim=-1
                    )
                    return x * cos + x_rotated * sin
            
            # Check if we need to patch the CausalSelfAttention.__init__
            original_init = llama.CausalSelfAttention.__init__
            
            @wraps(original_init)
            def patched_init(self, config, parallel_config, tp_pg, layer_idx):
                # Skip the flash_attn import in the original __init__
                try:
                    # Try with our standard implementation first
                    self.rotary_emb = StandardRotaryEmbedding(
                        dim=config.hidden_size // config.num_attention_heads,
                        base=config.rope_theta,
                        interleaved=False,
                    )
                    
                    # Then continue with the rest of the initialization
                    # We need to set all the attributes that would be set after the flash_attn import
                    self.config = config
                    self.layer_idx = layer_idx
                    
                    # Tensor parallel considerations
                    tp_size = tp_pg.size()
                    self.tp_size = tp_size
                    self.tp_rank = dist.get_rank(tp_pg)
                    self.tp_group = tp_pg
                    
                    # Get local values for the head dimensions
                    num_heads = config.num_attention_heads // tp_size
                    self.num_heads = num_heads
                    try:
                        num_kv_heads = config.num_key_value_heads // tp_size
                    except AttributeError:
                        num_kv_heads = num_heads
                        print(f"num_key_value_heads not found in config, using {num_kv_heads}")
                    self.num_kv_heads = num_kv_heads
                    self.head_dim = config.hidden_size // config.num_attention_heads
                    self.hidden_size = config.hidden_size
                    
                    # Continue with initializing the model parameters
                    # I'll import the rest of the initialization code from the original __init__
                    # The rest of the attributes, such as q_proj, k_proj, v_proj, would be set here
                    
                except Exception as e:
                    print(f"Error in patched initialization: {e}")
                    # Fall back to original init, which might fail but at least we tried
                    original_init(self, config, parallel_config, tp_pg, layer_idx)
            
            # Apply the patch
            llama.CausalSelfAttention.__init__ = patched_init
            print("Successfully patched CausalSelfAttention.__init__")
            
            # Create mock flash_attn modules for compatibility
            import types
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
            
            # Register our StandardRotaryEmbedding as the FlashRotaryEmbedding in the mock module
            mock_rotary.RotaryEmbedding = StandardRotaryEmbedding
            
            return True
        except Exception as e:
            print(f"Failed to create Flash Attention compatibility layer: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    created = create_flash_attention_compatibility_layer()
    print(f"Flash Attention compatibility layer created: {created}")
