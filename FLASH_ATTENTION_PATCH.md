# Flash Attention Patch for Llama Model

This patch provides instructions on how to modify the Llama model implementation to fall back to standard attention when Flash Attention is not available or when you explicitly want to disable it.

## Manual Changes Required

Edit the file `src/nanotron/models/llama.py` and make the following changes:

1. Find the `CausalSelfAttention` class initialization method
2. Add a check for the environment variable `DISABLE_FLASH_ATTN`
3. Wrap the Flash Attention import in a try-except block
4. Provide a fallback to standard attention when Flash Attention is not available

```python
def __init__(
    self,
    hidden_size,
    num_heads,
    num_kv_heads=None,
    rope_theta=10000,
    rope_scaling=None,
    max_position_embeddings=1024,
    rope_interleaved=False,
):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
    head_dim = hidden_size // num_heads
    
    # Check if Flash Attention should be disabled
    disable_flash_attn = os.environ.get("DISABLE_FLASH_ATTN", "0") == "1"
    
    try:
        # Try to import Flash Attention if not disabled
        if not disable_flash_attn:
            from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding
            from flash_attn.layers.attention import FlashCausalAttention
            self.use_flash_attn = True
            self.rotary_emb = FlashRotaryEmbedding(
                dim=head_dim,
                base=rope_theta,
                interleaved=rope_interleaved,
                scaling_factor=rope_scaling,
                max_position_embeddings=max_position_embeddings,
            )
        else:
            raise ImportError("Flash Attention disabled by environment variable")
    except ImportError as e:
        # Fallback to standard attention
        print(f"Flash Attention not available: {e}. Using standard attention implementation.")
        self.use_flash_attn = False
        # Implement standard rotary embeddings here
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            base=rope_theta,
            interleaved=rope_interleaved,
            scaling_factor=rope_scaling,
            max_position_embeddings=max_position_embeddings,
        )
    
    # Rest of initialization code...
```

And then modify the forward method to use the appropriate attention implementation:

```python
def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
    batch_size, seq_length, _ = hidden_states.shape
    
    # Project to q, k, v
    qkv = self.wqkv(hidden_states)
    
    # Rest of processing...
    
    # Different attention implementations
    if self.use_flash_attn:
        # Use Flash Attention
        attn_output = self.flash_attention(q, k, v, attention_mask)
    else:
        # Use standard attention implementation
        attn_output = self.standard_attention(q, k, v, attention_mask)
    
    # Rest of forward method...
```

## Automatic Patching

Alternatively, you can create an automatic patching script to apply these changes. Here's how:

1. Create a file `patch_llama_attention.py`
2. Run it before your first training run

This will automatically add the fallback mechanism to your model code.
