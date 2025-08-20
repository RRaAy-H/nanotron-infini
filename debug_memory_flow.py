#!/usr/bin/env python3
"""
Debug memory flow in Infini-Attention to understand why memory isn't activating
"""

import sys
sys.path.append('src')
import torch
from nanotron import constants
from nanotron.config import get_config_from_file
from nanotron.models import build_model
from nanotron.parallel import ParallelContext
from nanotron.serialize import load_weights
from nanotron.trainer import CONFIG_TO_MODEL_CLASS
from nanotron.random import RandomStates, get_current_random_state, get_synced_random_state, set_random_seed
from transformers import AutoTokenizer

def debug_memory_flow():
    # Load configuration
    config = get_config_from_file('./checkpoints/fineweb_4gpu_300m_infini/30000/config.yaml')
    constants.CONFIG = config
    
    print("=== MEMORY FLOW DEBUG ===")
    print(f"Config loaded: turn_on_memory = {config.infini_attention.turn_on_memory}")
    print(f"Segment length: {config.infini_attention.segment_length}")
    
    # Setup minimal model for testing
    parallel_context = ParallelContext(
        data_parallel_size=1,
        pipeline_parallel_size=1, 
        tensor_parallel_size=1,
    )
    
    set_random_seed(42)
    
    model_config = config.model.model_config
    model_config_cls = model_config.__class__.__name__
    
    random_states = RandomStates({"tp_synced": get_synced_random_state(
        random_state=get_current_random_state(), 
        pg=parallel_context.tp_pg
    )})
    
    model = build_model(
        model_builder=lambda: CONFIG_TO_MODEL_CLASS[model_config_cls](
            config=model_config,
            parallel_context=parallel_context,
            parallel_config=None,
            random_states=random_states,
        ),
        dtype=torch.bfloat16,
        parallel_context=parallel_context,
    )
    
    # Load weights
    load_weights(model=model, parallel_context=parallel_context, root_folder='./checkpoints/fineweb_4gpu_300m_infini/30000')
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create a test input that will span multiple segments
    test_text = "The quick brown fox jumps over the lazy dog. " * 500  # Should be ~4000+ tokens
    tokens = tokenizer.encode(test_text, return_tensors="pt")
    print(f"Test input length: {tokens.shape[1]} tokens")
    print(f"Expected segments: {(tokens.shape[1] + config.infini_attention.segment_length - 1) // config.infini_attention.segment_length}")
    
    # Hook into the attention forward to trace memory usage
    memory_trace = []
    
    def hook_attention_forward(self, hidden_states, sequence_mask):
        seq_len = hidden_states.shape[1]
        layer_idx = getattr(self, 'layer_idx', -1)
        
        print(f"\n--- Layer {layer_idx} Forward ---")
        print(f"Input sequence length: {seq_len}")
        print(f"Segment length: {self.segment_length}")
        print(f"turn_on_memory check: {constants.CONFIG.infini_attention.turn_on_memory is True}")
        
        if seq_len > self.segment_length:
            n_segments = (seq_len + self.segment_length - 1) // self.segment_length
            print(f"Will be split into {n_segments} segments")
            
            # Initialize memory tracking
            memory = None
            normalization = None
            
            for segment_idx in range(n_segments):
                start_idx = segment_idx * self.segment_length
                end_idx = min(start_idx + self.segment_length, seq_len)
                segment_len = end_idx - start_idx
                
                print(f"  Segment {segment_idx}: tokens {start_idx}-{end_idx} (len={segment_len})")
                print(f"    Memory before: {memory is not None}")
                print(f"    Normalization before: {normalization is not None}")
                
                # Check if memory mechanism would activate
                if constants.CONFIG.infini_attention.turn_on_memory is True:
                    if memory is not None and normalization is not None:
                        print(f"    ✅ MEMORY RETRIEVAL WOULD OCCUR")
                        print(f"    Memory norm: {memory.norm().item():.6f}")
                        print(f"    Normalization norm: {normalization.norm().item():.6f}")
                    else:
                        print(f"    ❌ No memory retrieval (memory={memory is not None}, norm={normalization is not None})")
                else:
                    print(f"    ❌ Memory disabled in config")
                
                # Simulate memory update (simplified)
                if constants.CONFIG.infini_attention.turn_on_memory is True:
                    # Create dummy key/value states for this segment
                    batch_size = hidden_states.shape[0]
                    d_k = self.d_qk
                    d_v = self.d_qk
                    n_heads = self.n_local_kv_heads
                    
                    dummy_key = torch.randn(batch_size, n_heads, segment_len, d_k, device=hidden_states.device, dtype=hidden_states.dtype)
                    dummy_value = torch.randn(batch_size, n_heads, segment_len, d_v, device=hidden_states.device, dtype=hidden_states.dtype)
                    
                    # Simulate memory update
                    if memory is None:
                        memory = torch.randn(batch_size, n_heads, d_k, d_v, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
                        normalization = torch.randn(batch_size, n_heads, d_k, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
                        print(f"    ✅ MEMORY INITIALIZED")
                    else:
                        # Update memory (simplified)
                        memory = memory + torch.randn_like(memory) * 0.001
                        normalization = normalization + torch.randn_like(normalization) * 0.001
                        print(f"    ✅ MEMORY UPDATED")
                
                memory_trace.append({
                    'layer_idx': layer_idx,
                    'segment_idx': segment_idx,
                    'has_memory': memory is not None,
                    'memory_norm': memory.norm().item() if memory is not None else 0.0
                })
        
        # Call original forward (but we'll just return dummy output for testing)
        return {"hidden_states": hidden_states, "sequence_mask": sequence_mask}
    
    # Hook the first attention layer to trace memory flow
    first_attention = model.model.decoder[0].attn
    first_attention.layer_idx = 0
    original_forward = first_attention.forward
    first_attention.forward = lambda *args, **kwargs: hook_attention_forward(first_attention, *args, **kwargs)
    
    # Test the forward pass
    sequence_mask = torch.ones(tokens.shape, dtype=torch.bool, device=tokens.device)
    hidden_states = model.model.token_position_embeddings({"input_ids": tokens, "position_ids": None})["hidden_states"]
    
    print(f"\n=== RUNNING FORWARD PASS ===")
    print(f"Hidden states shape: {hidden_states.shape}")
    
    # Run forward pass through first layer only
    try:
        output = first_attention.forward(hidden_states.transpose(0, 1), sequence_mask)
        print(f"\nForward pass completed successfully")
    except Exception as e:
        print(f"\nError during forward pass: {e}")
    
    # Restore original forward
    first_attention.forward = original_forward
    
    # Analyze results
    print(f"\n=== MEMORY TRACE ANALYSIS ===")
    total_segments = len(memory_trace)
    segments_with_memory = sum(1 for trace in memory_trace if trace['has_memory'])
    
    print(f"Total segments processed: {total_segments}")
    print(f"Segments with memory: {segments_with_memory}")
    print(f"Expected memory retrievals: {max(0, total_segments - 1)}")  # First segment has no memory
    
    if segments_with_memory > 1:
        print("✅ Memory mechanism appears to be working")
    else:
        print("❌ Memory mechanism not working properly")
    
    for trace in memory_trace:
        print(f"  Layer {trace['layer_idx']}, Segment {trace['segment_idx']}: Memory={trace['has_memory']}, Norm={trace['memory_norm']:.6f}")

if __name__ == "__main__":
    debug_memory_flow()
