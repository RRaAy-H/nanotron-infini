#!/usr/bin/env python3
"""
Simple memory debugging - add this to your existing test scripts
"""

# Add this patch to your scripts/debug_memory_usage.py or test_memory_comprehensive.py
# Right after loading the model (around line 320 in debug_memory_usage.py)

def patch_attention_for_debugging(model):
    """Patch attention layers to add memory flow debugging."""
    
    for layer_idx, layer in enumerate(model.model.decoder):
        if hasattr(layer, 'attn'):
            original_forward = layer.attn.forward
            
            def create_debug_forward(layer_idx, orig_forward):
                def debug_forward(self, hidden_states, sequence_mask):
                    seq_len = hidden_states.shape[1]
                    segment_length = getattr(self, 'segment_length', 1024)
                    
                    print(f"\n=== Layer {layer_idx} Debug ===")
                    print(f"Sequence length: {seq_len}")
                    print(f"Segment length: {segment_length}")
                    
                    if seq_len > segment_length:
                        n_segments = (seq_len + segment_length - 1) // segment_length
                        print(f"Expected segments: {n_segments}")
                        print(f"Expected memory retrievals: {max(0, n_segments - 1)}")
                        
                        # This is the critical check
                        memory = None
                        normalization = None
                        
                        for seg_idx in range(n_segments):
                            print(f"  Segment {seg_idx}:")
                            print(f"    Memory before: {memory is not None}")
                            
                            # Simulate what happens in the real forward
                            if constants.CONFIG.infini_attention.turn_on_memory is True:
                                if memory is not None and normalization is not None:
                                    print(f"    ✅ WOULD RETRIEVE MEMORY")
                                else:
                                    print(f"    ❌ NO MEMORY RETRIEVAL (first segment or bug)")
                                
                                # Simulate memory update
                                if memory is None:
                                    memory = "initialized"  # Placeholder
                                    normalization = "initialized"
                                    print(f"    📝 MEMORY INITIALIZED")
                                else:
                                    print(f"    📝 MEMORY UPDATED")
                        
                        print(f"  Final result: Memory={memory is not None}")
                    
                    # Call original forward
                    return orig_forward(hidden_states, sequence_mask)
                
                return debug_forward
            
            layer.attn.forward = create_debug_forward(layer_idx, original_forward).__get__(layer.attn, type(layer.attn))

# Usage instructions:
print("""
ADD THIS TO YOUR EXISTING TEST SCRIPT:

1. After loading the model (line ~320 in debug_memory_usage.py), add:
   
   # Add memory debugging
   patch_attention_for_debugging(model)

2. Then run your normal test:
   
   python3 scripts/debug_memory_usage.py --checkpoint ./checkpoints/fineweb_4gpu_300m_infini/30000 --context-lengths 2048 --num-samples 1

This will show you EXACTLY what's happening with memory between segments.
""")
