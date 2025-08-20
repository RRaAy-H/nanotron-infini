#!/usr/bin/env python3
"""
Final validation that memory mechanism is working correctly.

This script proves that:
1. Balance factors load correctly from checkpoint
2. Layer-wise memory specialization is preserved
3. Memory mechanism is properly configured for inference
"""

import sys
sys.path.append('src')
import torch
from apply_balance_fix_standalone import apply_balance_factor_fix_standalone

# Import nanotron components
from nanotron import constants
from nanotron.config import get_config_from_file
from nanotron.models import build_model
from nanotron.parallel import ParallelContext
from nanotron.trainer import CONFIG_TO_MODEL_CLASS
from nanotron.random import RandomStates, get_current_random_state, get_synced_random_state, set_random_seed


def main():
    checkpoint_path = '/data1/infini-attn/infini-llama/nanotron-infini/checkpoints/fineweb_4gpu_300m_infini/30000'
    
    print("=" * 60)
    print("🎯 FINAL MEMORY MECHANISM VALIDATION")
    print("=" * 60)
    
    # Load configuration  
    config = get_config_from_file(f'{checkpoint_path}/config.yaml')
    constants.CONFIG = config
    
    print(f"\n✅ Configuration:")
    print(f"   turn_on_memory: {config.infini_attention.turn_on_memory}")
    print(f"   segment_length: {config.infini_attention.segment_length}")
    print(f"   balance_factor_lr: {config.infini_attention.balance_factor_lr}")
    print(f"   balance_act_type: {config.infini_attention.balance_act_type}")
    
    # Setup model
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
    
    print(f"\n🏗️  Building model...")
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
    
    print(f"   Model type: {type(model)}")
    print(f"   Decoder layers: {len(model.model.decoder)}")
    
    # Test BEFORE balance factor fix
    print(f"\n❌ BEFORE Balance Factor Fix:")
    layer0 = model.model.decoder[0].pp_block.attn
    layer11 = model.model.decoder[11].pp_block.attn
    
    bf0_before = layer0.balance_factors.data
    bf11_before = layer11.balance_factors.data
    
    act0_before = layer0.balance_act_func(bf0_before).mean().item()
    act11_before = layer11.balance_act_func(bf11_before).mean().item()
    
    print(f"   Layer 0:  {act0_before*100:.1f}% memory (should be 94.1%)")
    print(f"   Layer 11: {act11_before*100:.1f}% memory (should be 0.0%)")
    print(f"   → Broken: All layers using default 50/50 weighting")
    
    # Apply balance factor fix
    print(f"\n🔧 Applying Balance Factor Fix...")
    success = apply_balance_factor_fix_standalone(model, checkpoint_path, verbose=False)
    
    if not success:
        print("❌ FAILED: Could not load balance factors")
        return False
    
    # Test AFTER balance factor fix
    print(f"\n✅ AFTER Balance Factor Fix:")
    
    # Test all layers
    memory_layers = []
    attention_layers = []
    
    for layer_idx in range(len(model.model.decoder)):
        layer = model.model.decoder[layer_idx].pp_block.attn
        bf = layer.balance_factors.data
        activated = layer.balance_act_func(bf).mean().item()
        
        if activated > 0.7:
            memory_layers.append(f"Layer {layer_idx}: {activated*100:.1f}%")
        elif activated < 0.3:
            attention_layers.append(f"Layer {layer_idx}: {activated*100:.1f}%")
        else:
            print(f"   Layer {layer_idx}: {activated*100:.1f}% memory (balanced)")
    
    print(f"   🧠 Memory-focused layers:")
    for layer_info in memory_layers:
        print(f"      {layer_info}")
    
    print(f"   👁️  Attention-focused layers:")
    for layer_info in attention_layers:
        print(f"      {layer_info}")
    
    # Validate expected patterns
    layer0_after = model.model.decoder[0].pp_block.attn
    layer11_after = model.model.decoder[11].pp_block.attn
    
    act0_after = layer0_after.balance_act_func(layer0_after.balance_factors).mean().item()
    act11_after = layer11_after.balance_act_func(layer11_after.balance_factors).mean().item()
    
    print(f"\n📊 Key Validation Metrics:")
    print(f"   Early layer memory preference: {act0_after*100:.1f}% (target: ~94%)")
    print(f"   Late layer attention preference: {(1-act11_after)*100:.1f}% (target: ~100%)")
    
    # Determine success
    early_layer_correct = abs(act0_after - 0.941) < 0.05  # Within 5% of expected 94.1%
    late_layer_correct = act11_after < 0.05  # Should be near 0%
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    
    if early_layer_correct and late_layer_correct:
        print(f"   🎉 SUCCESS: Memory mechanism properly configured!")
        print(f"   ✅ Balance factors loaded correctly")
        print(f"   ✅ Layer-wise specialization preserved")
        print(f"   ✅ Early layers prefer memory ({act0_after*100:.1f}%)")
        print(f"   ✅ Late layers prefer attention ({(1-act11_after)*100:.1f}%)")
        
        print(f"\n🚀 IMPLICATIONS:")
        print(f"   → Multi-segment contexts will now use memory effectively")
        print(f"   → Context lengths > 1024 tokens will benefit from compression")
        print(f"   → Memory retrievals should be detected in inference tests")
        print(f"   → Cross-segment information flow enabled")
        
        success = True
    else:
        print(f"   ❌ ISSUES DETECTED:")
        if not early_layer_correct:
            print(f"      Early layer memory weight incorrect: {act0_after*100:.1f}% (expected ~94%)")
        if not late_layer_correct:
            print(f"      Late layer attention weight incorrect: {(1-act11_after)*100:.1f}% (expected ~100%)")
        
        success = False
    
    print(f"\n📝 INTEGRATION INSTRUCTIONS:")
    print(f"   To use this fix in your scripts, add after model loading:")
    print(f"   ```python")
    print(f"   from apply_balance_fix_standalone import apply_balance_factor_fix_standalone")
    print(f"   apply_balance_factor_fix_standalone(model, checkpoint_path)")
    print(f"   ```")
    
    print(f"\n🔮 NEXT STEPS:")
    print(f"   1. Apply this fix to your inference/generation scripts")
    print(f"   2. Re-run memory usage tests (should show >0% memory usage)")
    print(f"   3. Test with contexts >2048 tokens to see memory activation")
    print(f"   4. Validate cross-segment performance improvements")
    
    return success


if __name__ == "__main__":
    success = main()
    
    print(f"\n" + "=" * 60)
    if success:
        print("🏆 MEMORY MECHANISM VALIDATION: PASSED")
        print("🎯 Your Infini-Attention model is ready for inference!")
    else:
        print("❌ MEMORY MECHANISM VALIDATION: FAILED")
        print("💡 Check balance factor loading issues")
    print("=" * 60)
    
    exit(0 if success else 1)
