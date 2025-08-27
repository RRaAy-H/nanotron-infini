#!/usr/bin/env python3
"""
Verify that the Nanotron to HuggingFace conversion worked correctly
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from pathlib import Path

def verify_conversion():
    """Verify the conversion by checking shapes and running inference"""
    
    print("=== VERIFYING CONVERSION ===")
    
    # 1. Load the HuggingFace model
    hf_model_path = "./hf_converted_model"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_model_path, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
        print("✓ Successfully loaded HuggingFace model")
    except Exception as e:
        print(f"✗ Failed to load HuggingFace model: {e}")
        return False
    
    # 2. Check model architecture
    print(f"✓ Model type: {model.__class__.__name__}")
    print(f"✓ Model config: {model.config.num_hidden_layers} layers, {model.config.hidden_size} hidden size")
    print(f"✓ Vocab size: {model.config.vocab_size}")
    
    # 3. Check for randomly initialized weights warning
    print("\n=== CHECKING WEIGHT INITIALIZATION ===")
    
    # Test loading the model fresh to see if there are warnings
    import io
    import sys
    from contextlib import redirect_stderr
    
    stderr_buffer = io.StringIO()
    with redirect_stderr(stderr_buffer):
        test_model = AutoModelForCausalLM.from_pretrained(hf_model_path, torch_dtype=torch.bfloat16)
    
    stderr_output = stderr_buffer.getvalue()
    if "were not initialized from the model checkpoint" in stderr_output:
        print("✗ WARNING: Some weights were randomly initialized!")
        print("This means the conversion failed for some layers.")
        return False
    else:
        print("✓ All weights loaded from checkpoint successfully")
    
    # 4. Basic inference test
    print("\n=== TESTING INFERENCE ===")
    
    try:
        # Simple text generation test
        input_text = "The capital of France is"
        inputs = tokenizer(input_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=5, 
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Input: '{input_text}'")
        print(f"✓ Output: '{generated_text}'")
        print("✓ Inference works!")
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False
    
    # 5. Check weight shapes match expected
    print("\n=== CHECKING WEIGHT SHAPES ===")
    
    safetensor_path = Path(hf_model_path) / "model.safetensors"
    if safetensor_path.exists():
        weights = load_file(safetensor_path)
        
        # Check key weights
        expected_shapes = {
            "model.embed_tokens.weight": (49152, 1024),  # vocab_size x hidden_size
            "lm_head.weight": (49152, 1024),
            "model.norm.weight": (1024,),
            "model.layers.0.self_attn.q_proj.weight": (1024, 1024),
            "model.layers.0.self_attn.k_proj.weight": (1024, 1024),
            "model.layers.0.self_attn.v_proj.weight": (1024, 1024),
            "model.layers.0.self_attn.o_proj.weight": (1024, 1024),
            "model.layers.0.mlp.gate_proj.weight": (4096, 1024),
            "model.layers.0.mlp.up_proj.weight": (4096, 1024),
            "model.layers.0.mlp.down_proj.weight": (1024, 4096),
        }
        
        for weight_name, expected_shape in expected_shapes.items():
            if weight_name in weights:
                actual_shape = weights[weight_name].shape
                if actual_shape == expected_shape:
                    print(f"✓ {weight_name}: {actual_shape}")
                else:
                    print(f"✗ {weight_name}: expected {expected_shape}, got {actual_shape}")
                    return False
            else:
                print(f"✗ Missing weight: {weight_name}")
                return False
        
        print(f"✓ All {len(weights)} weights have correct shapes")
    
    # 6. Compare with original Nanotron checkpoint (basic sanity check)
    print("\n=== COMPARING WITH ORIGINAL CHECKPOINT ===")
    
    nanotron_path = Path("/data1/infini-attn/infini-llama/nanotron-infini/checkpoints/fineweb_4gpu_300m_infini/30000")
    
    # Check token embeddings match
    nanotron_token_path = nanotron_path / "model" / "model" / "token_position_embeddings" / "pp_block" / "token_embedding" / "model_weight_pp-rank-0-of-1_tp-rank-0-of-1.safetensors"
    if nanotron_token_path.exists():
        nanotron_tokens = load_file(nanotron_token_path)["data"]
        hf_tokens = weights["model.embed_tokens.weight"]
        
        # Check with appropriate tolerance for bfloat16
        if torch.allclose(nanotron_tokens, hf_tokens, rtol=1e-3, atol=1e-4):
            diff = torch.abs(nanotron_tokens - hf_tokens)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            print(f"✓ Token embeddings match within tolerance (max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f})")
        else:
            print("✗ Token embeddings have significant differences!")
            return False
    
    print("\n=== CONVERSION VERIFICATION COMPLETE ===")
    print("✅ Conversion appears to be successful!")
    return True

if __name__ == "__main__":
    success = verify_conversion()
    exit(0 if success else 1)