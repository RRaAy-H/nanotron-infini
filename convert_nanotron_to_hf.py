#!/usr/bin/env python3
"""
Convert Nanotron checkpoint to HuggingFace Transformers format for evaluation.
"""

import argparse
import json
import torch
from pathlib import Path
from safetensors.torch import save_file, load_file
from transformers import LlamaConfig, AutoTokenizer
import yaml

def load_nanotron_checkpoint(checkpoint_path: Path):
    """Load Nanotron checkpoint structure with correct paths"""
    model_path = checkpoint_path / "model" / "model"
    
    weights = {}
    
    # Load token embeddings
    token_emb_path = model_path / "token_position_embeddings" / "pp_block" / "token_embedding" / "model_weight_pp-rank-0-of-1_tp-rank-0-of-1.safetensors"
    if token_emb_path.exists():
        token_weights = load_file(token_emb_path)
        weights["token_embedding.weight"] = token_weights["data"]
        print(f"Loaded token embeddings: {token_weights['data'].shape}")
    
    # Load decoder layers (0-11)
    for layer_idx in range(12):  # Assuming 12 layers based on config
        layer_path = model_path / "decoder" / str(layer_idx) / "pp_block"
        
        # Attention weights
        qkv_path = layer_path / "attn" / "qkv_proj" / "model_weight_pp-rank-0-of-1_tp-rank-0-of-1.safetensors"
        if qkv_path.exists():
            qkv_weights = load_file(qkv_path)
            weights[f"decoder.{layer_idx}.attn.qkv_proj.weight"] = qkv_weights["data"]
        
        o_proj_path = layer_path / "attn" / "o_proj" / "model_weight_pp-rank-0-of-1_tp-rank-0-of-1.safetensors"
        if o_proj_path.exists():
            o_proj_weights = load_file(o_proj_path)
            weights[f"decoder.{layer_idx}.attn.o_proj.weight"] = o_proj_weights["data"]
        
        # Balance factors (Infini-specific)
        balance_path = layer_path / "attn" / "model_balance_factors_pp-rank-0-of-1_tp-rank-0-of-1.safetensors"
        if balance_path.exists():
            balance_weights = load_file(balance_path)
            weights[f"decoder.{layer_idx}.attn.balance_factors.weight"] = balance_weights["data"]
        
        # MLP weights
        gate_up_path = layer_path / "mlp" / "gate_up_proj" / "model_weight_pp-rank-0-of-1_tp-rank-0-of-1.safetensors"
        if gate_up_path.exists():
            gate_up_weights = load_file(gate_up_path)
            weights[f"decoder.{layer_idx}.mlp.gate_up_proj.weight"] = gate_up_weights["data"]
        
        down_proj_path = layer_path / "mlp" / "down_proj" / "model_weight_pp-rank-0-of-1_tp-rank-0-of-1.safetensors"
        if down_proj_path.exists():
            down_proj_weights = load_file(down_proj_path)
            weights[f"decoder.{layer_idx}.mlp.down_proj.weight"] = down_proj_weights["data"]
        
        # Layer norms
        input_ln_path = layer_path / "input_layernorm" / "model_weight.safetensors"
        if input_ln_path.exists():
            input_ln_weights = load_file(input_ln_path)
            weights[f"decoder.{layer_idx}.input_layernorm.weight"] = input_ln_weights["data"]
        
        post_attn_ln_path = layer_path / "post_attention_layernorm" / "model_weight.safetensors"
        if post_attn_ln_path.exists():
            post_attn_ln_weights = load_file(post_attn_ln_path)
            weights[f"decoder.{layer_idx}.post_attention_layernorm.weight"] = post_attn_ln_weights["data"]
    
    # Load final layer norm
    final_ln_path = model_path / "final_layer_norm" / "pp_block" / "model_weight.safetensors"
    if final_ln_path.exists():
        final_ln_weights = load_file(final_ln_path)
        weights["final_layer_norm.weight"] = final_ln_weights["data"]
        print(f"Loaded final layer norm: {final_ln_weights['data'].shape}")
    
    # Load language model head
    lm_head_path = model_path / "lm_head" / "pp_block" / "model_weight_pp-rank-0-of-1_tp-rank-0-of-1.safetensors"
    if lm_head_path.exists():
        lm_head_weights = load_file(lm_head_path)
        weights["lm_head.weight"] = lm_head_weights["data"]
        print(f"Loaded lm_head: {lm_head_weights['data'].shape}")
    
    print(f"Total weights loaded: {len(weights)}")
    return weights

def convert_nanotron_to_hf_weights(nanotron_weights, config):
    """Convert Nanotron weight names to HuggingFace format"""
    hf_weights = {}
    
    print("Converting weights...")
    
    # Convert embeddings
    if "token_embedding.weight" in nanotron_weights:
        hf_weights["model.embed_tokens.weight"] = nanotron_weights["token_embedding.weight"]
        print("Mapped token_embedding.weight -> model.embed_tokens.weight")
    
    # Convert decoder layers
    for i in range(config.num_hidden_layers):
        # Self attention weights - handle QKV projection
        qkv_key = f"decoder.{i}.attn.qkv_proj.weight"
        if qkv_key in nanotron_weights:
            qkv_weight = nanotron_weights[qkv_key]
            hidden_size = config.hidden_size
            num_heads = config.num_attention_heads
            num_kv_heads = config.num_key_value_heads
            head_dim = hidden_size // num_heads
            
            # Split QKV weights
            q_weight = qkv_weight[:hidden_size]
            k_weight = qkv_weight[hidden_size:hidden_size + num_kv_heads * head_dim]
            v_weight = qkv_weight[hidden_size + num_kv_heads * head_dim:]
            
            hf_weights[f"model.layers.{i}.self_attn.q_proj.weight"] = q_weight
            hf_weights[f"model.layers.{i}.self_attn.k_proj.weight"] = k_weight
            hf_weights[f"model.layers.{i}.self_attn.v_proj.weight"] = v_weight
            print(f"Split QKV for layer {i}")
        
        # Output projection
        o_proj_key = f"decoder.{i}.attn.o_proj.weight"
        if o_proj_key in nanotron_weights:
            hf_weights[f"model.layers.{i}.self_attn.o_proj.weight"] = nanotron_weights[o_proj_key]
        
        # MLP weights - handle gate_up projection
        gate_up_key = f"decoder.{i}.mlp.gate_up_proj.weight"
        if gate_up_key in nanotron_weights:
            gate_up_weight = nanotron_weights[gate_up_key]
            intermediate_size = config.intermediate_size
            
            # Split gate and up projections
            gate_weight = gate_up_weight[:intermediate_size]
            up_weight = gate_up_weight[intermediate_size:]
            
            hf_weights[f"model.layers.{i}.mlp.gate_proj.weight"] = gate_weight
            hf_weights[f"model.layers.{i}.mlp.up_proj.weight"] = up_weight
            print(f"Split gate_up for layer {i}")
        
        down_proj_key = f"decoder.{i}.mlp.down_proj.weight"
        if down_proj_key in nanotron_weights:
            hf_weights[f"model.layers.{i}.mlp.down_proj.weight"] = nanotron_weights[down_proj_key]
        
        # Layer norms
        input_ln_key = f"decoder.{i}.input_layernorm.weight"
        if input_ln_key in nanotron_weights:
            hf_weights[f"model.layers.{i}.input_layernorm.weight"] = nanotron_weights[input_ln_key]
        
        post_attn_ln_key = f"decoder.{i}.post_attention_layernorm.weight"
        if post_attn_ln_key in nanotron_weights:
            hf_weights[f"model.layers.{i}.post_attention_layernorm.weight"] = nanotron_weights[post_attn_ln_key]
    
    # Final layer norm
    if "final_layer_norm.weight" in nanotron_weights:
        hf_weights["model.norm.weight"] = nanotron_weights["final_layer_norm.weight"]
        print("Mapped final_layer_norm.weight -> model.norm.weight")
    
    # Language model head
    if "lm_head.weight" in nanotron_weights:
        hf_weights["lm_head.weight"] = nanotron_weights["lm_head.weight"]
        print("Mapped lm_head.weight -> lm_head.weight")
    
    print(f"HF weights created: {len(hf_weights)}")
    return hf_weights

def create_hf_config(nanotron_config_path: Path):
    """Create HuggingFace LlamaConfig from Nanotron config"""
    with open(nanotron_config_path, 'r') as f:
        nanotron_config = yaml.safe_load(f)
    
    model_config = nanotron_config["model"]["model_config"]
    
    hf_config = LlamaConfig(
        vocab_size=model_config["vocab_size"],
        hidden_size=model_config["hidden_size"],
        intermediate_size=model_config["intermediate_size"],
        num_hidden_layers=model_config["num_hidden_layers"],
        num_attention_heads=model_config["num_attention_heads"],
        num_key_value_heads=model_config.get("num_key_value_heads", model_config["num_attention_heads"]),
        max_position_embeddings=model_config["max_position_embeddings"],
        rms_norm_eps=model_config["rms_norm_eps"],
        rope_scaling=model_config.get("rope_scaling"),
        hidden_act=model_config["hidden_act"],
        bos_token_id=model_config["bos_token_id"],
        eos_token_id=model_config["eos_token_id"],
        pad_token_id=model_config.get("pad_token_id"),
        tie_word_embeddings=model_config["tie_word_embeddings"],
        torch_dtype="bfloat16",
    )
    
    return hf_config

def main():
    parser = argparse.ArgumentParser(description="Convert Nanotron checkpoint to HuggingFace format")
    parser.add_argument("--nanotron_path", type=str, required=True, help="Path to Nanotron checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for HuggingFace model")
    parser.add_argument("--tokenizer", type=str, default="lvwerra/the-tokenizer-v1", help="Tokenizer to use")
    
    args = parser.parse_args()
    
    nanotron_path = Path(args.nanotron_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting Nanotron checkpoint from {nanotron_path} to HuggingFace format at {output_path}")
    
    # Load Nanotron config
    config_path = nanotron_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Create HuggingFace config
    hf_config = create_hf_config(config_path)
    
    # Save config
    hf_config.save_pretrained(output_path)
    print("Saved HuggingFace config")
    
    # Load and convert weights
    print("Loading Nanotron weights...")
    nanotron_weights = load_nanotron_checkpoint(nanotron_path)
    
    print("Converting weights to HuggingFace format...")
    hf_weights = convert_nanotron_to_hf_weights(nanotron_weights, hf_config)
    
    # Save weights as safetensors
    print("Saving converted weights...")
    save_file(hf_weights, output_path / "model.safetensors")
    
    # Copy tokenizer
    print("Setting up tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        tokenizer.save_pretrained(output_path)
        print(f"Successfully saved tokenizer: {tokenizer.__class__.__name__}")
    except Exception as e:
        print(f"Warning: Could not load tokenizer {args.tokenizer}: {e}")
        print("You may need to manually copy the tokenizer files")
    
    print(f"Conversion complete! HuggingFace model saved to {output_path}")
    print(f"You can now use: lm_eval --model hf --model_args pretrained={output_path} --tasks arc_easy")

if __name__ == "__main__":
    main()