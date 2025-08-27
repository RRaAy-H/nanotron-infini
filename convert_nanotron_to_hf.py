#!/usr/bin/env python3
"""
Convert Nanotron checkpoint to HuggingFace Transformers format for evaluation.
Usage: python convert_nanotron_to_hf.py --nanotron_path=/path/to/checkpoint --output_path=/path/to/hf_model
"""

import argparse
import json
import torch
from pathlib import Path
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaTokenizer
import yaml

def load_nanotron_checkpoint(checkpoint_path: Path):
    """Load Nanotron checkpoint structure"""
    model_path = checkpoint_path / "model"
    
    # Load weights from safetensors files
    weights = {}
    
    # Load token embeddings
    token_emb_path = model_path / "token_position_embeddings" / "pp-rank-00_tp-rank-00.safetensors"
    if token_emb_path.exists():
        from safetensors.torch import load_file
        token_weights = load_file(token_emb_path)
        weights.update(token_weights)
    
    # Load decoder layers
    decoder_path = model_path / "decoder"
    for layer_file in sorted(decoder_path.glob("*.safetensors")):
        layer_weights = load_file(layer_file)
        weights.update(layer_weights)
    
    # Load final layer norm
    norm_path = model_path / "final_layer_norm" / "pp-rank-00_tp-rank-00.safetensors"
    if norm_path.exists():
        norm_weights = load_file(norm_path)
        weights.update(norm_weights)
        
    # Load language model head
    lm_head_path = model_path / "lm_head" / "pp-rank-00_tp-rank-00.safetensors"
    if lm_head_path.exists():
        lm_head_weights = load_file(lm_head_path)
        weights.update(lm_head_weights)
    
    return weights

def convert_nanotron_to_hf_weights(nanotron_weights, config):
    """Convert Nanotron weight names to HuggingFace format"""
    hf_weights = {}
    
    # Convert embeddings
    if "token_embedding.weight" in nanotron_weights:
        hf_weights["model.embed_tokens.weight"] = nanotron_weights["token_embedding.weight"]
    
    # Convert decoder layers
    for i in range(config.num_hidden_layers):
        # Self attention weights
        if f"{i}.attn.qkv_proj.weight" in nanotron_weights:
            qkv_weight = nanotron_weights[f"{i}.attn.qkv_proj.weight"]
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
        
        # Output projection
        if f"{i}.attn.o_proj.weight" in nanotron_weights:
            hf_weights[f"model.layers.{i}.self_attn.o_proj.weight"] = nanotron_weights[f"{i}.attn.o_proj.weight"]
        
        # MLP weights
        if f"{i}.mlp.gate_up_proj.weight" in nanotron_weights:
            gate_up_weight = nanotron_weights[f"{i}.mlp.gate_up_proj.weight"]
            intermediate_size = config.intermediate_size
            
            # Split gate and up projections
            gate_weight = gate_up_weight[:intermediate_size]
            up_weight = gate_up_weight[intermediate_size:]
            
            hf_weights[f"model.layers.{i}.mlp.gate_proj.weight"] = gate_weight
            hf_weights[f"model.layers.{i}.mlp.up_proj.weight"] = up_weight
        
        if f"{i}.mlp.down_proj.weight" in nanotron_weights:
            hf_weights[f"model.layers.{i}.mlp.down_proj.weight"] = nanotron_weights[f"{i}.mlp.down_proj.weight"]
        
        # Layer norms
        if f"{i}.input_layernorm.weight" in nanotron_weights:
            hf_weights[f"model.layers.{i}.input_layernorm.weight"] = nanotron_weights[f"{i}.input_layernorm.weight"]
        
        if f"{i}.post_attention_layernorm.weight" in nanotron_weights:
            hf_weights[f"model.layers.{i}.post_attention_layernorm.weight"] = nanotron_weights[f"{i}.post_attention_layernorm.weight"]
    
    # Final layer norm
    if "ln_f.weight" in nanotron_weights:
        hf_weights["model.norm.weight"] = nanotron_weights["ln_f.weight"]
    
    # Language model head
    if "lm_head.weight" in nanotron_weights:
        hf_weights["lm_head.weight"] = nanotron_weights["lm_head.weight"]
    
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
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer)
        tokenizer.save_pretrained(output_path)
    except Exception as e:
        print(f"Warning: Could not load tokenizer {args.tokenizer}: {e}")
        print("You may need to manually copy the tokenizer files")
    
    print(f"Conversion complete! HuggingFace model saved to {output_path}")

if __name__ == "__main__":
    main()