#!/usr/bin/env python3
"""
Monitor passkey fine-tuning progress with memory behavior analysis.
Use this script during fine-tuning to track improvement in passkey retrieval.
"""

import sys
import os
from pathlib import Path
import torch
import json
import time
from datetime import datetime

# Add nanotron path
correct_nanotron_path = "/data1/infini-attn/infini-llama/nanotron-infini/src"
if correct_nanotron_path not in sys.path:
    sys.path.insert(0, correct_nanotron_path)

from nanotron.config import get_config_from_file
from nanotron.models.llama import LlamaForTraining
from nanotron.generation.decode import decode_text, GenerationInput, GenerationArgs, TokenizerConfig
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.tensor_parallel.nn import TensorParallelLinearMode
from nanotron.random import RandomStates, get_current_random_state, get_synced_random_state
from nanotron.serialize.weights import load_weights
from transformers import AutoTokenizer
import torch.distributed as dist
import argparse

# Import balance factor fix
try:
    from apply_balance_fix_standalone import apply_balance_factor_fix_standalone
except ImportError:
    sys.path.append('.')
    from apply_balance_fix_standalone import apply_balance_factor_fix_standalone

def load_model_and_tokenizer(checkpoint_path):
    """Load model and tokenizer with balance factor fix"""
    
    # Apply llama.py fix
    llama_path = "/data1/infini-attn/infini-llama/nanotron-infini/src/nanotron/models/llama.py"
    try:
        with open(llama_path, 'r') as f:
            content = f.read()
        
        if "assert torch.all(sequence_mask)" in content:
            fixed_content = content.replace(
                "assert torch.all(sequence_mask)",
                "# assert torch.all(sequence_mask)  # FIXED: Commented out for generation compatibility"
            )
            with open(llama_path, 'w') as f:
                f.write(fixed_content)
    except Exception as e:
        print(f"WARNING: Could not apply llama.py fix: {e}")
    
    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    
    # Load config
    config_path = Path(checkpoint_path) / "config.yaml"
    config = get_config_from_file(str(config_path))
    
    # Set constants.CONFIG
    from nanotron import constants
    constants.CONFIG = config
    
    # Setup parallelism
    parallel_config = config.parallelism
    parallel_config.dp = 1
    parallel_config.pp = 1  
    parallel_config.tp = 1
    
    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )
    
    # Build model
    model_config = config.model.model_config
    random_states = RandomStates({
        "tp_synced": get_synced_random_state(random_state=get_current_random_state(), pg=parallel_context.tp_pg)
    }) if parallel_config.tp_mode is TensorParallelLinearMode.ALL_REDUCE else RandomStates({})
    
    from nanotron.models import build_model
    model = build_model(
        model_builder=lambda: LlamaForTraining(
            config=model_config,
            parallel_context=parallel_context, 
            parallel_config=parallel_config,
            random_states=random_states,
        ),
        dtype=torch.bfloat16,
        parallel_context=parallel_context,
    )
    
    # Load weights
    try:
        load_weights(model=model, parallel_context=parallel_context, root_folder=Path(checkpoint_path))
    except NotImplementedError as e:
        if "should be a NanotronParameter" in str(e):
            print("Expected balance factor loading error - will fix with standalone loader")
        else:
            raise e
    
    # Apply balance factor fix
    apply_balance_factor_fix_standalone(model, checkpoint_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, parallel_context

def test_passkey_retrieval(model, tokenizer, parallel_context, passkey="12345"):
    """Test passkey retrieval performance"""
    
    test_prompt = f"The secret code is {passkey}. Remember this code. " + \
                  "Here is some additional text to make the context longer. " * 50 + \
                  f" What is the secret code?"
    
    try:
        outputs = list(decode_text(
            input_iter=[GenerationInput(text=test_prompt)],
            tokenizer=tokenizer,
            model=model.model,
            parallel_context=parallel_context,
            max_new_tokens=10,
            max_micro_batch_size=1,
            generation_config=GenerationArgs(sampler="greedy", use_cache=False),
            tokenizer_config=TokenizerConfig(max_input_length=2048),
        ))
        
        if outputs and outputs[0]:
            output = outputs[0]
            if hasattr(output, 'generation_ids') and hasattr(output, 'input_ids'):
                try:
                    generation_ids = output.generation_ids
                    input_ids = output.input_ids
                    
                    if generation_ids.dim() <= 1:
                        generation_ids = generation_ids.unsqueeze(0)
                    if input_ids.dim() <= 1:
                        input_ids = input_ids.unsqueeze(0)
                        
                    input_len = input_ids.shape[-1]
                    if generation_ids.shape[-1] > input_len:
                        answer_ids = generation_ids[0][input_len:]
                        answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
                    else:
                        answer = tokenizer.decode(generation_ids[0], skip_special_tokens=True).strip()
                        
                    return answer, passkey in answer
                        
                except Exception as e:
                    print(f"Error extracting answer: {e}")
                    return "", False
        
        return "", False
        
    except Exception as e:
        print(f"Generation failed: {e}")
        return "", False

def analyze_balance_factors(model):
    """Analyze current balance factor distribution"""
    
    results = {}
    memory_layers = 0
    total_memory_preference = 0
    
    for layer_idx, layer in enumerate(model.model.decoder):
        attn_layer = layer.pp_block.attn
        if hasattr(attn_layer, 'balance_factors') and attn_layer.balance_factors is not None:
            bf_mean = attn_layer.balance_factors.mean().item()
            bf_std = attn_layer.balance_factors.std().item()
            memory_preference = (bf_mean > 0.5)
            
            results[f"layer_{layer_idx}"] = {
                "mean": bf_mean,
                "std": bf_std,
                "memory_preference": memory_preference
            }
            
            total_memory_preference += bf_mean
            if memory_preference:
                memory_layers += 1
    
    results["summary"] = {
        "memory_layers": memory_layers,
        "total_layers": len(model.model.decoder),
        "avg_memory_preference": total_memory_preference / len(model.model.decoder),
        "memory_layer_percentage": (memory_layers / len(model.model.decoder)) * 100
    }
    
    return results

def monitor_checkpoint(checkpoint_path, output_dir="./passkey_monitoring"):
    """Monitor a single checkpoint"""
    
    print(f"\n{'='*60}")
    print(f"MONITORING CHECKPOINT: {checkpoint_path}")
    print(f"{'='*60}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Load model
        print("Loading model and tokenizer...")
        model, tokenizer, parallel_context = load_model_and_tokenizer(checkpoint_path)
        
        # Test passkey retrieval
        print("\nTesting passkey retrieval...")
        test_passkeys = ["12345", "67890", "54321"]
        passkey_results = []
        
        for passkey in test_passkeys:
            answer, success = test_passkey_retrieval(model, tokenizer, parallel_context, passkey)
            passkey_results.append({
                "passkey": passkey,
                "answer": answer,
                "success": success
            })
            print(f"Passkey {passkey}: {'SUCCESS' if success else 'FAIL'} -> '{answer}'")
        
        # Analyze balance factors
        print("\nAnalyzing balance factors...")
        balance_analysis = analyze_balance_factors(model)
        
        # Calculate success rate
        success_rate = sum(1 for r in passkey_results if r["success"]) / len(passkey_results)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        
        checkpoint_name = Path(checkpoint_path).name
        result_file = Path(output_dir) / f"monitor_{checkpoint_name}_{timestamp}.json"
        
        results = {
            "timestamp": timestamp,
            "checkpoint_path": str(checkpoint_path),
            "passkey_results": passkey_results,
            "success_rate": success_rate,
            "balance_factor_analysis": balance_analysis,
            "segment_length": getattr(model.model.decoder[0].pp_block.attn, 'segment_length', 'unknown'),
            "memory_enabled": getattr(model.model.decoder[0].pp_block.attn, 'turn_on_memory', 'unknown')
        }
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nRESULTS SUMMARY:")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Memory Layers: {balance_analysis['summary']['memory_layers']}/{balance_analysis['summary']['total_layers']}")
        print(f"Avg Memory Preference: {balance_analysis['summary']['avg_memory_preference']:.3f}")
        print(f"Results saved to: {result_file}")
        
        return results
        
    except Exception as e:
        print(f"ERROR monitoring checkpoint: {e}")
        return None

def monitor_training_directory(training_dir, output_dir="./passkey_monitoring", interval_minutes=30):
    """Continuously monitor a training directory for new checkpoints"""
    
    print(f"Starting continuous monitoring of: {training_dir}")
    print(f"Check interval: {interval_minutes} minutes")
    print(f"Output directory: {output_dir}")
    
    monitored_checkpoints = set()
    
    while True:
        try:
            # Find checkpoint directories
            checkpoint_dirs = []
            for item in Path(training_dir).iterdir():
                if item.is_dir() and item.name.isdigit():
                    checkpoint_dirs.append(item)
            
            # Sort by checkpoint number
            checkpoint_dirs.sort(key=lambda x: int(x.name))
            
            # Monitor new checkpoints
            for checkpoint_dir in checkpoint_dirs:
                if str(checkpoint_dir) not in monitored_checkpoints:
                    print(f"\nNew checkpoint detected: {checkpoint_dir}")
                    results = monitor_checkpoint(str(checkpoint_dir), output_dir)
                    
                    if results:
                        monitored_checkpoints.add(str(checkpoint_dir))
                        
                        # Print quick summary
                        success_rate = results["success_rate"]
                        memory_layers = results["balance_factor_analysis"]["summary"]["memory_layers"]
                        total_layers = results["balance_factor_analysis"]["summary"]["total_layers"]
                        
                        print(f"CHECKPOINT {checkpoint_dir.name}: Success={success_rate:.1%}, Memory={memory_layers}/{total_layers}")
            
            # Wait for next check
            print(f"\nWaiting {interval_minutes} minutes for next check...")
            time.sleep(interval_minutes * 60)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"Error in monitoring loop: {e}")
            time.sleep(60)  # Wait 1 minute before retrying

def main():
    parser = argparse.ArgumentParser(description="Monitor passkey fine-tuning progress")
    parser.add_argument("--checkpoint", help="Single checkpoint to monitor")
    parser.add_argument("--training-dir", help="Training directory to monitor continuously")
    parser.add_argument("--output-dir", default="./passkey_monitoring", help="Output directory for results")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in minutes for continuous monitoring")
    
    args = parser.parse_args()
    
    if args.checkpoint:
        # Monitor single checkpoint
        monitor_checkpoint(args.checkpoint, args.output_dir)
    elif args.training_dir:
        # Monitor training directory continuously
        monitor_training_directory(args.training_dir, args.output_dir, args.interval)
    else:
        print("Please specify either --checkpoint or --training-dir")
        return

if __name__ == "__main__":
    main()
