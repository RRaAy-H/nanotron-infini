#!/usr/bin/env python3
"""
Enhanced Passkey Evaluation with Memory Monitoring

This script enhances the existing passkey evaluation to show detailed memory usage
during real passkey retrieval tasks using existing datasets.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

# Ensure we're loading from the correct path
correct_path = "/data1/infini-attn/infini-llama/nanotron-infini/src"
if correct_path not in sys.path:
    sys.path.insert(0, correct_path)

# Remove any conflicting paths
sys.path = [p for p in sys.path if 'fiery/infini-nanotron' not in p]

import torch
from nanotron import constants, distributed as dist, logging
from nanotron.config import get_config_from_file, GenerationArgs, ParallelismArgs, LoggingArgs
from nanotron.generation.decode import GenerationInput, TokenizerConfig, decode_text
from nanotron.logging import log_rank, set_ranks_logging_level
from nanotron.models import build_model
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.parallel.pipeline_parallel.engine import OneForwardOneBackwardPipelineEngine
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.random import RandomStates, get_current_random_state, get_synced_random_state, set_random_seed
from nanotron.serialize import load_weights
from nanotron.trainer import CONFIG_TO_MODEL_CLASS, mark_tied_parameters
from transformers import AutoTokenizer

logger = logging.get_logger(__name__)

class DetailedMemoryMonitor:
    """Detailed memory monitoring for passkey evaluation."""
    
    def __init__(self):
        self.memory_calls = {'retrieve': 0, 'update': 0}
        self.detailed_stats = []
        self.hooked_blocks = []
        self.current_sample = None
        
    def start_sample(self, sample_info):
        """Start monitoring a new sample."""
        self.current_sample = sample_info
        self.memory_calls = {'retrieve': 0, 'update': 0}
        self.detailed_stats = []
        
    def hook_memory_functions(self, model):
        """Hook memory functions for detailed monitoring."""
        actual_model = model.model if hasattr(model, 'model') else model
        
        if not hasattr(actual_model, 'decoder'):
            return
        
        for layer_idx, pipeline_block in enumerate(actual_model.decoder):
            if hasattr(pipeline_block, 'pp_block') and hasattr(pipeline_block.pp_block, 'attn'):
                attn_layer = pipeline_block.pp_block.attn
                
                if hasattr(attn_layer, '_retrieve_from_memory') and hasattr(attn_layer, '_update_memory'):
                    original_forward = pipeline_block.forward
                    original_retrieve = attn_layer._retrieve_from_memory
                    original_update = attn_layer._update_memory
                    
                    def create_monitored_forward(layer_idx):
                        def monitored_forward(*args, **kwargs):
                            def counting_retrieve(query_states, prev_memory, prev_normalization):
                                self.memory_calls['retrieve'] += 1
                                has_memory = prev_memory is not None
                                memory_norm = prev_memory.norm().item() if has_memory else 0.0
                                
                                self.detailed_stats.append({
                                    'type': 'retrieve',
                                    'layer': layer_idx,
                                    'timestamp': time.time(),
                                    'has_prev_memory': has_memory,
                                    'memory_norm': memory_norm,
                                    'sample_info': self.current_sample
                                })
                                
                                return original_retrieve(query_states, prev_memory, prev_normalization)
                            
                            def counting_update(prev_memory, prev_normalization, key_states, value_states):
                                self.memory_calls['update'] += 1
                                prev_norm = prev_memory.norm().item() if prev_memory is not None else 0.0
                                
                                result = original_update(prev_memory, prev_normalization, key_states, value_states)
                                new_memory, new_normalization = result
                                new_norm = new_memory.norm().item()
                                
                                self.detailed_stats.append({
                                    'type': 'update',
                                    'layer': layer_idx,
                                    'timestamp': time.time(),
                                    'prev_memory_norm': prev_norm,
                                    'new_memory_norm': new_norm,
                                    'sample_info': self.current_sample
                                })
                                
                                return result
                            
                            attn_layer._retrieve_from_memory = counting_retrieve
                            attn_layer._update_memory = counting_update
                            
                            try:
                                result = original_forward(*args, **kwargs)
                                return result
                            finally:
                                attn_layer._retrieve_from_memory = original_retrieve
                                attn_layer._update_memory = original_update
                        
                        return monitored_forward
                    
                    pipeline_block.forward = create_monitored_forward(layer_idx)
                    self.hooked_blocks.append((layer_idx, pipeline_block, original_forward))
        
        log_rank(f"SUCCESS: Memory monitoring active on {len(self.hooked_blocks)} layers", logger=logger, level=logging.INFO, rank=0)
        
    def get_sample_summary(self):
        """Get summary for current sample."""
        return {
            'total_retrievals': self.memory_calls['retrieve'],
            'total_updates': self.memory_calls['update'],
            'memory_active': self.memory_calls['retrieve'] > 0 or self.memory_calls['update'] > 0,
            'layers_with_memory': len(set(stat['layer'] for stat in self.detailed_stats)),
            'retrieval_layers': list(set(stat['layer'] for stat in self.detailed_stats if stat['type'] == 'retrieve')),
            'update_layers': list(set(stat['layer'] for stat in self.detailed_stats if stat['type'] == 'update')),
            'avg_memory_norm': sum(s.get('memory_norm', 0) for s in self.detailed_stats if s['type'] == 'retrieve' and s.get('memory_norm', 0) > 0) / max(1, sum(1 for s in self.detailed_stats if s['type'] == 'retrieve' and s.get('memory_norm', 0) > 0)),
            'detailed_stats': self.detailed_stats
        }

def generate_with_memory_monitoring(args, model, tokenizer, inputs, parallel_context, monitor):
    """Enhanced generation function with memory monitoring."""
    
    responses = []
    answer_idxs = []
    memory_summaries = []
    
    for i, text in enumerate(inputs):
        # Start monitoring this sample
        monitor.start_sample({
            'sample_index': i,
            'prompt_length': len(tokenizer.encode(text)),
            'timestamp': time.time()
        })
        
        log_rank(f"Processing sample {i+1}/{len(inputs)} with memory monitoring...", 
                logger=logger, level=logging.INFO, rank=0)
        
        outputs = decode_text(
            input_iter=[GenerationInput(text=text)],
            tokenizer=tokenizer,
            model=model.model,
            parallel_context=parallel_context,
            max_new_tokens=args.max_new_tokens,
            max_micro_batch_size=1,
            generation_config=GenerationArgs(sampler="greedy", use_cache=False),
            tokenizer_config=TokenizerConfig(max_input_length=None),
            is_bench=os.environ.get("USE_BENCH", "0") == "1",
        )
        
        # Process output
        for output in outputs:
            input_ids = output.input_ids
            generated_ids = output.generation_ids
            
            answer_ids = generated_ids[len(input_ids):]
            decoded_answer = tokenizer.decode(answer_ids, clean_up_tokenization_spaces=False)
            
            if isinstance(input_ids, TensorPointer):
                assert isinstance(generated_ids, TensorPointer)
                continue
            assert isinstance(generated_ids, torch.Tensor)
            
            # Get memory summary for this sample
            memory_summary = monitor.get_sample_summary()
            memory_summaries.append(memory_summary)
            
            log_rank(f"""
            Memory Usage for Sample {i+1}:
               Retrievals: {memory_summary['total_retrievals']}
               Updates: {memory_summary['total_updates']}
               Active Layers: {memory_summary['layers_with_memory']}
               Avg Memory Norm: {memory_summary['avg_memory_norm']:.4f}
            Generated: {decoded_answer}""", 
                    logger=logger, level=logging.INFO, rank=0)
            
            responses.append(decoded_answer)
            answer_idxs.append(answer_ids.tolist())
    
    dist.barrier()
    return responses, answer_idxs, memory_summaries

def load_and_filter_dataset(eval_dataset_path, depth_percent, num_shots, num_digits, seed, num_samples):
    """Load and filter dataset (same as original)."""
    import random
    from datasets import load_dataset, load_from_disk
    
    random.seed(seed)
    
    if os.path.exists(eval_dataset_path):
        path = Path(eval_dataset_path)
        if path.is_file() and path.suffix == '.parquet':
            dataset = load_dataset("parquet", data_files=str(path), split="train")
        elif path.is_dir():
            parquet_files = list(path.glob("*.parquet"))
            if parquet_files:
                dataset = load_dataset("parquet", data_files=[str(f) for f in parquet_files], split="train")
            elif (path / "dataset_info.json").exists() or (path / "data-00000-of-00001.arrow").exists():
                dataset = load_from_disk(eval_dataset_path)
            else:
                dataset = load_dataset(eval_dataset_path, split="train")
        else:
            dataset = load_dataset(eval_dataset_path, split="train")
    else:
        dataset = load_dataset(eval_dataset_path, split="train")
    
    # Filter the dataset
    filtered_dataset = dataset.filter(lambda x: x["depth_percent"] == depth_percent and x["num_shots"] == num_shots)
    if num_digits > 0:
        filtered_dataset = filtered_dataset.filter(lambda x: x["num_digits"] == num_digits)
    
    shuffled_dataset = filtered_dataset.shuffle(seed=seed)
    final_dataset = shuffled_dataset.select(range(min(num_samples, len(shuffled_dataset))))
    
    return final_dataset

def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Enhanced Passkey Evaluation with Memory Monitoring")
    parser.add_argument("--ckpt-path", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--dp", type=int, default=0)
    parser.add_argument("--pp", type=int, default=0)
    parser.add_argument("--tp", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=15, help="Maximum number of new tokens to generate")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--eval_dataset_path", type=str, required=True)
    parser.add_argument("--num_shots", type=int, required=True)
    parser.add_argument("--num_digits", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--depth_percents", nargs="+", type=int, default=[0, 25, 50, 75, 100], 
                        help="Specific depth percentages to test")
    return parser.parse_args()

def main():
    args = get_args()
    
    log_rank("Enhanced Passkey Evaluation with Memory Monitoring", logger=logger, level=logging.INFO, rank=0)
    log_rank("=" * 60, logger=logger, level=logging.INFO, rank=0)
    
    assert args.ckpt_path.exists(), f"Checkpoint path {args.ckpt_path} does not exist"
    
    config = get_config_from_file((args.ckpt_path / "config.yaml").as_posix())
    constants.CONFIG = config
    model_config = config.model.model_config
    tokenizer_path = config.tokenizer.tokenizer_name_or_path
    
    parallel_config = ParallelismArgs(
        dp=args.dp or config.parallelism.dp,
        pp=args.pp or config.parallelism.pp,
        tp=args.tp or config.parallelism.tp,
        pp_engine=OneForwardOneBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        tp_linear_async_communication=False,
    )
    
    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )
    
    # Set log levels
    logging_config = LoggingArgs(log_level="info", log_level_replica="info")
    set_ranks_logging_level(parallel_context=parallel_context, logging_config=logging_config)
    
    set_random_seed(42)
    
    # Build model
    model_config_cls = model_config.__class__.__name__
    if parallel_config.tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        random_states = RandomStates({
            "tp_synced": get_synced_random_state(random_state=get_current_random_state(), pg=parallel_context.tp_pg)
        })
    else:
        random_states = RandomStates({})
    
    model = build_model(
        model_builder=lambda: CONFIG_TO_MODEL_CLASS[model_config_cls](
            config=model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=random_states,
        ),
        dtype=torch.bfloat16,
        parallel_context=parallel_context,
    )
    
    mark_tied_parameters(model=model, parallel_context=parallel_context, parallel_config=parallel_config)
    sanity_check(root_module=model)
    
    # Load checkpoint
    log_rank(f"Loading checkpoint from {args.ckpt_path}:", logger=logger, level=logging.INFO, rank=0)
    load_weights(model=model, parallel_context=parallel_context, root_folder=args.ckpt_path)
    
    # Apply balance factor fix
    log_rank("Applying balance factor fix...", logger=logger, level=logging.INFO, rank=0)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    
    try:
        from apply_balance_fix_standalone import apply_balance_factor_fix_standalone
        fix_success = apply_balance_factor_fix_standalone(model, args.ckpt_path, verbose=False)
        if fix_success:
            log_rank("SUCCESS: Balance factors loaded successfully", logger=logger, level=logging.INFO, rank=0)
        else:
            log_rank("WARNING: Balance factor fix may not have worked properly", logger=logger, level=logging.WARNING, rank=0)
    except Exception as e:
        log_rank(f"WARNING: Balance factor fix failed: {e}", logger=logger, level=logging.WARNING, rank=0)
    
    model.eval()
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    # Setup memory monitoring
    monitor = DetailedMemoryMonitor()
    monitor.hook_memory_functions(model)
    
    # Run evaluation with memory monitoring
    all_results = {}
    
    for depth_percent in args.depth_percents:
        log_rank(f"Testing depth_percent: {depth_percent}%", logger=logger, level=logging.INFO, rank=0)
        
        dataset = load_and_filter_dataset(
            args.eval_dataset_path, depth_percent, args.num_shots, args.num_digits, 
            seed=args.seed, num_samples=args.num_samples
        )
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        
        responses = []
        answer_idxs = []
        memory_summaries = []
        
        from tqdm import tqdm
        for batch in tqdm(dataloader, desc=f"Depth {depth_percent}%"):
            log_rank(f"Target answer: {batch['answer']}", logger=logger, level=logging.INFO, rank=0)
            
            texts = batch["prompt"]
            response, answer_ids, memory_summary = generate_with_memory_monitoring(
                args, model, tokenizer, texts, parallel_context, monitor
            )
            
            responses.extend(response)
            answer_idxs.extend(answer_ids)
            memory_summaries.extend(memory_summary)
        
        # Analyze results for this depth
        if dist.get_rank(parallel_context.dp_pg) == 0 and dist.get_rank(parallel_context.tp_pg) == 0 and dist.get_rank(parallel_context.pp_pg) == 0:
            
            # Check accuracy
            correct_answers = []
            for i, (response, expected) in enumerate(zip(responses, [b['answer'][0] for b in dataloader])):
                is_correct = str(expected).strip() in response.strip()
                correct_answers.append(is_correct)
            
            accuracy = sum(correct_answers) / len(correct_answers) * 100 if correct_answers else 0
            
            # Memory analysis
            avg_retrievals = sum(ms['total_retrievals'] for ms in memory_summaries) / len(memory_summaries) if memory_summaries else 0
            avg_updates = sum(ms['total_updates'] for ms in memory_summaries) / len(memory_summaries) if memory_summaries else 0
            
            depth_results = {
                'depth_percent': depth_percent,
                'accuracy': accuracy,
                'total_samples': len(responses),
                'correct_samples': sum(correct_answers),
                'memory_stats': {
                    'avg_retrievals': avg_retrievals,
                    'avg_updates': avg_updates,
                    'memory_active_samples': sum(1 for ms in memory_summaries if ms['memory_active']),
                },
                'responses': responses,
                'memory_summaries': memory_summaries
            }
            
            all_results[depth_percent] = depth_results
            
            log_rank(f"""
            Results for Depth {depth_percent}%:
               Accuracy: {accuracy:.1f}% ({sum(correct_answers)}/{len(correct_answers)})
               Avg Memory Retrievals: {avg_retrievals:.1f}
               Avg Memory Updates: {avg_updates:.1f}
               Memory Active: {sum(1 for ms in memory_summaries if ms['memory_active'])}/{len(memory_summaries)} samples
            """, logger=logger, level=logging.INFO, rank=0)
    
    # Save comprehensive results
    if dist.get_rank(parallel_context.dp_pg) == 0 and dist.get_rank(parallel_context.tp_pg) == 0 and dist.get_rank(parallel_context.pp_pg) == 0:
        
        log_rank("=" * 60, logger=logger, level=logging.INFO, rank=0)
        log_rank("COMPREHENSIVE MEMORY-AWARE PASSKEY ANALYSIS", logger=logger, level=logging.INFO, rank=0)
        log_rank("=" * 60, logger=logger, level=logging.INFO, rank=0)
        
        for depth in sorted(all_results.keys()):
            result = all_results[depth]
            log_rank(f"Depth {depth:3d}%: {result['accuracy']:5.1f}% accuracy, "
                    f"{result['memory_stats']['avg_retrievals']:5.1f} avg retrievals, "
                    f"{result['memory_stats']['memory_active_samples']:2d}/{result['total_samples']} memory active", 
                    logger=logger, level=logging.INFO, rank=0)
        
        # Save detailed results
        output_file = f"{args.save_path}/memory_aware_passkey_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'test_configuration': {
                    'checkpoint': str(args.ckpt_path),
                    'dataset': args.eval_dataset_path,
                    'num_shots': args.num_shots,
                    'num_digits': args.num_digits,
                    'num_samples': args.num_samples,
                    'depth_percents': args.depth_percents,
                    'seed': args.seed
                },
                'results_by_depth': all_results,
                'timestamp': time.time()
            }, f, indent=2)
        
        log_rank(f"Detailed results saved to: {output_file}", logger=logger, level=logging.INFO, rank=0)
        log_rank("Memory-aware passkey evaluation completed!", logger=logger, level=logging.INFO, rank=0)

if __name__ == "__main__":
    main()
