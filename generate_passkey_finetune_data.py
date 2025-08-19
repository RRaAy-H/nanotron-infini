#!/usr/bin/env python3
"""
Ultra-optimized synthetic passkey retrieval finetuning dataset generator for 300M Infini-Attention model.
Creates 10K token sequences with embedded numeric passkeys for the model to learn to retrieve.

Key optimizations:
- Batch processing to maximize CPU utilization
- Eliminates ALL tokenization bottlenecks
- Pre-computes token counts analytically 
- Uses pure string operations in all workers
- Designed for maximum 128-core CPU utilization
"""

import argparse
import json
import random
import os
import time
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import numpy as np
from multiprocessing import cpu_count, Manager
import threading

# Set tokenizer parallelism to avoid multiprocessing conflicts
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = lambda x, **kwargs: x

from datasets import Dataset
from transformers import AutoTokenizer


def generate_distractor_phrases(num_repeats: int = 50) -> str:
    """Generate repetitive distractor text."""
    distractor_phrases = [
        "The grass is green.",
        "The sky is blue.", 
        "The sun is yellow.",
        "Here we go.",
        "There and back again.",
    ]
    
    text = ""
    for _ in range(num_repeats):
        text += " ".join(distractor_phrases) + " "
    
    return text.strip()


def estimate_tokens_analytically(text: str, avg_tokens_per_char: float) -> int:
    """
    Estimate token count without tokenization using character-based approximation.
    This is much faster than actual tokenization.
    """
    return max(1, int(len(text) * avg_tokens_per_char))


def precompute_components(tokenizer, target_length: int = 10240) -> Dict:
    """
    Pre-compute and cache all components with analytical token estimation.
    """
    print("Pre-computing components with analytical token estimation...")
    
    # Base components
    instruction = (
        "There is an important info hidden inside a lot of irrelevant text. "
        "Find it and memorize them. I will quiz you about the important information there.\n"
    )
    question = "\nWhat is the pass key? The pass key is"
    
    # Tokenize fixed components once to establish baseline
    instruction_tokens = tokenizer.encode(instruction)
    question_tokens = tokenizer.encode(question)
    
    print(f"Instruction tokens: {len(instruction_tokens)}")
    print(f"Question tokens: {len(question_tokens)}")
    
    # Calculate tokens-per-character ratio for estimation
    sample_texts = [
        "The grass is green. The sky is blue.",
        "Here we go. There and back again.",
        "The sun is yellow. This is a test.",
        "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again." * 10,
    ]
    
    total_chars = 0
    total_tokens = 0
    for text in sample_texts:
        tokens = tokenizer.encode(text)
        total_chars += len(text)
        total_tokens += len(tokens)
    
    avg_tokens_per_char = total_tokens / total_chars
    print(f"Average tokens per character: {avg_tokens_per_char:.4f}")
    
    # Pre-generate large distractor text pools
    print("Pre-generating distractor text pools...")
    distractor_pools = {}
    
    # Create multiple large text pools for different token targets
    for target_tokens in [1000, 2000, 4000, 8000]:
        # Generate text that's approximately target_tokens long
        target_chars = int(target_tokens / avg_tokens_per_char)
        repeats_needed = max(1, target_chars // 150)  # Rough estimate
        
        pool_text = generate_distractor_phrases(repeats_needed)
        estimated_tokens = estimate_tokens_analytically(pool_text, avg_tokens_per_char)
        
        distractor_pools[target_tokens] = {
            'text': pool_text,
            'estimated_tokens': estimated_tokens,
            'chars': len(pool_text)
        }
        print(f"Generated {target_tokens}T pool: {estimated_tokens} tokens, {len(pool_text):,} chars")
    
    # Estimate needle token count
    sample_needle = " The pass key is 1234. Remember it. 1234 is the pass key. "
    actual_needle_tokens = len(tokenizer.encode(sample_needle))
    
    print(f"Sample needle tokens: {actual_needle_tokens}")
    
    # Calculate target distractor tokens
    fixed_tokens = len(instruction_tokens) + len(question_tokens) + actual_needle_tokens
    distractor_tokens_needed = target_length - fixed_tokens - 50  # 50 token buffer
    
    print(f"Fixed component tokens: {fixed_tokens}")
    print(f"Target distractor tokens: {distractor_tokens_needed}")
    
    return {
        'instruction': instruction,
        'question': question,
        'instruction_tokens': len(instruction_tokens),
        'question_tokens': len(question_tokens),
        'distractor_pools': distractor_pools,
        'needle_tokens': actual_needle_tokens,
        'target_distractor_tokens': distractor_tokens_needed,
        'avg_tokens_per_char': avg_tokens_per_char
    }


def create_distractor_text(components: Dict, target_tokens: int) -> Tuple[str, int]:
    """
    Create distractor text of approximately target_tokens length using pools.
    """
    pools = components['distractor_pools']
    avg_tokens_per_char = components['avg_tokens_per_char']
    
    # Find the best pool to start with
    best_pool_size = min(pools.keys(), key=lambda x: abs(x - target_tokens))
    base_pool = pools[best_pool_size]
    
    text = base_pool['text']
    current_tokens = base_pool['estimated_tokens']
    
    # Add more text if needed
    if current_tokens < target_tokens:
        # Calculate how much more we need
        chars_needed = int((target_tokens - current_tokens) / avg_tokens_per_char)
        # Use smallest pool to fill
        smallest_pool = pools[min(pools.keys())]
        
        while current_tokens < target_tokens:
            text += " " + smallest_pool['text']
            current_tokens += smallest_pool['estimated_tokens']
    
    # Trim if too long
    if current_tokens > target_tokens * 1.1:  # 10% tolerance
        # Estimate characters to keep
        chars_to_keep = int(target_tokens / avg_tokens_per_char)
        text = text[:chars_to_keep]
        current_tokens = estimate_tokens_analytically(text, avg_tokens_per_char)
    
    return text, current_tokens


def generate_single_example(passkey: int, depth_percent: float, target_length: int, 
                          components: Dict, seed: int) -> Dict:
    """
    Generate a single passkey example with analytical token counting.
    """
    # Set random seed for this example
    random.seed(seed)
    
    # Create the passkey needle
    needle = f" The pass key is {passkey}. Remember it. {passkey} is the pass key. "
    
    # Calculate target distractor tokens
    target_distractor_tokens = components['target_distractor_tokens']
    
    # Generate distractor text with token estimate
    distractor_text, distractor_token_count = create_distractor_text(components, target_distractor_tokens)
    
    # Insert needle at specified depth
    if depth_percent == 0:
        # Beginning
        full_text = components['instruction'] + needle + distractor_text + components['question']
    elif depth_percent == 100:
        # End  
        full_text = components['instruction'] + distractor_text + needle + components['question']
    else:
        # Middle - split distractor text
        distractor_words = distractor_text.split()
        if len(distractor_words) > 10:  # Avoid empty splits
            split_point = int(len(distractor_words) * (depth_percent / 100))
            first_part = " ".join(distractor_words[:split_point])
            second_part = " ".join(distractor_words[split_point:])
            full_text = components['instruction'] + first_part + needle + second_part + components['question']
        else:
            # Fallback to end if text too short
            full_text = components['instruction'] + distractor_text + needle + components['question']
    
    # Verify passkey is in text
    if str(passkey) not in full_text:
        print(f"WARNING: Passkey {passkey} not found in generated text")
    
    # Calculate estimated total token count analytically
    estimated_total_tokens = (
        components['instruction_tokens'] + 
        components['question_tokens'] + 
        components['needle_tokens'] + 
        distractor_token_count
    )
    
    return {
        "prompt": full_text,
        "answer": str(passkey),
        "depth_percent": depth_percent,
        "passkey": passkey,
        "token_count": estimated_total_tokens
    }


def generate_batch_examples(batch_args: Tuple) -> List[Dict]:
    """
    Generate a batch of examples in a single worker process.
    This is the key function that maximizes CPU utilization.
    """
    batch_params, components, batch_id = batch_args
    
    examples = []
    
    # Process all examples in this batch
    for passkey, depth_percent, target_length, seed in batch_params:
        try:
            example = generate_single_example(
                passkey=passkey,
                depth_percent=depth_percent, 
                target_length=target_length,
                components=components,
                seed=seed
            )
            examples.append(example)
            
        except Exception as e:
            print(f"Error in batch {batch_id} generating example {passkey}: {e}")
            continue
    
    print(f"Batch {batch_id} completed: {len(examples)} examples")
    return examples


def batch_validate_samples(tokenizer, examples: List[Dict], batch_size: int = 1000) -> None:
    """
    Validate a small sample of examples to verify analytical estimates are reasonable.
    """
    if not examples:
        return
        
    sample_size = min(batch_size, len(examples))
    print(f"Validating token estimates on {sample_size} sample examples...")
    
    sample_examples = random.sample(examples, sample_size)
    
    total_error = 0
    max_error = 0
    
    for example in tqdm(sample_examples, desc="Validating samples"):
        actual_tokens = len(tokenizer.encode(example['prompt']))
        estimated_tokens = example['token_count']
        error = abs(actual_tokens - estimated_tokens)
        
        total_error += error
        max_error = max(max_error, error)
    
    avg_error = total_error / len(sample_examples)
    print(f"Validation complete:")
    print(f"  Average token estimation error: {avg_error:.1f} tokens")
    print(f"  Maximum token estimation error: {max_error} tokens")
    print(f"  Estimation accuracy: {100 * (1 - avg_error / 10240):.1f}%")


def generate_dataset(
    tokenizer_path: str,
    num_examples: int = 20000,
    target_length: int = 10240,
    seed: int = 42,
    num_workers: int = None
) -> Dataset:
    """
    Generate full passkey retrieval dataset using batch-optimized parallel processing.
    """
    if num_workers is None:
        num_workers = min(128, cpu_count())
    
    print(f"Starting BATCH-OPTIMIZED dataset generation with {num_workers} workers...")
    print(f"Target: {num_examples} examples with ~{target_length} tokens each")
    
    # Load tokenizer once in main process (only for pre-computation)
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Pre-compute all components with analytical estimation
    components = precompute_components(tokenizer, target_length)
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Define depth percentages
    depth_percentages = [0, 25, 50, 75, 100]
    examples_per_depth = num_examples // len(depth_percentages)
    
    # Generate all example parameters efficiently
    print("Generating example parameters...")
    all_example_params = []
    
    # Pre-generate all unique passkeys at once to avoid collision detection bottleneck
    total_examples = sum(examples_per_depth for _ in depth_percentages)
    print(f"Pre-generating {total_examples} unique passkeys...")
    
    # Generate more than needed and shuffle to avoid collision loops
    passkey_pool = list(range(1000, 10000))  # All possible 4-digit numbers
    random.shuffle(passkey_pool)
    
    if total_examples > len(passkey_pool):
        # If we need more than 9000 examples, allow duplicates
        print(f"Warning: Need {total_examples} examples but only {len(passkey_pool)} unique passkeys available")
        print("Allowing duplicate passkeys...")
        passkey_pool = passkey_pool * ((total_examples // len(passkey_pool)) + 1)
    
    selected_passkeys = passkey_pool[:total_examples]
    
    # Distribute passkeys across depths
    passkey_idx = 0
    for depth in depth_percentages:
        print(f"Creating {examples_per_depth} examples for depth {depth}%...")
        for i in tqdm(range(examples_per_depth), desc=f"Depth {depth}%", leave=False):
            passkey = selected_passkeys[passkey_idx]
            example_seed = seed + len(all_example_params)
            
            all_example_params.append((passkey, depth, target_length, example_seed))
            passkey_idx += 1
    
    # Shuffle for better load distribution
    random.shuffle(all_example_params)
    
    # Create batches for workers - THIS IS THE KEY OPTIMIZATION
    batch_size = max(1, len(all_example_params) // num_workers)
    batches = []
    
    print("Creating worker batches...")
    for i in tqdm(range(0, len(all_example_params), batch_size), desc="Creating batches"):
        batch_params = all_example_params[i:i + batch_size]
        batch_id = len(batches)
        batches.append((batch_params, components, batch_id))
    
    print(f"Created {len(batches)} batches with ~{batch_size} examples each")
    print(f"Starting CPU-intensive batch processing...")
    
    # Process batches in parallel with maximum CPU utilization
    start_time = time.time()
    all_examples = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit batch tasks
        future_to_batch = {
            executor.submit(generate_batch_examples, batch): batch[2] 
            for batch in batches
        }
        
        # Collect results with progress bar
        if TQDM_AVAILABLE:
            futures = tqdm(as_completed(future_to_batch), total=len(future_to_batch), 
                          desc="Processing batches", unit="batch")
        else:
            futures = as_completed(future_to_batch)
        
        completed_batches = 0
        for future in futures:
            try:
                batch_examples = future.result()
                all_examples.extend(batch_examples)
                completed_batches += 1
                
                if TQDM_AVAILABLE:
                    # Update description with progress
                    current_rate = len(all_examples) / (time.time() - start_time)
                    futures.set_description(
                        f"Processing batches ({len(all_examples):,} examples, {current_rate:.0f}/s)"
                    )
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
    
    generation_time = time.time() - start_time
    print(f"\nBatch processing completed in {generation_time:.1f} seconds")
    print(f"Speed: {len(all_examples) / generation_time:.1f} examples/second")
    print(f"Generated {len(all_examples)} examples")
    
    # Validate token estimates on a small sample
    batch_validate_samples(tokenizer, all_examples, batch_size=min(1000, len(all_examples) // 10))
    
    # Shuffle final examples
    random.shuffle(all_examples)
    
    # Create dataset
    dataset_dict = {
        "prompt": [ex["prompt"] for ex in all_examples],
        "answer": [ex["answer"] for ex in all_examples],
        "depth_percent": [ex["depth_percent"] for ex in all_examples],
        "token_count": [ex["token_count"] for ex in all_examples],
        "passkey": [ex["passkey"] for ex in all_examples],
    }
    
    return Dataset.from_dict(dataset_dict)


def main():
    parser = argparse.ArgumentParser(description="Generate ultra-optimized passkey retrieval finetuning dataset")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="lvwerra/the-tokenizer-v1",
        help="Path to tokenizer (same as used in training)"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=20000,
        help="Number of examples to generate"
    )
    parser.add_argument(
        "--target_length",
        type=int,
        default=10240,
        help="Target sequence length in tokens (default 10240 for ~10K)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./passkey_finetune_data_10k",
        help="Path to save the dataset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=128,
        help="Number of parallel workers (default: 128, max: CPU count)"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "--hub_repo",
        type=str,
        default="your-username/passkey-finetune-10k",
        help="HuggingFace Hub repository name"
    )
    
    args = parser.parse_args()
    
    # Limit workers to available CPU count
    max_workers = min(args.num_workers, cpu_count())
    
    # Display configuration
    print("=" * 70)
    print("BATCH-OPTIMIZED PASSKEY DATASET GENERATOR")
    print("=" * 70)
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Examples: {args.num_examples:,}")
    print(f"Target length: {args.target_length:,} tokens")
    print(f"Workers: {max_workers} (requested: {args.num_workers})")
    print(f"CPU cores available: {cpu_count()}")
    print(f"Batch size per worker: ~{args.num_examples // max_workers}")
    print(f"Seed: {args.seed}")
    print(f"TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM', 'not set')}")
    print("=" * 70)
    
    # Generate dataset
    total_start = time.time()
    dataset = generate_dataset(
        tokenizer_path=args.tokenizer_path,
        num_examples=args.num_examples,
        target_length=args.target_length,
        seed=args.seed,
        num_workers=max_workers
    )
    total_time = time.time() - total_start
    
    # Save dataset
    parquet_path = f"{args.save_path}.parquet"
    dataset.to_parquet(parquet_path)
    print(f"\nSaved dataset as parquet: {parquet_path}")
    
    # Print comprehensive statistics
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"Total examples: {len(dataset):,}")
    print(f"Total generation time: {total_time:.1f} seconds")
    print(f"Overall speed: {len(dataset) / total_time:.1f} examples/second")
    print(f"Average token count: {sum(dataset['token_count']) / len(dataset):.1f}")
    print(f"Min token count: {min(dataset['token_count']):,}")
    print(f"Max token count: {max(dataset['token_count']):,}")
    
    # Show depth distribution
    depth_counts = Counter(dataset['depth_percent'])
    print("\nDepth distribution:")
    for depth in sorted(depth_counts.keys()):
        print(f"  {depth}%: {depth_counts[depth]:,} examples")
    
    # Show sample
    print(f"\nSample example (first 500 chars):")
    print(dataset['prompt'][0][:500] + "...")
    print(f"Answer: {dataset['answer'][0]}")
    print(f"Tokens: {dataset['token_count'][0]:,}")
    
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.hub_repo}...")
        dataset.push_to_hub(args.hub_repo)
        print("Dataset pushed successfully!")
    
    print("\n" + "=" * 70)
    print("BATCH-OPTIMIZED GENERATION COMPLETE!")
    print(f"Generated {len(dataset):,} examples in {total_time:.1f} seconds")
    print(f"Final speed: {len(dataset) / total_time:.1f} examples/second")
    print("=" * 70)


if __name__ == "__main__":
    main()