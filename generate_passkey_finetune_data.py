#!/usr/bin/env python3
"""
Optimized synthetic passkey retrieval finetuning dataset generator for 300M Infini-Attention model.
Creates 10K token sequences with embedded numeric passkeys for the model to learn to retrieve.
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


def precompute_components(tokenizer, target_length: int = 10240) -> Dict:
    """
    Pre-compute and cache all tokenized components to avoid tokenization in workers.
    """
    print("Pre-computing tokenized components...")
    
    # Base components
    instruction = (
        "There is an important info hidden inside a lot of irrelevant text. "
        "Find it and memorize them. I will quiz you about the important information there.\n"
    )
    question = "\nWhat is the pass key? The pass key is"
    
    # Tokenize fixed components once
    instruction_tokens = tokenizer.encode(instruction)
    question_tokens = tokenizer.encode(question)
    
    print(f"Instruction tokens: {len(instruction_tokens)}")
    print(f"Question tokens: {len(question_tokens)}")
    
    # Generate various sized distractor chunks and tokenize them
    print("Generating distractor text chunks...")
    distractor_chunks = []
    chunk_token_counts = []
    
    # Create chunks of different sizes to accommodate different target lengths
    for repeat_count in [5, 10, 20, 30, 40, 50, 75, 100, 150, 200]:
        chunk_text = generate_distractor_phrases(repeat_count)
        chunk_tokens = tokenizer.encode(chunk_text)
        distractor_chunks.append(chunk_text)
        chunk_token_counts.append(len(chunk_tokens))
        print(f"Chunk {repeat_count} repeats: {len(chunk_tokens)} tokens")
    
    # Pre-generate passkey needles for common passkeys
    print("Pre-computing passkey needles...")
    passkey_needles = {}
    needle_token_counts = {}
    
    # Sample some passkeys to pre-compute (we'll compute others on demand)
    sample_passkeys = [1000, 1234, 5678, 9999]
    for passkey in sample_passkeys:
        needle = f" The pass key is {passkey}. Remember it. {passkey} is the pass key. "
        needle_tokens = tokenizer.encode(needle)
        passkey_needles[passkey] = needle
        needle_token_counts[passkey] = len(needle_tokens)
    
    # Estimate typical needle token count for planning
    avg_needle_tokens = sum(needle_token_counts.values()) // len(needle_token_counts)
    print(f"Average needle tokens: {avg_needle_tokens}")
    
    # Calculate how much distractor text we typically need
    fixed_tokens = len(instruction_tokens) + len(question_tokens) + avg_needle_tokens
    distractor_tokens_needed = target_length - fixed_tokens - 50  # 50 token buffer
    
    print(f"Fixed component tokens: {fixed_tokens}")
    print(f"Target distractor tokens: {distractor_tokens_needed}")
    
    return {
        'instruction': instruction,
        'question': question,
        'instruction_tokens': len(instruction_tokens),
        'question_tokens': len(question_tokens),
        'distractor_chunks': distractor_chunks,
        'chunk_token_counts': chunk_token_counts,
        'passkey_needles': passkey_needles,
        'needle_token_counts': needle_token_counts,
        'avg_needle_tokens': avg_needle_tokens,
        'target_distractor_tokens': distractor_tokens_needed
    }


def create_distractor_text(components: Dict, target_tokens: int) -> str:
    """
    Create distractor text of approximately target_tokens length using pre-computed chunks.
    """
    chunks = components['distractor_chunks']
    chunk_counts = components['chunk_token_counts']
    
    # Find the best combination of chunks to reach target tokens
    selected_text = ""
    current_tokens = 0
    
    # Start with largest chunks and work down
    for i in range(len(chunks) - 1, -1, -1):
        while current_tokens + chunk_counts[i] <= target_tokens:
            selected_text += chunks[i] + " "
            current_tokens += chunk_counts[i]
    
    # Fill remaining space with smallest chunks
    while current_tokens < target_tokens and chunks:
        selected_text += chunks[0] + " "
        current_tokens += chunk_counts[0]
    
    return selected_text.strip()


def generate_single_example(args: Tuple) -> Dict:
    """
    Generate a single passkey example using only string operations (no tokenizer).
    This function runs in worker processes.
    """
    passkey, depth_percent, target_length, components, worker_seed = args
    
    # Set random seed for this example
    random.seed(worker_seed)
    
    # Create the passkey needle
    needle = f" The pass key is {passkey}. Remember it. {passkey} is the pass key. "
    
    # Calculate target distractor tokens
    estimated_needle_tokens = components['avg_needle_tokens']
    target_distractor_tokens = target_length - components['instruction_tokens'] - components['question_tokens'] - estimated_needle_tokens - 50
    
    # Generate distractor text
    distractor_text = create_distractor_text(components, target_distractor_tokens)
    
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
        split_point = int(len(distractor_words) * (depth_percent / 100))
        
        first_part = " ".join(distractor_words[:split_point])
        second_part = " ".join(distractor_words[split_point:])
        
        full_text = components['instruction'] + first_part + needle + second_part + components['question']
    
    # Verify passkey is in text
    assert str(passkey) in full_text, f"Passkey {passkey} not found in generated text"
    
    return {
        "prompt": full_text,
        "answer": str(passkey),
        "depth_percent": depth_percent,
        "passkey": passkey,
        "estimated_tokens": target_length  # We'll calculate actual tokens later in main process
    }


def validate_and_count_tokens(tokenizer, examples: List[Dict]) -> List[Dict]:
    """
    Validate examples and count actual tokens in main process.
    """
    print("Validating examples and counting tokens...")
    
    if TQDM_AVAILABLE:
        examples = tqdm(examples, desc="Counting tokens")
    
    for example in examples:
        # Count actual tokens
        actual_tokens = len(tokenizer.encode(example['prompt']))
        example['token_count'] = actual_tokens
        
        # Remove estimated_tokens field
        del example['estimated_tokens']
    
    return examples


def generate_dataset(
    tokenizer_path: str,
    num_examples: int = 20000,
    target_length: int = 10240,
    seed: int = 42,
    num_workers: int = 128
) -> Dataset:
    """
    Generate full passkey retrieval dataset using optimized parallel processing.
    """
    print(f"Starting optimized dataset generation with {num_workers} workers...")
    print(f"Target: {num_examples} examples with ~{target_length} tokens each")
    
    # Set environment variable for tokenizer parallelism
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Load tokenizer once in main process
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Pre-compute all components
    components = precompute_components(tokenizer, target_length)
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Define depth percentages
    depth_percentages = [0, 25, 50, 75, 100]
    examples_per_depth = num_examples // len(depth_percentages)
    
    # Generate all passkeys and parameters
    print("Generating example parameters...")
    example_args = []
    used_passkeys = set()
    
    for depth in depth_percentages:
        for i in range(examples_per_depth):
            # Generate unique passkey
            while True:
                passkey = random.randint(1000, 9999)
                if passkey not in used_passkeys:
                    used_passkeys.add(passkey)
                    break
            
            # Create unique seed for this example
            worker_seed = seed + len(example_args)
            
            example_args.append((
                passkey,
                depth,
                target_length,
                components,
                worker_seed
            ))
    
    # Shuffle for better load distribution
    random.shuffle(example_args)
    
    print(f"Generated {len(example_args)} example parameters")
    print(f"Starting parallel generation...")
    
    # Process examples in parallel
    start_time = time.time()
    examples = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_args = {
            executor.submit(generate_single_example, args): args 
            for args in example_args
        }
        
        # Collect results with progress bar
        if TQDM_AVAILABLE:
            futures = tqdm(as_completed(future_to_args), total=len(future_to_args), 
                          desc="Generating examples", unit="ex")
        else:
            futures = as_completed(future_to_args)
        
        for future in futures:
            try:
                example = future.result()
                examples.append(example)
            except Exception as e:
                print(f"Error generating example: {e}")
                continue
    
    generation_time = time.time() - start_time
    print(f"\nParallel generation completed in {generation_time:.1f} seconds")
    print(f"Speed: {len(examples) / generation_time:.1f} examples/second")
    print(f"Generated {len(examples)} examples")
    
    # Validate and count actual tokens in main process
    examples = validate_and_count_tokens(tokenizer, examples)
    
    # Shuffle final examples
    random.shuffle(examples)
    
    # Create dataset
    dataset_dict = {
        "prompt": [ex["prompt"] for ex in examples],
        "answer": [ex["answer"] for ex in examples],
        "depth_percent": [ex["depth_percent"] for ex in examples],
        "token_count": [ex["token_count"] for ex in examples],
        "passkey": [ex["passkey"] for ex in examples],
    }
    
    return Dataset.from_dict(dataset_dict)


def main():
    parser = argparse.ArgumentParser(description="Generate optimized passkey retrieval finetuning dataset")
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
        help="Number of parallel workers"
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
    
    # Display configuration
    print("=" * 60)
    print("OPTIMIZED PASSKEY DATASET GENERATOR")
    print("=" * 60)
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Examples: {args.num_examples}")
    print(f"Target length: {args.target_length} tokens")
    print(f"Workers: {args.num_workers}")
    print(f"Seed: {args.seed}")
    print(f"TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM', 'not set')}")
    print("=" * 60)
    
    # Generate dataset
    total_start = time.time()
    dataset = generate_dataset(
        tokenizer_path=args.tokenizer_path,
        num_examples=args.num_examples,
        target_length=args.target_length,
        seed=args.seed,
        num_workers=args.num_workers
    )
    total_time = time.time() - total_start
    
    # Save dataset
    parquet_path = f"{args.save_path}.parquet"
    dataset.to_parquet(parquet_path)
    print(f"\nSaved dataset as parquet: {parquet_path}")
    
    # Print comprehensive statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total examples: {len(dataset)}")
    print(f"Total generation time: {total_time:.1f} seconds")
    print(f"Overall speed: {len(dataset) / total_time:.1f} examples/second")
    print(f"Average token count: {sum(dataset['token_count']) / len(dataset):.1f}")
    print(f"Min token count: {min(dataset['token_count'])}")
    print(f"Max token count: {max(dataset['token_count'])}")
    
    # Show depth distribution
    depth_counts = Counter(dataset['depth_percent'])
    print("\nDepth distribution:")
    for depth in sorted(depth_counts.keys()):
        print(f"  {depth}%: {depth_counts[depth]} examples")
    
    # Show sample
    print(f"\nSample example (first 500 chars):")
    print(dataset['prompt'][0][:500] + "...")
    print(f"Answer: {dataset['answer'][0]}")
    print(f"Tokens: {dataset['token_count'][0]}")
    
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.hub_repo}...")
        dataset.push_to_hub(args.hub_repo)
        print("Dataset pushed successfully!")
    
    print("=" * 60)
    print("GENERATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()