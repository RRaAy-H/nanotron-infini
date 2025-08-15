#!/usr/bin/env python3
"""
Generate synthetic passkey retrieval finetuning dataset for 300M Infini-Attention model.
Creates 10K token sequences with embedded numeric passkeys for the model to learn to retrieve.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict
import os

from datasets import Dataset
from transformers import AutoTokenizer


def generate_distractor_text(num_repeats: int = 50) -> str:
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


def generate_passkey_example(
    tokenizer,
    passkey: int,
    depth_percent: float,
    target_length: int = 10240,
    is_eval: bool = False
) -> Dict[str, any]:
    """
    Generate a single passkey retrieval example.
    
    Args:
        tokenizer: Tokenizer to count tokens
        passkey: The numeric passkey to embed (4-digit number)
        depth_percent: Where to place the passkey (0-100, where 0 is beginning, 100 is end)
        target_length: Target sequence length in tokens (default 10240 for ~10K)
        is_eval: If True, don't include answer in the prompt (for evaluation)
    
    Returns:
        Dictionary with 'prompt' and 'answer' keys
    """
    # Create the instruction prompt
    instruction = (
        "There is an important info hidden inside a lot of irrelevant text. "
        "Find it and memorize them. I will quiz you about the important information there.\n"
    )
    
    # Create the passkey needle
    needle = f" The pass key is {passkey}. Remember it. {passkey} is the pass key. "
    
    # Create the question
    if is_eval:
        question = "\nWhat is the pass key? The pass key is"
    else:
        question = f"\nWhat is the pass key? The pass key is {passkey}."
    
    # Calculate how much distractor text we need
    instruction_tokens = len(tokenizer.encode(instruction))
    needle_tokens = len(tokenizer.encode(needle))
    question_tokens = len(tokenizer.encode(question))
    
    # Leave some buffer for tokenization differences
    distractor_tokens_needed = target_length - instruction_tokens - needle_tokens - question_tokens - 50
    
    # Generate distractor text
    distractor_text = ""
    while len(tokenizer.encode(distractor_text)) < distractor_tokens_needed:
        distractor_text += generate_distractor_text(10) + " "
    
    # Trim to approximate size
    distractor_tokens = tokenizer.encode(distractor_text)
    if len(distractor_tokens) > distractor_tokens_needed:
        distractor_text = tokenizer.decode(distractor_tokens[:distractor_tokens_needed])
    
    # Insert needle at the specified depth
    if depth_percent == 0:
        # Beginning
        full_text = instruction + needle + distractor_text + question
    elif depth_percent == 100:
        # End
        full_text = instruction + distractor_text + needle + question
    else:
        # Middle - split distractor text
        distractor_words = distractor_text.split()
        split_point = int(len(distractor_words) * (depth_percent / 100))
        
        first_part = " ".join(distractor_words[:split_point])
        second_part = " ".join(distractor_words[split_point:])
        
        full_text = instruction + first_part + needle + second_part + question
    
    # Verify the passkey is in the text
    assert str(passkey) in full_text, f"Passkey {passkey} not found in generated text"
    
    # Check final token count
    final_tokens = len(tokenizer.encode(full_text))
    
    return {
        "prompt": full_text,
        "answer": str(passkey),
        "depth_percent": depth_percent,
        "token_count": final_tokens,
        "passkey": passkey
    }


def generate_dataset(
    tokenizer,
    num_examples: int = 2000,
    target_length: int = 10240,
    seed: int = 42
) -> Dataset:
    """
    Generate full passkey retrieval dataset.
    
    Args:
        tokenizer: Tokenizer to use
        num_examples: Total number of examples to generate
        target_length: Target sequence length in tokens
        seed: Random seed for reproducibility
    
    Returns:
        HuggingFace Dataset object
    """
    random.seed(seed)
    
    # Define depth percentages to cover
    depth_percentages = [0, 25, 50, 75, 100]
    
    examples = []
    used_passkeys = set()
    
    # Generate examples evenly distributed across depths
    examples_per_depth = num_examples // len(depth_percentages)
    
    for depth in depth_percentages:
        for _ in range(examples_per_depth):
            # Generate unique 4-digit passkey
            while True:
                passkey = random.randint(1000, 9999)
                if passkey not in used_passkeys:
                    used_passkeys.add(passkey)
                    break
            
            # Generate example
            example = generate_passkey_example(
                tokenizer=tokenizer,
                passkey=passkey,
                depth_percent=depth,
                target_length=target_length,
                is_eval=False  # Training data includes answer
            )
            
            examples.append(example)
    
    # Shuffle examples
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
    parser = argparse.ArgumentParser(description="Generate passkey retrieval finetuning dataset")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="lvwerra/the-tokenizer-v1",
        help="Path to tokenizer (same as used in training)"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=2000,
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
    
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    print(f"Generating {args.num_examples} examples with ~{args.target_length} tokens each...")
    dataset = generate_dataset(
        tokenizer=tokenizer,
        num_examples=args.num_examples,
        target_length=args.target_length,
        seed=args.seed
    )
    
    # Save dataset
    print(f"Saving dataset to {args.save_path}...")
    dataset.save_to_disk(args.save_path)
    
    # Also save as parquet for easier loading
    parquet_path = f"{args.save_path}.parquet"
    dataset.to_parquet(parquet_path)
    print(f"Also saved as parquet: {parquet_path}")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total examples: {len(dataset)}")
    print(f"Average token count: {sum(dataset['token_count']) / len(dataset):.1f}")
    print(f"Min token count: {min(dataset['token_count'])}")
    print(f"Max token count: {max(dataset['token_count'])}")
    
    # Show depth distribution
    from collections import Counter
    depth_counts = Counter(dataset['depth_percent'])
    print("\nDepth distribution:")
    for depth in sorted(depth_counts.keys()):
        print(f"  {depth}%: {depth_counts[depth]} examples")
    
    # Show a sample
    print("\nSample example (first 500 chars):")
    print(dataset['prompt'][0][:500] + "...")
    print(f"Answer: {dataset['answer'][0]}")
    
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.hub_repo}...")
        dataset.push_to_hub(args.hub_repo)
        print("Dataset pushed successfully!")


if __name__ == "__main__":
    main()