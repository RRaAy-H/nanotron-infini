#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/parquet_data_loader.py

"""
Parquet Data Loader for Infini-Llama Training

This script handles loading parquet files from a directory and preparing them
for training with Infini-Llama. It integrates with the standard preprocessing
and training pipelines.

Usage:
```
python parquet_data_loader.py --data-dir /path/to/parquet/files --output-dir processed_data
```
"""

import os
import sys
import time
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Union

# Add the project root to Python path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

# Import datasets
try:
    from datasets import load_dataset, Dataset, DatasetDict
except ImportError:
    print("Error: Could not import datasets library. Please install it with: pip install datasets")
    sys.exit(1)

# Import tokenizer
try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: Could not import transformers library. Please install it with: pip install transformers")
    sys.exit(1)

def load_parquet_datasets(parquet_dir: str, split: str = "train") -> Dataset:
    """
    Load parquet files from a directory into a Dataset.
    
    Args:
        parquet_dir: Directory containing parquet files
        split: Dataset split name to use
        
    Returns:
        Dataset object containing the data from parquet files
    """
    print(f"Loading parquet files from {parquet_dir}")
    
    # Find all parquet files in the directory
    parquet_files = list(Path(parquet_dir).glob("*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {parquet_dir}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Load the dataset
    try:
        dataset = load_dataset(
            "parquet", 
            data_files=[str(f) for f in parquet_files],
            split=split
        )
        print(f"Loaded dataset with {len(dataset)} examples")
        
        # Create dataset dict with train split
        dataset_dict = DatasetDict({split: dataset})
        return dataset_dict
    except Exception as e:
        print(f"Error loading parquet files: {e}")
        raise

def preprocess_dataset(
    dataset: Union[Dataset, DatasetDict],
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf", 
    max_seq_length: int = 2048,
    output_dir: str = "processed_data",
    use_gpu: bool = True
) -> str:
    """
    Tokenize and preprocess a dataset for training.
    
    Args:
        dataset: The dataset to process
        tokenizer_name: The name or path of the tokenizer to use
        max_seq_length: The maximum sequence length to use for tokenization
        output_dir: Directory to save processed data
        use_gpu: Whether to use GPU for processing
        
    Returns:
        Path to the processed dataset
    """
    print(f"Processing dataset with tokenizer {tokenizer_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Process with the right device
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"preprocessed_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    
    # Check if the dataset has the required text column
    if "text" not in dataset["train"].column_names:
        print(f"Warning: 'text' column not found in dataset. Available columns: {dataset['train'].column_names}")
        # Try to find a suitable text column
        text_candidates = ["content", "document", "sentence", "input_text"]
        found_column = None
        for column in text_candidates:
            if column in dataset["train"].column_names:
                found_column = column
                break
        
        if found_column:
            print(f"Using '{found_column}' as the text column")
            # Rename the column to 'text'
            dataset["train"] = dataset["train"].rename_column(found_column, "text")
        else:
            raise ValueError("Could not find a suitable text column in the dataset")
    
    # Tokenize the dataset
    start_time = time.time()
    print("Tokenizing dataset...")
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"], 
            truncation=True,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_special_tokens_mask=False,
            return_token_type_ids=False,
        )
        return tokenized

    # Process dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing",
        remove_columns=dataset["train"].column_names
    )
    
    # Save tokenizer
    tokenizer_path = os.path.join(output_path, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path}")
    
    # Save dataset
    dataset_path = os.path.join(output_path, "train_dataset")
    tokenized_dataset["train"].save_to_disk(dataset_path)
    print(f"Saved tokenized dataset to {dataset_path}")
    
    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "tokenizer": tokenizer_name,
        "max_seq_length": max_seq_length,
        "num_examples": len(tokenized_dataset["train"]),
        "data_source": "parquet_files",
        "preprocessed": True
    }
    
    with open(os.path.join(output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    elapsed_time = time.time() - start_time
    print(f"Dataset processing completed in {elapsed_time:.2f} seconds")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Load and preprocess parquet data for Infini-Llama training")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing parquet files")
    parser.add_argument("--output-dir", type=str, default="processed_data", help="Directory to save preprocessed data")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf", help="Tokenizer name or path")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Maximum sequence length for tokenization")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration for preprocessing")
    parser.add_argument("--split", type=str, default="train", help="Dataset split name to use")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load parquet datasets
    dataset = load_parquet_datasets(args.data_dir, split=args.split)
    
    # Preprocess dataset
    output_path = preprocess_dataset(
        dataset=dataset,
        tokenizer_name=args.tokenizer,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        use_gpu=not args.no_gpu
    )
    
    print("\nPreprocessing completed successfully!")
    print(f"Output directory: {output_path}")
    print("You can now train using the preprocessed data with:")
    print(f"  python scripts/training/train_infini_llama.py --config-file custom_infini_config_gpu.yaml --data-dir {output_path}")

if __name__ == "__main__":
    main()
