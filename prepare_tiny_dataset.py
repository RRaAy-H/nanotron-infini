#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/prepare_tiny_dataset.py

"""
Dataset preparation script for the Infini-Llama model training.

This script checks the structure of the tiny dataset and prepares it for use with the Infini-Llama model.
It ensures the dataset has the correct structure and adds any missing columns.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add the project root and src directories to Python path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, os.path.join(str(root_dir), 'src'))

try:
    import datasets
    from transformers import AutoTokenizer
    from datasets import Dataset, DatasetDict, load_from_disk
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install the required packages with: pip install datasets transformers pandas numpy")
    sys.exit(1)

def prepare_dataset(data_dir, tokenizer_name="meta-llama/Llama-2-7b-hf", force=False):
    """
    Prepare the dataset for training with Infini-Llama.
    
    Args:
        data_dir: Path to the dataset directory
        tokenizer_name: Name of the tokenizer to use
        force: Force reprocessing even if the dataset already seems prepared
    """
    data_path = Path(data_dir)
    
    print(f"Checking dataset at: {data_path}")
    
    # Check if the directory exists
    if not data_path.exists():
        print(f"Error: Dataset directory {data_dir} does not exist.")
        return False
    
    try:
        # Try to load the dataset
        print(f"Loading dataset from {data_path}...")
        try:
            dataset = load_from_disk(str(data_path))
            print(f"Dataset loaded successfully: {type(dataset)}")
        except Exception as e:
            print(f"Could not load dataset from {data_path}: {e}")
            print("Checking for Arrow files...")
            arrow_files = list(data_path.glob("*.arrow"))
            if arrow_files:
                print(f"Found {len(arrow_files)} arrow files.")
                try:
                    # Try to load the Arrow files directly
                    from datasets import load_dataset
                    dataset = load_dataset("arrow", data_files=[str(f) for f in arrow_files])
                    print(f"Successfully loaded Arrow files: {dataset}")
                except Exception as arrow_e:
                    print(f"Error loading Arrow files directly: {arrow_e}")
                    # Try a different approach with pyarrow
                    try:
                        import pyarrow as pa
                        import pyarrow.parquet as pq
                        dfs = []
                        for file in arrow_files:
                            try:
                                # Read Arrow file
                                reader = pa.ipc.open_file(file)
                                table = reader.read_all()
                                df = table.to_pandas()
                                dfs.append(df)
                            except Exception as e:
                                print(f"Error loading {file}: {e}")
                        
                        if dfs:
                            # Concatenate the dataframes
                            df = pd.concat(dfs)
                            # Create a dataset
                            dataset = Dataset.from_pandas(df)
                            print(f"Created dataset from Arrow files: {len(dataset)} examples")
                        else:
                            print("No valid Arrow files could be processed.")
                            return False
                    except ImportError:
                        print("PyArrow not available. Trying parquet files instead...")
                        # Fall back to parquet files
                        parquet_files = list(data_path.glob("*.parquet"))
                        if parquet_files:
                            print(f"Found {len(parquet_files)} parquet files.")
                            # Load the parquet files
                            dfs = []
                            for file in parquet_files:
                                try:
                                    df = pd.read_parquet(file)
                                    dfs.append(df)
                                except Exception as e:
                                    print(f"Error loading {file}: {e}")
                            if dfs:
                                # Concatenate the dataframes
                                df = pd.concat(dfs)
                                # Create a dataset
                                dataset = Dataset.from_pandas(df)
                                print(f"Created dataset from parquet files: {len(dataset)} examples")
                            else:
                                print("No valid parquet files found.")
                                return False
                        else:
                            print("No parquet files found.")
                            return False
            else:
                print("No Arrow files found. Checking for parquet files...")
                parquet_files = list(data_path.glob("*.parquet"))
                if parquet_files:
                    print(f"Found {len(parquet_files)} parquet files.")
                    # Load the parquet files
                    dfs = []
                    for file in parquet_files:
                        try:
                            df = pd.read_parquet(file)
                            dfs.append(df)
                        except Exception as e:
                            print(f"Error loading {file}: {e}")
                    if dfs:
                        # Concatenate the dataframes
                        df = pd.concat(dfs)
                        # Create a dataset
                        dataset = Dataset.from_pandas(df)
                        print(f"Created dataset from parquet files: {len(dataset)} examples")
                    else:
                        print("No valid parquet files found.")
                        return False
                else:
                    print("No Arrow or parquet files found.")
                    return False
        
        # Check if this is a DatasetDict or a Dataset
        if isinstance(dataset, DatasetDict):
            print(f"Dataset splits: {list(dataset.keys())}")
            # Check if there's a train split
            if "train" not in dataset:
                print("Warning: No 'train' split found. Creating one from available splits.")
                # Create a train split from the first available split
                first_split = next(iter(dataset.values()))
                dataset = DatasetDict({"train": first_split})
        else:
            # If it's just a Dataset, wrap it in a DatasetDict with a train split
            print("Converting Dataset to DatasetDict with 'train' split")
            dataset = DatasetDict({"train": dataset})
        
        # Check if the dataset has the text column
        if "text" not in dataset["train"].column_names:
            print("Warning: No 'text' column found.")
            
            # Try to find a suitable text column
            potential_text_columns = ["content", "sentence", "input_text", "document", "article"]
            text_column = None
            
            for col in dataset["train"].column_names:
                if col.lower() in potential_text_columns or "text" in col.lower():
                    text_column = col
                    break
            
            if text_column:
                print(f"Using '{text_column}' as text column")
                # Rename the column to 'text'
                dataset["train"] = dataset["train"].rename_column(text_column, "text")
            else:
                # If no suitable column is found, look for the first string column
                for col in dataset["train"].column_names:
                    if dataset["train"].features[col].dtype == "string":
                        print(f"Using string column '{col}' as text column")
                        dataset["train"] = dataset["train"].rename_column(col, "text")
                        break
                else:
                    print("Error: Could not find a suitable text column.")
                    return False
        
        # Check if the dataset has been tokenized
        if "input_ids" not in dataset["train"].column_names or force:
            print("Dataset needs tokenization. Loading tokenizer...")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            # Process the dataset
            def tokenize_function(examples):
                return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=2048)
            
            print("Tokenizing dataset...")
            tokenized_dataset = dataset.map(
                tokenize_function, 
                batched=True, 
                num_proc=4,
                remove_columns=[col for col in dataset["train"].column_names if col != "text"]
            )
            
            # Save the tokenizer
            tokenizer_path = data_path / "tokenizer"
            if not tokenizer_path.exists() or force:
                print(f"Saving tokenizer to {tokenizer_path}...")
                tokenizer.save_pretrained(str(tokenizer_path))
            
            # Save the processed dataset
            processed_dataset_path = data_path / "processed_dataset"
            print(f"Saving processed dataset to {processed_dataset_path}...")
            tokenized_dataset.save_to_disk(str(processed_dataset_path))
            
            # Save metadata
            metadata = {
                "tokenizer": tokenizer_name,
                "max_length": 2048,
                "processed_date": str(pd.Timestamp.now())
            }
            metadata_path = data_path / "metadata.json"
            print(f"Saving metadata to {metadata_path}...")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            print("Dataset preparation completed successfully!")
            return True
        else:
            print("Dataset already has input_ids column. Skipping tokenization.")
            
            # Save metadata if it doesn't exist
            metadata_path = data_path / "metadata.json"
            if not metadata_path.exists() or force:
                metadata = {
                    "tokenizer": tokenizer_name,
                    "max_length": 2048,
                    "processed_date": str(pd.Timestamp.now())
                }
                print(f"Saving metadata to {metadata_path}...")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
            
            print("Dataset already prepared.")
            return True
            
    except Exception as e:
        import traceback
        print(f"Error preparing dataset: {e}")
        print(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Infini-Llama training")
    parser.add_argument("--data-dir", type=str, required=True,
                      help="Path to the dataset directory")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf",
                      help="Name of the tokenizer to use")
    parser.add_argument("--force", action="store_true",
                      help="Force reprocessing even if the dataset already seems prepared")
    args = parser.parse_args()
    
    success = prepare_dataset(args.data_dir, args.tokenizer, args.force)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
