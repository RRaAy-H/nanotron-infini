#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/prepare_data.py

import pandas as pd
import os
import glob
from datasets import Dataset, DatasetDict
import traceback

def prepare_data():
    """Prepare the parquet data file for training with nanotron."""
    data_dir = "/Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/data"
    output_dir = "/Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/data/processed"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"Loading parquet files from {data_dir}")
        
        # Look for all parquet files in the data directory
        parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
        print(f"Found {len(parquet_files)} parquet files: {parquet_files}")
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")
        
        # Read all parquet files and concatenate them
        dfs = []
        for file in parquet_files:
            try:
                print(f"Reading file: {file}")
                df = pd.read_parquet(file)
                dfs.append(df)
                print(f"File {file} columns: {df.columns.tolist()}")
            except Exception as e:
                print(f"Error reading {file}: {e}")
                traceback.print_exc()
        
        if not dfs:
            raise ValueError("Could not read any parquet files")
        
        df = pd.concat(dfs, ignore_index=True)
        
        # Ensure 'text' column exists
        if 'text' not in df.columns:
            # Try to find a suitable text column
            text_like_columns = [col for col in df.columns if any(name in col.lower() for name in ['text', 'content', 'body', 'message'])]
            if text_like_columns:
                print(f"Using column '{text_like_columns[0]}' as text")
                df['text'] = df[text_like_columns[0]]
            else:
                raise ValueError(f"No 'text' column found in the data. Available columns: {df.columns.tolist()}")
        
        # Keep only necessary columns and filter out any empty texts
        df = df[["text"]].copy()  # Create a copy to avoid SettingWithCopyWarning
        
        # Handle missing values
        df = df.dropna(subset=['text'])
        
        # Remove any rows with empty text
        df['text'] = df['text'].astype(str).str.strip()
        df = df[df["text"] != ""]
        
        print(f"Data shape after cleaning: {df.shape}")
        print("Sample text:")
        if not df.empty:
            sample_text = df["text"].iloc[0]
            print(sample_text[:500] + ("..." if len(sample_text) > 500 else ""))
        else:
            print("No data available after cleaning.")
        
        # Create train/validation split
        train_df = df.sample(frac=0.95, random_state=42)
        val_df = df.drop(train_df.index)
        
        # Convert to Hugging Face dataset
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        # Create a DatasetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
        
        # Save as Arrow dataset
        dataset_dict.save_to_disk(output_dir)
        
        print(f"Dataset saved to {output_dir}")
        print("Dataset info:")
        print(dataset_dict)
    except Exception as e:
        print(f"Error processing data: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    prepare_data()
