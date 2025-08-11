#!/usr/bin/env python3
"""
Test script to verify the local FineWeb dataset integration works correctly.
This script tests:
1. Loading the modified configuration
2. Loading the local dataset
3. Verifying the dataset structure
4. Testing tokenization
"""

import os
import sys
import traceback

def test_configuration_loading():
    """Test that the configuration file loads correctly."""
    print("=" * 60)
    print("TEST 1: Configuration Loading")
    print("=" * 60)
    
    try:
        from nanotron.config import get_config_from_file
        
        config_path = "fineweb_local_200m_infini_config.yaml"
        if not os.path.exists(config_path):
            print(f"❌ Error: Config file not found: {config_path}")
            return False
            
        config = get_config_from_file(config_path)
        print("✅ Configuration loaded successfully!")
        
        # Check key configuration elements
        print(f"Model hidden size: {config.model.model_config.hidden_size}")
        print(f"Model layers: {config.model.model_config.num_hidden_layers}")
        print(f"Infini-attention segment length: {config.infini_attention.segment_length}")
        print(f"Infini-attention memory enabled: {config.infini_attention.turn_on_memory}")
        print(f"Dataset path: {config.data_stages[0].data.dataset.data_dir}")
        print(f"Text column: {config.data_stages[0].data.dataset.text_column_name}")
        print(f"Tokenizer: {config.tokenizer.tokenizer_name_or_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        traceback.print_exc()
        return False

def test_local_dataset_access():
    """Test that the local dataset can be accessed."""
    print("\n" + "=" * 60)
    print("TEST 2: Local Dataset Access")
    print("=" * 60)
    
    try:
        import glob
        
        # Check if dataset directory exists
        dataset_path = "data1/dataset/HuggingFaceFW/fineweb/"
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset directory not found: {dataset_path}")
            print(f"Please update the path in fineweb_local_200m_infini_config.yaml")
            return False
            
        # Check for parquet files
        parquet_files = glob.glob(os.path.join(dataset_path, "*.parquet"))
        print(f"Found {len(parquet_files)} parquet files in {dataset_path}")
        
        if len(parquet_files) == 0:
            print("❌ No parquet files found in dataset directory")
            return False
            
        # List first few files
        print("Sample files:")
        for i, file in enumerate(parquet_files[:5]):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"  {os.path.basename(file)}: {size_mb:.1f} MB")
            
        return True
        
    except Exception as e:
        print(f"❌ Error accessing local dataset: {e}")
        traceback.print_exc()
        return False

def test_dataset_loading():
    """Test that the dataset can be loaded with HuggingFace datasets."""
    print("\n" + "=" * 60)
    print("TEST 3: Dataset Loading with HuggingFace")
    print("=" * 60)
    
    try:
        from datasets import load_dataset
        
        dataset_path = "data1/dataset/HuggingFaceFW/fineweb/"
        
        # Test loading a small sample
        print("Loading small sample (10 examples)...")
        dataset = load_dataset("parquet", data_dir=dataset_path, split="train[:10]")
        
        print(f"✅ Dataset loaded successfully!")
        print(f"Number of samples: {len(dataset)}")
        print(f"Columns: {dataset.column_names}")
        
        # Check text column
        if "text" in dataset.column_names:
            sample_text = dataset[0]["text"]
            print(f"Sample text (first 200 chars): {sample_text[:200]}...")
            print(f"Sample text length: {len(sample_text)} characters")
        else:
            print(f"❌ Warning: 'text' column not found. Available columns: {dataset.column_names}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        traceback.print_exc()
        return False

def test_nanotron_integration():
    """Test that the dataset loads through nanotron's get_datasets function."""
    print("\n" + "=" * 60)
    print("TEST 4: Nanotron Integration")
    print("=" * 60)
    
    try:
        from nanotron.dataloader import get_datasets
        
        dataset_path = "data1/dataset/HuggingFaceFW/fineweb/"
        
        print("Testing nanotron's get_datasets function...")
        raw_datasets = get_datasets(
            hf_dataset_or_datasets="parquet",
            hf_dataset_config_name=None,
            splits="train",
            data_dir=dataset_path,
            data_files=None,
        )
        
        train_dataset = raw_datasets["train"]
        print(f"✅ Nanotron integration successful!")
        print(f"Dataset type: {type(train_dataset)}")
        print(f"Number of samples: {len(train_dataset)}")
        print(f"Columns: {train_dataset.column_names}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with nanotron integration: {e}")
        traceback.print_exc()
        return False

def test_tokenizer():
    """Test that the tokenizer works correctly."""
    print("\n" + "=" * 60)
    print("TEST 5: Tokenizer Test")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer
        
        tokenizer_name = "lvwerra/the-tokenizer-v1"
        print(f"Loading tokenizer: {tokenizer_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        # Test tokenization
        test_text = "Hello world! This is a test of the tokenizer."
        tokens = tokenizer(test_text)
        
        print(f"✅ Tokenizer loaded successfully!")
        print(f"Test text: {test_text}")
        print(f"Tokens: {tokens['input_ids']}")
        print(f"Number of tokens: {len(tokens['input_ids'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with tokenizer: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Local FineWeb Dataset Integration")
    print("This script verifies that all modifications work correctly.\n")
    
    tests = [
        test_configuration_loading,
        test_local_dataset_access,
        test_dataset_loading,
        test_nanotron_integration,
        test_tokenizer,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    test_names = [
        "Configuration Loading",
        "Local Dataset Access", 
        "Dataset Loading",
        "Nanotron Integration",
        "Tokenizer Test"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! You're ready to start training with:")
        print("python run_train.py --config-file fineweb_local_200m_infini_config.yaml")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the issues above.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)