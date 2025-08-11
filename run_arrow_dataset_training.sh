#!/usr/bin/env bash
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/run_arrow_dataset_training.sh

# Script to train the Infini-Llama model with an Arrow dataset

set -e  # Exit on any error

# Set the correct working directory
WORKING_DIR="/home/data/daal_insight/fiery/Infini-attention/nanotron-infini"

# Go to the working directory
cd "$WORKING_DIR" || { echo "Cannot change to working directory $WORKING_DIR"; exit 1; }
echo "Changed to working directory: $(pwd)"

# Set up environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Install required packages if needed
pip install -q pyarrow datasets

echo "======= Infini-Llama Training with Arrow Dataset ======="
echo "Current directory: $(pwd)"
echo "Python path: $PYTHONPATH"

# Data directory to use
DATA_DIR="/data1/dataset/HuggingFaceFW/processed/tiny"
echo "Using Arrow dataset directory: $DATA_DIR"

# Create directories if they don't exist
mkdir -p tensorboard_logs
mkdir -p infini_llama_checkpoints
mkdir -p models

# Check if we need to install additional dependencies
if ! python -c "import pyarrow" &> /dev/null; then
    echo "PyArrow not found, installing..."
    pip install pyarrow
fi

# Step 1: Check and prepare the dataset
echo ""
echo "Step 1: Checking and preparing Arrow dataset..."
python - <<EOF
import os
import sys
from pathlib import Path
import json

# Add the project root and src directories to Python path
root_dir = "$WORKING_DIR"
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

try:
    import datasets
    from transformers import AutoTokenizer
    from datasets import Dataset, DatasetDict, load_dataset
    
    # Data directory to use
    data_dir = "$DATA_DIR"
    data_path = Path(data_dir)
    
    print(f"Checking Arrow dataset at: {data_path}")
    
    # Check for Arrow files
    arrow_files = list(data_path.glob("*.arrow"))
    if arrow_files:
        print(f"Found {len(arrow_files)} Arrow files.")
        
        # Try to load the Arrow files
        try:
            # Load directly using load_dataset
            dataset = load_dataset("arrow", data_files=[str(f) for f in arrow_files])
            print(f"Successfully loaded Arrow dataset: {dataset}")
            
            # Check if we have a train split
            if "train" not in dataset:
                print("Creating 'train' split from the dataset")
                dataset = DatasetDict({"train": dataset["train" if "train" in dataset else list(dataset.keys())[0]]})
            
            # Save as a datasets format
            output_path = data_path / "processed_dataset"
            print(f"Saving processed dataset to {output_path}")
            dataset.save_to_disk(str(output_path))
            
            # Create metadata
            metadata = {
                "tokenizer": "meta-llama/Llama-2-7b-hf",
                "processed_date": str(datasets.utils.datetime.datetime.now())
            }
            
            # Save metadata
            metadata_path = data_path / "metadata.json"
            print(f"Saving metadata to {metadata_path}")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            print("Dataset preparation completed successfully!")
        except Exception as e:
            print(f"Error processing Arrow files: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("No Arrow files found. Please check your dataset path.")
        sys.exit(1)
except Exception as e:
    print(f"Error importing or processing dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "Arrow dataset preparation failed. Exiting."
    exit 1
fi

echo "Arrow dataset preparation completed successfully."

# Step 2: Run the training script
echo ""
echo "Step 2: Running the training script..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null
then
    echo "CUDA is available. Using GPU for training."
    GPU_FLAG=""
else
    echo "CUDA is not available. Using CPU for training."
    GPU_FLAG="--cpu-only"
fi

# Run the fixed training script with the custom configuration
echo "Starting training with Arrow dataset..."
python train_infini_llama_fixed.py \
    --config-file custom_tiny_infini_config.yaml \
    --data-dir "$DATA_DIR/processed_dataset" \
    --tensorboard-dir tensorboard_logs \
    --use-gpu-dataloader \
    --verbose \
    $GPU_FLAG

if [ $? -ne 0 ]; then
    echo "Training failed. Exiting."
    exit 1
fi

echo "Training completed successfully!"
echo "========================================"
echo "The entire pipeline has completed successfully!"
echo "Checkpoints saved in: $(pwd)/infini_llama_checkpoints"
echo "TensorBoard logs saved in: $(pwd)/tensorboard_logs"
