# Local Dataset Integration for Nanotron

This document explains the modifications made to support local datasets (specifically your FineWeb parquet files) with Nanotron's training pipeline.

## 🎯 What Was Modified

### 1. Configuration Support (`src/nanotron/config/config.py`)
**Added fields to `PretrainDatasetsArgs` class:**
```python
data_dir: Optional[str] = None  # Directory containing local dataset files
data_files: Optional[Union[str, list]] = None  # Specific files or patterns
```

### 2. Dataset Loading (`src/nanotron/dataloader.py`)
**Enhanced `get_datasets` function to:**
- Accept `data_dir` and `data_files` parameters
- Automatically detect local vs remote datasets
- Handle local parquet files with proper error handling
- Maintain backward compatibility with existing configs

### 3. Training Pipeline (`run_train.py`)
**Updated dataset loading call to:**
- Pass new local dataset parameters from config
- Use `getattr` for backward compatibility

## 🚀 Ready-to-Use Files Created

### 1. `fineweb_local_200m_infini_config.yaml`
Complete configuration for your local FineWeb dataset:
- 200M parameter Llama model
- Infini-attention with 64 segment length
- Optimized for single GPU training
- Uses your local path: `data1/dataset/HuggingFaceFW/fineweb/`

### 2. `test_local_fineweb.py`
Comprehensive test script that verifies:
- Configuration loading
- Local dataset access
- HuggingFace dataset integration
- Nanotron integration
- Tokenizer functionality

## 📋 How to Use

### Step 1: Test Your Setup
```bash
python test_local_fineweb.py
```

This will verify everything is working correctly.

### Step 2: Adjust Configuration (if needed)
Edit `fineweb_local_200m_infini_config.yaml`:

**For your GPU setup:**
```yaml
parallelism:
  dp: 1  # Single GPU
  # dp: 4  # For 4 GPUs
  
tokens:
  micro_batch_size: 4  # Adjust based on GPU memory
```

**For different dataset path:**
```yaml
data_stages:
  - name: "Local FineWeb Training Stage"
    data:
      dataset:
        data_dir: "your/path/to/fineweb/"  # Update this
```

### Step 3: Start Training
```bash
# Single GPU
python run_train.py --config-file fineweb_local_200m_infini_config.yaml

# Multi-GPU (adjust nproc_per_node to your GPU count)
torchrun --nproc_per_node=4 run_train.py --config-file fineweb_local_200m_infini_config.yaml
```

## 🔧 Configuration Details

### Local Dataset Configuration
```yaml
data_stages:
  - name: "Local FineWeb Training Stage"
    start_training_step: 1
    data:
      dataset:
        hf_dataset_or_datasets: "parquet"  # Use parquet loader
        data_dir: "data1/dataset/HuggingFaceFW/fineweb/"  # Your local path
        text_column_name: "text"  # FineWeb text column
```

### Infini-Attention Configuration
```yaml
infini_attention:
  segment_length: 64          # Segment size for infini-attention
  turn_on_memory: true        # Enable infini-attention memory
  balance_factor_lr: 0.00015  # Learning rate for balance factors
  balance_act_type: hard_sigmoid  # Activation for balance factors
  balance_init_type: zeros    # Initialization for balance factors
```

### Model Configuration (200M Parameters)
```yaml
model:
  model_config:
    hidden_size: 1024
    intermediate_size: 4096
    num_hidden_layers: 6
    num_attention_heads: 8
    max_position_embeddings: 256
    vocab_size: 49152
```

## 🧪 Verification Commands

```bash
# Test dataset loading
python -c "
from datasets import load_dataset
dataset = load_dataset('parquet', data_dir='data1/dataset/HuggingFaceFW/fineweb/', split='train[:10]')
print(f'✅ Loaded {len(dataset)} samples with columns: {dataset.column_names}')
"

# Test configuration
python -c "
from nanotron.config import get_config_from_file
config = get_config_from_file('fineweb_local_200m_infini_config.yaml')
print('✅ Configuration loaded successfully')
print(f'Dataset path: {config.data_stages[0].data.dataset.data_dir}')
"

# Test nanotron integration
python -c "
from nanotron.dataloader import get_datasets
raw_datasets = get_datasets(
    hf_dataset_or_datasets='parquet',
    data_dir='data1/dataset/HuggingFaceFW/fineweb/',
    splits='train'
)
print(f'✅ Nanotron integration works: {len(raw_datasets[\"train\"])} samples')
"
```

## 🔥 Key Benefits

✅ **No Pre-tokenization Required**: Uses on-the-fly tokenization  
✅ **Direct Local File Access**: No network dependency during training  
✅ **Backward Compatible**: Existing configs still work  
✅ **Infini-Attention Ready**: Includes complete infini-attention setup  
✅ **200M Model Optimized**: Based on proven exp53 configuration  
✅ **GPU Flexible**: Easy to adjust for different GPU setups  

## 🐛 Troubleshooting

### Dataset Path Issues
```bash
# Check if files exist
ls -la data1/dataset/HuggingFaceFW/fineweb/*.parquet | head -5

# Check file count
ls data1/dataset/HuggingFaceFW/fineweb/*.parquet | wc -l
```

### Memory Issues
```bash
# Reduce batch size in config
tokens:
  micro_batch_size: 2  # Reduce from 4

# Enable gradient checkpointing
model:
  ddp_bucket_cap_mb: 10  # Reduce from 25
```

### GPU Issues
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

## 📝 What's Next

1. **Run the test script** to verify everything works
2. **Start with a small training run** (reduce `train_steps` to 100 for testing)
3. **Monitor GPU usage** and adjust `micro_batch_size` if needed
4. **Scale up** training steps once everything is working

Your local FineWeb dataset is now fully integrated with Nanotron's training pipeline! 🎉