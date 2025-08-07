# Training Llama with Infi-Attention

This document provides step-by-step instructions for training the Llama model with Infi-Attention using your custom dataset.

## Files Created for Training

1. **`prepare_data.py`**: Script to process your parquet dataset for training
2. **`custom_infini_config.yaml`**: Standard configuration file for training
3. **`custom_infini_config_cpu.yaml`**: Configuration file for CPU training (smaller model)
4. **`custom_infini_config_gpu.yaml`**: Optimized configuration file for GPU training
5. **`run_train_cpu.py`**: Modified training script compatible with CPU
6. **`train_infini_llama_cpu.sh`**: Shell script to run the CPU training
7. **`train_gpu_with_tensorboard.py`**: Script for GPU training with TensorBoard integration
8. **`train_infini_llama_gpu.sh`**: Shell script to run GPU training with TensorBoard
9. **`TRAINING_GUIDE.md`**: Detailed guide with troubleshooting tips

## Steps to Train the Model

### Option 1: Using a Conda Environment (Recommended)

1. Create a conda environment with Python 3.10:

```bash
conda create -y -n infi-llama python=3.10
conda activate infi-llama
```

2. Install dependencies:

```bash
cd /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini
pip install -e .
pip install datasets transformers huggingface_hub pyarrow pandas
```

3. Prepare the dataset:

```bash
python prepare_data.py
```

4. Make the training script executable:

```bash
chmod +x train_infini_llama_cpu.sh
```

5. Run the training script:

```bash
./train_infini_llama_cpu.sh
```

### Option 2: Direct Execution

If you prefer not to use conda, you can try running the scripts directly:

1. Prepare the dataset:

```bash
python prepare_data.py
```

2. Run the CPU training script:

```bash
export CUDA_VISIBLE_DEVICES=""
python run_train_cpu.py --config-file custom_infini_config_cpu.yaml
```

## Model Configuration

The Infi-Attention model is configured with:
- Segment length: 64
- Memory enabled: True
- Balance initialization: zeros
- Balance activation: orig_sigmoid

## Modifying Model Size

For faster training or to accommodate resource limitations, you can modify:
- `hidden_size`: Controls the dimensionality of hidden layers
- `num_hidden_layers`: Controls the depth of the model
- `num_attention_heads`: Controls the number of attention heads
- `intermediate_size`: Controls the size of feed-forward network

## Using GPU with TensorBoard (Recommended)

For faster training and better monitoring, we've provided a dedicated script that uses GPU acceleration and TensorBoard for tracking training metrics:

### Step 1: Prepare for GPU Training

1. Install additional dependencies:
   ```bash
   pip install torch>=1.13.1 flash-attn>=2.5.0 tensorboard torchvision tqdm
   ```

2. Ensure your dataset is prepared:
   ```bash
   python prepare_data.py
   ```

### Step 2: Run Training with GPU

1. Make the GPU training script executable:
   ```bash
   chmod +x train_infini_llama_gpu.sh
   ```

2. Run the GPU training script, specifying which GPU to use (0 by default):
   ```bash
   ./train_infini_llama_gpu.sh 0  # Use GPU 0
   ```
   
   You can also specify a custom TensorBoard log directory:
   ```bash
   ./train_infini_llama_gpu.sh 0 /path/to/tensorboard_logs
   ```

3. The script will automatically:
   - Set the CUDA_VISIBLE_DEVICES environment variable
   - Prepare the dataset if not already done
   - Start TensorBoard in the background
   - Launch the training process with GPU acceleration

4. TensorBoard will automatically start at http://localhost:6006, allowing you to monitor in real-time:
   - Training loss (per step)
   - Learning rate schedule
   - Model parameters and gradients
   - Training throughput (samples/sec)
   - Elapsed training time
   - Validation metrics (when validation runs)

5. When training is complete, you can view the logs later with:
   ```bash
   tensorboard --logdir=/path/to/tensorboard_logs
   ```

### Step 3: Understanding the TensorBoard Interface

The TensorBoard dashboard provides several tabs for monitoring your training:

- **SCALARS**: Shows graphs of metrics over time (loss, learning rate, throughput)
- **IMAGES**: Any visualizations generated during training
- **GRAPHS**: Model architecture visualization
- **DISTRIBUTIONS/HISTOGRAMS**: Parameter value distributions over time
- **TIME SERIES**: Detailed time series data for selected metrics

### Step 4: Analyzing Training Results

Look for these signs of successful training:

1. **Decreasing Loss**: The training loss should generally decrease over time.
2. **Learning Rate Schedule**: Should follow the configured warmup and decay pattern.
3. **Throughput**: Higher values indicate efficient GPU utilization.
4. **Validation Metrics**: Should improve as training progresses.

### GPU Configuration Details

The GPU configuration file (`custom_infini_config_gpu.yaml`) is optimized for modern GPUs with:

#### 1. Performance Optimizations
- **Mixed Precision**: Uses `bfloat16` data type to reduce memory usage and increase training speed
- **Fused Optimizer**: Enables `torch_adam_is_fused: true` for faster optimizer operations
- **Flash Attention**: Leverages optimized attention implementation for better throughput
- **Parallel Processing**: Configured for optimal tensor parallelism with `tp_linear_async_communication: true`

#### 2. Model Architecture
- **Full-Size Model**: Uses larger dimensions (`hidden_size: 1024`, `intermediate_size: 4096`)
- **Attention Configuration**: 8 attention heads with 8 key-value heads
- **Infini-Attention Settings**:
  - `segment_length: 64`
  - `turn_on_memory: true`
  - `balance_init_type: zeros`
  - `balance_act_type: orig_sigmoid`

#### 3. Training Hyperparameters
- **Batch Size**: Increased `micro_batch_size: 4` for better GPU utilization
- **Learning Rate**: Optimized learning rate schedule with warmup
- **Sequence Length**: 256 tokens per training example
- **Validation Frequency**: Checks validation metrics every 500 steps
- **Checkpoint Interval**: Saves model checkpoints every 1000 steps

#### 4. Data Processing
- **Parallel Processing**: Uses multiple workers for dataset processing and loading
- **Tokenization**: Configured to use the Meta Llama-2-7b tokenizer
- **Gradient Clipping**: Set to 1.0 to prevent exploding gradients

## Monitoring Training with TensorBoard

TensorBoard provides powerful visualization tools for tracking your model's training progress. When you run the GPU training script, TensorBoard automatically starts at http://localhost:6006.

### Key Metrics to Monitor

1. **Training Loss**
   - Should steadily decrease over time
   - Sudden spikes may indicate issues with learning rate or data
   - Plateaus might suggest need for learning rate adjustment

2. **Learning Rate**
   - Visualizes the warmup and decay schedule
   - Confirms proper implementation of the learning rate scheduler

3. **Throughput (Samples/sec)**
   - Higher values indicate better GPU utilization
   - Consistent values suggest stable training process
   - Drops may indicate resource contention or data loading bottlenecks

4. **Elapsed Training Time**
   - Helps estimate total training duration
   - Useful for planning and resource allocation

5. **Validation Metrics**
   - Loss and perplexity on validation set
   - Key indicators of model generalization
   - Should be checked regularly to detect overfitting

### Using TensorBoard Effectively

1. **Real-time Monitoring**:
   - Keep TensorBoard open in a browser tab during training
   - Refresh periodically to see the latest metrics

2. **Comparing Runs**:
   - TensorBoard allows comparing multiple training runs
   - Use different directories for different configurations
   - Helpful for hyperparameter optimization

3. **Sharing Results**:
   - Export graphs as PNG files for documentation
   - Save TensorBoard logs for future reference

4. **Debugging Training Issues**:
   - Use parameter histograms to detect weight saturation
   - Check gradient norms for explosion/vanishing problems
   - Monitor GPU memory usage to optimize batch size

5. **Remote Monitoring**:
   - For remote servers, use port forwarding:
     ```bash
     ssh -L 6006:localhost:6006 user@remote-server
     ```

## Advanced Training Customization

### Modifying the GPU Configuration

You can customize the GPU training process by editing `custom_infini_config_gpu.yaml`:

1. **Scaling Model Size**:
   - For larger GPUs (>24GB VRAM): Increase `hidden_size` (e.g., to 2048) and `num_hidden_layers` (e.g., to 12)
   - For smaller GPUs (<12GB VRAM): Decrease `hidden_size` (e.g., to 768) and `micro_batch_size` (e.g., to 2)

2. **Precision Settings**:
   - Use `dtype: float16` for GPUs without bfloat16 support
   - Use `dtype: float32` for highest precision (at the cost of speed)

3. **Optimization Parameters**:
   - Adjust `learning_rate` based on your model size (larger models often need smaller learning rates)
   - Modify `lr_warmup_steps` and `lr_decay_style` to customize the learning rate schedule
   - Experiment with `weight_decay` values between 0.01-0.1

4. **Training Duration**:
   - Modify `train_steps` based on your dataset size and training budget
   - Adjust `val_check_interval` to control validation frequency

### Custom TensorBoard Monitoring

You can modify `train_gpu_with_tensorboard.py` to track additional metrics:

1. **Add Custom Metrics**:
   - Add new metrics to the `on_step_end` method in the `TensorBoardCallback` class
   - Track custom statistics like gradient norms, parameter distributions, etc.

2. **Custom Visualizations**:
   - Add code to generate and log visualizations of model outputs
   - Create custom plots for attention patterns or embedding representations

### Multi-GPU Training

For multi-GPU setups, modify `train_infini_llama_gpu.sh`:

1. **Use Multiple GPUs**:
   ```bash
   # Example: using GPUs 0 and 1
   export CUDA_VISIBLE_DEVICES=0,1
   torchrun --nproc_per_node=2 train_gpu_with_tensorboard.py --config-file custom_infini_config_gpu.yaml
   ```

2. **Distributed Training Configuration**:
   In `custom_infini_config_gpu.yaml`, adjust:
   ```yaml
   parallelism:
     dp: 2  # Data parallel size (number of GPUs)
     pp: 1  # Pipeline parallel size
     tp: 1  # Tensor parallel size
   ```

## Troubleshooting

If you encounter errors:

1. **Python Version**: Ensure you're using Python 3.10, not 3.13
2. **Package Installation**: Check that nanotron is installed with `pip install -e .`
3. **Memory Issues**: Reduce model size in the configuration file or lower the batch size
4. **GPU Compatibility**: Ensure your GPU drivers are up-to-date and compatible with PyTorch
5. **CUDA Errors**: Check CUDA toolkit installation and environment variables
6. **Dataset Issues**: Verify the dataset is properly processed in `/data/processed`
7. **TensorBoard Problems**: 
   - If you see "TensorBoard could not bind to port 6006", the port is already in use. The script will now try the next available port automatically.
   - You can manually specify a different port: `tensorboard --logdir=tensorboard_logs --port=6007`

8. **Tokenizer Issues**:
   - If you see errors about invalid repository IDs like `HFValidationError: Repo id must be in the form 'repo_name'`, the tokenizer path is invalid
   - Check that you have access to the specified HuggingFace model (e.g., "meta-llama/Llama-2-7b-hf")
   - For Llama models, you may need to request access on the HuggingFace website first

9. **Path Issues**:
   - If you see file not found errors, check that all paths in config files match your actual directory structure
   - The training script now uses environment variables for better path handling

For more troubleshooting details, see TRAINING_GUIDE.md.
