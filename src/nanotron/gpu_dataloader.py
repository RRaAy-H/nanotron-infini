#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/src/nanotron/gpu_dataloader.py

"""
GPU-accelerated data processing for Nanotron.
This module provides GPU-accelerated versions of the data processing functions
in nanotron.dataloader for faster text processing.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

try:
    import datasets
    from datasets import Dataset, Features, Sequence, Value
    from transformers import PreTrainedTokenizerBase
except ImportError:
    raise ImportError("Datasets and/or Transformers are required for the GPU dataloader.")

from nanotron import distributed as dist
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer


def gpu_clm_process(
    raw_dataset: "Dataset",
    tokenizer: "PreTrainedTokenizerBase",
    text_column_name: str,
    dataset_processing_num_proc_per_process: int,
    dataset_overwrite_cache: bool,
    sequence_length: int,
    device: Optional[str] = None,
    batch_size: int = 1024,  # Process this many texts at a time
):
    """
    GPU-accelerated version of clm_process.
    Concatenate all texts from raw_dataset and generate chunks of `sequence_length + 1`,
    where chunks overlap by a single token, using GPU for faster processing.
    
    Args:
        raw_dataset: The raw dataset to process
        tokenizer: The tokenizer to use
        text_column_name: The column name in the dataset that contains the text
        dataset_processing_num_proc_per_process: Number of processes for CPU operations (unused in GPU version)
        dataset_overwrite_cache: Whether to overwrite the cache
        sequence_length: The sequence length to use
        device: The GPU device to use (defaults to cuda:0 if available)
        batch_size: Number of texts to process at once on the GPU
    """
    # Default to CUDA:0 if device not specified
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    # Check if we're actually using GPU
    using_gpu = "cuda" in device
    if not using_gpu:
        print(f"Warning: GPU acceleration requested but no CUDA device available. Using {device} instead.")
    else:
        print(f"Using GPU acceleration for text processing on {device}")
    
    # Tokenize and group texts on GPU
    def gpu_tokenize_and_group_texts(texts: List[str]) -> Dict[str, List[np.ndarray]]:
        # Process in batches to avoid OOM
        all_token_ids = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize (this is still done on CPU as it's not the bottleneck)
            tokenized_batch = tokenizer.batch_encode_plus(
                batch_texts, 
                return_attention_mask=False, 
                return_token_type_ids=False
            )
            
            # Convert to tensors
            input_ids = [torch.tensor(ids, dtype=torch.long) for ids in tokenized_batch["input_ids"]]
            
            # Move to GPU if available
            if using_gpu:
                input_ids = [ids.to(device) for ids in input_ids]
            
            all_token_ids.extend(input_ids)
        
        return gpu_group_texts({"input_ids": all_token_ids}, device, sequence_length)
    
    # Process dataset with GPU acceleration
    train_dataset = raw_dataset.map(
        gpu_tokenize_and_group_texts,
        input_columns=text_column_name,
        remove_columns=raw_dataset.column_names,
        features=Features({"input_ids": Sequence(feature=Value(dtype="int64"), length=sequence_length + 1)}),
        batched=True,
        batch_size=1000,  # Process this many examples at once with map()
        load_from_cache_file=not dataset_overwrite_cache,
        desc=f"GPU tokenizing and grouping texts in chunks of {sequence_length+1}",
    )
    
    return train_dataset


def gpu_group_texts(examples: Dict[str, List[torch.Tensor]], device: str, sequence_length: int) -> Dict[str, List[np.ndarray]]:
    """
    GPU-accelerated text grouping function.
    
    Args:
        examples: Dictionary containing tensors of tokenized texts
        device: The device to use for processing
        sequence_length: The sequence length to use
    """
    # Get the key (should be "input_ids")
    key = next(iter(examples.keys()))
    
    # Concatenate all examples using GPU
    all_ids = torch.cat(examples[key], dim=0)
    
    # Calculate total length after removing remainders
    total_length = all_ids.size(0)
    if total_length >= sequence_length + 1:
        total_length = ((total_length - 1) // sequence_length) * sequence_length + 1
    
    # Use the GPU to split by chunks of sequence_length
    result = {}
    
    # Check if we have enough tokens to create at least one chunk
    if total_length >= sequence_length + 1:
        # Split into chunks of sequence_length + 1
        chunks = []
        for i in range(0, total_length - sequence_length, sequence_length):
            chunk = all_ids[i:i + sequence_length + 1]
            # Move back to CPU and convert to numpy for HuggingFace dataset compatibility
            chunks.append(chunk.cpu().numpy())
        
        result[key] = chunks
    else:
        # Not enough tokens to create any chunks
        result[key] = []
    
    return result


@dataclass
class GPUDataCollatorForCLM:
    """
    GPU-accelerated data collator for causal language modeling.
    
    This collator moves data to the specified GPU device earlier in the process
    for faster data processing.
    """
    sequence_length: int
    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext
    device: str = "cuda:0"  # Default device

    def __call__(self, examples: List[Dict[str, List[np.ndarray]]]) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # Process the case when current rank doesn't require data
        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)
        if current_pp_rank not in [self.input_pp_rank, self.output_pp_rank]:
            assert all(len(example) == 0 for example in examples)
            return {
                "input_ids": TensorPointer(self.input_pp_rank),
                "input_mask": TensorPointer(self.input_pp_rank),
                "label_ids": TensorPointer(self.output_pp_rank),
                "label_mask": TensorPointer(self.output_pp_rank),
            }

        # Make sure we load only what's necessary
        assert all(list(example.keys()) == ["input_ids"] for example in examples)

        # Stack examples - directly move to GPU after stacking if possible
        if isinstance(examples[0]["input_ids"], torch.Tensor):
            input_ids = torch.stack([examples[i]["input_ids"] for i in range(len(examples))]).to(self.device)
        else:
            # Convert from numpy if needed
            input_ids = torch.tensor(np.vstack([examples[i]["input_ids"] for i in range(len(examples))]), 
                                    device=self.device)
            
        batch_size, expanded_input_length = input_ids.shape

        result = {}
        result["input_ids"] = TensorPointer(group_rank=self.input_pp_rank)
        result["input_mask"] = TensorPointer(group_rank=self.input_pp_rank)
        result["label_ids"] = TensorPointer(group_rank=self.output_pp_rank)
        result["label_mask"] = TensorPointer(group_rank=self.output_pp_rank)

        assert expanded_input_length == self.sequence_length + 1, \
            f"Samples should be of length {self.sequence_length + 1} (seq_len+1), but got {expanded_input_length}"

        # Process inputs: last token is for the label
        if current_pp_rank == self.input_pp_rank:
            result["input_ids"] = input_ids[:, :-1]
            result["input_mask"] = torch.ones((batch_size, self.sequence_length), 
                                            dtype=torch.bool, 
                                            device=self.device)

        # Process labels: shift them to the left
        if current_pp_rank == self.output_pp_rank:
            result["label_ids"] = input_ids[:, 1:]
            result["label_mask"] = torch.ones((batch_size, self.sequence_length), 
                                            dtype=torch.bool,
                                            device=self.device)

        # Validation checks
        if isinstance(result["input_ids"], torch.Tensor) and result["input_ids"].shape[-1] != self.sequence_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {result['input_ids'].shape[-1]}, but should be"
                f" {self.sequence_length}."
            )
        if isinstance(result["label_ids"], torch.Tensor) and result["label_ids"].shape[-1] != self.sequence_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {result['label_ids'].shape[-1]}, but should be"
                f" {self.sequence_length}."
            )

        return result


def get_gpu_train_dataloader(
    train_dataset: "Dataset",
    sequence_length: int,
    parallel_context: ParallelContext,
    input_pp_rank: int,
    output_pp_rank: int,
    micro_batch_size: int,
    consumed_train_samples: int,
    dataloader_num_workers: int,
    seed_worker: int,
    gpu_device: str = "cuda:0",
    dataloader_drop_last: bool = True,
    dataloader_pin_memory: bool = True,
    use_loop_to_round_batch_size: bool = False,
):
    """
    GPU-accelerated version of get_train_dataloader.
    Creates a dataloader with GPU-accelerated data processing.
    
    Args:
        Same as get_train_dataloader, plus:
        gpu_device: The GPU device to use for data processing
    """
    from nanotron.dataloader import (
        _get_train_sampler, 
        get_dataloader_worker_init,
        EmptyInfiniteDataset
    )
    from torch.utils.data import DataLoader
    
    if not isinstance(train_dataset, datasets.Dataset):
        raise ValueError(f"training requires a datasets.Dataset, but got {type(train_dataset)}")

    # Case of ranks requiring data
    if dist.get_rank(parallel_context.pp_pg) in [input_pp_rank, output_pp_rank]:
        train_dataset = train_dataset.with_format(type="numpy", columns=["input_ids"], output_all_columns=True)
    else:
        # Case of ranks not requiring data
        assert train_dataset.column_names == ["input_ids"]
        dataset_length = len(train_dataset)
        train_dataset = train_dataset.remove_columns(column_names="input_ids")
        assert len(train_dataset) == 0
        train_dataset = EmptyInfiniteDataset(length=dataset_length)
        dataloader_num_workers = 0

    # Use GPU-accelerated data collator
    data_collator = GPUDataCollatorForCLM(
        sequence_length=sequence_length,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        parallel_context=parallel_context,
        device=gpu_device,
    )

    # Compute size and rank of dataloader workers
    dp_ranks_size = parallel_context.dp_pg.size()
    dp_rank = parallel_context.dp_pg.rank()

    train_sampler = _get_train_sampler(
        dl_rank=dp_rank,
        dl_ranks_size=dp_ranks_size,
        train_dataset=train_dataset,
        seed=seed_worker,
        use_loop_to_round_batch_size=use_loop_to_round_batch_size,
        micro_batch_size=micro_batch_size,
        drop_last=dataloader_drop_last,
        consumed_train_samples=consumed_train_samples,
    )

    # Create dataloader with GPU-accelerated collation
    return DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        drop_last=dataloader_drop_last,
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory,
        worker_init_fn=get_dataloader_worker_init(dp_rank=dp_rank),
    )
