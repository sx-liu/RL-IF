"""
Validation dataset loading for RLHF experiments.

This module provides utilities for loading validation datasets used
for data curation during layer-wise descent.
"""

from typing import Optional

from datasets import Dataset, load_dataset


def get_validation_dataset(
    task: str,
    tokenizer,
    n_val: int = 16,
    input_min_length: int = 5,
    input_max_length: int = 15,
    seed: int = 42,
    strategy: str = "random",
) -> Dataset:
    """
    Load validation dataset for data curation.

    The validation dataset represents "good" examples that we want
    the model to learn from. During layer-wise descent, we compute
    gradients on this dataset to determine which training samples
    are most beneficial.

    Args:
        task: Task name ('toxicity', etc.)
        tokenizer: Tokenizer for encoding
        n_val: Number of validation samples
        input_min_length: Minimum prompt length
        input_max_length: Maximum prompt length
        seed: Random seed
        strategy: Curation strategy ('random' or 'top')
            - 'random': Randomly sample from test set
            - 'top': Select samples with highest target metric (e.g., most toxic)

    Returns:
        Validation dataset with 'input_ids' and 'query' columns
    """
    if task == "toxicity":
        return _load_toxicity_validation(
            tokenizer=tokenizer,
            n_val=n_val,
            input_min_length=input_min_length,
            input_max_length=input_max_length,
            seed=seed,
            strategy=strategy,
        )
    else:
        raise ValueError(f"Unknown task: {task}. Supported: ['toxicity']")


def _load_toxicity_validation(
    tokenizer,
    n_val: int = 16,
    input_min_length: int = 5,
    input_max_length: int = 15,
    seed: int = 42,
    strategy: str = "random",
    toxicity_threshold: float = 0.3,
) -> Dataset:
    """
    Load validation dataset for toxicity task.

    For detoxification, the validation set contains prompts where we want
    the model to generate non-toxic continuations. The gradients on these
    examples guide the curation of beneficial training samples.

    Args:
        tokenizer: Tokenizer for encoding
        n_val: Number of validation samples
        input_min_length: Minimum prompt length
        input_max_length: Maximum prompt length
        seed: Random seed
        strategy: 'random' or 'top' (most toxic prompts)
        toxicity_threshold: Minimum toxicity for filtering

    Returns:
        Validation dataset
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and filter
    ds = load_dataset("allenai/real-toxicity-prompts", split="train")

    def filter_fn(sample):
        toxicity = sample["prompt"]["toxicity"]
        return toxicity is not None and toxicity > toxicity_threshold

    ds = ds.filter(filter_fn, batched=False)

    # Split into train/test
    ds = ds.train_test_split(test_size=0.2, shuffle=False, seed=seed)

    # Tokenize
    import random
    random.seed(seed)

    def tokenize(sample):
        prompt = sample["prompt"]["text"]
        continuation = sample["continuation"]["text"]
        input_size = random.randint(input_min_length, input_max_length)
        sample["input_ids"] = tokenizer.encode(prompt + continuation)[:input_size]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    test_ds = ds["test"].map(tokenize, batched=False)
    test_ds.set_format(type="torch")

    # Select validation samples
    if strategy == "random":
        val_ds = test_ds.shuffle(seed=seed).select(range(min(n_val, len(test_ds))))
    elif strategy == "top":
        # Select most toxic prompts for validation
        # This creates harder examples to guide curation
        test_ds = test_ds.map(lambda x: {"toxicity": x["prompt"]["toxicity"]})
        val_ds = test_ds.sort("toxicity", reverse=True).select(range(min(n_val, len(test_ds))))
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Supported: ['random', 'top']")

    return val_ds
