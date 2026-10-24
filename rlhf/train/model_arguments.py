"""
Model arguments for RLHF experiments.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments for model/tokenizer configuration.
    """

    model_name_or_path: Optional[str] = field(
        default="EleutherAI/gpt-neo-2.7B",
        metadata={"help": "The model checkpoint for weights initialization."},
    )
    reward_model_name: Optional[str] = field(
        default="facebook/roberta-hate-speech-dynabench-r4-target",
        metadata={
            "help": (
                "Reward model name or path. "
                "Default: 'facebook/roberta-hate-speech-dynabench-r4-target'."
            )
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store pretrained models from huggingface.co"},
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": "Model dtype",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    # LoRA arguments
    lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for training"},
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank"},
    )
    lora_alpha: float = field(
        default=32,
        metadata={"help": "LoRA alpha"},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"},
    )
    lora_target_modules: List[str] = field(
        default_factory=list,  # Empty = PEFT auto-detects correct modules for each model
        metadata={"help": "Target modules for LoRA (empty = auto-detect, use out_proj for GPT-Neo, o_proj for LLaMA)"},
    )

    # Flash attention
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "Whether to use Flash Attention 2"},
    )


def add_padding_to_tokenizer(tokenizer):
    """Add padding token to tokenizer if not present and set left-padding for generation."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-padding is required for decoder-only models during generation
    tokenizer.padding_side = "left"
