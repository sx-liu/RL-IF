"""
Reward model utilities for RLHF experiments.

This module provides utilities for loading and using reward models
for PPO training. Currently supports:
- Toxicity classification (facebook/roberta-hate-speech-dynabench-r4-target)
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


def load_reward_model(
    model_name: str,
    device: str = "cuda",
    task: str = "toxicity",
) -> Tuple[AutoTokenizer, nn.Module]:
    """
    Load a reward model for RLHF.

    Args:
        model_name: Model name or path
        device: Device to load model on
        task: Task type ('toxicity')

    Returns:
        Tuple of (tokenizer, model)

    Raises:
        ValueError: If task is not supported
    """
    if task != "toxicity":
        raise ValueError(
            f"Unsupported task: '{task}'. Currently only 'toxicity' is supported."
        )
    return load_toxicity_model(model_name, device)


def load_toxicity_model(
    model_name: str = "facebook/roberta-hate-speech-dynabench-r4-target",
    device: str = "cuda",
) -> Tuple[AutoTokenizer, nn.Module]:
    """
    Load the toxicity classification model.

    The model outputs logits for [nothate, hate] classes.
    Higher logit for nothate = less toxic = higher reward.

    Args:
        model_name: Model name
        device: Device to load on

    Returns:
        Tuple of (tokenizer, model)
    """
    from transformers import RobertaTokenizer, RobertaForSequenceClassification

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)

    logger.info(f"Loaded toxicity model: {model_name}")
    return tokenizer, model


def compute_toxicity_rewards(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: str = "cuda",
    max_length: int = 512,
) -> torch.Tensor:
    """
    Compute toxicity-based rewards for generated texts.

    Lower toxicity = higher reward.
    The model has labels: class 0 = "nothate", class 1 = "hate".
    We use the "nothate" logit as the reward.

    Args:
        model: Toxicity classification model
        tokenizer: Tokenizer for the model
        texts: List of generated texts
        device: Device
        max_length: Maximum sequence length

    Returns:
        Tensor of rewards [batch_size]
    """
    # Tokenize
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    # Get logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.float()

    # Use nothate logit as reward (higher = less toxic = better)
    rewards = logits[:, 0]

    return rewards


class RewardModelWrapper(nn.Module):
    """
    Wrapper for reward models that handles tokenization and reward computation.

    This wrapper allows using the reward model with the policy model's tokenizer,
    handling the vocabulary mismatch between models.
    """

    def __init__(
        self,
        reward_model: nn.Module,
        reward_tokenizer: AutoTokenizer,
        policy_tokenizer: AutoTokenizer,
        device: str = "cuda",
        max_length: int = 512,
        task: str = "toxicity",
    ):
        """
        Initialize the wrapper.

        Args:
            reward_model: The reward model
            reward_tokenizer: Tokenizer for the reward model
            policy_tokenizer: Tokenizer for the policy model
            device: Device
            max_length: Maximum sequence length
            task: Task type ('toxicity')
        """
        super().__init__()
        if task != "toxicity":
            raise ValueError(
                f"Unsupported task: '{task}'. Currently only 'toxicity' is supported."
            )
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer
        self.policy_tokenizer = policy_tokenizer
        self.device = device
        self.max_length = max_length
        self.task = task

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute rewards for input sequences.

        Args:
            input_ids: Token IDs from the policy tokenizer
            attention_mask: Attention mask (optional)

        Returns:
            Reward tensor [batch_size]
        """
        # Decode using policy tokenizer
        texts = self.policy_tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=True,
        )

        # Re-tokenize using reward tokenizer
        reward_inputs = self.reward_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        # Get rewards
        with torch.no_grad():
            outputs = self.reward_model(**reward_inputs)
            logits = outputs.logits.float()

        # Assume class 0 is "good" (e.g., nothate for toxicity)
        rewards = logits[:, 0]

        return rewards

    def compute_rewards(
        self,
        queries: List[str],
        responses: List[str],
    ) -> torch.Tensor:
        """
        Compute rewards for query-response pairs.

        For toxicity: Score ONLY the response text (not query+response).
        This matches the reference implementation (rl_utils.py line 839):
            if 'hate' in rmname:
                rewards = process_reward(batch["response"], rmname, ...)
        Scoring query+response would dilute the signal because the query
        may be toxic by design.

        Args:
            queries: Query texts
            responses: Response texts

        Returns:
            Reward tensor [batch_size]
        """
        # For toxicity, score ONLY the response
        texts = responses

        # Tokenize with reward tokenizer
        inputs = self.reward_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        # Get rewards
        with torch.no_grad():
            outputs = self.reward_model(**inputs)
            logits = outputs.logits.float()

        # For toxicity models, output is binary classification
        rewards = logits[:, 0]  # [batch_size] - nothate class

        return rewards
