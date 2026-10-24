#!/usr/bin/env python
# coding=utf-8
"""
Training script for RLHF experiments.

Trainers:
  1. kl_only  — RLHFKLTrainer:      KL-penalized objective, no clipping, no filtering.
  2. tracin   — RLHFTracInTrainer:   KL + TracIn gradient-alignment filtering (baseline).
  3. rlif    —                         KL + full RLIF trajectory influence filtering.
  4. clipped  — RLHFClippedTrainer:  Clipped PPO, no KL penalty, no filtering.

method arg mapping:
  NA      → kl_only
  Clipped → clipped
  TracIn  → tracin
  RLIF    → rlif
"""

import logging
import os
import sys
import warnings

import torch
from peft import LoraConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, HfArgumentParser, set_seed, get_scheduler
from trl import AutoModelForCausalLMWithValueHead
from trl.models import create_reference_model

# Suppress torch.compile warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")

from data.get_prompts import get_prompt_dataset, get_validation_prompt_dataset, collator
from train.model_arguments import ModelArguments, add_padding_to_tokenizer
from train.training_arguments import TrainingArguments
from train.rewards import load_reward_model, RewardModelWrapper
from train.evaluator import create_evaluator
from train.rlhf_trainer import (
    RLHFKLTrainer,
    RLHFTracInTrainer,
    RLHFIIFTracInTrainer,
    RLHFRLIFTrainer,
    RLHFClippedTrainer,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Monkey-patch GPT-Neo Flash Attention (same as train/train.py)
# ---------------------------------------------------------------------------

def _patch_gpt_neo_flash_attention():
    """
    Monkey-patch fix for GPT-Neo Flash Attention 2 bug in transformers.

    Bug: GPTNeoFlashAttention2.forward assumes attention_mask is 4D, but when
    using FA2 with padded batches, _update_causal_mask returns the 2D mask directly.
    This causes: IndexError: too many indices for tensor of dimension 2

    Fix: fully reimplement forward, skipping the 4D-only slice when mask is 2D.
    """
    try:
        from transformers.models.gpt_neo import modeling_gpt_neo

        if not hasattr(modeling_gpt_neo, "GPTNeoFlashAttention2"):
            return  # FA2 class doesn't exist in this version

        def patched_forward(self, hidden_states, attention_mask=None, layer_past=None,
                           head_mask=None, use_cache=False, output_attentions=False,
                           cache_position=None):
            bsz, _, _ = hidden_states.size()

            query = self.q_proj(hidden_states)
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)

            query = self._split_heads(query, self.num_heads, self.head_dim)
            key = self._split_heads(key, self.num_heads, self.head_dim)
            value = self._split_heads(value, self.num_heads, self.head_dim)

            if layer_past is not None:
                cache_kwargs = {"cache_position": cache_position}
                key, value = layer_past.update(key, value, self.layer_id, cache_kwargs)

            query_length = query.shape[2]
            tgt_len = key.shape[2]

            query = query.transpose(1, 2).view(bsz, query_length, self.num_heads, self.head_dim)
            key = key.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)
            value = value.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)

            attn_dropout = self.config.attention_dropout if self.training else 0.0

            # FIX: Only slice if attention_mask is 4D (the bug was assuming 4D always)
            if attention_mask is not None and attention_mask.dim() == 4:
                attention_mask = attention_mask[:, :, :, : key.shape[1]]
            # For 2D masks, pass as-is (Flash Attention handles them correctly)

            # Handle dtype casting for PEFT
            device_type = query.device.type if query.device.type != "mps" else "cpu"
            if query.dtype == torch.float32:
                if torch.is_autocast_enabled():
                    target_dtype = (
                        torch.get_autocast_dtype(device_type)
                        if hasattr(torch, "get_autocast_dtype")
                        else torch.get_autocast_gpu_dtype()
                    )
                elif hasattr(self.config, "_pre_quantization_dtype"):
                    target_dtype = self.config._pre_quantization_dtype
                else:
                    target_dtype = self.q_proj.weight.dtype

                query = query.to(target_dtype)
                key = key.to(target_dtype)
                value = value.to(target_dtype)

            from transformers.modeling_flash_attention_utils import _flash_attention_forward

            attn_output = _flash_attention_forward(
                query,
                key,
                value,
                attention_mask,
                query_length,
                dropout=attn_dropout,
                softmax_scale=1.0,
                is_causal=self.is_causal,
                use_top_left_mask=self._flash_attn_uses_top_left_mask,
            )

            attn_weights_reshaped = attn_output.reshape(bsz, query_length, self.num_heads * self.head_dim)
            attn_output = self.out_proj(attn_weights_reshaped)
            attn_output = self.resid_dropout(attn_output)

            return attn_output, attn_weights_reshaped

        modeling_gpt_neo.GPTNeoFlashAttention2.forward = patched_forward
        logger.info("Applied GPT-Neo Flash Attention 2 monkey-patch fix")
    except Exception as e:
        logger.warning(f"Failed to apply GPT-Neo FA2 patch: {e}")


_patch_gpt_neo_flash_attention()


# ---------------------------------------------------------------------------
# Helper: find LoRA layer names
# ---------------------------------------------------------------------------

def find_trainable_layers(model, lora_only: bool = True):
    """Return names of trainable Linear layers (LoRA-only by default)."""
    layer_names = []
    target = model
    prefix = ""
    if hasattr(model, "pretrained_model"):
        target = model.pretrained_model
        prefix = "pretrained_model."

    for name, module in target.named_modules():
        if lora_only:
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                if hasattr(module.lora_A, "default"):
                    layer_names.append(f"{prefix}{name}.lora_A.default")
                elif isinstance(module.lora_A, torch.nn.Linear):
                    layer_names.append(f"{prefix}{name}.lora_A")
                if hasattr(module.lora_B, "default"):
                    layer_names.append(f"{prefix}{name}.lora_B.default")
                elif isinstance(module.lora_B, torch.nn.Linear):
                    layer_names.append(f"{prefix}{name}.lora_B")
        else:
            if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
                layer_names.append(f"{prefix}{name}")
    return layer_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    # Determine trainer type from --method argument
    method_to_trainer = {
        "NA":        "kl_only",
        "Clipped":   "clipped",
        "TracIn":    "tracin",
        "IIFTracIn": "iif_tracin",
        "RLIF":      "rlif",
    }
    trainer_type = method_to_trainer[training_args.method]  # validated in __post_init__

    logger.info("=" * 60)
    logger.info("RLHF Training")
    logger.info(f"  Trainer type: {trainer_type}")
    logger.info(f"  Method arg:   {training_args.method}")
    logger.info(f"  Model:        {model_args.model_name_or_path}")
    logger.info(f"  LoRA:         {model_args.lora}")
    logger.info("=" * 60)

    set_seed(training_args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    add_padding_to_tokenizer(tokenizer)

    # Datasets
    logger.info("Loading datasets...")
    train_dataset = get_prompt_dataset(
        task=training_args.task,
        tokenizer=tokenizer,
        seed=training_args.seed,
    )
    logger.info(f"  Train: {len(train_dataset)} samples")

    val_dataset = None
    if training_args.use_validation_set and trainer_type in ("tracin", "iif_tracin", "rlif"):
        val_dataset = get_validation_prompt_dataset(
            task=training_args.task,
            tokenizer=tokenizer,
            n_val=training_args.n_val,
            seed=training_args.seed,
        )
        logger.info(f"  Val:   {len(val_dataset)} samples")

    # Held-out test prompts from wiki_toxic (different from training distribution)
    test_prompts = None
    n_test_eval = getattr(training_args, 'n_test_eval', 0)
    if n_test_eval > 0:
        from datasets import load_dataset as _load_dataset
        logger.info(f"Loading wiki_toxic test prompts (n={n_test_eval})...")
        _wiki = _load_dataset("OxAISH-AL-LLM/wiki_toxic", split="test")
        _wiki = _wiki.filter(lambda x: x["label"] == 1)
        _wiki = _wiki.select(range(min(n_test_eval, len(_wiki))))
        test_prompts = [ex["comment_text"][:2000] for ex in _wiki]
        logger.info(f"  Test:  {len(test_prompts)} wiki_toxic prompts")

    # Policy model
    logger.info("Loading policy model...")
    model_kwargs = {"torch_dtype": getattr(torch, model_args.torch_dtype)}
    if model_args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    peft_config = None
    if model_args.lora:
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=(
                model_args.lora_target_modules if model_args.lora_target_modules
                else None
            ),
        )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_args.model_name_or_path,
        peft_config=peft_config,
        summary_dropout_prob=0.0,
        **model_kwargs,
    ).to(device)

    if model_args.lora:
        logger.info(f"Applied LoRA (r={model_args.lora_r})")
        model.pretrained_model.print_trainable_parameters()
        if hasattr(model.pretrained_model, "enable_input_require_grads"):
            model.pretrained_model.enable_input_require_grads()

    # Reference model
    ref_model = None
    if not model_args.lora:
        logger.info("Creating frozen reference model...")
        ref_model = create_reference_model(model)
    else:
        logger.info("Using disable_adapter() for reference log-probs (PEFT)")

    # Reward model
    logger.info("Loading reward model...")
    reward_tokenizer, reward_model_raw = load_reward_model(
        model_args.reward_model_name,
        device=device,
        task=training_args.task,
    )
    reward_model = RewardModelWrapper(
        reward_model=reward_model_raw,
        reward_tokenizer=reward_tokenizer,
        policy_tokenizer=tokenizer,
        device=device,
        task=training_args.task,
    )

    # Gradient hook (only for RLIF — TracIn manages its own hooks internally)
    layer_names = find_trainable_layers(model, lora_only=model_args.lora)
    logger.info(f"Found {len(layer_names)} trainable layers")

    grad_hook = None
    selection_strategy = None

    # Optimizer
    policy_params, vhead_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "v_head" in name:
            vhead_params.append(param)
        else:
            policy_params.append(param)

    vhead_lr = (
        training_args.learning_rate_vhead
        if training_args.learning_rate_vhead is not None
        else training_args.learning_rate
    )
    param_groups = [
        {"params": policy_params, "lr": training_args.learning_rate},
        {"params": vhead_params, "lr": vhead_lr},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
    )
    logger.info("Using AdamW optimizer")

    # Dataloader & LR scheduler
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    if training_args.max_steps > 0:
        n_steps = training_args.max_steps
    else:
        n_steps = len(train_dataloader) * int(training_args.num_train_epochs)

    warmup = (
        training_args.warmup_steps
        if training_args.warmup_steps > 0
        else int(n_steps * training_args.warmup_ratio)
    )
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup,
        num_training_steps=n_steps,
    )

    # Evaluator
    evaluator = None
    if training_args.enable_eval:
        logger.info(f"Creating evaluator (task={training_args.task})...")
        evaluator = create_evaluator(
            task=training_args.task,
            device=device,
            batch_size=training_args.eval_batch_size,
        )

    # ------------------------------------------------------------------
    # Construct the right trainer
    # ------------------------------------------------------------------
    if trainer_type == "kl_only":
        logger.info("Using RLHFKLTrainer (Task 1: KL penalty, no clipping)")
        trainer = RLHFKLTrainer(
            model=model,
            ref_model=ref_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            args=training_args,
            grad_hook=None,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            evaluator=evaluator,
            test_prompts=test_prompts,
        )

    elif trainer_type == "tracin":
        logger.info("Using RLHFTracInTrainer (TracIn gradient-alignment filtering + KL)")
        trainer = RLHFTracInTrainer(
            model=model,
            ref_model=ref_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            args=training_args,
            grad_hook=grad_hook,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            evaluator=evaluator,
            val_dataset=val_dataset,
            test_prompts=test_prompts,
        )

    elif trainer_type == "iif_tracin":
        logger.info("Using RLHFIIFTracInTrainer (IIF-TracIn rollout-level filtering + KL)")
        trainer = RLHFIIFTracInTrainer(
            model=model,
            ref_model=ref_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            args=training_args,
            grad_hook=grad_hook,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            evaluator=evaluator,
            val_dataset=val_dataset,
            test_prompts=test_prompts,
        )

    elif trainer_type == "rlif":
        logger.info("Using RLHFRLIFTrainer (RLIF trajectory-influence filtering + KL)")
        trainer = RLHFRLIFTrainer(
            model=model,
            ref_model=ref_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            args=training_args,
            grad_hook=None,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            evaluator=evaluator,
            val_dataset=val_dataset,
            test_prompts=test_prompts,
        )

    elif trainer_type == "clipped":
        logger.info("Using RLHFClippedTrainer (Baseline: clipped PPO, no KL)")
        trainer = RLHFClippedTrainer(
            model=model,
            ref_model=ref_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            args=training_args,
            grad_hook=None,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            evaluator=evaluator,
            test_prompts=test_prompts,
        )

    else:  # pragma: no cover
        raise ValueError(f"Unknown trainer_type: {trainer_type!r}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    log_interval = int(training_args.logging_steps)
    max_steps = training_args.max_steps if training_args.max_steps > 0 else None

    history = trainer.train(
        train_dataloader=train_dataloader,
        num_epochs=int(training_args.num_train_epochs),
        max_steps=max_steps,
        log_interval=log_interval,
    )

    # Save model
    final_dir = os.path.join(training_args.output_dir, "final")
    trainer.save_model(final_dir)

    # Save history
    import json
    history_file = os.path.join(training_args.output_dir, "training_history.json")
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"History saved to {history_file}")

    logger.info("Training complete!")
    return history


if __name__ == "__main__":
    main()
