# Reproducing Experiments

This repository contains the code for the toxicity-reduction RLHF experiments comparing three methods:

| Method | Description |
|--------|-------------|
| **Standard** | KL-penalized PPO, no data filtering |
| **IIF** | TracIn gradient-alignment filtering |
| **RLIF** | Trajectory influence filtering |

---


---

## Requirements

**Hardware:** 1 × NVIDIA A40 (48 GB) or equivalent per job. Up to 4 jobs run in parallel.

**Software:**

```bash
pip install torch==2.6.0 transformers==4.51.3 trl==0.17.0 peft==0.15.2 \
            datasets==3.5.0 accelerate==1.6.0 scipy numpy
pip install flash-attn --no-build-isolation
```

| Package | Version |
|---------|---------|
| Python | 3.10 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 4.51.3 |
| TRL | 0.17.0 |
| PEFT | 0.15.2 |
| Datasets | 3.5.0 |
| Accelerate | 1.6.0 |

---

## Models and Data

| Role | Identifier |
|------|-----------|
| Policy model | `EleutherAI/gpt-neo-2.7B` |
| Reward model | `facebook/roberta-hate-speech-dynabench-r4-target` |
| Dataset | `allenai/real-toxicity-prompts` (train split, toxicity > 0.3) |

Both models are downloaded automatically from Hugging Face on first run.

---

## Running the Experiments

```bash
# Standard
CUDA_VISIBLE_DEVICES=0 python -m train.train_rlif \
    --model_name_or_path="EleutherAI/gpt-neo-2.7B" \
    --reward_model_name="facebook/roberta-hate-speech-dynabench-r4-target" \
    --task="toxicity" --method="NA" \
    --lora=True --lora_r=16 --lora_alpha=32 --lora_dropout=0.05 \
    --use_flash_attention=True --torch_dtype=bfloat16 \
    --per_device_train_batch_size=64 --mini_batch_size=8 \
    --ppo_epochs=4 --max_steps=300 \
    --learning_rate=1e-5 --learning_rate_vhead=5e-4 \
    --lr_scheduler_type=constant --weight_decay=0.0 --max_grad_norm=1.0 \
    --init_kl_coef=0.4 --adap_kl_ctrl=False \
    --kl_estimator=k1 --target_kl=0.1 --early_stopping=False \
    --max_new_tokens=30 --temperature=1.0 --top_p=1.0 \
    --enable_eval=True --eval_interval=1 --n_test_eval=200 --eval_batch_size=32 \
    --eval_on_step_generations=True --filter_frac=0.1 \
    --seed=42 --logging_steps=1 --report_to=none --bf16=True \
    --output_dir="results/kl_b04/seed_42"
```

To run other methods, change `--method` and add method-specific flags:

| Method | `--method` | Extra flags |
|--------|-----------|-------------|
| Standard | `NA` | — |
| IIF | `IIFTracIn` | — |
| RLIF | `RLIF` | `--rlif_horizon=64 --rlif_u_scope=last_rollout` |

---

## Output Format

Each run produces `results/<name>/seed_<seed>/training_history.json` with per-step metrics:

| Key | Description |
|-----|-------------|
| `reward`, `toxicity_prob` | On-policy training metrics |
| `kl_div`, `policy_loss`, `value_loss` | PPO diagnostics |
| `frac_kept` | Fraction of samples retained (IIF / RLIF only) |
| `test_eval_steps`, `test_toxicity_prob` | Held-out evaluation (every step, n=200) |
| `elapsed_time` | Cumulative wall-clock time (seconds) |
| `timing/rollout_s`, `timing/forward_s`, `timing/optimize_s` | Per-phase timing |
