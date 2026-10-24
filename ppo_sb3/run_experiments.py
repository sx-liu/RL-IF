import argparse
import os
import yaml
import subprocess
from pathlib import Path

from rl_attribution.trainer import PPOTrainer

def load_config(config_path: str):
    """Load experiment configuration from a YAML file."""
    if config_path is None:
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def apply_overrides(base_config: dict, overrides: dict):
    """Apply overrides from main config to sub-configs."""
    if not overrides:
        return base_config

    for key, value in overrides.items():
        if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):

            base_config[key] = apply_overrides(base_config[key], value)
        else:
            base_config[key] = value

    return base_config

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train PPO agent on a given environment")
    parser.add_argument("config", type=str, help="Path to the experiment config YAML file")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="List of seeds for experiments",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run experiments"
    )

    args = parser.parse_args()
    config = load_config(args.config)
    training_config = config["training_config"]

    ppo_config = load_config(config.get("ppo_config", None))
    attribution_config = load_config(config.get("attribution_config", None))
    eval_config = load_config(config.get("eval_config", None))

    if "ppo_override" in config and ppo_config:
        ppo_config = apply_overrides(ppo_config, config["ppo_override"])
    if "attribution_override" in config and attribution_config:
        attribution_config = apply_overrides(attribution_config, config["attribution_override"])
    if "eval_override" in config and eval_config:
        eval_config = apply_overrides(eval_config, config["eval_override"])

    trainer = PPOTrainer(
        training_config=training_config,
        ppo_config=ppo_config,
        attribution_config=attribution_config,
        eval_config=eval_config,
        seeds=args.seeds,
    )
    trainer.run_experiments(device=args.device)

if __name__ == "__main__":
    main()
