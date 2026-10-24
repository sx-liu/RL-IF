import torch
import torch.nn as nn

from rl_attribution.models.cnn import get_cnn_cls
from rl_attribution.trainers.ppo import PPOWithLocalDataSelection

from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn

ACTIVATION_MAPPING = {
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh,
    "ELU": nn.ELU,
    "LeakyReLU": nn.LeakyReLU,
    "GELU": nn.GELU,
    "SiLU": nn.SiLU,
}

def get_ppo_model(env, ppo_config, seed=42, device="cpu"):
    policy_kwargs = dict(
        optimizer_class=torch.optim.SGD if ppo_config.optimizer_class == "SGD" else torch.optim.Adam,
    )

    if ppo_config.policy_type == "CnnPolicy":
        if ppo_config.policy_name != "none":
            policy_kwargs["features_extractor_class"] = get_cnn_cls(ppo_config.policy_name)
        policy_kwargs["features_extractor_kwargs"] = dict(features_dim=ppo_config.features_dim)

    if ppo_config.policy_kwargs is not None and len(ppo_config.policy_kwargs) > 0:
        print(ppo_config.policy_kwargs)
    policy_kwargs.update(ppo_config.policy_kwargs)
    if "activation_fn" in policy_kwargs:
        act_fn_name = policy_kwargs["activation_fn"]
        if act_fn_name in ACTIVATION_MAPPING:
            policy_kwargs["activation_fn"] = ACTIVATION_MAPPING[act_fn_name]
        else:
            raise ValueError(f"Unsupported activation function: {act_fn_name}")

    learning_schedule = get_linear_fn(
        start=ppo_config.learning_rate,
        end=ppo_config.final_learning_rate,
        end_fraction=1.0,
    ) if ppo_config.final_learning_rate is not None else ppo_config.learning_rate

    clip_schedule = get_linear_fn(
        start=ppo_config.clip_range,
        end=ppo_config.final_clip_range,
        end_fraction=1.0,
    ) if ppo_config.final_clip_range is not None else ppo_config.clip_range

    ppo_class = PPOWithLocalDataSelection

    model = ppo_class(
        ppo_config.policy_type,
        env,
        verbose=1,
        device=device,
        n_epochs=ppo_config.n_epochs,
        n_steps=ppo_config.n_steps,
        learning_rate=learning_schedule,
        gamma=ppo_config.gamma,
        ent_coef=ppo_config.ent_coef,
        vf_coef=ppo_config.vf_coef,
        kl_coef=ppo_config.kl_coef,
        clip_range=clip_schedule,
        batch_size=ppo_config.batch_size,
        gae_lambda=ppo_config.gae_lambda,
        max_grad_norm=ppo_config.max_grad_norm,
        normalize_advantage=ppo_config.normalize_advantage,
        seed=seed,
        policy_kwargs=policy_kwargs,
    )

    return model
