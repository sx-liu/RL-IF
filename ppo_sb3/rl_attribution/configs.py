from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import argparse

@dataclass
class Config:
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PPOConfig":
        """Initialize config from a plain Python dictionary."""
        return cls(**data)

@dataclass
class TrainingConfig(Config):
    env_name: str = "LunarLander-v2"
    experiment_name: str = "default_experiment"
    experiment_type: str = "training"
    batches_dirs: List[str] = None
    initial_policy_dir: str = None
    data_selection_method: str = "none"
    cache_grad: bool = False
    cache_rollouts: bool = False
    clear_cache: bool = False
    n_envs: int = 8
    total_rollouts: int = 20
    rollout_begin: int = 0
    rollout_end: int = 20
    render_rollouts: bool = False

@dataclass
class PPOConfig(Config):
    policy_type: str = "MlpPolicy"
    policy_name: str = "none"
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)

    total_timesteps: int = 40_960
    n_epochs: int = 10
    n_steps: int = 2_048

    batch_size: int = 64
    learning_rate: float = 1e-2
    final_learning_rate: Optional[float] = None
    features_dim: Optional[int] = None

    gamma: float = 0.99
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    kl_coef: float = 0.0
    clip_range: float = 0.2
    final_clip_range: Optional[float] = None
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    optimizer_class: str = "SGD"

@dataclass
class AttributionConfig(Config):
    method: str = "none"
    ensemble: int = 0
    target_model: str = "final"

    step_size: int = 5
    drop_percentile: float = 50.0
    discard_mode: str = "neg-bottom"
    selection_mode: str = "remove"
    order: str = "fixed"
    group_by: str = "none"
    w_min: float = 0.0

    dynamic_horizon: bool = True
    adjust_log_prob: bool = True

    target_buffer: str = "reuse"
    target_rollout_size: int = 10240

    use_local_buffer: bool = False

    max_return_factor: float = 1.0

    max_kl_divergence: float = None
    kl_bound: float = float("inf")

    kl_penalty_coef: float = 0.0
    off_policy_clipping_coeff: float = 0.0

@dataclass
class EvaluationConfig(Config):
    eval_n_envs: int = 1
    n_eval_episodes: int = 1000
    step_size: int = 1
