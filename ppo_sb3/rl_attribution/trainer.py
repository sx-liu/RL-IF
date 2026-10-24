import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import yaml
import shutil

import re
from dataclasses import is_dataclass, asdict
from pathlib import Path
from tqdm.rich import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from rl_attribution.attribution import build_dataset_from_influence
from rl_attribution.utils.envs import make_vec_env
from rl_attribution.configs import PPOConfig, TrainingConfig, AttributionConfig, EvaluationConfig
from rl_attribution.training import train_ppo, train_ppo_with_selected_data
from rl_attribution.utils.envs import make_env
from rl_attribution.eval import evaluate_policy, compute_surrogate_validation_loss
from rl_attribution.utils.advantage import (
    evalute_advantage_estimate_with_cache,
    evalute_advantage_estimate_with_sampling,
    evalute_advantage_estimate_with_repeated_sampling,
)
from rl_attribution.utils.logger import get_logger
from rl_attribution.utils.buffers import collect_training_buffers, sample_rollouts_buffers

logger = get_logger()

from rl_attribution.trainers.ppo import (
    PPOWithLocalDataSelection,
    PPOWithGlobalDataSelection,
    PPOWithIterativeGlobalDataSelection,
)

trainer_dict = {
    "none": PPO,
    "local": PPOWithLocalDataSelection,
    "global": PPOWithGlobalDataSelection,
    "global_iterative": PPOWithIterativeGlobalDataSelection,
}

ACTIVATION_MAPPING = {
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh,
    "ELU": nn.ELU,
    "LeakyReLU": nn.LeakyReLU,
    "GELU": nn.GELU,
    "SiLU": nn.SiLU,
}

class PPOTrainer:

    def __init__(
        self,
        ppo_config,
        training_config,
        attribution_config,
        eval_config,
        seeds=None,
    ):
        if seeds is None:
            seeds = [2, 22, 42, 62, 82]
        self.seeds = seeds
        self.ppo_config = PPOConfig.from_dict(ppo_config)
        self.training_config = TrainingConfig.from_dict(training_config)
        self.attribution_config = AttributionConfig.from_dict(attribution_config)
        self.eval_config = EvaluationConfig.from_dict(eval_config)
        self.check_consistency()
        logger.info(self.ppo_config)
        logger.info(self.training_config)

        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._dump_unified_yaml(self.root_dir / "config.yaml")

        env_name = self.training_config.env_name
        n_envs_train = self.training_config.n_envs

        render_mode = "rgb_array" if self.training_config.render_rollouts else None

        self.train_env = make_vec_env(env_name, n_envs_train, render_mode=render_mode)

    def check_consistency(self):
        total_rollouts = self.training_config.total_rollouts
        total_timesteps = self.ppo_config.total_timesteps
        n_steps = self.ppo_config.n_steps
        n_envs = self.training_config.n_envs

        assert total_rollouts * n_steps * n_envs == total_timesteps, (
            f"Inconsistent config: total_rollouts ({total_rollouts}) * n_steps ({n_steps}) "
            f"!= total_timesteps ({total_timesteps})"
        )

        batch_size = self.ppo_config.batch_size
        assert n_steps % batch_size == 0, (
            f"n_steps ({n_steps}) must be divisible by batch_size ({batch_size})"
        )

    @staticmethod
    def _cfg_to_dict(obj):
        """Convert config objects to plain dicts robustly."""
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            return obj.to_dict()
        if is_dataclass(obj):
            return asdict(obj)
        if hasattr(obj, "__dict__"):

            return dict(obj.__dict__)

        return json.loads(json.dumps(obj, default=str))

    def _dump_unified_yaml(self, path: Path):
        unified = {
            "seeds": list(self.seeds),
            "ppo_config": self._cfg_to_dict(self.ppo_config),
            "training_config": self._cfg_to_dict(self.training_config),
            "attribution_config": self._cfg_to_dict(self.attribution_config),
            "eval_config": self._cfg_to_dict(self.eval_config),
        }
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(unified, f, sort_keys=False, allow_unicode=True)

    @property
    def root_dir(self):
        log_root = Path("./ppo_results")
        env_name = self.training_config.env_name
        experiment_name = self.training_config.experiment_name
        return log_root / env_name / experiment_name

    def _loo_retrain_scores(
        self,
        target_model,
        target_rollout,
        n_batches,
        model_path,
        begin_step,
        end_step,
        total_updates,
        device="cpu",
        att_scores=None,
        log_freq: int = 5,
        sample_proportion: float = 0.005,
        leave_out_weight: float = 1.0,
        do_resample: bool = True,
        num_retrain: int = 1,
        seed=42,
    ):

        rollout_begin = begin_step // self.n_updates
        rollout_end = end_step // self.n_updates
        assert rollout_begin != rollout_end
        initial_policy = model_path / f"policy_grad_{begin_step}.zip"

        base_save_path = self.root_dir / f"seed-{seed}" / self.model_path(
            "loo_retrain",
            rollout_begin,
            rollout_end,
        )

        train_ppo_with_selected_data(
            self.training_config.env_name,
            self.ppo_config,
            model_path,
            rollout_begin,
            rollout_end,
            save_path=base_save_path / "base",
            initial_policy=initial_policy,
            cache_grad=False,
            n_envs=self.training_config.n_envs,
            device=device,
        )

        idx = rollout_end * self.n_updates
        model = PPO.load(base_save_path / "base" / f"policy_grad_{idx}.zip", device=device)

        eval_results = compute_surrogate_validation_loss(model, target_rollout)
        base_loss = eval_results[0]

        retrain_base_losses = []
        if rollout_end > rollout_begin + 1 and do_resample:
            for i in range(num_retrain):
                train_ppo(
                    self.train_env,
                    self.ppo_config,
                    rollout_begin + 1,
                    rollout_end,
                    save_path=base_save_path / f"resample_{i}",
                    initial_policy=base_save_path / "base" / f"policy_grad_{(rollout_begin + 1) * self.n_updates}.zip",
                    seed=seed + i + 1,
                    cache_grad=self.training_config.cache_grad,
                    cache_rollouts=self.training_config.cache_rollouts,
                    n_envs=self.training_config.n_envs,
                    device=device,
                )
                final_model_idx = rollout_end * self.n_updates
                retrain_model = PPO.load(
                    base_save_path / f"resample_{i}" / f"policy_grad_{final_model_idx}.zip",
                    device=device,
                )
                retrain_eval_results = compute_surrogate_validation_loss(retrain_model, target_rollout)
                retrain_base_losses.append(retrain_eval_results[0])

        loo_scores = {}
        processed_keys = []

        all_samples = []
        for batch in range(self.n_updates * rollout_begin, self.n_updates * (rollout_begin + 1)):
            for sample in range(self.ppo_config.batch_size):
                all_samples.append((batch, sample))

        import random
        random.seed(seed)

        n_samples_to_compute = len(all_samples)
        samples_to_compute = random.sample(all_samples, n_samples_to_compute)

        logger.info(f"Computing LOO scores for {n_samples_to_compute}/{len(all_samples)} samples (proportion={sample_proportion})")

        pbar = tqdm(samples_to_compute, desc="Computing LOO scores", unit="sample")
        for batch, sample in pbar:
            pbar.set_postfix({"batch": batch, "sample": sample})
            save_path = base_save_path / f"batch_{batch}_sample_{sample}"
            os.makedirs(save_path, exist_ok=True)

            train_ppo_with_selected_data(
                self.training_config.env_name,
                self.ppo_config,
                model_path,
                rollout_begin,
                rollout_begin + 1,
                save_path=save_path / "retrain",
                initial_policy=initial_policy,
                cache_grad=False,
                n_envs=self.training_config.n_envs,
                device=device,
                leave_out_idx=(batch, sample),
                leave_out_weight=leave_out_weight,
            )

            if rollout_end > rollout_begin + 1:
                idx = (rollout_begin + 1) * self.n_updates
                if do_resample:
                    for i in range(num_retrain):
                        train_ppo(
                            self.train_env,
                            self.ppo_config,
                            rollout_begin + 1,
                            rollout_end,
                            save_path=save_path / f"resample_{i}",
                            initial_policy=save_path / "retrain" / f"policy_grad_{idx}.zip",
                            seed=seed + i + 1,
                            cache_grad=self.training_config.cache_grad,
                            cache_rollouts=self.training_config.cache_rollouts,
                            n_envs=self.training_config.n_envs,
                            device=device,
                        )
                    final_model_idx = rollout_end * self.n_updates
                    final_model_path = [
                        save_path / f"resample_{i}" / f"policy_grad_{final_model_idx}.zip"
                        for i in range(num_retrain)
                    ]
                else:
                    train_ppo_with_selected_data(
                        self.training_config.env_name,
                        self.ppo_config,
                        model_path,
                        rollout_begin + 1,
                        rollout_end,
                        save_path=save_path / "retrain",
                        initial_policy=save_path / "retrain" / f"policy_grad_{idx}.zip",
                        cache_grad=False,
                        n_envs=self.training_config.n_envs,
                        device=device,

                    )
                    final_model_idx = rollout_end * self.n_updates
                    final_model_path = [save_path / "retrain" / f"policy_grad_{final_model_idx}.zip"]
            else:
                final_model_idx = rollout_end * self.n_updates
                final_model_path = [save_path / "retrain" / f"policy_grad_{final_model_idx}.zip"]

            eval_results = []
            for path in final_model_path:
                model = PPO.load(path, device=device)

                loo_loss = compute_surrogate_validation_loss(model, target_rollout)[0]
                eval_results.append(loo_loss)

            self.remove_cache(save_path)

            key = (batch, sample)
            loo_scores[key] = [loo_loss - base_loss for loo_loss in eval_results]
            processed_keys.append(key)

            if att_scores is not None and (len(processed_keys) % max(1, log_freq) == 0):
                paired_att = []
                paired_loo = []
                for k in processed_keys:
                    if k in att_scores:
                        paired_att.append(float(att_scores[k]))
                        paired_loo.append(float(loo_scores[k]))

                if len(paired_att) >= 2:
                    try:
                        corr = float(
                            np.corrcoef(np.array(paired_att), np.array(paired_loo))[0, 1]
                        )
                    except Exception:
                        corr = float(np.nan)
                    logger.info(
                        f"LOO vs attribution correlation after {len(processed_keys)} processed: {corr:.4f}"
                    )

            if self.training_config.clear_cache:
                self.remove_cache(save_path)

        return {
            "loo_scores": loo_scores,
            "base_loss": base_loss,
            "retrain_base_losses": retrain_base_losses,
        }

    def _loo_retrain_scores_parallel(
        self,
        target_model,
        target_rollout,
        n_batches,
        model_path,
        begin_step,
        end_step,
        total_updates,
        device="cpu",
        att_scores=None,
        log_freq: int = 5,
        sample_proportion: float = 0.005,
        leave_out_weight: float = 1.0,
        do_resample: bool = True,
        num_retrain: int = 1,
        seed=42,
        n_workers: int = 8,
    ):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        rollout_begin = begin_step // self.n_updates
        rollout_end = end_step // self.n_updates
        assert rollout_begin != rollout_end
        initial_policy = model_path / f"policy_grad_{begin_step}.zip"

        base_save_path = self.root_dir / f"seed-{seed}" / self.model_path(
            "loo_retrain",
            rollout_begin,
            rollout_end,
        )

        train_ppo_with_selected_data(
            self.training_config.env_name,
            self.ppo_config,
            model_path,
            rollout_begin,
            rollout_end,
            save_path=base_save_path / "base",
            initial_policy=initial_policy,
            cache_grad=False,
            n_envs=self.training_config.n_envs,
            device=device,
        )

        idx = rollout_end * self.n_updates
        model = PPO.load(base_save_path / "base" / f"policy_grad_{idx}.zip", device=device)

        eval_results = compute_surrogate_validation_loss(model, target_rollout)
        base_loss = eval_results[0]

        retrain_base_losses = []
        if rollout_end > rollout_begin + 1 and do_resample:
            for i in range(num_retrain):
                train_ppo(
                    self.train_env,
                    self.ppo_config,
                    rollout_begin + 1,
                    rollout_end,
                    save_path=base_save_path / f"resample_{i}",
                    initial_policy=base_save_path / "base" / f"policy_grad_{(rollout_begin + 1) * self.n_updates}.zip",
                    seed=seed + i + 1,
                    cache_grad=self.training_config.cache_grad,
                    cache_rollouts=self.training_config.cache_rollouts,
                    n_envs=self.training_config.n_envs,
                    device=device,
                )
                final_model_idx = rollout_end * self.n_updates
                retrain_model = PPO.load(
                    base_save_path / f"resample_{i}" / f"policy_grad_{final_model_idx}.zip",
                    device=device,
                )
                retrain_eval_results = compute_surrogate_validation_loss(retrain_model, target_rollout)
                retrain_base_losses.append(retrain_eval_results[0])

        all_samples = []
        for batch in range(self.n_updates * rollout_begin, self.n_updates * (rollout_begin + 1)):
            for sample in range(self.ppo_config.batch_size):
                all_samples.append((batch, sample))

        import random
        random.seed(seed)
        n_samples_to_compute = len(all_samples)
        samples_to_compute = random.sample(all_samples, n_samples_to_compute)

        logger.info(
            f"Computing LOO scores for {n_samples_to_compute}/{len(all_samples)} samples "
            f"(proportion={sample_proportion}) with {n_workers} workers"
        )

        def _process_sample(batch, sample):
            save_path = base_save_path / f"batch_{batch}_sample_{sample}"
            os.makedirs(save_path, exist_ok=True)

            train_ppo_with_selected_data(
                self.training_config.env_name,
                self.ppo_config,
                model_path,
                rollout_begin,
                rollout_begin + 1,
                save_path=save_path / "retrain",
                initial_policy=initial_policy,
                cache_grad=False,
                n_envs=self.training_config.n_envs,
                device=device,
                leave_out_idx=(batch, sample),
                leave_out_weight=leave_out_weight,
            )

            if rollout_end > rollout_begin + 1:
                idx = (rollout_begin + 1) * self.n_updates
                if do_resample:
                    for i in range(num_retrain):
                        train_ppo(
                            self.train_env,
                            self.ppo_config,
                            rollout_begin + 1,
                            rollout_end,
                            save_path=save_path / f"resample_{i}",
                            initial_policy=save_path / "retrain" / f"policy_grad_{idx}.zip",
                            seed=seed + i + 1,
                            cache_grad=self.training_config.cache_grad,
                            cache_rollouts=self.training_config.cache_rollouts,
                            n_envs=self.training_config.n_envs,
                            device=device,
                        )
                    final_model_idx = rollout_end * self.n_updates
                    final_model_path = [
                        save_path / f"resample_{i}" / f"policy_grad_{final_model_idx}.zip"
                        for i in range(num_retrain)
                    ]
                else:
                    train_ppo_with_selected_data(
                        self.training_config.env_name,
                        self.ppo_config,
                        model_path,
                        rollout_begin + 1,
                        rollout_end,
                        save_path=save_path / "retrain",
                        initial_policy=save_path / "retrain" / f"policy_grad_{idx}.zip",
                        cache_grad=False,
                        n_envs=self.training_config.n_envs,
                        device=device,
                    )
                    final_model_idx = rollout_end * self.n_updates
                    final_model_path = [save_path / "retrain" / f"policy_grad_{final_model_idx}.zip"]
            else:
                final_model_idx = rollout_end * self.n_updates
                final_model_path = [save_path / "retrain" / f"policy_grad_{final_model_idx}.zip"]

            sample_eval_results = []
            for path in final_model_path:
                loaded_model = PPO.load(path, device=device)
                loo_loss = compute_surrogate_validation_loss(loaded_model, target_rollout)[0]
                sample_eval_results.append(loo_loss)

            self.remove_cache(save_path)

            if self.training_config.clear_cache:
                self.remove_cache(save_path)

            return (batch, sample), [loo_loss - base_loss for loo_loss in sample_eval_results]

        loo_scores = {}
        processed_keys = []
        log_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_sample, batch, sample): (batch, sample)
                for batch, sample in samples_to_compute
            }
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Computing LOO scores (parallel)", unit="sample")
            for future in pbar:
                batch, sample = futures[future]
                key, scores = future.result()
                pbar.set_postfix({"batch": batch, "sample": sample})

                with log_lock:
                    loo_scores[key] = scores
                    processed_keys.append(key)

                    if att_scores is not None and (len(processed_keys) % max(1, log_freq) == 0):
                        paired_att = []
                        paired_loo = []
                        for k in processed_keys:
                            if k in att_scores:
                                paired_att.append(float(att_scores[k]))
                                paired_loo.append(float(loo_scores[k]))

                        if len(paired_att) >= 2:
                            try:
                                corr = float(
                                    np.corrcoef(np.array(paired_att), np.array(paired_loo))[0, 1]
                                )
                            except Exception:
                                corr = float(np.nan)
                            logger.info(
                                f"LOO vs attribution correlation after {len(processed_keys)} processed: {corr:.4f}"
                            )

        return {
            "loo_scores": loo_scores,
            "base_loss": base_loss,
            "retrain_base_losses": retrain_base_losses,
        }

    @property
    def attribution_func(self):
        if self.attribution_config.method == "none":
            return None
        elif self.attribution_config.method == "TracIn":
            from rl_attribution.attribution import compute_tracin_scores
            return compute_tracin_scores
        elif self.attribution_config.method == "HYDRA":
            from rl_attribution.attribution import compute_hydra_scores
            return compute_hydra_scores
        elif self.attribution_config.method == "RLIF-opt":
            from rl_attribution.attribution import compute_rlif_opt_scores
            return compute_rlif_opt_scores
        elif self.attribution_config.method == "RLIF":
            from rl_attribution.attribution import compute_rlif_scores
            return compute_rlif_scores
        elif self.attribution_config.method == "random":
            def random_scores(
                target_model,
                target_rollout,
                n_batches,
                model_path,
                begin_step,
                end_step,
                total_updates,
                device="cpu",
            ):
                influence_scores = {}
                logger.info("Generating random scores.")
                for t in range(begin_step, end_step):
                    for j in range(self.ppo_config.batch_size):
                        influence_scores[(t, j)] = np.random.randn()
                return {"random_scores": influence_scores}
            return random_scores
        elif self.attribution_config.method == "LOO":
            return self._loo_retrain_scores
        else:
            raise NotImplementedError

    @property
    def embedding_func(self):
        if self.attribution_config.method == "none":
            return None
        elif self.attribution_config.method == "TracIn":
            from rl_attribution.attribution import compute_tracin_embeddings
            return compute_tracin_embeddings
        elif self.attribution_config.method == "RLIF-opt":
            from rl_attribution.attribution import compute_rlif_opt_embeddings
            return compute_rlif_opt_embeddings
        else:
            raise NotImplementedError

    @property
    def n_batches(self):
        """Number of batches within one rollout."""
        return (
            self.ppo_config.n_steps *
            self.training_config.n_envs //
            self.ppo_config.batch_size
        )

    @property
    def n_updates(self):
        """Number of updates for one rollout. `n_batches * n_epoches`"""
        return self.n_batches * self.ppo_config.n_epochs

    @property
    def total_updates(self):
        """Number of total updates for all rollouts."""
        return self.n_updates * self.training_config.total_rollouts

    @staticmethod
    def model_path(
        prefix,
        rollout_begin,
        rollout_end,
    ):
        model_root = Path("./models")
        return model_root / prefix / f"{rollout_begin}-{rollout_end}"

    @staticmethod
    def attribution_path(
        prefix,
        rollout_begin,
        rollout_end,
    ):

        attribution_root = Path("./attribution")
        return attribution_root / prefix / f"{rollout_begin}-{rollout_end}"

    @staticmethod
    def dataset_path(
        rollout_begin,
        rollout_end,
    ):
        dataset_root = Path("./datasets")
        return dataset_root / f"{rollout_begin}-{rollout_end}"

    @staticmethod
    def eval_path(
        rollout_begin,
        rollout_end,
    ):
        dataset_root = Path("./eval")
        return dataset_root / f"{rollout_begin}-{rollout_end}"

    @staticmethod
    def remove_cache(
        cache_dirs,
    ):
        if not isinstance(cache_dirs, list):
            cache_dirs = [cache_dirs]

        for dir in cache_dirs:
            shutil.rmtree(dir)

    def preprocess(self, seed, device="cpu"):
        if self.attribution_config.target_model not in ["initial", "final"]:
            target_model_path = self.attribution_config.target_model.format(seed)
            target_model = PPO.load(target_model_path, device=device)
            logger.info(f"Using oracle model {target_model_path}")
            self.oracle_rollout, _ = sample_rollouts_buffers(
                self.train_env,
                self.attribution_config.target_rollout_size,
                rollout_model=target_model,
            )

    def get_target_model_idx(
        self,
        rollout_begin,
        rollout_end,
    ):
        if self.attribution_config.target_model == "final":
            return rollout_end * self.n_updates
        else:
            return rollout_begin * self.n_updates

    def get_target_model(
        self,
        model_path,
        rollout_begin,
        rollout_end,
        device="cpu",
    ):
        if isinstance(model_path, str):
            model_path = Path(model_path)
        idx = self.get_target_model_idx(rollout_begin, rollout_end)
        target_model_path = model_path / f"policy_grad_{idx}.zip"
        logger.info(f"Using target model: {target_model_path}")
        target_model = PPO.load(target_model_path, device=device)

        return target_model

    def run_experiments(
        self,
        device="cpu",
    ):
        if self.training_config.experiment_type == "training":
            for seed in self.seeds:
                logger.info(f"Running experiment with seed {seed}")
                self.run_training(
                    seed=seed,
                    device=device,
                )
        elif self.training_config.experiment_type == "loo_retrain":
            for seed in self.seeds:
                logger.info(f"Running experiment with seed {seed}")
                initial_policy_dir = self.training_config.initial_policy_dir
                initial_policy = None
                assert self.training_config.rollout_begin == 0 or initial_policy_dir is not None
                if initial_policy_dir is not None:
                    ckpt_name = f"policy_grad_{self.training_config.rollout_begin * self.n_updates}.zip"
                    initial_policy = Path(initial_policy_dir.format(seed)) / ckpt_name

                self.run_global_loo_training_with_sampling(
                    initial_policy=initial_policy,
                    seed=seed,
                    device=device,
                )
        elif self.training_config.experiment_type == "single_rollout_perturbation":
            for seed in self.seeds:
                logger.info(f"Running experiment with seed {seed}")
                initial_policy_dir = self.training_config.initial_policy_dir
                initial_policy = None
                assert self.training_config.rollout_begin == 0 or initial_policy_dir is not None
                if initial_policy_dir is not None:
                    ckpt_name = f"policy_grad_{self.training_config.rollout_begin * self.n_updates}.zip"
                    initial_policy = Path(initial_policy_dir.format(seed)) / ckpt_name

                self.run_single_rollout_perturbation_training(
                    initial_policy=initial_policy,
                    seed=seed,
                    device=device,
                )
        elif self.training_config.experiment_type == "single_rollout_perturbation_with_sampling":
            for seed in self.seeds:
                logger.info(f"Running experiment with seed {seed}")
                initial_policy_dir = self.training_config.initial_policy_dir
                initial_policy = None
                assert self.training_config.rollout_begin == 0 or initial_policy_dir is not None
                if initial_policy_dir is not None:
                    ckpt_name = f"policy_grad_{self.training_config.rollout_begin * self.n_updates}.zip"
                    initial_policy = Path(initial_policy_dir.format(seed)) / ckpt_name

                self.run_single_rollout_perturbation_training_with_sampling(
                    initial_policy=initial_policy,
                    seed=seed,
                    device=device,
                )
        elif self.training_config.experiment_type == "lds_retraining":
            for seed in self.seeds:
                logger.info(f"Running experiment with seed {seed}")
                initial_policy_dir = self.training_config.initial_policy_dir
                initial_policy = None
                assert self.training_config.rollout_begin == 0 or initial_policy_dir is not None
                if initial_policy_dir is not None:
                    ckpt_name = f"policy_grad_{self.training_config.rollout_begin * self.n_updates}.zip"
                    initial_policy = Path(initial_policy_dir.format(seed)) / ckpt_name

                self.run_lds_retraining(
                    initial_policy=initial_policy,
                    seed=seed,
                    device=device,
                )
        elif self.training_config.experiment_type == "training_with_seleced_data":
            for seed in self.seeds:
                logger.info(f"Running experiment with seed {seed}")
                initial_policy_dir = self.training_config.initial_policy_dir
                initial_policy = None
                assert self.training_config.rollout_begin == 0 or initial_policy_dir is not None
                if initial_policy_dir is not None:
                    ckpt_name = f"policy_grad_{self.training_config.rollout_begin * self.n_updates}.zip"
                    initial_policy = Path(initial_policy_dir.format(seed)) / ckpt_name

                self.run_train_with_selected_data(
                    initial_policy=initial_policy,
                    seed=seed,
                    device=device,
                )
        elif self.training_config.experiment_type == "compute_influence":
            for seed in self.seeds:
                logger.info(f"Running experiment with seed {seed}")
                initial_policy_dir = self.training_config.initial_policy_dir
                initial_policy = None
                assert self.training_config.rollout_begin == 0 or initial_policy_dir is not None
                if initial_policy_dir is not None:
                    ckpt_name = f"policy_grad_{self.training_config.rollout_begin * self.n_updates}.zip"
                    initial_policy = Path(initial_policy_dir.format(seed)) / ckpt_name

                self.run_influence_computation(
                    initial_policy=initial_policy,
                    seed=seed,
                    device=device,
                )
        elif self.training_config.experiment_type == "compute_embedding":
            for batches_dir in self.training_config.batches_dirs:
                self.run_embedding_computation(
                    batches_dir=Path(batches_dir),
                    device=device,
                )
        elif self.training_config.experiment_type == "evaluate":
            for seed in self.seeds:
                rollout_begin = self.training_config.rollout_begin
                rollout_end = self.training_config.rollout_end
                save_path = self.root_dir / f"seed-{seed}"/ self.model_path(
                    "base",
                    rollout_begin,
                    rollout_end,
                )
                eval_save_path = self.root_dir / f"seed-{seed}" / self.eval_path(
                    rollout_begin,
                    rollout_end,
                )
                self.evaluate_return(
                    save_path,
                    eval_save_path,
                    rollout_begin,
                    rollout_end,
                    device=device,
                )

                save_path = self.root_dir / f"seed-{seed}"/ self.model_path(
                    "base_local",
                    rollout_begin,
                    rollout_end,
                )
                if os.path.exists(save_path):
                    eval_save_path = self.root_dir / f"seed-{seed}" / self.eval_path(
                        rollout_begin,
                        rollout_end,
                    ) / "local"
                    self.evaluate_return(
                        save_path,
                        eval_save_path,
                        rollout_begin,
                        rollout_end,
                        device=device,
                    )
        else:
            raise NotImplementedError()

    def evaluate_return(
        self,
        save_path,
        eval_save_path,
        rollout_begin,
        rollout_end,
        device="cpu",
    ):
        step_size = self.eval_config.step_size

        os.makedirs(eval_save_path, exist_ok=True)
        eval_rollouts = list(range(rollout_begin, rollout_end, step_size))
        eval_rollouts.append(rollout_end)
        logger.info(f"Evaluating rollouts: {eval_rollouts}")
        for k in eval_rollouts:
            idx = k * self.n_updates
            model_path = save_path / f"policy_grad_{idx}.zip"
            if not os.path.exists(model_path):
                logger.warning(f"Model path {model_path} does not exist, skipping.")
                continue
            model = PPO.load(model_path, device=device)
            logger.info(f"Evaluating {model_path}")
            eval_results = evaluate_policy(
                model,
                self.training_config.env_name,
                self.eval_config.n_eval_episodes,
                n_envs=self.eval_config.eval_n_envs,
            )
            eval_file = eval_save_path / f"eval_{idx}.json"
            with open(eval_file, "w") as f:
                json.dump(eval_results, f)

    def run_influence_computation(
        self,
        initial_policy=None,
        seed=42,
        device="cpu",
    ):
        initial_policy_dir = Path(self.training_config.initial_policy_dir.format(seed))
        rollout_begin = self.training_config.rollout_begin
        rollout_end = self.training_config.rollout_end

        att_save_path = self.root_dir / f"seed-{seed}" / self.attribution_path(
            self.attribution_config.method,
            rollout_begin,
            rollout_end,
        )
        os.makedirs(att_save_path, exist_ok=True)
        att_file = att_save_path / f"scores_{rollout_begin}_{rollout_end}.pkl"
        rollout_file = att_save_path / f"target_rollout_{rollout_begin}_{rollout_end}.pkl"

        idx = self.get_target_model_idx(rollout_begin, rollout_end)
        target_model_path = initial_policy_dir / f"policy_grad_{idx}.zip"
        logger.info(f"Using target model: {target_model_path}")
        target_model = PPO.load(target_model_path, device=device)

        scores, target_rollout = self.compute_influence(
            target_model,
            initial_policy_dir,
            rollout_begin,
            rollout_end,
            buffer_rollout_begin=rollout_end-1,
            device=device,
            seed=seed,
        )

        with open(att_file, "wb") as f:
            pickle.dump(scores, f)

        with open(rollout_file, "wb") as f:
            pickle.dump(target_rollout, f)

    def compute_influence(
        self,
        target_model,
        batches_dir,
        rollout_begin,
        rollout_end,
        buffer_rollout_begin=None,
        buffer_rollout_end=None,
        seed=42,
        device="cpu",
    ):
        if self.attribution_config.method == "random":
            target_rollout = None

        elif self.attribution_config.target_buffer == "reuse":
            if buffer_rollout_begin is None:
                buffer_rollout_begin = rollout_begin
            if buffer_rollout_end is None:
                buffer_rollout_end = rollout_end

            target_rollout, _ = collect_training_buffers(
                batches_dir,
                buffer_rollout_begin,
                buffer_rollout_end,
                n_batches=self.n_batches,
                n_epochs=self.ppo_config.n_epochs,
            )

        elif self.attribution_config.target_buffer == "sample":
            target_rollout, _ = sample_rollouts_buffers(
                self.train_env,
                self.attribution_config.target_rollout_size,
                rollout_model=target_model,
            )

        elif self.attribution_config.target_buffer == "oracle":
            rollout_model_path = self.attribution_config.target_model.format(seed)
            rollout_target_model = PPO.load(rollout_model_path, device=device)
            target_rollout, _ = sample_rollouts_buffers(
                self.train_env,
                self.attribution_config.target_rollout_size,
                rollout_model=rollout_target_model,
            )
            del rollout_target_model

        else:
            raise NotImplementedError()

        attribution_kwargs = {
            "target_model": target_model,
            "target_rollout": target_rollout,
            "n_batches": self.n_updates,
            "model_path": batches_dir,
            "begin_step": rollout_begin * self.n_updates,
            "end_step": rollout_end * self.n_updates,
            "total_updates": self.total_updates,
            "device": device,
        }
        if self.attribution_config.method == "LOO":
            attribution_kwargs["seed"] = seed

        scores = self.attribution_func(
            **attribution_kwargs
        )

        if self.attribution_config.ensemble > 1:
            assert rollout_begin + 1 < rollout_end
            ensemble_save_path = self.root_dir / f"seed-{seed}" / self.model_path(
                f"{self.attribution_config.method}_ensemble",
                rollout_begin,
                rollout_end,
            )
            os.makedirs(ensemble_save_path, exist_ok=True)

            model_ckpt_file = batches_dir / f"policy_grad_{rollout_begin * self.n_updates}.zip"
            shutil.copy(model_ckpt_file, ensemble_save_path)
            for i in range(rollout_begin * self.n_updates, (rollout_begin + 1) * self.n_updates):
                model_ckpt_file = batches_dir / f"policy_grad_{i + 1}.zip"
                batch_file = batches_dir / f"batch_{i + 1}.pkl"
                shutil.copy(model_ckpt_file, ensemble_save_path)
                shutil.copy(batch_file, ensemble_save_path)

            for i in range(self.attribution_config.ensemble - 1):
                idx = (rollout_begin + 1) * self.n_updates
                train_ppo(
                    self.train_env,
                    self.ppo_config,
                    rollout_begin + 1,
                    rollout_end,
                    save_path=ensemble_save_path,
                    initial_policy=ensemble_save_path / f"policy_grad_{idx}.zip",
                    seed=seed + i + 1,
                    cache_grad=self.training_config.cache_grad,
                    cache_rollouts=self.training_config.cache_rollouts,
                    n_envs=self.training_config.n_envs,
                    device=device,
                )

                target_rollout["observations"] = target_rollout["observations"].unsqueeze(1)
                ensemble_scores = self.attribution_func(
                    target_model,
                    target_rollout,
                    self.n_updates,
                    ensemble_save_path,
                    rollout_begin * self.n_updates,
                    rollout_end * self.n_updates,
                    self.total_updates,
                    device=device,
                )

                for i in range((rollout_begin + 1) * self.n_updates, rollout_end * self.n_updates):
                    model_ckpt_file = ensemble_save_path / f"policy_grad_{i + 1}.zip"
                    batch_file = ensemble_save_path / f"batch_{i + 1}.pkl"
                    os.remove(model_ckpt_file)
                    os.remove(batch_file)

                for key in scores["influence_scores"]:
                    assert key in ensemble_scores["influence_scores"], f"Key {key} not found in ensemble scores"
                    if not isinstance(scores["influence_scores"][key], list):
                        scores["influence_scores"][key] = [scores["influence_scores"][key]]
                    scores["influence_scores"][key].append(ensemble_scores["influence_scores"][key])

        return scores, target_rollout

    def run_embedding_computation(
        self,
        batches_dir,
        device="cpu",
    ):
        match = re.search(r"seed-(\d+)", str(batches_dir))
        if match:
            seed_number = int(match.group(1))

        rollout_begin = self.training_config.rollout_begin
        rollout_end = self.training_config.rollout_end

        att_save_path = self.root_dir / f"seed-{seed_number}" / self.attribution_path(
            self.attribution_config.method,
            rollout_begin,
            rollout_end,
        )
        os.makedirs(att_save_path, exist_ok=True)
        att_file = att_save_path / f"embeddings_{rollout_begin}_{rollout_end}.pkl"

        idx = self.get_target_model_idx(rollout_begin, rollout_end)
        target_model_path = batches_dir / f"policy_grad_{idx}.zip"
        logger.info(f"Using target model: {target_model_path}")
        target_model = PPO.load(target_model_path, device=device)

        embeddings = self.compute_embeddings(
            target_model,
            batches_dir,
            rollout_begin,
            rollout_end,
            device=device,
        )

        with open(att_file, "wb") as f:
            pickle.dump(embeddings, f)

    def compute_embeddings(
        self,
        target_model,
        batches_dir,
        rollout_begin,
        rollout_end,
        buffer_rollout_begin=None,
        buffer_rollout_end=None,
        device="cpu",
    ):
        if buffer_rollout_begin is None:
            buffer_rollout_begin = rollout_begin
        if buffer_rollout_end is None:
            buffer_rollout_end = rollout_end

        target_rollout, _ = collect_training_buffers(
            batches_dir,
            buffer_rollout_begin,
            buffer_rollout_end,
            n_batches=self.n_batches,
            n_epochs=self.ppo_config.n_epochs,
        )

        embeddings = self.embedding_func(
            target_model,
            target_rollout,
            self.n_updates,
            batches_dir,
            rollout_begin * self.n_updates,
            rollout_end * self.n_updates,
            self.total_updates,
            device=device,
        )

        return embeddings

    def run_training(
        self,
        seed=42,
        device="cpu",
    ):
        method = self.training_config.data_selection_method
        initial_policy_dir = self.training_config.initial_policy_dir
        initial_policy = None
        assert self.training_config.rollout_begin == 0 or initial_policy_dir is not None
        if initial_policy_dir is not None:
            ckpt_name = f"policy_grad_{self.training_config.rollout_begin * self.n_updates}.zip"
            initial_policy = Path(initial_policy_dir.format(seed)) / ckpt_name

        if method == "none":
            self.run_baseline_training(
                initial_policy=initial_policy,
                seed=seed,
                device=device,
            )
        elif method == "local":
            self.run_local_attribution_training(
                initial_policy=initial_policy,
                seed=seed,
            )
        elif method == "global":
            self.run_global_attribution_training(
                initial_policy=initial_policy,
                seed=seed,
            )
        elif method == "global_iterative":
            self.run_global_iterative_attribution_training(
                initial_policy=initial_policy,
                seed=seed,
            )
        else:
            raise NotImplementedError()

    def run_baseline_training(
        self,
        initial_policy=None,
        seed=42,
        device="cpu",
    ):
        from rl_attribution.trainers.ppo import PPOWithLocalDataSelection
        from stable_baselines3.common.utils import get_linear_fn

        rollout_begin = self.training_config.rollout_begin
        rollout_end = self.training_config.rollout_end

        save_path = self.root_dir / f"seed-{seed}"
        model_save_path = save_path / self.model_path(
            "base",
            rollout_begin,
            rollout_end,
        )
        eval_save_path = save_path / self.eval_path(
            rollout_begin,
            rollout_end,
        )
        os.makedirs(eval_save_path, exist_ok=True)

        policy_kwargs = dict(
            optimizer_class=torch.optim.SGD if self.ppo_config.optimizer_class == "SGD" else torch.optim.Adam,
        )

        if self.ppo_config.policy_type == "CnnPolicy":
            from rl_attribution.models.cnn import get_cnn_cls
            if self.ppo_config.policy_name != "none":
                policy_kwargs["features_extractor_class"] = get_cnn_cls(self.ppo_config.policy_name)
                policy_kwargs["features_extractor_kwargs"] = dict(features_dim=self.ppo_config.features_dim)

        if self.ppo_config.policy_kwargs is not None and len(self.ppo_config.policy_kwargs) > 0:
            print(self.ppo_config.policy_kwargs)
            policy_kwargs.update(self.ppo_config.policy_kwargs)

        if "activation_fn" in policy_kwargs:
            act_fn_name = policy_kwargs["activation_fn"]
            if act_fn_name in ACTIVATION_MAPPING:
                policy_kwargs["activation_fn"] = ACTIVATION_MAPPING[act_fn_name]
            else:
                raise ValueError(f"Unsupported activation function: {act_fn_name}")

        learning_schedule = get_linear_fn(
            start=self.ppo_config.learning_rate,
            end=self.ppo_config.final_learning_rate,
            end_fraction=1.0,
        ) if self.ppo_config.final_learning_rate is not None else self.ppo_config.learning_rate

        clip_schedule = get_linear_fn(
            start=self.ppo_config.clip_range,
            end=self.ppo_config.final_clip_range,
            end_fraction=1.0,
        ) if self.ppo_config.final_clip_range is not None else self.ppo_config.clip_range

        trainer = PPOWithLocalDataSelection(
            policy=self.ppo_config.policy_type,
            env=self.train_env,
            attribution_func=None,
            attribution_config=None,
            learning_rate=learning_schedule,
            n_steps=self.ppo_config.n_steps,
            batch_size=self.ppo_config.batch_size,
            n_epochs=self.ppo_config.n_epochs,
            gamma=self.ppo_config.gamma,
            gae_lambda=self.ppo_config.gae_lambda,
            clip_range=clip_schedule,
            clip_range_vf=None,
            normalize_advantage=self.ppo_config.normalize_advantage,
            ent_coef=self.ppo_config.ent_coef,
            vf_coef=self.ppo_config.vf_coef,
            kl_coef=self.ppo_config.kl_coef,
            max_grad_norm=self.ppo_config.max_grad_norm,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            stats_window_size=100,
            tensorboard_log=None,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
            device=device,
            save_path=save_path,
            cache_grad=self.training_config.cache_grad,
            cache_rollouts=self.training_config.cache_rollouts,
            render_rollouts=self.training_config.render_rollouts,
        )

        if initial_policy is not None:
            model_init = PPO.load(initial_policy, device=device)
            trainer.policy.load_state_dict(model_init.policy.state_dict())
            logger.info(f'Loaded initial policy: {initial_policy}')
            del model_init

        total_timesteps = self.ppo_config.total_timesteps
        trainer.learn(
            total_timesteps=total_timesteps,
            rollout_begin=rollout_begin,
            rollout_end=rollout_end,
        )

        self.evaluate_return(
            model_save_path,
            eval_save_path,
            rollout_begin,
            rollout_end,
            device=device,
        )

        return save_path

    def run_loo_training(
        self,
        batches_dir,
        device="cpu",
    ):
        match = re.search(r"seed-(\d+)", str(batches_dir))
        if match:
            seed_number = int(match.group(1))

        rollout_begin = self.training_config.rollout_begin
        rollout_end = self.training_config.rollout_end

        target_rollout, _ = collect_training_buffers(
            batches_dir,
            rollout_begin,
            rollout_end,
            n_batches=self.n_batches,
            n_epochs=self.ppo_config.n_epochs,
        )

        base_save_path = self.root_dir / f"seed-{seed_number}" / self.model_path(
            "loo_retrain",
            rollout_begin,
            rollout_end,
        )
        initial_policy = batches_dir / f"policy_grad_{rollout_begin * self.n_updates}.zip"

        train_ppo_with_selected_data(
            self.training_config.env_name,
            self.ppo_config,
            batches_dir,
            rollout_begin,
            rollout_end,
            save_path=base_save_path / "base",
            initial_policy=initial_policy,
            cache_grad=False,
            n_envs=self.training_config.n_envs,
            device=device,
        )
        idx = rollout_end * self.n_updates
        model = PPO.load(base_save_path / "base" / f"policy_grad_{idx}.zip", device=device)
        eval_save_path = self.root_dir / f"seed-{seed_number}" / self.eval_path(
            rollout_begin,
            rollout_end,
        ) / "base"
        os.makedirs(eval_save_path, exist_ok=True)

        eval_file = eval_save_path / f"loss_{idx}.json"
        eval_results = compute_surrogate_validation_loss(model, target_rollout)
        with open(eval_file, "w") as f:
            json.dump(eval_results, f)

        for batch in range(self.n_updates * rollout_begin, self.n_updates * rollout_end):
            for sample in range(self.ppo_config.batch_size):
                logger.info(f"Retraining without batch {batch}, sample {sample}")
                save_path = base_save_path / f"batch_{batch}_sample_{sample}"
                eval_save_path = self.root_dir / f"seed-{seed_number}" / self.eval_path(
                    rollout_begin,
                    rollout_end,
                ) / f"batch_{batch}_sample_{sample}"
                os.makedirs(save_path, exist_ok=True)
                os.makedirs(eval_save_path, exist_ok=True)

                train_ppo_with_selected_data(
                    self.training_config.env_name,
                    self.ppo_config,
                    batches_dir,
                    rollout_begin,
                    rollout_end,
                    save_path=save_path,
                    initial_policy=initial_policy,
                    cache_grad=False,
                    n_envs=self.training_config.n_envs,
                    device=device,
                    leave_out_idx=(batch, sample),
                )

                idx = rollout_end * self.n_updates
                model_path = save_path / f"policy_grad_{idx}.zip"
                model = PPO.load(model_path, device=device)

                eval_file = eval_save_path / f"loss_{idx}.json"
                eval_results = compute_surrogate_validation_loss(model, target_rollout)
                with open(eval_file, "w") as f:
                    json.dump(eval_results, f)

                if self.training_config.clear_cache:
                    self.remove_cache(save_path)

    def run_single_rollout_perturbation_training(
        self,
        initial_policy=None,
        num_retrain=32,
        seed=42,
        device="cpu",
    ):
        initial_policy_dir = Path(self.training_config.initial_policy_dir.format(seed))
        rollout_begin = self.training_config.rollout_begin
        rollout_end = self.training_config.rollout_end

        perturbation_save_path = self.root_dir / f"seed-{seed}" / self.model_path(
            "single_rollout_perturbation",
            rollout_begin,
            rollout_end,
        )

        base_save_path = self.root_dir / f"seed-{seed}" / self.model_path(
            "baseline_retrain",
            rollout_begin,
            rollout_end,
        )

        baseline_losses = []
        perturbed_losses = []
        for i in range(num_retrain):
            idx = (rollout_begin + 1) * self.n_updates
            os.makedirs(perturbation_save_path / f"resample_{i}", exist_ok=True)
            train_ppo(
                self.train_env,
                self.ppo_config,
                rollout_begin,
                rollout_end,
                save_path=base_save_path / f"resample_{i}",
                initial_policy=initial_policy_dir / f"policy_grad_{idx}.zip",
                seed=seed + i + 1,
                cache_grad=True,
                cache_rollouts=self.training_config.cache_rollouts,
                n_envs=self.training_config.n_envs,
                device=device,
            )

            target_model = self.get_target_model(
                base_save_path / f"resample_{i}",
                rollout_begin,
                rollout_end,
            )

            scores, target_rollout = self.compute_influence(
                target_model,
                base_save_path / f"resample_{i}",
                rollout_begin,
                rollout_end,
                buffer_rollout_begin=rollout_end-1,
                device=device,
                seed=seed,
            )

            final_model_idx = rollout_end * self.n_updates
            final_model_path = base_save_path / f"resample_{i}" / f"policy_grad_{final_model_idx}.zip"
            model = PPO.load(final_model_path, device=device)

            loo_loss = compute_surrogate_validation_loss(model, target_rollout)[0]
            baseline_losses.append(loo_loss)

            influence_scores = scores["influence_scores"]
            drop_percentile = self.attribution_config.drop_percentile

            sorted_samples = sorted(influence_scores.items(), key=lambda x: x[1])

            n_samples_to_drop = int(len(sorted_samples) * drop_percentile / 100)

            leave_out_list = [idx_tuple for idx_tuple, score in sorted_samples[:n_samples_to_drop]]

            logger.info(f"Dropping {n_samples_to_drop} samples ({drop_percentile:.1f}%) with lowest influence scores")
            logger.info(f"Bottom score: {sorted_samples[0][1] if sorted_samples else 'N/A'}")
            logger.info(f"Top dropped score: {sorted_samples[n_samples_to_drop-1][1] if n_samples_to_drop > 0 else 'N/A'}")

            train_ppo_with_selected_data(
                self.training_config.env_name,
                self.ppo_config,
                base_save_path / f"resample_{i}",
                rollout_begin,
                rollout_end,
                save_path=perturbation_save_path / f"resample_{i}",
                initial_policy=initial_policy,
                cache_grad=False,
                n_envs=self.training_config.n_envs,
                device=device,
                leave_out_idx=leave_out_list,
                leave_out_weight=1.0,
            )

            final_model_idx = rollout_end * self.n_updates
            final_model_path = perturbation_save_path / f"resample_{i}" / f"policy_grad_{final_model_idx}.zip"
            model = PPO.load(final_model_path, device=device)

            loo_loss = compute_surrogate_validation_loss(model, target_rollout)[0]
            perturbed_losses.append(loo_loss)

        att_save_path = self.root_dir / f"seed-{seed}" / self.attribution_path(
            self.attribution_config.method,
            rollout_begin,
            rollout_end,
        )
        os.makedirs(att_save_path, exist_ok=True)
        att_file = att_save_path / f"perturbation_{rollout_begin}_{rollout_end}.pkl"

        with open(att_file, "wb") as f:
            pickle.dump({
                "perturbed_losses": perturbed_losses,
                "baseline_losses": baseline_losses,
            }, f)

    def run_single_rollout_perturbation_training_with_sampling(
        self,
        initial_policy=None,
        num_retrain=256,
        seed=42,
        device="cpu",
    ):
        initial_policy_dir = Path(self.training_config.initial_policy_dir.format(seed))
        rollout_begin = self.training_config.rollout_begin
        rollout_end = self.training_config.rollout_end

        perturbation_save_path = self.root_dir / f"seed-{seed}" / self.model_path(
            "single_rollout_perturbation",
            rollout_begin,
            rollout_end,
        )

        target_model = self.get_target_model(
            initial_policy_dir,
            rollout_begin,
            rollout_end,
        )

        scores, target_rollout = self.compute_influence(
            target_model,
            initial_policy_dir,
            rollout_begin,
            rollout_end,
            buffer_rollout_begin=rollout_end-1,
            device=device,
            seed=seed,
        )

        influence_scores = scores["influence_scores"]
        drop_percentile = self.attribution_config.drop_percentile

        sorted_samples = sorted(influence_scores.items(), key=lambda x: x[1])

        n_samples_to_drop = int(len(sorted_samples) * drop_percentile / 100)

        leave_out_list = [idx_tuple for idx_tuple, score in sorted_samples[:n_samples_to_drop]]

        logger.info(f"Dropping {n_samples_to_drop} samples ({drop_percentile:.1f}%) with lowest influence scores")
        logger.info(f"Bottom score: {sorted_samples[0][1] if sorted_samples else 'N/A'}")
        logger.info(f"Top dropped score: {sorted_samples[n_samples_to_drop-1][1] if n_samples_to_drop > 0 else 'N/A'}")

        train_ppo_with_selected_data(
            self.training_config.env_name,
            self.ppo_config,
            initial_policy_dir,
            rollout_begin,
            rollout_begin + 1,
            save_path=perturbation_save_path / "retrain",
            initial_policy=initial_policy,
            cache_grad=False,
            n_envs=self.training_config.n_envs,
            device=device,
            leave_out_idx=leave_out_list,
            leave_out_weight=1.0,
        )

        if rollout_end > rollout_begin + 1:
            idx = (rollout_begin + 1) * self.n_updates
            for i in tqdm(range(num_retrain)):
                train_ppo(
                    self.train_env,
                    self.ppo_config,
                    rollout_begin + 1,
                    rollout_end,
                    save_path=perturbation_save_path / f"resample_{i}",
                    initial_policy=perturbation_save_path / "retrain" / f"policy_grad_{idx}.zip",
                    seed=seed + i + 1,
                    cache_grad=self.training_config.cache_grad,
                    cache_rollouts=self.training_config.cache_rollouts,
                    n_envs=self.training_config.n_envs,
                    device=device,
                )
            final_model_idx = rollout_end * self.n_updates
            final_model_path = [
                perturbation_save_path / f"resample_{i}" / f"policy_grad_{final_model_idx}.zip"
                for i in range(num_retrain)
            ]
        else:
            final_model_idx = rollout_end * self.n_updates
            final_model_path = [perturbation_save_path / "retrain" / f"policy_grad_{final_model_idx}.zip"]

        perturbed_losses = []
        for path in final_model_path:
            model = PPO.load(path, device=device)

            loo_loss = compute_surrogate_validation_loss(model, target_rollout)[0]
            perturbed_losses.append(loo_loss)

        base_save_path = self.root_dir / f"seed-{seed}" / self.model_path(
            "baseline_retrain",
            rollout_begin,
            rollout_end,
        )

        train_ppo_with_selected_data(
            self.training_config.env_name,
            self.ppo_config,
            initial_policy_dir,
            rollout_begin,
            rollout_begin + 1,
            save_path=base_save_path / "retrain",
            initial_policy=initial_policy,
            cache_grad=False,
            n_envs=self.training_config.n_envs,
            device=device,
        )

        if rollout_end > rollout_begin + 1:
            idx = (rollout_begin + 1) * self.n_updates
            for i in tqdm(range(num_retrain)):
                train_ppo(
                    self.train_env,
                    self.ppo_config,
                    rollout_begin + 1,
                    rollout_end,
                    save_path=base_save_path / f"resample_{i}",
                    initial_policy=base_save_path / "retrain" / f"policy_grad_{idx}.zip",
                    seed=seed + i + 1,
                    cache_grad=self.training_config.cache_grad,
                    cache_rollouts=self.training_config.cache_rollouts,
                    n_envs=self.training_config.n_envs,
                    device=device,
                )
            final_model_idx = rollout_end * self.n_updates
            final_model_path = [
                base_save_path / f"resample_{i}" / f"policy_grad_{final_model_idx}.zip"
                for i in range(num_retrain)
            ]
        else:
            final_model_idx = rollout_end * self.n_updates
            final_model_path = [base_save_path / "retrain" / f"policy_grad_{final_model_idx}.zip"]

        baseline_losses = []
        for path in final_model_path:
            model = PPO.load(path, device=device)

            loo_loss = compute_surrogate_validation_loss(model, target_rollout)[0]
            baseline_losses.append(loo_loss)

        att_save_path = self.root_dir / f"seed-{seed}" / self.attribution_path(
            self.attribution_config.method,
            rollout_begin,
            rollout_end,
        )
        os.makedirs(att_save_path, exist_ok=True)
        att_file = att_save_path / f"perturbation_{rollout_begin}_{rollout_end}.pkl"

        with open(att_file, "wb") as f:
            pickle.dump({
                "perturbed_losses": perturbed_losses,
                "baseline_losses": baseline_losses,
            }, f)

        self.remove_cache([perturbation_save_path, base_save_path])

    def run_lds_retraining(
        self,
        initial_policy=None,
        num_retrain=32,
        seed=42,
        device="cpu",
    ):
        import random
        assert initial_policy is not None, "LDS retraining requires an initial policy checkpoint"

        initial_policy_dir = Path(self.training_config.initial_policy_dir.format(seed))
        rollout_begin = self.training_config.rollout_begin
        rollout_end = self.training_config.rollout_end

        perturbation_save_path = self.root_dir / f"seed-{seed}" / self.model_path(
            "single_rollout_perturbation",
            rollout_begin,
            rollout_end,
        )
        os.makedirs(perturbation_save_path, exist_ok=True)

        target_rollout, _ = collect_training_buffers(
            initial_policy_dir,
            rollout_end-1,
            rollout_end,
            n_batches=self.n_batches,
            n_epochs=self.ppo_config.n_epochs,
        )

        all_samples = []
        for batch in range(self.n_updates * rollout_begin, self.n_updates * (rollout_begin + 1)):
            for sample in range(self.ppo_config.batch_size):
                all_samples.append((batch, sample))

        n_samples_to_drop = getattr(self.attribution_config, "n_samples_to_drop", None)
        if n_samples_to_drop is None:
            drop_percentile = 5.0
            n_samples_to_drop = int(len(all_samples) * drop_percentile / 100)

        if n_samples_to_drop <= 0:
            raise ValueError(f"n_samples_to_drop must be > 0, got {n_samples_to_drop}")
        if n_samples_to_drop > len(all_samples):
            raise ValueError(
                f"n_samples_to_drop ({n_samples_to_drop}) is larger than available samples ({len(all_samples)})"
            )

        n_subsets = 100
        lds_losses = []
        sampled_indices = []
        lds_resample_seeds = []

        logger.info(
            f"Running LDS retraining with {n_subsets} random subsets, "
            f"dropping {n_samples_to_drop}/{len(all_samples)} samples each, "
            f"with {num_retrain} continuation runs per subset"
        )

        for subset_id in tqdm(range(n_subsets), desc="LDS subsets", unit="subset"):
            rng = random.Random(seed + subset_id + 100000)
            leave_out_list = rng.sample(all_samples, n_samples_to_drop)
            sampled_indices.append(leave_out_list)
            subset_losses = []
            subset_seeds = []

            subset_save_path = perturbation_save_path / f"subset_{subset_id}"
            retrain_save_path = subset_save_path / "retrain"
            os.makedirs(subset_save_path, exist_ok=True)

            train_ppo_with_selected_data(
                self.training_config.env_name,
                self.ppo_config,
                initial_policy_dir,
                rollout_begin,
                rollout_begin + 1,
                save_path=retrain_save_path,
                initial_policy=initial_policy,
                cache_grad=False,
                n_envs=self.training_config.n_envs,
                device=device,
                leave_out_idx=leave_out_list,

            )

            if rollout_end > rollout_begin + 1:
                idx = (rollout_begin + 1) * self.n_updates
                for retrain_id in range(num_retrain):
                    resample_seed = seed + 10000 + subset_id * num_retrain + retrain_id
                    resample_save_path = subset_save_path / f"resample_{retrain_id}"
                    train_ppo(
                        self.train_env,
                        self.ppo_config,
                        rollout_begin + 1,
                        rollout_end,
                        save_path=resample_save_path,
                        initial_policy=retrain_save_path / f"policy_grad_{idx}.zip",
                        seed=resample_seed,
                        cache_grad=self.training_config.cache_grad,
                        cache_rollouts=self.training_config.cache_rollouts,
                        n_envs=self.training_config.n_envs,
                        device=device,
                    )
                    final_model_path = resample_save_path / f"policy_grad_{rollout_end * self.n_updates}.zip"
                    model = PPO.load(final_model_path, device=device)
                    subset_loss = compute_surrogate_validation_loss(model, target_rollout)[0]
                    subset_losses.append(float(subset_loss))
                    subset_seeds.append(resample_seed)

                    if self.training_config.clear_cache:
                        self.remove_cache(resample_save_path)
            else:
                final_model_path = retrain_save_path / f"policy_grad_{rollout_end * self.n_updates}.zip"
                model = PPO.load(final_model_path, device=device)
                subset_loss = compute_surrogate_validation_loss(model, target_rollout)[0]
                subset_losses.append(float(subset_loss))
                subset_seeds.append(seed)

            lds_losses.append(subset_losses)
            lds_resample_seeds.append(subset_seeds)

            if self.training_config.clear_cache:
                self.remove_cache(retrain_save_path)

        base_save_path = self.root_dir / f"seed-{seed}" / self.model_path(
            "baseline_retrain",
            rollout_begin,
            rollout_end,
        )
        os.makedirs(base_save_path, exist_ok=True)

        train_ppo_with_selected_data(
            self.training_config.env_name,
            self.ppo_config,
            initial_policy_dir,
            rollout_begin,
            rollout_begin + 1,
            save_path=base_save_path / "retrain",
            initial_policy=initial_policy,
            cache_grad=False,
            n_envs=self.training_config.n_envs,
            device=device,
        )

        baseline_losses = []
        baseline_resample_seeds = []
        if rollout_end > rollout_begin + 1:
            idx = (rollout_begin + 1) * self.n_updates
            for retrain_id in range(num_retrain):
                resample_seed = seed + 1000000 + retrain_id
                resample_save_path = base_save_path / f"resample_{retrain_id}"
                train_ppo(
                    self.train_env,
                    self.ppo_config,
                    rollout_begin + 1,
                    rollout_end,
                    save_path=resample_save_path,
                    initial_policy=base_save_path / "retrain" / f"policy_grad_{idx}.zip",
                    seed=resample_seed,
                    cache_grad=self.training_config.cache_grad,
                    cache_rollouts=self.training_config.cache_rollouts,
                    n_envs=self.training_config.n_envs,
                    device=device,
                )
                final_model_path = resample_save_path / f"policy_grad_{rollout_end * self.n_updates}.zip"
                model = PPO.load(final_model_path, device=device)
                baseline_loss = compute_surrogate_validation_loss(model, target_rollout)[0]
                baseline_losses.append(float(baseline_loss))
                baseline_resample_seeds.append(resample_seed)

                if self.training_config.clear_cache:
                    self.remove_cache(resample_save_path)
        else:
            final_model_path = base_save_path / "retrain" / f"policy_grad_{rollout_end * self.n_updates}.zip"
            model = PPO.load(final_model_path, device=device)
            baseline_loss = compute_surrogate_validation_loss(model, target_rollout)[0]
            baseline_losses.append(float(baseline_loss))
            baseline_resample_seeds.append(seed)

        if self.training_config.clear_cache:
            self.remove_cache(base_save_path / "retrain")

        att_save_path = self.root_dir / f"seed-{seed}" / self.attribution_path(
            "LDS",
            rollout_begin,
            rollout_end,
        )
        os.makedirs(att_save_path, exist_ok=True)
        att_file = att_save_path / f"lds_{rollout_begin}_{rollout_end}.pkl"

        with open(att_file, "wb") as f:
            pickle.dump({
                "n_subsets": n_subsets,
                "num_retrain": num_retrain,
                "n_samples_to_drop": n_samples_to_drop,
                "sampled_indices": sampled_indices,
                "lds_losses": lds_losses,
                "lds_resample_seeds": lds_resample_seeds,
                "baseline_losses": baseline_losses,
                "baseline_resample_seeds": baseline_resample_seeds,
            }, f)

        if self.training_config.clear_cache:
            self.remove_cache(perturbation_save_path)
            self.remove_cache(base_save_path)

    def run_train_with_selected_data(
        self,
        initial_policy=None,
        seed=42,
        device="cpu",
    ):
        data_dir = Path(self.training_config.initial_policy_dir.format(seed))
        rollout_begin = self.training_config.rollout_begin
        rollout_end = self.training_config.rollout_end

        target_rollout, _ = collect_training_buffers(
            data_dir,
            rollout_begin,
            rollout_end,
            n_batches=self.n_batches,
            n_epochs=self.ppo_config.n_epochs,
        )

        main_save_path = self.root_dir / f"seed-{seed}" / self.model_path(
            "selected_data_retrain",
            rollout_begin,
            rollout_end,
        )
        att_save_path = self.root_dir / f"seed-{seed}" / self.attribution_path(
            self.attribution_config.method,
            rollout_begin,
            rollout_end,
        )
        data_save_path = self.root_dir / f"seed-{seed}" / self.dataset_path(
            rollout_begin,
            rollout_end,
        )
        eval_save_path = self.root_dir / f"seed-{seed}" / self.eval_path(
            rollout_begin,
            rollout_end,
        )
        os.makedirs(main_save_path, exist_ok=True)
        os.makedirs(att_save_path, exist_ok=True)
        os.makedirs(data_save_path, exist_ok=True)
        os.makedirs(eval_save_path, exist_ok=True)

        trainer_class = trainer_dict[self.training_config.data_selection_method]
        initial_model = trainer_class.load(initial_policy, device=device)

        class DummyLogger:
            def record(self, *args, **kwargs):
                pass

            def dump(self, *args, **kwargs):
                pass
        initial_model._logger = DummyLogger()

        initial_model.attribution_func = self.attribution_func
        initial_model.attribution_config = self.attribution_config

        target_model_idx = rollout_end * self.n_updates
        target_model_path = data_dir / f"policy_grad_{target_model_idx}.zip"
        target_model = PPO.load(target_model_path, device=device)

        for att_begin in range(rollout_begin, rollout_end):
            att_end = att_begin + 1
            global_att_methods = ["RLIF-opt", "RLIF"]
            is_global = self.attribution_config.method in global_att_methods
            attribution_kwargs = {
                "target_model": target_model,
                "target_rollout": target_rollout,
                "n_batches": self.n_updates,
                "model_path": data_dir,
                "begin_step": att_begin * self.n_updates,
                "end_step": att_end * self.n_updates if not is_global else rollout_end * self.n_updates,
                "total_updates": self.training_config.total_rollouts * self.n_updates,
                "device": device,
            }

            if not is_global or att_begin == rollout_begin:
                attribution_scores = self.attribution_func(
                    **attribution_kwargs
                )

                att_file = att_save_path / f"attribution_scores.pkl"
                with open(att_file, "wb") as f:
                    pickle.dump(attribution_scores, f)

            logger.info(f"Phase 3: Building selected dataset")
            build_dataset_from_influence(
                folder=att_save_path,
                model_path=data_dir,
                k=att_begin,
                ed=att_end,
                n_epochs=self.ppo_config.n_epochs,
                n_updates=self.n_updates,
                batch_size=self.ppo_config.batch_size,
                method=self.attribution_config.method,
                drop_percentile=self.attribution_config.drop_percentile,
                mode=self.attribution_config.discard_mode,
                order=self.attribution_config.order,
                group_by=self.attribution_config.group_by,
                out_dir=data_save_path,
                seed=seed,
            )

            logger.info(f"Phase 4: Training with selected data")

            initial_model._train_with_selected_batches(
                data_save_path,
                main_save_path,
                att_begin,
                att_end,
                self.n_updates,
                rollout_pos=att_begin,
            )

        self.evaluate_return(
            main_save_path,
            eval_save_path,
            rollout_begin,
            rollout_end,
            device=device,
        )

    def run_local_attribution_training(self,
        initial_policy=None,
        seed=42,
        device="cpu",
    ):
        from rl_attribution.trainers.ppo import PPOWithLocalDataSelection
        from stable_baselines3.common.utils import get_linear_fn

        rollout_begin = self.training_config.rollout_begin
        rollout_end = self.training_config.rollout_end

        save_path = self.root_dir / f"seed-{seed}"
        eval_save_path = self.root_dir / f"seed-{seed}" / self.eval_path(
            rollout_begin,
            rollout_end,
        )
        os.makedirs(eval_save_path, exist_ok=True)

        policy_kwargs = dict(
            optimizer_class=torch.optim.SGD if self.ppo_config.optimizer_class == "SGD" else torch.optim.Adam,
        )

        if self.ppo_config.policy_type == "CnnPolicy":
            from rl_attribution.models.cnn import get_cnn_cls
            if self.ppo_config.policy_name != "none":
                policy_kwargs["features_extractor_class"] = get_cnn_cls(self.ppo_config.policy_name)
                policy_kwargs["features_extractor_kwargs"] = dict(features_dim=self.ppo_config.features_dim)

        if self.ppo_config.policy_kwargs is not None and len(self.ppo_config.policy_kwargs) > 0:
            print(self.ppo_config.policy_kwargs)
            policy_kwargs.update(self.ppo_config.policy_kwargs)

        if "activation_fn" in policy_kwargs:
            act_fn_name = policy_kwargs["activation_fn"]
            if act_fn_name in ACTIVATION_MAPPING:
                policy_kwargs["activation_fn"] = ACTIVATION_MAPPING[act_fn_name]
            else:
                raise ValueError(f"Unsupported activation function: {act_fn_name}")

        learning_schedule = get_linear_fn(
            start=self.ppo_config.learning_rate,
            end=self.ppo_config.final_learning_rate,
            end_fraction=1.0,
        ) if self.ppo_config.final_learning_rate is not None else self.ppo_config.learning_rate

        clip_schedule = get_linear_fn(
            start=self.ppo_config.clip_range,
            end=self.ppo_config.final_clip_range,
            end_fraction=1.0,
        ) if self.ppo_config.final_clip_range is not None else self.ppo_config.clip_range

        trainer = PPOWithLocalDataSelection(
            policy=self.ppo_config.policy_type,
            env=self.train_env,
            attribution_func=self.attribution_func,
            attribution_config=self.attribution_config,
            learning_rate=learning_schedule,
            n_steps=self.ppo_config.n_steps,
            batch_size=self.ppo_config.batch_size,
            n_epochs=self.ppo_config.n_epochs,
            gamma=self.ppo_config.gamma,
            gae_lambda=self.ppo_config.gae_lambda,
            clip_range=clip_schedule,
            clip_range_vf=None,
            normalize_advantage=self.ppo_config.normalize_advantage,
            ent_coef=self.ppo_config.ent_coef,
            vf_coef=self.ppo_config.vf_coef,
            kl_coef=self.ppo_config.kl_coef,
            max_grad_norm=self.ppo_config.max_grad_norm,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            stats_window_size=100,
            tensorboard_log=None,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
            device=device,
            save_path=save_path,
            cache_grad=self.training_config.cache_grad,
            cache_rollouts=self.training_config.cache_rollouts,
            clear_cache=self.training_config.clear_cache,
            render_rollouts=self.training_config.render_rollouts,
        )

        if initial_policy is not None:
            model_init = PPO.load(initial_policy, device=device)
            trainer.policy.load_state_dict(model_init.policy.state_dict())
            logger.info(f'Loaded initial policy: {initial_policy}')
            del model_init

        total_timesteps = self.ppo_config.total_timesteps
        trainer.learn(
            total_timesteps=total_timesteps,
            rollout_begin=rollout_begin,
            rollout_end=rollout_end,
        )

        main_save_path = save_path / self.model_path("base", rollout_begin, rollout_end)

        self.evaluate_return(
            main_save_path,
            eval_save_path,
            rollout_begin,
            rollout_end,
            device=device,
        )

        return save_path

    def run_global_attribution_training(
        self,
        initial_policy=None,
        seed=42,
        device="cpu",
    ):
        from rl_attribution.trainers.ppo import PPOWithGlobalDataSelection
        from stable_baselines3.common.utils import get_linear_fn

        rollout_begin = self.training_config.rollout_begin
        rollout_end = self.training_config.rollout_end

        save_path = self.root_dir / f"seed-{seed}"
        eval_save_path = self.root_dir / f"seed-{seed}" / self.eval_path(
            rollout_begin,
            rollout_end,
        )
        os.makedirs(eval_save_path, exist_ok=True)

        policy_kwargs = dict(
            optimizer_class=torch.optim.SGD if self.ppo_config.optimizer_class == "SGD" else torch.optim.Adam,
        )

        if self.ppo_config.policy_type == "CnnPolicy":
            from rl_attribution.models.cnn import get_cnn_cls
            if self.ppo_config.policy_name != "none":
                policy_kwargs["features_extractor_class"] = get_cnn_cls(self.ppo_config.policy_name)
                policy_kwargs["features_extractor_kwargs"] = dict(features_dim=self.ppo_config.features_dim)

        if self.ppo_config.policy_kwargs is not None and len(self.ppo_config.policy_kwargs) > 0:
            print(self.ppo_config.policy_kwargs)
            policy_kwargs.update(self.ppo_config.policy_kwargs)

        if "activation_fn" in policy_kwargs:
            act_fn_name = policy_kwargs["activation_fn"]
            if act_fn_name in ACTIVATION_MAPPING:
                policy_kwargs["activation_fn"] = ACTIVATION_MAPPING[act_fn_name]
            else:
                raise ValueError(f"Unsupported activation function: {act_fn_name}")

        learning_schedule = get_linear_fn(
            start=self.ppo_config.learning_rate,
            end=self.ppo_config.final_learning_rate,
            end_fraction=1.0,
        ) if self.ppo_config.final_learning_rate is not None else self.ppo_config.learning_rate

        clip_schedule = get_linear_fn(
            start=self.ppo_config.clip_range,
            end=self.ppo_config.final_clip_range,
            end_fraction=1.0,
        ) if self.ppo_config.final_clip_range is not None else self.ppo_config.clip_range

        trainer = PPOWithGlobalDataSelection(
            policy=self.ppo_config.policy_type,
            env=self.train_env,
            attribution_func=self.attribution_func,
            attribution_config=self.attribution_config,
            learning_rate=learning_schedule,
            n_steps=self.ppo_config.n_steps,
            batch_size=self.ppo_config.batch_size,
            n_epochs=self.ppo_config.n_epochs,
            gamma=self.ppo_config.gamma,
            gae_lambda=self.ppo_config.gae_lambda,
            clip_range=clip_schedule,
            clip_range_vf=None,
            normalize_advantage=self.ppo_config.normalize_advantage,
            ent_coef=self.ppo_config.ent_coef,
            vf_coef=self.ppo_config.vf_coef,
            kl_coef=self.ppo_config.kl_coef,
            max_grad_norm=self.ppo_config.max_grad_norm,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            stats_window_size=100,
            tensorboard_log=None,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
            device=device,
            save_path=save_path,
            cache_grad=self.training_config.cache_grad,
            cache_rollouts=self.training_config.cache_rollouts,
            clear_cache=self.training_config.clear_cache,
            render_rollouts=self.training_config.render_rollouts,
        )

        if initial_policy is not None:
            model_init = PPO.load(initial_policy, device=device)
            trainer.policy.load_state_dict(model_init.policy.state_dict())
            logger.info(f'Loaded initial policy: {initial_policy}')
            del model_init

        total_timesteps = self.ppo_config.total_timesteps
        trainer.learn(
            total_timesteps=total_timesteps,
            rollout_begin=rollout_begin,
            rollout_end=rollout_end,
        )

        main_save_path = save_path / self.model_path("base", rollout_begin, rollout_end)

        self.evaluate_return(
            main_save_path,
            eval_save_path,
            rollout_begin,
            rollout_end,
            device=device,
        )

        return save_path

    def run_global_iterative_attribution_training(
        self,
        initial_policy=None,
        seed=42,
        device="cpu",
    ):
        from rl_attribution.trainers.ppo import PPOWithIterativeGlobalDataSelection
        from stable_baselines3.common.utils import get_linear_fn

        rollout_begin = self.training_config.rollout_begin
        rollout_end = self.training_config.rollout_end

        save_path = self.root_dir / f"seed-{seed}"
        eval_save_path = self.root_dir / f"seed-{seed}" / self.eval_path(
            rollout_begin,
            rollout_end,
        )
        os.makedirs(eval_save_path, exist_ok=True)

        policy_kwargs = dict(
            optimizer_class=torch.optim.SGD if self.ppo_config.optimizer_class == "SGD" else torch.optim.Adam,
        )

        if self.ppo_config.policy_type == "CnnPolicy":
            from rl_attribution.models.cnn import get_cnn_cls
            if self.ppo_config.policy_name != "none":
                policy_kwargs["features_extractor_class"] = get_cnn_cls(self.ppo_config.policy_name)
                policy_kwargs["features_extractor_kwargs"] = dict(features_dim=self.ppo_config.features_dim)

        if self.ppo_config.policy_kwargs is not None and len(self.ppo_config.policy_kwargs) > 0:
            print(self.ppo_config.policy_kwargs)
            policy_kwargs.update(self.ppo_config.policy_kwargs)

        if "activation_fn" in policy_kwargs:
            act_fn_name = policy_kwargs["activation_fn"]
            if act_fn_name in ACTIVATION_MAPPING:
                policy_kwargs["activation_fn"] = ACTIVATION_MAPPING[act_fn_name]
            else:
                raise ValueError(f"Unsupported activation function: {act_fn_name}")

        learning_schedule = get_linear_fn(
            start=self.ppo_config.learning_rate,
            end=self.ppo_config.final_learning_rate,
            end_fraction=1.0,
        ) if self.ppo_config.final_learning_rate is not None else self.ppo_config.learning_rate

        clip_schedule = get_linear_fn(
            start=self.ppo_config.clip_range,
            end=self.ppo_config.final_clip_range,
            end_fraction=1.0,
        ) if self.ppo_config.final_clip_range is not None else self.ppo_config.clip_range

        trainer = PPOWithIterativeGlobalDataSelection(
            policy=self.ppo_config.policy_type,
            env=self.train_env,
            attribution_func=self.attribution_func,
            attribution_config=self.attribution_config,
            learning_rate=learning_schedule,
            n_steps=self.ppo_config.n_steps,
            batch_size=self.ppo_config.batch_size,
            n_epochs=self.ppo_config.n_epochs,
            gamma=self.ppo_config.gamma,
            gae_lambda=self.ppo_config.gae_lambda,
            clip_range=clip_schedule,
            clip_range_vf=None,
            normalize_advantage=self.ppo_config.normalize_advantage,
            ent_coef=self.ppo_config.ent_coef,
            vf_coef=self.ppo_config.vf_coef,
            kl_coef=self.ppo_config.kl_coef,
            max_grad_norm=self.ppo_config.max_grad_norm,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            stats_window_size=100,
            tensorboard_log=None,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
            device=device,
            save_path=save_path,
            cache_grad=self.training_config.cache_grad,
            cache_rollouts=self.training_config.cache_rollouts,
            clear_cache=self.training_config.clear_cache,
            render_rollouts=self.training_config.render_rollouts,
        )

        if initial_policy is not None:
            model_init = PPO.load(initial_policy, device=device)
            trainer.policy.load_state_dict(model_init.policy.state_dict())
            logger.info(f'Loaded initial policy: {initial_policy}')
            del model_init

        total_timesteps = self.ppo_config.total_timesteps
        trainer.learn(
            total_timesteps=total_timesteps,
            rollout_begin=rollout_begin,
            rollout_end=rollout_end,
        )

        main_save_path = save_path / self.model_path("base", rollout_begin, rollout_end)

        self.evaluate_return(
            main_save_path,
            eval_save_path,
            rollout_begin,
            rollout_end,
            device=device,
        )

        return save_path
