import os
import pickle
import json
import sys
import time
import copy
import random
import shutil
import numpy as np
import torch
from typing import Optional, Union, Any, TypeVar
from pathlib import Path
from gymnasium import spaces
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.utils import set_random_seed

from rl_attribution.attribution import build_dataset_from_influence
from rl_attribution.utils.logger import get_logger
from rl_attribution.eval import evaluate_policy

logger = get_logger()

SelfPPO = TypeVar("SelfPPO", bound="PPOWithLocalDataSelection")

class FrameRecorderCallback(BaseCallback):
    """Record rendered frames during rollout collection."""

    def __init__(
        self,
        env: VecEnv,
        output_path: Path,
    ) -> None:
        super().__init__()
        self.env = env
        self.output_path = output_path
        self.frames = []

    def _on_rollout_start(self) -> None:
        """Capture the initial observation at the start of each rollout."""
        try:
            frame = self.env.render()
            if frame is not None:
                if not isinstance(frame, (list, tuple)):
                    frame = [frame]
                self.frames.append(frame)
        except Exception as exc:
            logger.warning(f"Failed to render initial frame: {exc}")

    def _on_step(self) -> bool:
        try:
            frame = self.env.render()
        except Exception as exc:
            logger.warning(f"Failed to render frame: {exc}")
            return True

        if frame is not None:
            if not isinstance(frame, (list, tuple)):

                frame = [frame]
            self.frames.append(frame)

        return True

    def _on_rollout_end(self) -> None:
        if not self.frames:
            raise ValueError("No frames were recorded during the rollout.")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "wb") as f:
            pickle.dump(self.frames, f)

        self.frames = []

def _wrap_callback_with_frame_recorder(
    model: PPO,
    callback: BaseCallback,
    output_path: Path,
) -> CallbackList:
    recorder = FrameRecorderCallback(
        env=model.env,
        output_path=output_path,
    )
    rollout_callback = CallbackList([callback, recorder])
    rollout_callback.init_callback(model)
    return rollout_callback

class PPOWithLocalDataSelection(PPO):
    """
    PPO with local data selection. Overrides the learn() method to:
    1. Collect rollout data (auxiliary training phase)
    2. Compute attribution scores for each rollout
    3. Select influential data points
    4. Train with selected data (main training phase)

    This avoids multiple calls to learn() which causes environment randomness issues.
    """

    def __init__(
        self,
        policy: Union[str, type],
        env: Union[GymEnv, str],
        attribution_func=None,
        attribution_config=None,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        kl_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,

        save_path: Optional[Path] = None,
        cache_grad=False,
        cache_rollouts=False,
        clear_cache=True,
        render_rollouts: bool = False,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

        self.attribution_func = attribution_func
        self.attribution_config = attribution_config
        self.base_save_path = Path(save_path) if save_path else Path("./ppo_checkpoints")

        self.cache_grad = cache_grad
        self.cache_rollouts = cache_rollouts
        self.clear_cache = clear_cache
        self._update_counter = 0
        self._current_rollout_idx = 0
        self._rollout_begin = 0
        self._auxiliary_batches = []
        self.kl_coef = kl_coef

        self.seed = seed
        self.render_rollouts = render_rollouts
        self.render_save_path = self.base_save_path / "frames"

    def _excluded_save_params(self) -> list[str]:
        """
        Exclude non-serializable attributes from being saved.
        This prevents pickling errors with attribution functions and configs.
        """
        excluded = super()._excluded_save_params()
        excluded.extend([
            "attribution_func",
            "attribution_config",
            "_auxiliary_batches",
        ])
        return excluded

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        rollout_begin: int = 0,
        rollout_end: Optional[int] = None,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        """
        Overridden learn method that performs local data selection within a single continuous session.

        This avoids the environment randomness issues caused by calling learn() multiple times.

        :param total_timesteps: Total timesteps for training
        :param rollout_begin: Starting rollout index
        :param rollout_end: Ending rollout index (computed if None)
        :param callback: Callback for training events
        :param log_interval: Logging interval
        :param tb_log_name: Tensorboard log name
        :param reset_num_timesteps: Whether to reset timesteps
        :param progress_bar: Whether to show progress bar
        """

        total_rollouts = total_timesteps // (self.n_steps * self.env.num_envs)
        if rollout_end is None:
            rollout_end = total_rollouts

        self._rollout_begin = rollout_begin
        n_updates_per_rollout = self.env.num_envs * self.n_steps // self.batch_size * self.n_epochs

        iteration = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        global_start_time = time.time()

        rollout_times = {}

        for rollout_idx in range(rollout_begin, rollout_end):

            logger.info(f"Setting random seed for rollout {rollout_idx}: {self.seed + rollout_idx}")
            set_random_seed(self.seed + rollout_idx)
            self._current_rollout_idx = rollout_idx

            rollout_times.setdefault(
                rollout_idx,
                {
                    "rollout_collection_time": 0.0,
                    "training_time": 0.0,
                    "data_attribution_time": 0.0,
                    "replay_time": 0.0,
                },
            )

            logger.info("=" * 60)
            logger.info(f"Processing Rollout {rollout_idx + 1}/{rollout_end}")
            logger.info("=" * 60)

            curr_begin = rollout_idx
            curr_end = rollout_idx + 1

            main_save_path = self.base_save_path / "models" / "base" / f"{rollout_begin}-{rollout_end}"
            auxiliary_save_path = self.base_save_path / "models" / "auxiliary" / f"{curr_begin}-{curr_end}"
            att_save_path = self.base_save_path / "attribution" / f"{curr_begin}-{curr_end}"
            data_save_path = self.base_save_path / "dataset" / f"{curr_begin}-{curr_end}"

            all_save_pathes = [main_save_path]
            if self.attribution_func is not None:
                all_save_pathes.extend([auxiliary_save_path, att_save_path, data_save_path])

            for path in all_save_pathes:
                path.mkdir(parents=True, exist_ok=True)

            self._update_counter = n_updates_per_rollout * curr_begin

            logger.info(f"Phase 1: Continual training to collect data")
            self._auxiliary_batches = []

            if rollout_idx == rollout_begin:
                init_main_path = main_save_path / f"policy_grad_{self._update_counter}.zip"
                self.save(init_main_path)

            save_path = main_save_path if self.attribution_func is None else auxiliary_save_path
            rollout_callback = callback
            if self.render_rollouts:
                frame_dir = self.render_save_path / f"{rollout_begin}-{rollout_end}"
                frame_file = frame_dir / f"rollout_{rollout_idx}.pkl"
                logger.info(f"Rendering rollout frames to {frame_file}")
                rollout_callback = _wrap_callback_with_frame_recorder(
                    self,
                    callback,
                    frame_file,
                )
            continue_training = self._auxiliary_training_iteration(
                save_path,
                rollout_callback,
                iteration,
                log_interval,
            )

            rollout_times[rollout_idx]["rollout_collection_time"] += getattr(self, "_last_collection_time", 0.0)
            rollout_times[rollout_idx]["training_time"] += getattr(self, "_last_aux_train_time", 0.0)

            if not continue_training:
                break

            iteration += 1

            if self.cache_rollouts:
                rollout_file = save_path / f"rollout_{rollout_idx}.pkl"
                with open(rollout_file, "wb") as f:
                    pickle.dump(self.rollout_buffer, f)

            if self.attribution_func is not None:

                attribution_start = time.time()
                logger.info(f"Phase 2: Computing attribution scores")

                target_model_idx = curr_end * n_updates_per_rollout
                target_model_path = auxiliary_save_path / f"policy_grad_{target_model_idx}.zip"
                target_model = PPO.load(target_model_path, device=self.device)

                attribution_scores = self.attribution_func(
                    target_model,
                    self.rollout_buffer,
                    n_updates_per_rollout,
                    auxiliary_save_path,
                    curr_begin * n_updates_per_rollout,
                    curr_end * n_updates_per_rollout,
                    total_rollouts * n_updates_per_rollout,
                    device=self.device,
                )

                del target_model

                att_file = att_save_path / f"attribution_scores.pkl"
                with open(att_file, "wb") as f:
                    pickle.dump(attribution_scores, f)

                logger.info(f"Phase 3: Building selected dataset")
                build_dataset_from_influence(
                    folder=att_save_path,
                    model_path=auxiliary_save_path,
                    k=curr_begin,
                    ed=curr_end,
                    n_epochs=self.n_epochs,
                    n_updates=n_updates_per_rollout,
                    batch_size=self.batch_size,
                    method=self.attribution_config.method,
                    drop_percentile=self.attribution_config.drop_percentile,
                    mode=self.attribution_config.discard_mode,
                    selection_mode=self.attribution_config.selection_mode,
                    w_min=self.attribution_config.w_min,
                    order=self.attribution_config.order,
                    group_by=self.attribution_config.group_by,
                    out_dir=data_save_path,
                    seed=self.seed,
                )
                attribution_end = time.time()
                rollout_times[rollout_idx]["data_attribution_time"] += attribution_end - attribution_start

                logger.info(f"Phase 4: Training with selected data")

                prev_checkpoint_idx = curr_begin * n_updates_per_rollout
                prev_checkpoint_path = main_save_path / f"policy_grad_{prev_checkpoint_idx}.zip"
                prev_model = PPO.load(prev_checkpoint_path, device=self.device)
                self.policy.load_state_dict(prev_model.policy.state_dict())
                self.policy.optimizer.load_state_dict(prev_model.policy.optimizer.state_dict())
                del prev_model

                replay_start = time.time()
                self._train_with_selected_batches(
                    data_save_path,
                    main_save_path,
                    curr_begin,
                    curr_end,
                    n_updates_per_rollout,
                )
                replay_end = time.time()
                rollout_times[rollout_idx]["replay_time"] += replay_end - replay_start

                if self.clear_cache:
                    logger.info(f"Phase 5: Cleaning up cache for rollout {rollout_idx + 1}")
                    self._cleanup_rollout_cache(auxiliary_save_path, att_save_path, data_save_path)

            if self.clear_cache:
                self._cleanup_main_save_path(main_save_path, n_updates_per_rollout)

            rollout_end_time = time.time()
            rollout_duration = rollout_end_time - global_start_time
            rollout_times[rollout_idx]["total_time"] = rollout_duration
            logger.info(f"Rollout {rollout_idx + 1} completed in {rollout_duration:.2f} seconds")

        rollout_times_path = self.base_save_path / "rollout_times.pkl"
        with open(rollout_times_path, "wb") as f:
            pickle.dump(rollout_times, f)
        logger.info(f"Saved rollout runtime log to {rollout_times_path}")

        if rollout_times:
            total_time = sum(entry.get("total_time", 0.0) for entry in rollout_times.values())
            avg_time = total_time / len(rollout_times)
            logger.info(f"Total training time: {total_time:.2f} seconds")
            logger.info(f"Average time per rollout: {avg_time:.2f} seconds")

        callback.on_training_end()

        return self

    def _auxiliary_training_iteration(
        self,
        save_path: Path,
        callback: BaseCallback,
        iteration: int,
        log_interval: int,
    ) -> bool:
        """
        Perform one auxiliary training iteration (collect rollout + train).
        This is similar to one iteration of the standard PPO learn loop.
        """

        collection_start = time.time()
        continue_training = self.collect_rollouts(
            self.env,
            callback,
            self.rollout_buffer,
            n_rollout_steps=self.n_steps,
        )
        self._last_collection_time = time.time() - collection_start

        if not continue_training:
            return False

        iteration += 1
        self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
        logger.info(f"Current progress remaining: {self._current_progress_remaining}")

        if log_interval is not None and iteration % log_interval == 0:
            assert self.ep_info_buffer is not None
            time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
            fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
            if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            self.logger.record("time/fps", fps)
            self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(step=self.num_timesteps)

        train_start = time.time()
        self._auxiliary_train_with_batch_saving(save_path)
        self._last_aux_train_time = time.time() - train_start

        return True

    def buffer_get(self, batch_size):
        buffer = self.rollout_buffer
        assert buffer.full, ""
        indices = np.random.permutation(buffer.buffer_size * buffer.n_envs)

        if not buffer.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                buffer.__dict__[tensor] = buffer.swap_and_flatten(buffer.__dict__[tensor])
            buffer.generator_ready = True

        if batch_size is None:
            batch_size = buffer.buffer_size * buffer.n_envs

        start_idx = 0
        while start_idx < buffer.buffer_size * buffer.n_envs:
            sampled_indices = indices[start_idx : start_idx + batch_size]
            yield buffer._get_samples(sampled_indices), buffer.episode_starts.flatten()[sampled_indices], sampled_indices
            start_idx += batch_size

    def _auxiliary_train_with_batch_saving(self, save_path: Path):
        """
        Train on the current rollout buffer and save each mini-batch for later attribution.
        This is the auxiliary training phase.
        """

        if self.cache_grad or self.attribution_func is not None:
            init_checkpoint = save_path / f"policy_grad_{self._update_counter}.zip"
            self.save(init_checkpoint)

        self.policy.set_training_mode(True)

        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)

        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            for rollout_data, episode_starts, indices in self.buffer_get(self.batch_size):
                self._update_counter += 1

                if self.cache_grad or self.attribution_func is not None:
                    batch_filename = save_path / f"batch_{self._update_counter}.pkl"

                    rollout_dict = rollout_data._asdict()
                    rollout_dict["episode_starts"] = episode_starts
                    rollout_dict["indices"] = indices
                    with open(batch_filename, "wb") as f:
                        pickle.dump(rollout_dict, f)

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = torch.nn.functional.mse_loss(rollout_data.returns, values_pred)

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                log_ratio = log_prob - rollout_data.old_log_prob
                approx_kl_loss = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.kl_coef * approx_kl_loss

                self.policy.optimizer.zero_grad()
                loss.backward()

                self.policy.optimizer.step()

                if self.cache_grad or self.attribution_func is not None:
                    checkpoint_filename = save_path / f"policy_grad_{self._update_counter}.zip"
                    self.save(checkpoint_filename)

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

            self._n_updates += 1
            if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                break

        checkpoint_filename = save_path / f"policy_grad_{self._update_counter}.zip"
        self.save(checkpoint_filename)

    def _train_with_selected_batches(
        self,
        data_save_path: Path,
        main_save_path: Path,
        rollout_begin: int,
        rollout_end: int,
        n_updates: int,
    ):
        """
        Train with pre-selected data batches (main training phase).
        This performs offline training on the selected dataset.
        """
        self.policy.train()
        logger.info(f"Current progress remaining: {self._current_progress_remaining}")

        for t in range(n_updates * rollout_begin, n_updates * rollout_end):
            batch_path = data_save_path / f"batch_{t + 1}.pkl"

            with open(batch_path, "rb") as f:
                rollout_data = pickle.load(f)

            if rollout_data is None:
                logger.info(f"Batch data at step {t} is None. Quitting...")
                break

            self._train_step_with_batch(rollout_data, t, n_updates, rollout_end)

            if self.cache_grad or (t + 1) % n_updates == 0:
                self.save(main_save_path / f"policy_grad_{t + 1}.zip")

        final_idx = n_updates * rollout_end
        final_model_path = main_save_path / f"policy_grad_{final_idx}.zip"
        self.save(final_model_path)
        logger.info(f"Saved model checkpoint: {final_model_path}")

    def _cleanup_rollout_cache(
        self,
        auxiliary_save_path: Path,
        att_save_path: Path,
        data_save_path: Path,
    ):
        """
        Clean up temporary cache directories after training on a rollout.
        Removes auxiliary models, attribution scores, and selected datasets.
        """
        import shutil

        cache_dirs = [auxiliary_save_path, att_save_path, data_save_path]

        for cache_dir in cache_dirs:
            if cache_dir.exists():
                try:
                    shutil.rmtree(cache_dir)
                    logger.info(f"Removed cache directory: {cache_dir}")
                except Exception as e:
                    logger.warning(f"Failed to remove cache directory {cache_dir}: {e}")

    def _cleanup_main_save_path(
        self,
        main_save_path: Path,
        n_updates_per_rollout: int,
    ) -> None:
        """
        Remove non-rollout-end checkpoints and all batch data in main_save_path.
        Keeps checkpoints at rollout boundaries (multiples of n_updates_per_rollout).
        """
        for path in main_save_path.glob("batch_*.pkl"):
            try:
                path.unlink()
            except Exception as exc:
                logger.warning(f"Failed to remove batch file {path}: {exc}")

        for path in main_save_path.glob("policy_grad_*.zip"):
            name = path.name
            idx_str = name[len("policy_grad_") : -len(".zip")]
            if not idx_str.isdigit():
                continue
            idx = int(idx_str)
            if idx % n_updates_per_rollout == 0:
                continue
            try:
                path.unlink()
            except Exception as exc:
                logger.warning(f"Failed to remove checkpoint {path}: {exc}")

    def _train_step_with_batch(self, rollout_data, t, n_updates, rollout_end):
        """Perform a single training step with a batch of data."""

        if isinstance(rollout_data, dict):
            actions = rollout_data["actions"]
            observations = rollout_data["observations"]
            advantages = rollout_data["advantages"]
            returns = rollout_data["returns"]
            old_log_prob = rollout_data["old_log_prob"]
            weights = rollout_data.get("weights", torch.ones_like(advantages))
        else:
            actions = rollout_data.actions
            observations = rollout_data.observations
            advantages = rollout_data.advantages
            returns = rollout_data.returns
            old_log_prob = rollout_data.old_log_prob
            weights = getattr(rollout_data, "weights", torch.ones_like(advantages))

        actions = actions.to(self.device)
        observations = observations.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        old_log_prob = old_log_prob.to(self.device)

        if isinstance(self.action_space, spaces.Discrete):
            actions = actions.long().flatten()

        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)

        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)
        values = values.flatten()

        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(log_prob - old_log_prob)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2)

        if self.clip_range_vf is None:
            values_pred = values
        else:
            values_pred = rollout_data.old_values + torch.clamp(
                values - rollout_data.old_values, -clip_range_vf, clip_range_vf
            )
        value_loss = torch.nn.functional.mse_loss(returns, values_pred, reduction="none")

        if entropy is None:

            entropy_loss = log_prob
        else:

            entropy_loss = -entropy

        log_ratio = log_prob - old_log_prob
        approx_kl_losses = (torch.exp(log_ratio) - 1) - log_ratio

        losses = policy_loss + self.ent_coef * entropy_loss + self.kl_coef * approx_kl_losses
        assert losses.shape[0] == len(weights), (losses.shape[0], len(weights))
        loss = (losses * weights + self.vf_coef * value_loss).mean()

        batch_size = len(observations)
        if batch_size != self.batch_size:
            logger.info(f'rescale batch t={t}: {batch_size}->{self.batch_size}.')
            for param in self.policy.parameters():
                if param.grad is not None:
                    param.grad.data *= batch_size / self.batch_size

        self.policy.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()

class PPOWithGlobalDataSelection(PPO):
    """
    PPO with local data selection. Overrides the learn() method to:
    1. Collect rollout data (auxiliary training phase)
    2. Compute attribution scores for each rollout
    3. Select influential data points
    4. Train with selected data (main training phase)

    This avoids multiple calls to learn() which causes environment randomness issues.
    """

    def __init__(
        self,
        policy: Union[str, type],
        env: Union[GymEnv, str],
        attribution_func=None,
        attribution_config=None,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        kl_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,

        save_path: Optional[Path] = None,
        cache_grad=False,
        cache_rollouts=False,
        clear_cache=True,
        render_rollouts: bool = False,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

        self.attribution_func = attribution_func
        self.attribution_config = attribution_config
        self.base_save_path = Path(save_path) if save_path else Path("./ppo_checkpoints")

        self.cache_grad = cache_grad
        self.cache_rollouts = cache_rollouts
        self.clear_cache = clear_cache
        self._update_counter = 0
        self._current_rollout_idx = 0
        self._rollout_begin = 0
        self._auxiliary_batches = []
        self._cached_buffers = []
        self.kl_coef = kl_coef

        self.max_kl_div = None
        self.seed = seed
        self.render_rollouts = render_rollouts
        self.render_save_path = self.base_save_path / "frames"

    def _excluded_save_params(self) -> list[str]:
        """
        Exclude non-serializable attributes from being saved.
        This prevents pickling errors with attribution functions and configs.
        """
        excluded = super()._excluded_save_params()
        excluded.extend([
            "attribution_func",
            "attribution_config",
            "_auxiliary_batches",
        ])
        return excluded

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        rollout_begin: int = 0,
        rollout_end: Optional[int] = None,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        """
        Overridden learn method that performs local data selection within a single continuous session.

        This avoids the environment randomness issues caused by calling learn() multiple times.

        :param total_timesteps: Total timesteps for training
        :param rollout_begin: Starting rollout index
        :param rollout_end: Ending rollout index (computed if None)
        :param callback: Callback for training events
        :param log_interval: Logging interval
        :param tb_log_name: Tensorboard log name
        :param reset_num_timesteps: Whether to reset timesteps
        :param progress_bar: Whether to show progress bar
        """

        total_rollouts = total_timesteps // (self.n_steps * self.env.num_envs)
        if rollout_end is None:
            rollout_end = total_rollouts

        self._rollout_begin = rollout_begin
        n_updates_per_rollout = self.env.num_envs * self.n_steps // self.batch_size * self.n_epochs

        iteration = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None
        global_start_time = time.time()

        def mean_episode_return(rollout_buffer):
            rewards = rollout_buffer.rewards
            episode_starts = rollout_buffer.episode_starts

            n_steps, n_envs = rewards.shape
            episode_returns = []

            for env in range(n_envs):
                current_return = 0.0

                for t in range(n_steps):
                    if episode_starts[t, env] and t > 0:
                        episode_returns.append(current_return)
                        current_return = 0.0

                    current_return += rewards[t, env]

                episode_returns.append(current_return)

            return np.mean(episode_returns)

        step_size = self.attribution_config.step_size

        curr_begin = rollout_begin

        horizon_log = {}
        kl_div_log = {}
        return_log = {}
        eval_return_log = {}
        rollout_times = {}
        prev_max_kl = 0.0
        while curr_begin < rollout_end:

            curr_end = min(curr_begin + step_size, rollout_end)

            main_save_path = self.base_save_path / "models" / "base" / f"{rollout_begin}-{rollout_end}"
            auxiliary_save_path = self.base_save_path / "models" / "auxiliary" / f"{curr_begin}-{curr_end}"
            att_save_path = self.base_save_path / "attribution" / f"{curr_begin}-{curr_end}"
            data_save_path = self.base_save_path / "dataset" / f"{curr_begin}-{curr_end}"

            all_save_pathes = [main_save_path]
            if self.attribution_func is not None:
                all_save_pathes.extend([auxiliary_save_path, att_save_path, data_save_path])

            for path in all_save_pathes:
                path.mkdir(parents=True, exist_ok=True)

            self._update_counter = n_updates_per_rollout * curr_begin

            logger.info(f"Phase 1: Continual training to collect data")
            self._auxiliary_batches = []
            self._cached_buffers = []

            if curr_begin == rollout_begin:
                init_main_path = main_save_path / f"policy_grad_{self._update_counter}.zip"
                self.save(init_main_path)

            save_path = main_save_path if self.attribution_func is None else auxiliary_save_path
            self.max_kl_div = {}
            max_return = -float("inf")
            for rollout_idx in range(curr_begin, curr_end):
                stopped_idx = rollout_idx + 1

                logger.info(f"Setting random seed for rollout {rollout_idx}: {self.seed + rollout_idx}")
                set_random_seed(self.seed + rollout_idx)
                self._current_rollout_idx = rollout_idx

                logger.info("=" * 60)
                logger.info(f"Processing Rollout {rollout_idx + 1}/{rollout_end}")
                logger.info("=" * 60)

                rollout_times.setdefault(
                    rollout_idx,
                    {
                        "rollout_collection_time": 0.0,
                        "training_time": 0.0,
                        "data_attribution_time": 0.0,
                        "replay_time": 0.0,
                    },
                )

                rollout_callback = callback
                if self.render_rollouts:
                    frame_dir = self.render_save_path / f"{rollout_begin}-{rollout_end}"
                    frame_file = frame_dir / f"rollout_{rollout_idx}.pkl"
                    rollout_callback = _wrap_callback_with_frame_recorder(
                        self,
                        callback,
                        frame_file,
                    )

                continue_training = self._auxiliary_training_iteration(
                    save_path,
                    rollout_callback,
                    iteration,
                    log_interval,
                )

                rollout_times[rollout_idx]["rollout_collection_time"] += getattr(self, "_last_collection_time", 0.0)
                rollout_times[rollout_idx]["training_time"] += getattr(self, "_last_aux_train_time", 0.0)

                if self.attribution_config.use_local_buffer:
                    self._cached_buffers.append(copy.deepcopy(self.rollout_buffer))

                if self.cache_rollouts:
                    rollout_file = save_path / f"rollout_{rollout_idx}.pkl"
                    with open(rollout_file, "wb") as f:
                        pickle.dump(self.rollout_buffer, f)

                if rollout_idx == curr_begin:
                    curr_return = mean_episode_return(self.rollout_buffer)
                    return_log[(curr_begin, rollout_idx)] = curr_return

                    rollout_end_time = time.time()
                    rollout_duration = rollout_end_time - global_start_time
                    rollout_times[rollout_idx]["total_time"] = rollout_duration
                    logger.info(f"Rollout {rollout_idx} completed in {rollout_duration:.2f} seconds")

                    max_return = curr_return

                else:
                    curr_return = mean_episode_return(self.rollout_buffer)
                    return_log[(curr_begin, rollout_idx)] = curr_return

                    rollout_end_time = time.time()
                    rollout_duration = rollout_end_time - global_start_time
                    rollout_times[rollout_idx]["total_time"] = rollout_duration
                    logger.info(f"Rollout {rollout_idx} completed in {rollout_duration:.2f} seconds")

                    use_dynamic_horizon = self.attribution_config.dynamic_horizon
                    if use_dynamic_horizon and curr_return < self.attribution_config.max_return_factor * max_return:
                        self.num_timesteps = prev_num_timesteps
                        stopped_idx -= 1
                        break

                    max_return = curr_return

                if not continue_training:
                    break

                iteration += 1
                prev_num_timesteps = self.num_timesteps

            logger.info(f"Max KL divergence per rollout: {self.max_kl_div}")

            curr_end = stopped_idx
            horizon_log[curr_begin] = curr_end

            if self.attribution_func is not None:

                prev_checkpoint_idx = curr_begin * n_updates_per_rollout
                prev_checkpoint_path = main_save_path / f"policy_grad_{prev_checkpoint_idx}.zip"
                prev_model = PPO.load(prev_checkpoint_path, device=self.device)
                self.policy.load_state_dict(prev_model.policy.state_dict())
                self.policy.optimizer.load_state_dict(prev_model.policy.optimizer.state_dict())
                del prev_model

                logger.info(f"Phase 2: Computing attribution scores")

                target_model_idx = curr_end * n_updates_per_rollout
                target_model_path = auxiliary_save_path / f"policy_grad_{target_model_idx}.zip"
                target_model = PPO.load(target_model_path, device=self.device)

                highest_return = -float('inf')
                for att_begin in range(curr_begin, curr_end):
                    rollout_times.setdefault(
                        att_begin,
                        {
                            "rollout_collection_time": 0.0,
                            "training_time": 0.0,
                            "data_attribution_time": 0.0,
                            "replay_time": 0.0,
                        },
                    )
                    attribution_start = time.time()
                    if self.attribution_config.target_model == "stage":

                        logger.info(f"Loading target model for attribution at step {att_begin}: {target_model_path}")
                        target_model_idx = (att_begin + 1) * n_updates_per_rollout
                        target_model_path = auxiliary_save_path / f"policy_grad_{target_model_idx}.zip"
                        target_model = PPO.load(target_model_path, device=self.device)

                    att_end = att_begin + 1
                    global_att_methods = ["RLIF-opt", "RLIF"]
                    is_global = self.attribution_config.method in global_att_methods
                    attribution_kwargs = {
                        "target_model": target_model,
                        "target_rollout": self.rollout_buffer if not self.attribution_config.use_local_buffer else self._cached_buffers[att_begin - curr_begin],
                        "n_batches": n_updates_per_rollout,
                        "model_path": auxiliary_save_path,
                        "begin_step": att_begin * n_updates_per_rollout,
                        "end_step": att_end * n_updates_per_rollout if not is_global else curr_end * n_updates_per_rollout,
                        "total_updates": total_rollouts * n_updates_per_rollout,
                        "device": self.device,
                    }

                    if not is_global or att_begin == curr_begin:
                        attribution_scores = self.attribution_func(
                            **attribution_kwargs
                        )

                        att_file = att_save_path / f"attribution_scores.pkl"
                        with open(att_file, "wb") as f:
                            pickle.dump(attribution_scores, f)

                    logger.info(f"Phase 3: Building selected dataset")
                    build_dataset_from_influence(
                        folder=att_save_path,
                        model_path=auxiliary_save_path,
                        k=att_begin,
                        ed=att_end,
                        n_epochs=self.n_epochs,
                        n_updates=n_updates_per_rollout,
                        batch_size=self.batch_size,
                        method=self.attribution_config.method,
                        drop_percentile=self.attribution_config.drop_percentile,
                        mode=self.attribution_config.discard_mode,
                        selection_mode=self.attribution_config.selection_mode,
                        w_min=self.attribution_config.w_min,
                        order=self.attribution_config.order,
                        group_by=self.attribution_config.group_by,
                        out_dir=data_save_path,
                        seed=self.seed,
                    )
                    attribution_end = time.time()
                    rollout_times[att_begin]["data_attribution_time"] += attribution_end - attribution_start

                    logger.info(f"Phase 4: Training with selected data")

                    replay_start = time.time()
                    self._train_with_selected_batches(
                        data_save_path,
                        main_save_path,
                        att_begin,
                        att_end,
                        n_updates_per_rollout,
                        rollout_pos=att_begin,
                    )
                    replay_end = time.time()
                    rollout_times[att_begin]["replay_time"] += replay_end - replay_start

                with open(self.base_save_path / f"log_statistics.pkl", "wb") as f:
                    pickle.dump({
                        "horizon_log": horizon_log,
                        "return_log": return_log,
                        "eval_return_log": eval_return_log,

                    }, f)

                if self.clear_cache:
                    logger.info(f"Phase 5: Cleaning up cache for rollout {curr_begin} - {curr_end}")
                    self._cleanup_rollout_cache(auxiliary_save_path, att_save_path, data_save_path)

            curr_begin = stopped_idx

            if self.clear_cache:
                self._cleanup_main_save_path(main_save_path, n_updates_per_rollout)

        rollout_times_path = self.base_save_path / "rollout_times.pkl"
        with open(rollout_times_path, "wb") as f:
            pickle.dump(rollout_times, f)
        logger.info(f"Saved rollout runtime log to {rollout_times_path}")

        if rollout_times:
            total_time = sum(entry.get("total_time", 0.0) for entry in rollout_times.values())
            avg_time = total_time / len(rollout_times)
            logger.info(f"Total training time: {total_time:.2f} seconds")
            logger.info(f"Average time per rollout: {avg_time:.2f} seconds")

        callback.on_training_end()

        return self

    def _auxiliary_training_iteration(
        self,
        save_path: Path,
        callback: BaseCallback,
        iteration: int,
        log_interval: int,
    ) -> bool:
        """
        Perform one auxiliary training iteration (collect rollout + train).
        This is similar to one iteration of the standard PPO learn loop.
        """

        collection_start = time.time()
        continue_training = self.collect_rollouts(
            self.env,
            callback,
            self.rollout_buffer,
            n_rollout_steps=self.n_steps,
        )
        self._last_collection_time = time.time() - collection_start

        if not continue_training:
            return False

        iteration += 1
        self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
        logger.info(f"Current progress remaining: {self._current_progress_remaining}")

        if log_interval is not None and iteration % log_interval == 0:
            assert self.ep_info_buffer is not None
            time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
            fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
            if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            self.logger.record("time/fps", fps)
            self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(step=self.num_timesteps)

        train_start = time.time()
        self._auxiliary_train_with_batch_saving(save_path)
        self._last_aux_train_time = time.time() - train_start

        return True

    def buffer_get(self, batch_size):
        buffer = self.rollout_buffer
        assert buffer.full, ""
        indices = np.random.permutation(buffer.buffer_size * buffer.n_envs)

        if not buffer.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                buffer.__dict__[tensor] = buffer.swap_and_flatten(buffer.__dict__[tensor])
            buffer.generator_ready = True

        if batch_size is None:
            batch_size = buffer.buffer_size * buffer.n_envs

        start_idx = 0
        while start_idx < buffer.buffer_size * buffer.n_envs:
            sampled_indices = indices[start_idx : start_idx + batch_size]
            yield buffer._get_samples(sampled_indices), buffer.episode_starts.flatten()[sampled_indices], sampled_indices
            start_idx += batch_size

    def _auxiliary_train_with_batch_saving(self, save_path: Path):
        """
        Train on the current rollout buffer and save each mini-batch for later attribution.
        This is the auxiliary training phase.
        """

        if self.cache_grad or self.attribution_func is not None:
            init_checkpoint = save_path / f"policy_grad_{self._update_counter}.zip"
            self.save(init_checkpoint)

        self.policy.set_training_mode(True)

        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)

        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            for rollout_data, episode_starts, indices in self.buffer_get(self.batch_size):
                self._update_counter += 1

                if self.cache_grad or self.attribution_func is not None:
                    batch_filename = save_path / f"batch_{self._update_counter}.pkl"

                    rollout_dict = rollout_data._asdict()
                    rollout_dict["episode_starts"] = episode_starts
                    rollout_dict["indices"] = indices
                    with open(batch_filename, "wb") as f:
                        pickle.dump(rollout_dict, f)

                self._auxiliary_batches.append((self._update_counter, rollout_data))

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = torch.nn.functional.mse_loss(rollout_data.returns, values_pred)

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                log_ratio = log_prob - rollout_data.old_log_prob
                approx_kl_loss = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.kl_coef * approx_kl_loss

                self.policy.optimizer.zero_grad()
                loss.backward()

                self.policy.optimizer.step()

                if self.cache_grad or self.attribution_func is not None:
                    checkpoint_filename = save_path / f"policy_grad_{self._update_counter}.zip"
                    self.save(checkpoint_filename)

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                    rollout_idx = (self._update_counter - 1) // (self.n_epochs * (self.env.num_envs * self.n_steps // self.batch_size))
                    self.max_kl_div[rollout_idx] = max(self.max_kl_div.get(rollout_idx, 0.0), approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

            self._n_updates += 1
            if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                break

        checkpoint_filename = save_path / f"policy_grad_{self._update_counter}.zip"
        self.save(checkpoint_filename)

    def _train_with_selected_batches(
        self,
        data_save_path: Path,
        main_save_path: Path,
        rollout_begin: int,
        rollout_end: int,
        n_updates: int,
        rollout_pos: int,
    ):
        """
        Train with pre-selected data batches (main training phase).
        This performs offline training on the selected dataset.
        """
        self.policy.train()

        pbar = tqdm(range(n_updates * rollout_begin, n_updates * rollout_end), desc="Training")
        highest_return = -float('inf')
        for t in pbar:
            pbar.set_postfix({"progress_remaining": f"{self._current_progress_remaining:.4f}"})

            if t % n_updates == 0 and self.attribution_config.adjust_log_prob:
                logger.info(f"Adjusting old_log_prob at {t}/{n_updates * rollout_end}")
                self._update_log_prob(data_save_path, t, t + n_updates)

            batch_path = data_save_path / f"batch_{t + 1}.pkl"

            with open(batch_path, "rb") as f:
                rollout_data = pickle.load(f)

            if rollout_data is None:
                logger.info(f"Batch data at step {t} is None. Quitting...")
                break

            continue_training = self._train_step_with_batch(rollout_data, t, n_updates, rollout_end, rollout_pos)

            if not continue_training:
                break

            if self.cache_grad or (t + 1) % n_updates == 0:
                self.save(main_save_path / f"policy_grad_{t + 1}.zip")

        final_idx = n_updates * rollout_end
        final_model_path = main_save_path / f"policy_grad_{final_idx}.zip"
        self.save(final_model_path)
        logger.info(f"Saved model checkpoint: {final_model_path}")

    def _update_log_prob(
        self,
        data_save_path: Path,
        begin_idx: int,
        end_idx: int,
    ):
        """
        Adjust old_log_prob values in the selected dataset to match the current policy.
        This ensures consistency between the stored log probabilities and the updated policy.
        """
        for t in range(begin_idx, end_idx):
            batch_path = data_save_path / f"batch_{t + 1}.pkl"

            with open(batch_path, "rb") as f:
                rollout_data = pickle.load(f)

            if rollout_data is None:
                logger.info(f"Batch data at step {t} is None. Skipping adjustment...")
                continue

            if isinstance(rollout_data, dict):
                observations = rollout_data["observations"].to(self.device)
                actions = rollout_data["actions"].to(self.device)
            else:
                observations = rollout_data.observations.to(self.device)
                actions = rollout_data.actions.to(self.device)

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.long().flatten()

            with torch.no_grad():
                _, log_prob, _ = self.policy.evaluate_actions(observations, actions)

            if isinstance(rollout_data, dict):
                rollout_data["adjusted_log_prob"] = log_prob.cpu()
            else:
                rollout_data.adjusted_log_prob = log_prob.cpu()

            with open(batch_path, "wb") as f:
                pickle.dump(rollout_data, f)

    def _cleanup_rollout_cache(
        self,
        auxiliary_save_path: Path,
        att_save_path: Path,
        data_save_path: Path,
    ):
        """
        Clean up temporary cache directories after training on a rollout.
        Removes auxiliary models, attribution scores, and selected datasets.
        """
        import shutil

        cache_dirs = [auxiliary_save_path, att_save_path, data_save_path]

        for cache_dir in cache_dirs:
            if cache_dir.exists():
                try:
                    shutil.rmtree(cache_dir)
                    logger.info(f"Removed cache directory: {cache_dir}")
                except Exception as e:
                    logger.warning(f"Failed to remove cache directory {cache_dir}: {e}")

    def _cleanup_main_save_path(
        self,
        main_save_path: Path,
        n_updates_per_rollout: int,
    ) -> None:
        """
        Remove non-rollout-end checkpoints and all batch data in main_save_path.
        Keeps checkpoints at rollout boundaries (multiples of n_updates_per_rollout).
        """
        for path in main_save_path.glob("batch_*.pkl"):
            try:
                path.unlink()
            except Exception as exc:
                logger.warning(f"Failed to remove batch file {path}: {exc}")

        for path in main_save_path.glob("policy_grad_*.zip"):
            name = path.name
            idx_str = name[len("policy_grad_") : -len(".zip")]
            if not idx_str.isdigit():
                continue
            idx = int(idx_str)
            if idx % n_updates_per_rollout == 0:
                continue
            try:
                path.unlink()
            except Exception as exc:
                logger.warning(f"Failed to remove checkpoint {path}: {exc}")

    def _train_step_with_batch(
        self,
        rollout_data,
        t,
        n_updates,
        rollout_end,
        rollout_pos,
    ):
        """Perform a single training step with a batch of data."""

        if isinstance(rollout_data, dict):
            actions = rollout_data["actions"]
            observations = rollout_data["observations"]
            advantages = rollout_data["advantages"]
            returns = rollout_data["returns"]
            old_log_prob = rollout_data["old_log_prob"]
            if self.attribution_config.adjust_log_prob:
                adjusted_old_log_prob = rollout_data["adjusted_log_prob"]
            else:
                adjusted_old_log_prob = old_log_prob
            weights = rollout_data.get("weights", torch.ones_like(advantages))
        else:
            actions = rollout_data.actions
            observations = rollout_data.observations
            advantages = rollout_data.advantages
            returns = rollout_data.returns
            old_log_prob = rollout_data.old_log_prob
            if self.attribution_config.adjust_log_prob:
                adjusted_old_log_prob = rollout_data.adjusted_log_prob
            else:
                adjusted_old_log_prob = old_log_prob
            weights = getattr(rollout_data, "weights", torch.ones_like(advantages))

        actions = actions.to(self.device)
        observations = observations.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        old_log_prob = old_log_prob.to(self.device)

        if isinstance(self.action_space, spaces.Discrete):
            actions = actions.long().flatten()

        total_updates = self._total_timesteps // self.batch_size * self.n_epochs
        self._current_progress_remaining = 1 - (
            (int(t / n_updates) + 1) / (total_updates / n_updates)
        )

        self._update_learning_rate(self.policy.optimizer)

        r = self.attribution_config.off_policy_clipping_coeff
        clip_range = self.clip_range(self._current_progress_remaining) * (1.0 + rollout_pos * r)

        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)
        values = values.flatten()

        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(log_prob - old_log_prob)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2)

        if self.clip_range_vf is None:
            values_pred = values
        else:
            values_pred = rollout_data.old_values + torch.clamp(
                values - rollout_data.old_values, -clip_range_vf, clip_range_vf
            )
        value_loss = torch.nn.functional.mse_loss(returns, values_pred, reduction="none")

        if entropy is None:

            entropy_loss = log_prob
        else:

            entropy_loss = -entropy

        log_ratio = log_prob - adjusted_old_log_prob
        approx_kl_losses = (torch.exp(log_ratio) - 1) - log_ratio

        losses = policy_loss + self.ent_coef * entropy_loss + self.kl_coef * approx_kl_losses

        assert losses.shape[0] == len(weights), (losses.shape[0], len(weights))
        loss = (losses * weights + self.vf_coef * value_loss).mean()

        batch_size = len(observations)
        if batch_size != self.batch_size:
            logger.info(f'rescale batch t={t}: {batch_size}->{self.batch_size}.')
            for param in self.policy.parameters():
                if param.grad is not None:
                    param.grad.data *= batch_size / self.batch_size

        self.policy.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()

        with torch.no_grad():
            log_ratio = log_prob - old_log_prob
            approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()

        rollout_idx = t // n_updates

        return True

class PPOWithIterativeGlobalDataSelection(PPO):
    """
    PPO with iterative global data selection. Overrides the learn() method to:
    1. Collect rollout data with nested loop structure (auxiliary training phase)
    2. Compute attribution scores for each rollout
    3. Select influential data points
    4. Train with selected data (main training phase)

    This avoids multiple calls to learn() which causes environment randomness issues.
    """

    def __init__(
        self,
        policy: Union[str, type],
        env: Union[GymEnv, str],
        attribution_func=None,
        attribution_config=None,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        kl_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,

        save_path: Optional[Path] = None,
        cache_grad=False,
        cache_rollouts=False,
        clear_cache=True,
        render_rollouts: bool = False,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

        self.attribution_func = attribution_func
        self.attribution_config = attribution_config
        self.base_save_path = Path(save_path) if save_path else Path("./ppo_checkpoints")

        self.cache_grad = cache_grad
        self.cache_rollouts = cache_rollouts
        self.clear_cache = clear_cache
        self._update_counter = 0
        self._current_rollout_idx = 0
        self._rollout_begin = 0
        self._auxiliary_batches = []
        self.kl_coef = kl_coef

        self.seed = seed
        self.render_rollouts = render_rollouts
        self.render_save_path = self.base_save_path / "frames"

    def _excluded_save_params(self) -> list[str]:
        """
        Exclude non-serializable attributes from being saved.
        This prevents pickling errors with attribution functions and configs.
        """
        excluded = super()._excluded_save_params()
        excluded.extend([
            "attribution_func",
            "attribution_config",
            "_auxiliary_batches",
        ])
        return excluded

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        rollout_begin: int = 0,
        rollout_end: Optional[int] = None,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        """
        Overridden learn method that performs iterative global data selection within a single continuous session.

        This avoids the environment randomness issues caused by calling learn() multiple times.

        :param total_timesteps: Total timesteps for training
        :param rollout_begin: Starting rollout index
        :param rollout_end: Ending rollout index (computed if None)
        :param callback: Callback for training events
        :param log_interval: Logging interval
        :param tb_log_name: Tensorboard log name
        :param reset_num_timesteps: Whether to reset timesteps
        :param progress_bar: Whether to show progress bar
        """

        total_rollouts = total_timesteps // (self.n_steps * self.env.num_envs)
        if rollout_end is None:
            rollout_end = total_rollouts

        self._rollout_begin = rollout_begin
        n_updates_per_rollout = self.env.num_envs * self.n_steps // self.batch_size * self.n_epochs

        iteration = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None
        global_start_time = time.time()

        def mean_episode_return(rollout_buffer):
            rewards = rollout_buffer.rewards
            episode_starts = rollout_buffer.episode_starts

            n_steps, n_envs = rewards.shape
            episode_returns = []

            for env in range(n_envs):
                current_return = 0.0

                for t in range(n_steps):
                    if episode_starts[t, env] and t > 0:
                        episode_returns.append(current_return)
                        current_return = 0.0

                    current_return += rewards[t, env]

                episode_returns.append(current_return)

            return np.mean(episode_returns)

        return_log = {}
        rollout_times = {}

        step_size = self.attribution_config.step_size
        for curr_begin in range(rollout_begin, rollout_end):
            stage_target_rollout = curr_begin + step_size
            curr_end = curr_begin + 1

            main_save_path = self.base_save_path / "models" / "base" / f"{rollout_begin}-{rollout_end}"
            auxiliary_save_path = self.base_save_path / "models" / "auxiliary" / f"{curr_begin}-{stage_target_rollout}"
            att_save_path = self.base_save_path / "attribution" / f"{curr_begin}-{stage_target_rollout}"
            data_save_path = self.base_save_path / "dataset" / f"{curr_begin}-{stage_target_rollout}"

            all_save_pathes = [main_save_path]
            if self.attribution_func is not None:
                all_save_pathes.extend([auxiliary_save_path, att_save_path, data_save_path])

            for path in all_save_pathes:
                path.mkdir(parents=True, exist_ok=True)

            self._update_counter = n_updates_per_rollout * curr_begin

            logger.info(f"Phase 1: Auxiliary training from rollout {curr_begin} to {stage_target_rollout}")
            self._auxiliary_batches = []

            if curr_begin == rollout_begin:
                init_main_path = main_save_path / f"policy_grad_{self._update_counter}.zip"
                self.save(init_main_path)
            else:

                prev_checkpoint_idx = curr_begin * n_updates_per_rollout
                prev_checkpoint_path = main_save_path / f"policy_grad_{prev_checkpoint_idx}.zip"
                prev_model = PPO.load(prev_checkpoint_path, device=self.device)
                self.policy.load_state_dict(prev_model.policy.state_dict())
                self.policy.optimizer.load_state_dict(prev_model.policy.optimizer.state_dict())
                del prev_model

            save_path = main_save_path if self.attribution_func is None else auxiliary_save_path
            for rollout_idx in range(curr_begin, stage_target_rollout):
                stopped_idx = rollout_idx + 1

                logger.info(f"Setting random seed for rollout {rollout_idx}: {self.seed + rollout_idx}")
                set_random_seed(self.seed + rollout_idx)
                self._current_rollout_idx = rollout_idx

                logger.info("=" * 60)
                logger.info(f"Processing Rollout {rollout_idx + 1}/{rollout_end}")
                logger.info("=" * 60)

                rollout_callback = callback
                if self.render_rollouts:
                    frame_dir = self.render_save_path / f"{rollout_begin}-{rollout_end}"
                    frame_file = frame_dir / f"rollout_{rollout_idx}.pkl"
                    rollout_callback = _wrap_callback_with_frame_recorder(
                        self,
                        callback,
                        frame_file,
                    )

                continue_training = self._auxiliary_training_iteration(
                    save_path,
                    rollout_callback,
                    iteration,
                    log_interval,
                )

                if self.cache_rollouts:
                    rollout_file = save_path / f"rollout_{rollout_idx}.pkl"
                    with open(rollout_file, "wb") as f:
                        pickle.dump(self.rollout_buffer, f)

                if rollout_idx == curr_begin:
                    curr_return = mean_episode_return(self.rollout_buffer)
                    return_log[(curr_begin, rollout_idx)] = curr_return

                    rollout_end_time = time.time()
                    rollout_duration = rollout_end_time - global_start_time
                    rollout_times[rollout_idx] = rollout_duration
                    logger.info(f"Rollout {rollout_idx} completed in {rollout_duration:.2f} seconds")

                    max_return = curr_return

                else:
                    curr_return = mean_episode_return(self.rollout_buffer)
                    return_log[(curr_begin, rollout_idx)] = curr_return

                    rollout_end_time = time.time()
                    rollout_duration = rollout_end_time - global_start_time
                    rollout_times[rollout_idx] = rollout_duration
                    logger.info(f"Rollout {rollout_idx} completed in {rollout_duration:.2f} seconds")

                    if curr_return < self.attribution_config.max_return_factor * max_return:
                        stopped_idx -= 1
                        break

                    max_return = curr_return
                if not continue_training:
                    break

                iteration += 1

            stage_target_rollout = stopped_idx

            if self.attribution_func is not None:
                logger.info(f"Phase 2: Computing attribution scores")

                target_model_idx = curr_end * n_updates_per_rollout
                target_model_path = auxiliary_save_path / f"policy_grad_{target_model_idx}.zip"
                target_model = PPO.load(target_model_path, device=self.device)

                attribution_scores = self.attribution_func(
                    target_model,
                    self.rollout_buffer,
                    n_updates_per_rollout,
                    auxiliary_save_path,
                    curr_begin * n_updates_per_rollout,
                    stage_target_rollout * n_updates_per_rollout,

                    total_rollouts * n_updates_per_rollout,
                    device=self.device,
                )

                del target_model

                att_file = att_save_path / f"attribution_scores.pkl"
                with open(att_file, "wb") as f:
                    pickle.dump(attribution_scores, f)

                logger.info(f"Phase 3: Building selected dataset")
                build_dataset_from_influence(
                    folder=att_save_path,
                    model_path=auxiliary_save_path,
                    k=curr_begin,
                    ed=curr_end,
                    n_epochs=self.n_epochs,
                    n_updates=n_updates_per_rollout,
                    batch_size=self.batch_size,
                    method=self.attribution_config.method,
                    drop_percentile=self.attribution_config.drop_percentile,
                    mode=self.attribution_config.discard_mode,
                    selection_mode=self.attribution_config.selection_mode,
                    w_min=self.attribution_config.w_min,
                    order=self.attribution_config.order,
                    group_by=self.attribution_config.group_by,
                    out_dir=data_save_path,
                    seed=self.seed,
                )

                logger.info(f"Phase 4: Training with selected data")

                prev_checkpoint_idx = curr_begin * n_updates_per_rollout
                prev_checkpoint_path = main_save_path / f"policy_grad_{prev_checkpoint_idx}.zip"
                prev_model = PPO.load(prev_checkpoint_path, device=self.device)
                self.policy.load_state_dict(prev_model.policy.state_dict())
                self.policy.optimizer.load_state_dict(prev_model.policy.optimizer.state_dict())
                del prev_model

                self._train_with_selected_batches(
                    data_save_path,
                    main_save_path,
                    curr_begin,
                    curr_end,
                    n_updates_per_rollout,
                )

                if self.clear_cache:
                    logger.info(f"Phase 5: Cleaning up cache for rollout {curr_begin} - {stage_target_rollout}")
                    self._cleanup_rollout_cache(auxiliary_save_path, att_save_path, data_save_path)
                    self._cleanup_main_save_path(main_save_path, n_updates_per_rollout)

            rollout_end_time = time.time()
            rollout_duration = rollout_end_time - global_start_time
            rollout_times[curr_begin] = rollout_duration
            logger.info(f"Rollout {curr_begin} completed in {rollout_duration:.2f} seconds")

        rollout_times_path = self.base_save_path / "rollout_times.pkl"
        with open(rollout_times_path, "wb") as f:
            pickle.dump(rollout_times, f)
        logger.info(f"Saved rollout runtime log to {rollout_times_path}")

        if rollout_times:
            total_time = sum(rollout_times.values())
            avg_time = total_time / len(rollout_times)
            logger.info(f"Total training time: {total_time:.2f} seconds")
            logger.info(f"Average time per rollout: {avg_time:.2f} seconds")

        callback.on_training_end()

        return self

    def _auxiliary_training_iteration(
        self,
        save_path: Path,
        callback: BaseCallback,
        iteration: int,
        log_interval: int,
    ) -> bool:
        """
        Perform one auxiliary training iteration (collect rollout + train).
        This is similar to one iteration of the standard PPO learn loop.
        """

        continue_training = self.collect_rollouts(
            self.env,
            callback,
            self.rollout_buffer,
            n_rollout_steps=self.n_steps,
        )

        if not continue_training:
            return False

        iteration += 1
        self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
        logger.info(f"Current progress remaining: {self._current_progress_remaining}")

        if log_interval is not None and iteration % log_interval == 0:
            assert self.ep_info_buffer is not None
            time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
            fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
            if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            self.logger.record("time/fps", fps)
            self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(step=self.num_timesteps)

        self._auxiliary_train_with_batch_saving(save_path)

        return True

    def _auxiliary_train_with_batch_saving(self, save_path: Path):
        """
        Train on the current rollout buffer and save each mini-batch for later attribution.
        This is the auxiliary training phase.
        """

        if self.cache_grad or self.attribution_func is not None:
            init_checkpoint = save_path / f"policy_grad_{self._update_counter}.zip"
            self.save(init_checkpoint)

        self.policy.set_training_mode(True)

        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)

        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                self._update_counter += 1

                if self.cache_grad or self.attribution_func is not None:
                    batch_filename = save_path / f"batch_{self._update_counter}.pkl"

                    rollout_dict = rollout_data._asdict()
                    with open(batch_filename, "wb") as f:
                        pickle.dump(rollout_dict, f)

                self._auxiliary_batches.append((self._update_counter, rollout_data))

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = torch.nn.functional.mse_loss(rollout_data.returns, values_pred)

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                log_ratio = log_prob - rollout_data.old_log_prob
                approx_kl_loss = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.kl_coef * approx_kl_loss

                self.policy.optimizer.zero_grad()
                loss.backward()

                self.policy.optimizer.step()

                if self.cache_grad or self.attribution_func is not None:
                    checkpoint_filename = save_path / f"policy_grad_{self._update_counter}.zip"
                    self.save(checkpoint_filename)

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

            self._n_updates += 1
            if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                break

        checkpoint_filename = save_path / f"policy_grad_{self._update_counter}.zip"
        self.save(checkpoint_filename)

    def _train_with_selected_batches(
        self,
        data_save_path: Path,
        main_save_path: Path,
        rollout_begin: int,
        rollout_end: int,
        n_updates: int,
    ):
        """
        Train with pre-selected data batches (main training phase).
        This performs offline training on the selected dataset.
        """
        self.policy.train()

        pbar = tqdm(range(n_updates * rollout_begin, n_updates * rollout_end), desc="Training")
        for t in pbar:
            pbar.set_postfix({"progress_remaining": f"{self._current_progress_remaining:.4f}"})

            batch_path = data_save_path / f"batch_{t + 1}.pkl"

            with open(batch_path, "rb") as f:
                rollout_data = pickle.load(f)

            if rollout_data is None:
                logger.info(f"Batch data at step {t} is None. Quitting...")
                break

            self._train_step_with_batch(rollout_data, t, n_updates, rollout_end)

            if self.cache_grad or (t + 1) % n_updates == 0:
                self.save(main_save_path / f"policy_grad_{t + 1}.zip")

        final_idx = n_updates * rollout_end
        final_model_path = main_save_path / f"policy_grad_{final_idx}.zip"
        self.save(final_model_path)
        logger.info(f"Saved model checkpoint: {final_model_path}")

    def _cleanup_rollout_cache(
        self,
        auxiliary_save_path: Path,
        att_save_path: Path,
        data_save_path: Path,
    ):
        """
        Clean up temporary cache directories after training on a rollout.
        Removes auxiliary models, attribution scores, and selected datasets.
        """
        import shutil

        cache_dirs = [auxiliary_save_path, att_save_path, data_save_path]

        for cache_dir in cache_dirs:
            if cache_dir.exists():
                try:
                    shutil.rmtree(cache_dir)
                    logger.info(f"Removed cache directory: {cache_dir}")
                except Exception as e:
                    logger.warning(f"Failed to remove cache directory {cache_dir}: {e}")

    def _cleanup_main_save_path(
        self,
        main_save_path: Path,
        n_updates_per_rollout: int,
    ) -> None:
        """
        Remove non-rollout-end checkpoints and all batch data in main_save_path.
        Keeps checkpoints at rollout boundaries (multiples of n_updates_per_rollout).
        """
        for path in main_save_path.glob("batch_*.pkl"):
            try:
                path.unlink()
            except Exception as exc:
                logger.warning(f"Failed to remove batch file {path}: {exc}")

        for path in main_save_path.glob("policy_grad_*.zip"):
            name = path.name
            idx_str = name[len("policy_grad_") : -len(".zip")]
            if not idx_str.isdigit():
                continue
            idx = int(idx_str)
            if idx % n_updates_per_rollout == 0:
                continue
            try:
                path.unlink()
            except Exception as exc:
                logger.warning(f"Failed to remove checkpoint {path}: {exc}")

    def _train_step_with_batch(self, rollout_data, t, n_updates, rollout_end):
        """Perform a single training step with a batch of data."""

        if isinstance(rollout_data, dict):
            actions = rollout_data["actions"]
            observations = rollout_data["observations"]
            advantages = rollout_data["advantages"]
            returns = rollout_data["returns"]
            old_log_prob = rollout_data["old_log_prob"]
            weights = rollout_data.get("weights", torch.ones_like(advantages))
        else:
            actions = rollout_data.actions
            observations = rollout_data.observations
            advantages = rollout_data.advantages
            returns = rollout_data.returns
            old_log_prob = rollout_data.old_log_prob
            weights = getattr(rollout_data, "weights", torch.ones_like(advantages))

        actions = actions.to(self.device)
        observations = observations.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        old_log_prob = old_log_prob.to(self.device)

        if isinstance(self.action_space, spaces.Discrete):
            actions = actions.long().flatten()

        total_updates = self._total_timesteps // self.batch_size * self.n_epochs
        self._current_progress_remaining = 1 - (
            (int(t / n_updates) + 1) / (total_updates / n_updates)
        )

        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)

        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)
        values = values.flatten()

        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(log_prob - old_log_prob)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2)

        if self.clip_range_vf is None:
            values_pred = values
        else:
            values_pred = rollout_data.old_values + torch.clamp(
                values - rollout_data.old_values, -clip_range_vf, clip_range_vf
            )
        value_loss = torch.nn.functional.mse_loss(returns, values_pred, reduction="none")

        if entropy is None:

            entropy_loss = log_prob
        else:

            entropy_loss = -entropy

        log_ratio = log_prob - old_log_prob
        approx_kl_losses = (torch.exp(log_ratio) - 1) - log_ratio

        losses = policy_loss + self.ent_coef * entropy_loss + self.kl_coef * approx_kl_losses
        assert losses.shape[0] == len(weights), (losses.shape[0], len(weights))
        loss = (losses * weights + self.vf_coef * value_loss).mean()

        batch_size = len(observations)
        if batch_size != self.batch_size:
            logger.info(f'rescale batch t={t}: {batch_size}->{self.batch_size}.')
            for param in self.policy.parameters():
                if param.grad is not None:
                    param.grad.data *= batch_size / self.batch_size

        self.policy.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()
