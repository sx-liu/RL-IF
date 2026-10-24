import os
import numpy as np
import torch
import torch.nn.functional as F

import pickle
import wandb
from tqdm import tqdm
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.utils import update_learning_rate
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from rl_attribution.utils.callbacks import SaveStatesCallback
from rl_attribution.utils.envs import make_env
from rl_attribution.utils.logger import get_logger
from rl_attribution.models.ppo_model import get_ppo_model

from rl_attribution.trainers.ppo import PPOWithLocalDataSelection

PPOClass = PPO

logger = get_logger()

def train_ppo(
    env,
    ppo_config,
    rollout_begin,
    rollout_end,
    save_path,
    initial_policy=None,
    cache_grad=False,
    cache_rollouts=False,
    seed=42,
    n_envs=8,
    device="cpu",
    use_wandb=False,
    project_name="rl_attribution",
    run_name=None
):
    assert env.num_envs == n_envs, (env.num_envs, n_envs)

    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(save_path, exist_ok=True)

    if use_wandb:
        wandb.init(
            project=project_name,
            name=run_name,
            config=ppo_config,
            sync_tensorboard=True,
            save_code=True,
        )

    logger.info('env:', env)
    logger.info('observation space:', env.observation_space)
    logger.info('number of actions:', env.action_space.n if hasattr(env.action_space, 'n') else None)
    logger.info(f"Using device: {device}")

    model = get_ppo_model(
        env,
        ppo_config,
        seed=seed,
        device=device,
    )

    if initial_policy is not None:
        model_init = PPOClass.load(
            initial_policy,
            device=device,
            seed=seed,
        )
        model.policy = model_init.policy
        logger.info('Loaded model with initial policy:', initial_policy)

    total_timesteps = ppo_config.total_timesteps
    global_steps = n_envs * ppo_config.n_steps * rollout_begin

    n_updates = n_envs * ppo_config.n_steps // ppo_config.batch_size * ppo_config.n_epochs
    model._update_counter = n_updates * rollout_begin

    init_save_path = os.path.join(save_path, f"policy_grad_{model._update_counter}.zip")
    if os.path.exists(init_save_path):
        def compare_models(m1, m2):
            m1_dict = dict(m1.named_parameters())
            m2_dict = dict(m2.named_parameters())

            diff = []
            for n in m1_dict:
                assert n in m2_dict
                if not torch.allclose(m1_dict[n], m2_dict[n]):
                    diff.append(n)

            return diff
        model_init = PPOClass.load(init_save_path, device=device, seed=seed)
        assert not compare_models(model.policy, model_init.policy), f"Model parameters do not match at {init_save_path}!"

    model.save(init_save_path)

    model.num_timesteps = global_steps

    original_optimizer_step = model.policy.optimizer.step
    def patched_optimizer_step(*args_, **kwargs):
        result = original_optimizer_step(*args_, **kwargs)
        if cache_grad:
            checkpoint_filename = os.path.join(save_path, f"policy_grad_{model._update_counter}.zip")
            model.save(checkpoint_filename)
        return result
    model.policy.optimizer.step = patched_optimizer_step

    save_states_callback = SaveStatesCallback(save_path=save_path if cache_rollouts else None, verbose=1)
    callback = CallbackList([WandbCallback(verbose=2), save_states_callback]) if use_wandb else save_states_callback

    logger.info('===============================================================')
    logger.info(f'Begin training at rollout_idx = {rollout_begin} and global_steps = {global_steps}')
    logger.info(f'Target rollouts = {rollout_end} with total global_steps = {total_timesteps}')

    original_get = model.rollout_buffer.get
    def patched_get(batch_size):
        for mini_batch in original_get(batch_size):
            model._update_counter += 1
            if cache_grad:
                batch_filename = os.path.join(save_path, f"batch_{model._update_counter}.pkl")
                with open(batch_filename, "wb") as f:
                    pickle.dump(mini_batch, f)
            yield mini_batch
    model.rollout_buffer.get = patched_get

    for rollout_idx in range(rollout_begin, rollout_end):
        logger.info(f"Training on rollout {rollout_idx+1}/{rollout_end} with {model.n_steps} transitions.")

        global_steps += model.n_steps
        logger.info(f"Global progress: {1.0 - (global_steps / total_timesteps):.3f}")

        model.learn(
            total_timesteps=ppo_config.total_timesteps,
            rollout_begin=rollout_idx,
            rollout_end=rollout_idx + 1,
            callback=callback,
        )

        print(model._update_counter)
        model.save(os.path.join(save_path, f"policy_grad_{model._update_counter}.zip"))

    final_model_path = os.path.join(save_path, "final_policy.zip")
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    if use_wandb:
        wandb.finish()

def train_ppo_with_selected_data(
    env_name,
    ppo_config,
    batches_dir,
    rollout_begin,
    rollout_end,
    save_path,
    leave_out_idx=None,
    leave_out_weight=1.0,
    initial_policy=None,
    cache_grad=False,
    seed=42,
    n_envs=8,
    device="cpu",
    use_wandb=False,
    project_name="rl_attribution",
    run_name=None
):

    os.makedirs(save_path, exist_ok=True)

    if use_wandb:
        wandb.init(
            project=project_name,
            name=run_name,
            config=ppo_config,
            sync_tensorboard=True,
            save_code=True,
        )

    env = make_env(env_name, None)()

    logger.info('env is :', env_name)
    logger.info('observation space:', env.observation_space)
    logger.info('number of actions:', env.action_space.n if hasattr(env.action_space, 'n') else None)

    model = get_ppo_model(
        env,
        ppo_config,
        seed=seed,
        device=device,
    )

    if initial_policy:
        model_init = PPOClass.load(initial_policy, device=device, seed=seed)
        model.policy = model_init.policy
        logger.info(f"Loaded initial policy from {initial_policy}")

    logger.info("normalize_advantage:", model.normalize_advantage)
    model.policy.train()

    n_updates = n_envs * ppo_config.n_steps // ppo_config.batch_size * ppo_config.n_epochs
    model._update_counter = n_updates * rollout_begin

    init_save_path = os.path.join(save_path, f"policy_grad_{model._update_counter}.zip")
    if os.path.exists(init_save_path):
        def compare_models(m1, m2):
            m1_dict = dict(m1.named_parameters())
            m2_dict = dict(m2.named_parameters())

            diff = []
            for n in m1_dict:
                assert n in m2_dict
                if not torch.allclose(m1_dict[n], m2_dict[n]):
                    diff.append(n)

            return diff
        model_init = PPOClass.load(init_save_path, device=device, seed=seed)
        assert not compare_models(model.policy, model_init.policy), f"Model parameters do not match at {init_save_path}!"

    model.save(init_save_path)

    sanity_check = False

    if leave_out_idx is not None:
        if not isinstance(leave_out_idx, list):
            leave_out_idx = [leave_out_idx]

        leave_out_dict = {}
        for i, j in leave_out_idx:
            if i not in leave_out_dict:
                leave_out_dict[i] = []
            leave_out_dict[i].append(j)

    for t in tqdm(range(n_updates * rollout_begin, n_updates * rollout_end)):
        batch_path = os.path.join(batches_dir, f"batch_{t+1}.pkl")
        with open(batch_path, "rb") as f:
            rollout_data = pickle.load(f)

        if rollout_data is None:
            logger.info(f'encountered None batch at t={t}, exiting')

            model.save(os.path.join(save_path, f"policy_grad_{n_updates * rollout_end}.zip"))
            break

        total_updates = ppo_config.total_timesteps // ppo_config.batch_size * ppo_config.n_epochs
        progress = 1 - ((int(t / n_updates) + 1) / (total_updates / n_updates))

        optimizers = model.policy.optimizer
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, model.lr_schedule(progress))

        clip_range = model.clip_range(progress) if callable(model.clip_range) else model.clip_range

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

        actions = actions.to(device)
        observations = observations.to(device)
        advantages = advantages.to(device)
        returns = returns.to(device)
        old_log_prob = old_log_prob.to(device)

        if isinstance(model.action_space, spaces.Discrete):
            actions = actions.long().flatten()

        batch_size = len(observations)

        values, log_prob, entropy = model.policy.evaluate_actions(observations, actions)
        values = values.flatten()

        if model.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(log_prob - old_log_prob)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2)

        value_loss = F.mse_loss(returns, values, reduction="none")
        if entropy is None:

            entropy_loss = log_prob
        else:

            entropy_loss = -entropy

        losses = policy_loss + model.ent_coef * entropy_loss

        log_ratio = log_prob - old_log_prob
        approx_kl_losses = (torch.exp(log_ratio) - 1) - log_ratio

        if leave_out_idx is not None and t in leave_out_dict:

            for j in leave_out_dict[t]:
                weights[j] *= (1.0 - leave_out_weight)
            sanity_check = True

        loss = (losses * weights + model.vf_coef * value_loss + model.kl_coef * approx_kl_losses * weights).mean()

        model.policy.optimizer.zero_grad()
        loss.backward()

        if batch_size != ppo_config.batch_size:
            logger.info(f'rescale batch t={t}: {batch_size}->{ppo_config.batch_size}.')
            for param in model.policy.parameters():
                if param.grad is not None:
                    param.grad.data *= batch_size / ppo_config.batch_size
        model.policy.optimizer.step()

        if cache_grad or (t + 1) % n_updates == 0:
            model.save(os.path.join(save_path, f"policy_grad_{t+1}.zip"))

    final_model_path = os.path.join(save_path, "final_policy.zip")
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    if leave_out_idx is not None:
        assert sanity_check

    if use_wandb:
        wandb.finish()
    env.close()
