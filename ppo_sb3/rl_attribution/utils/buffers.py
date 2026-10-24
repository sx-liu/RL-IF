import os
import pickle
import torch
import numpy as np
import gymnasium as gym

from tqdm import tqdm
from types import SimpleNamespace
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecTransposeImage

def collect_training_buffers(
    model_path: str,
    k: int,
    ed: int,
    n_batches: int = 32,
    n_epochs: int = 10,
    save_file: str = None,
):
    """
    Collects PPO rollout batches between [k, ed) and merges into one buffer file.

    Args:
        model_path (str): Path containing saved batch_*.pkl files.
        k (int): Start rollout index (inclusive).
        ed (int): End rollout index (exclusive).
        env_short (str): Environment short name (unused, for logging consistency).
        n_updates (int): Number of updates per epoch.
        n_epochs (int): Number of epochs per rollout.
        save_file (str): Output filename to save the combined buffer.
    """

    obs_list, act_list, adv_list, prob_list, ret_list = [], [], [], [], []

    for i in tqdm(range(k, ed), desc=f"Collecting rollouts {k}-{ed}"):
        for j in range(i * n_batches * n_epochs, i * n_batches * n_epochs + n_batches):
            batch_file = os.path.join(model_path, f"batch_{j+1}.pkl")
            with open(batch_file, "rb") as f:
                data = pickle.load(f)

            if isinstance(data, dict):
                obs_list.append(data["observations"])
                act_list.append(data["actions"])
                adv_list.append(data["advantages"])
                prob_list.append(data["old_log_prob"])
                ret_list.append(data["returns"])
            else:
                obs_list.append(data.observations)
                act_list.append(data.actions)
                adv_list.append(data.advantages)
                prob_list.append(data.old_log_prob)
                ret_list.append(data.returns)

    obs = torch.cat(obs_list, dim=0)
    act = torch.cat(act_list, dim=0)
    adv = torch.cat(adv_list, dim=0)
    prob = torch.cat(prob_list, dim=0)
    ret = torch.cat(ret_list, dim=0)

    buffer_save = {
        "observations": obs,
        "actions": act,
        "advantages": adv,
        "log_probs": prob,
        "returns": ret,
    }
    buffer_ns = SimpleNamespace(**buffer_save)

    print(f"Buffer shape: {obs.shape}, {act.shape}, {adv.shape}, {prob.shape}")

    if save_file is not None:
        with open(save_file, "wb") as f:
            pickle.dump(buffer_ns, f)

        print(f"Saved buffer to {save_file}")

    return buffer_save, buffer_ns

def sample_rollouts_buffers(env, n_rollout_steps, rollout_model, adv_estimate_model=None, gamma=0.99):
    """
    Sample rollouts using the rollout_model's policy but compute value estimates
    (and hence advantages) using the adv_estimate_model.

    :param env: A vectorized environment.
    :param n_rollout_steps: Total number of steps to collect.
    :param rollout_model: The model used to roll out episodes (for actions, log_probs).
    :param adv_estimate_model: The model used to compute value estimates (for advantage calculation).
    :param gamma: Discount factor.
    :return: A filled RolloutBuffer and a list of episode returns.
    """
    if hasattr(env, "envs") and env.envs[0].spec.id == "MiniGrid-Empty-8x8-v0":
        env = VecTransposeImage(env)

    rollout_model.policy.set_training_mode(False)
    assert adv_estimate_model is None

    n_envs = env.num_envs
    rollout_steps_per_env = (n_rollout_steps + n_envs - 1) // n_envs

    rollout_buffer = RolloutBuffer(
        rollout_steps_per_env,
        env.observation_space,
        env.action_space,
        device=rollout_model.device,
        gamma=gamma,
        n_envs=n_envs,
    )

    obs = env.reset()
    last_obs = obs

    last_episode_starts = np.ones((n_envs,), dtype=bool)

    ep_returns = np.zeros(n_envs)
    episode_returns = []

    n_steps = 0

    print(f"Sampling {rollout_steps_per_env} steps on {n_envs} envs.")
    while n_steps < rollout_steps_per_env:
        with torch.no_grad():

            obs_tensor = obs_as_tensor(last_obs, rollout_model.device)

            actions, values, log_probs = rollout_model.policy(obs_tensor)
        actions = actions.cpu().numpy()

        if isinstance(env.action_space, gym.spaces.Box):
            if rollout_model.policy.squash_output:
                clipped_actions = rollout_model.policy.unscale_action(actions)
            else:
                clipped_actions = np.clip(actions, env.action_space.low, env.action_space.high)
        else:
            clipped_actions = actions

        new_obs, rewards, dones, infos = env.step(clipped_actions)
        n_steps += 1

        rewards = np.array(rewards)
        ep_returns += rewards

        for idx, done in enumerate(dones):
            if done:
                episode_returns.append(ep_returns[idx])
                ep_returns[idx] = 0.0

        for idx, done in enumerate(dones):
            if done and infos[idx].get("terminal_observation") is not None and infos[idx].get("TimeLimit.truncated", False):
                terminal_obs = rollout_model.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                with torch.no_grad():
                    terminal_value = rollout_model.policy.predict_values(terminal_obs)[0]
                rewards[idx] += gamma * terminal_value

        rollout_buffer.add(last_obs, actions, rewards, last_episode_starts, values, log_probs)
        last_obs = new_obs
        last_episode_starts = dones

        if n_steps * n_envs % 10000 == 0:
            print(f"Collected {n_steps * n_envs} steps so far...")

    with torch.no_grad():

        last_values = rollout_model.policy.predict_values(
            obs_as_tensor(new_obs, rollout_model.device)
        )
    rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)

    def _flatten(x):
        if x is None:
            return None
        return x.reshape((-1,) + x.shape[2:])

    rollout_buffer_dict = {
        "observations": _flatten(rollout_buffer.observations),
        "actions": _flatten(rollout_buffer.actions),
        "advantages": rollout_buffer.advantages.reshape(-1),
    }
    del rollout_buffer

    print("Collected rollout buffer:",
        rollout_buffer_dict["observations"].shape,
        rollout_buffer_dict["actions"].shape,
        rollout_buffer_dict["advantages"].shape,
    )

    return rollout_buffer_dict, episode_returns
