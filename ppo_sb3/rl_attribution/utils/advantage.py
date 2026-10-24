import os
from typing import Any, Optional, Sequence

import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
import minigrid
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.buffers import RolloutBuffer

import pickle

from tqdm import tqdm
from collections import defaultdict

import numpy as np
from typing import Any, Callable, Dict, List, Tuple

from .buffers import sample_rollouts_buffers
from .tensor import flatten_ob_act
from .envs import CloneableLunarLander

def _row_keys(x: np.ndarray) -> np.ndarray:
    """
    Make hashable per-row keys using a void view (exact-match grouping).
    Returns a 1D array of dtype voidN (N = bytes in one row).
    """
    x2 = np.ascontiguousarray(np.atleast_2d(x))

    xb = x2.view(np.uint8).reshape(x2.shape[0], -1)
    return xb.view(np.dtype((np.void, xb.shape[1]))).ravel()

def mc_returns(rewards: np.ndarray, episode_starts: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """
    Per-step Monte Carlo returns G_t within episodes, no bootstrapping across boundaries.
    `episode_starts[t]` must be True/1 at the FIRST step of an episode.
    """
    r = rewards.astype(np.float32)
    starts = (episode_starts.astype(np.int8) != 0)

    N = r.shape[0]
    G = np.zeros_like(r, dtype=np.float32)

    next_start = np.empty_like(starts)
    next_start[:-1] = starts[1:]
    next_start[-1] = True

    ret = 0.0
    for t in range(N - 1, -1, -1):
        if next_start[t]:
            ret = 0.0
        ret = r[t] + gamma * ret
        G[t] = ret
    return G

def group_means(inv: np.ndarray, values: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Given inverse indices from np.unique and per-item values, compute group means
    and broadcast them back to item order.
    """
    sums = np.bincount(inv, weights=values, minlength=n_groups)
    counts = np.bincount(inv, minlength=n_groups)
    means = sums / np.maximum(counts, 1)
    return means[inv].astype(np.float32)

def mc_grouped_QV_and_advantages(
    obs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    episode_starts: np.ndarray,
    gamma: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute MC returns G_t, then every-visit grouped estimates:
      V_hat(s) = E[G_t | s_t = s]
      Q_hat(s,a) = E[G_t | s_t = s, a_t = a]
    and advantages A_hat = Q_hat - V_hat, all aligned with the input order.

    Returns:
        Q_hat: shape (N,), grouped by (state, action)
        V_hat: shape (N,), grouped by (state)
        A_hat: shape (N,), Q_hat - V_hat
    """

    G = mc_returns(rewards, episode_starts, gamma=gamma)

    state_keys = _row_keys(obs)

    actions_keys = actions.ravel()
    sa_keys = np.empty(state_keys.shape[0], dtype=[('s', state_keys.dtype), ('a', np.int64)])
    sa_keys['s'] = state_keys
    sa_keys['a'] = actions_keys

    uniq_s, inv_s = np.unique(state_keys, return_inverse=True)
    uniq_sa, inv_sa = np.unique(sa_keys, return_inverse=True)

    V_hat = group_means(inv_s, G, n_groups=len(uniq_s))
    Q_hat = group_means(inv_sa, G, n_groups=len(uniq_sa))

    A_hat = (Q_hat - V_hat).astype(np.float32)

    return Q_hat, V_hat, A_hat

def build_advantage_dict(obs_array, action_array, advantage_array):
    """
    Create a dictionary mapping (obs, action) -> advantage.
    Each obs row is converted into a tuple of numbers (so it's hashable).

    Args:
        obs_array: (N, D) NumPy array of observations.
        action_array: (N,) NumPy array of discrete actions.
        advantage_array: (N,) NumPy array of advantages.

    Returns:
        dict mapping (tuple(obs), int(action)) -> float(advantage)
    """
    adv_dict = {}
    for obs, act, adv in zip(obs_array, action_array, advantage_array):
        key = flatten_ob_act(obs, act)
        if key in adv_dict:
            assert adv_dict[key] == float(adv), "Inconsistent advantages for the same (obs, action) pair."
        adv_dict[key] = float(adv)
    return adv_dict

def evalute_advantage_estimate(
    obs_array,
    action_array,
    rewards_array,
    advantage_array,
    starts_array,
):
    Q_hat, V_hat, A_hat = mc_grouped_QV_and_advantages(
        obs_array, action_array, rewards_array, starts_array, gamma=0.99
    )

    assert A_hat.shape == advantage_array.shape, (A_hat.shape, advantage_array.shape)
    abs_error = np.abs(advantage_array - A_hat)
    dot_product = advantage_array * A_hat
    sign_agreement = np.sign(advantage_array) == np.sign(A_hat)

    return {
        "abs_error": abs_error,
        "dot_product": dot_product,
        "sign_agreement": sign_agreement,
    }

def evalute_advantage_estimate_with_cache(
    batches_dir,
    rollout_idx,
):
    obs_array = np.load(os.path.join(batches_dir, f"r{rollout_idx}_states.npy")).squeeze(1)
    action_array = np.load(os.path.join(batches_dir, f"r{rollout_idx}_actions.npy")).flatten()
    rewards_array = np.load(os.path.join(batches_dir, f"r{rollout_idx}_rewards.npy")).flatten()
    advantage_array = np.load(os.path.join(batches_dir, f"r{rollout_idx}_advantages.npy")).flatten()
    starts_array = np.load(os.path.join(batches_dir, f"r{rollout_idx}_episode_starts.npy")).flatten()

    return evalute_advantage_estimate(
        obs_array,
        action_array,
        rewards_array,
        advantage_array,
        starts_array,
    )

def evalute_advantage_estimate_with_sampling(
    env,
    n_rollout_steps,
    batches_dir,
    model_idx,
):
    model_path = batches_dir / f"policy_grad_{model_idx}.zip"
    model = PPO.load(model_path)

    rollout_buffer, _ = sample_rollouts_buffers(
        env=env,
        n_rollout_steps=n_rollout_steps,
        rollout_model=model,
    )

    obs_array = rollout_buffer.observations.squeeze(1)
    action_array = rollout_buffer.actions.flatten()
    rewards_array = rollout_buffer.rewards.flatten()
    advantage_array = rollout_buffer.advantages.flatten()
    starts_array = rollout_buffer.episode_starts.flatten()

    return rollout_buffer, evalute_advantage_estimate(
        obs_array,
        action_array,
        rewards_array,
        advantage_array,
        starts_array,
    )

def evalute_advantage_estimate_with_repeated_sampling(
    env,

    batches_dir,
    model_idx,
):
    print("Env Name:", env.envs[0].spec.id)
    if "LunarLander" in env.envs[0].spec.id:
        env.envs[0] = CloneableLunarLander(env.envs[0])

    model_path = batches_dir / f"policy_grad_{model_idx}.zip"
    model = PPO.load(model_path)

    rollout_buffer, mc_estimation = estimate_advantages_by_repeats(
        env,
        model,
        n_collect_steps = 1000,
        n_pairs = 16,
        repeats_q = 128,
        repeats_v = 128,
        horizon=1000,
    )

    advantage_array = rollout_buffer["advantages"]
    A_hat = mc_estimation["A_hat"]

    abs_error = np.abs(advantage_array - A_hat)
    dot_product = advantage_array * A_hat
    sign_agreement = np.sign(advantage_array) == np.sign(A_hat)

    return rollout_buffer, {
        "abs_error": abs_error,
        "dot_product": dot_product,
        "sign_agreement": sign_agreement,
    }

def _get_base_env_and_index(env: VecEnv, i_env: int = 0):
    if isinstance(env, DummyVecEnv):
        return env.envs[i_env], i_env, "dummy"
    if isinstance(env, SubprocVecEnv):
        return env, i_env, "subproc"
    base = getattr(env, "envs", [None])[i_env]
    if base is None:
        raise RuntimeError("Unsupported VecEnv type; cannot access underlying env.")
    return base, i_env, "unknown"

def _auto_state_io(env: VecEnv, i_env: int = 0) -> Tuple[Callable[[], Any], Callable[[Any], None]]:
    """
    Auto-detect get_state()/set_state() for sub-env i_env.
    Works directly for DummyVecEnv; for SubprocVecEnv the worker must expose methods.
    """
    base, idx, kind = _get_base_env_and_index(env, i_env)

    if kind == "subproc":
        def _has(name: str) -> bool:
            try:
                _ = env.env_method(name, indices=[idx])
                return True
            except Exception:
                return False

        if _has("get_state") and _has("set_state"):
            def get_state(): return env.env_method("get_state", indices=[idx])[0]
            def set_state(s): env.env_method("set_state", s, indices=[idx])
            return get_state, set_state
        if _has("clone_state") and _has("restore_state"):
            def get_state(): return env.env_method("clone_state", indices=[idx])[0]
            def set_state(s): env.env_method("restore_state", s, indices=[idx])
            return get_state, set_state
        if _has("clone_full_state") and _has("restore_full_state"):
            def get_state(): return env.env_method("clone_full_state", indices=[idx])[0]
            def set_state(s): env.env_method("restore_full_state", s, indices=[idx])
            return get_state, set_state

        raise RuntimeError(
            "SubprocVecEnv: no state cloning interface detected. "
            "Expose get_state/set_state (or clone/restore) on the worker, "
            "or switch to DummyVecEnv for this evaluation."
        )

    u = base
    for g, s in [("get_state", "set_state"), ("clone_state", "restore_state")]:
        if hasattr(u, g) and hasattr(u, s):
            def _g(u=u, g=g): return getattr(u, g)()
            def _s(st, u=u, s=s): return getattr(u, s)(st)
            return _g, _s

    if hasattr(u, "clone_full_state") and hasattr(u, "restore_full_state"):
        return u.clone_full_state, u.restore_full_state

    if hasattr(u, "physics") and hasattr(u.physics, "get_state") and hasattr(u.physics, "set_state"):
        def get_state(): return u.physics.get_state()
        def set_state(st): u.physics.set_state(st)
        return get_state, set_state

    if hasattr(u, "sim") and hasattr(u.sim, "get_state") and hasattr(u.sim, "set_state"):
        def get_state(): return u.sim.get_state()
        def set_state(st):
            u.sim.set_state(st)
            if hasattr(u.sim, "forward"): u.sim.forward()
        return get_state, set_state

    if hasattr(u, "state") and isinstance(getattr(u, "state"), np.ndarray):
        def get_state(): return u.state.copy()
        def set_state(st): setattr(u, "state", st.copy())
        return get_state, set_state

    raise RuntimeError(
        "No state cloning API found. Please add get_state/set_state (or clone/restore) to your env."
    )

def _pack_action_for_vecenv(env, i_env, action):
    """
    Build a batch action for VecEnv with exactly one env (i_env) set to `action`
    and others set to a neutral/default (zeros). Handles Discrete, MultiDiscrete, Box.
    """
    space = env.action_space
    if isinstance(space, gym.spaces.Discrete):
        batch = np.zeros((env.num_envs,), dtype=np.int64)
        batch[i_env] = int(action)
        return batch

    if isinstance(space, gym.spaces.MultiDiscrete):
        batch = np.zeros((env.num_envs, len(space.nvec)), dtype=np.int64)
        batch[i_env] = np.asarray(action, dtype=np.int64)
        return batch

    if isinstance(space, gym.spaces.Box):
        batch = np.zeros((env.num_envs,) + space.shape, dtype=np.float32)
        batch[i_env] = np.asarray(action, dtype=np.float32)

        return batch

    raise TypeError(f"Unsupported action space: {type(space)}")

def estimate_advantages_by_repeats(
    env: VecEnv,
    rollout_model,
    adv_estimate_model=None,
    n_collect_steps: int = 10_000,
    n_pairs: int = 256,
    repeats_q: int = 16,
    repeats_v: int = 16,
    gamma: float = 0.99,
    horizon: int = None,
    gae_lambda: float = 0.95,
    i_env: int = 0,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Outputs both:
      - MC estimates: Q_hat(s,a), V_hat(s), A_hat = Q_hat - V_hat (for n_pairs targets)
      - Value-net-based: returns_vnet, advantages_vnet, values_vnet (full stream),
        plus their slices at the chosen targets.

    Notes:
      * MC estimates reuse exact saved simulator states (auto-detected snapshot/restore).
      * Value-net paths follow SB3's RolloutBuffer.compute_returns_and_advantage.
    """
    if adv_estimate_model is None:
        adv_estimate_model = rollout_model

    get_state, set_state = _auto_state_io(env, i_env=i_env)

    rollout_model.policy.set_training_mode(False)
    adv_estimate_model.policy.set_training_mode(False)

    obs = env.reset()
    last_obs = obs
    last_starts = np.ones((env.num_envs,), dtype=bool)

    buffer = RolloutBuffer(
        n_collect_steps,
        env.observation_space,
        env.action_space,
        device=rollout_model.device,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    cand_states: List[Any] = []
    cand_obs: List[np.ndarray] = []
    cand_actions: List[np.ndarray] = []
    cand_starts: List[bool] = []
    cand_buffer_idx: List[int] = []

    steps = 0
    print("Collecting candidate (s,) and (s, a)...")
    while steps < n_collect_steps:
        with torch.no_grad():
            obs_tensor = obs_as_tensor(last_obs, rollout_model.device)

            actions_t, values_t, logprobs_t = rollout_model.policy(obs_tensor)

        actions_np = actions_t.cpu().numpy()

        try:
            snap = get_state()
            cand_states.append(snap)
            cand_obs.append(np.array(last_obs[i_env]))
            cand_actions.append(np.array(actions_np[i_env]))
            cand_starts.append(bool(last_starts[i_env]))
            cand_buffer_idx.append(steps)
        except Exception as e:
            if verbose:
                print(f"[collect] snapshot failed at step {steps}: {e}")

        obs_next, rewards, dones, infos = env.step(actions_np)

        buffer.add(
            last_obs,
            actions_t,
            rewards,
            last_starts,
            values_t,
            logprobs_t,
        )

        last_obs, last_starts = obs_next, dones
        steps += 1
        if verbose and steps % 5000 == 0:
            print(f"[collect] {steps}/{n_collect_steps} collected")

    with torch.no_grad():
        all_obs = obs_as_tensor(buffer.observations, adv_estimate_model.device)

        all_obs = all_obs.squeeze(1) if all_obs.dim() >= 3 else all_obs
        vpred = adv_estimate_model.policy.predict_values(all_obs)
    buffer.values = vpred.detach().cpu().numpy()

    with torch.no_grad():
        last_values = adv_estimate_model.policy.predict_values(
            obs_as_tensor(last_obs, adv_estimate_model.device)
        )
    buffer.compute_returns_and_advantage(last_values=last_values, dones=last_starts)

    returns_vnet = buffer.returns.flatten().astype(np.float32)
    advantages_vnet = buffer.advantages.flatten().astype(np.float32)
    values_vnet = buffer.values.flatten().astype(np.float32)

    if len(cand_states) == 0:
        raise RuntimeError("No candidate states were collected; state cloning not functional.")
    sel = np.random.choice(len(cand_states), size=min(n_pairs, len(cand_states)), replace=False)
    sel.sort()

    target_obs = np.stack([cand_obs[i] for i in sel], axis=0)
    target_actions = np.stack([cand_actions[i] for i in sel], axis=0)
    target_starts = np.array([cand_starts[i] for i in sel], dtype=bool)

    step_idx = np.array([cand_buffer_idx[i] for i in sel], dtype=np.int64)

    Q_hat = np.zeros((len(sel),), dtype=np.float32)
    V_hat = np.zeros((len(sel),), dtype=np.float32)

    for k, ci in tqdm(enumerate(sel), total=len(sel), desc="Sampling from (s,) and (s, a)..."):

        q_returns = []
        for _ in range(repeats_q):
            set_state(cand_states[ci])
            ret, disc, t, done = 0.0, 1.0, 0, False

            a0 = _pack_action_for_vecenv(env, i_env, target_actions[k])
            obs1, r1, d1, _ = env.step(a0)
            ret += disc * float(r1[i_env]); disc *= gamma; done = bool(d1[i_env]); t += 1

            while not done and (horizon is None or t < horizon):
                with torch.no_grad():
                    a_t, _, _ = rollout_model.policy(obs_as_tensor(obs1, rollout_model.device))
                a_np = a_t.cpu().numpy()
                obs1, r, d, _ = env.step(a_np)
                ret += disc * float(r[i_env]); disc *= gamma; done = bool(d[i_env]); t += 1
            q_returns.append(ret)
        Q_hat[k] = float(np.mean(q_returns, dtype=np.float64))

        v_returns = []
        for _ in range(repeats_v):
            set_state(cand_states[ci])
            ret, disc, t, done = 0.0, 1.0, 0, False

            with torch.no_grad():
                a0_t, _, _ = rollout_model.policy(
                    obs_as_tensor(target_obs[k][None, ...], rollout_model.device)
                )

            a0_np = _pack_action_for_vecenv(env, i_env, a0_t.cpu().numpy().squeeze(0))
            obs2, r2, d2, _ = env.step(a0_np)
            ret += disc * float(r2[i_env]); disc *= gamma; done = bool(d2[i_env]); t += 1

            while not done and (horizon is None or t < horizon):
                with torch.no_grad():
                    a_t, _, _ = rollout_model.policy(obs_as_tensor(obs2, rollout_model.device))
                a_np = a_t.cpu().numpy()
                obs2, r, d, _ = env.step(a_np)
                ret += disc * float(r[i_env]); disc *= gamma; done = bool(d[i_env]); t += 1
            v_returns.append(ret)
        V_hat[k] = float(np.mean(v_returns, dtype=np.float64))

        if verbose and (k + 1) % max(1, len(sel) // 10) == 0:
            print(f"[eval] {k+1}/{len(sel)} targets")

    A_hat = (Q_hat - V_hat).astype(np.float32)

    return_array = returns_vnet[step_idx]
    advantage_array = advantages_vnet[step_idx]
    value_array = values_vnet[step_idx]

    rollout_buffer = {

        "indices": sel,
        "obs": target_obs,
        "actions": target_actions,
        "starts": target_starts,
        "buffer_step_idx": step_idx,

        "returns": return_array,
        "advantages": advantage_array,
        "values": value_array,

    }

    mc_estimation = {

        "Q_hat": Q_hat,
        "V_hat": V_hat,
        "A_hat": A_hat,
    }

    return rollout_buffer, mc_estimation
