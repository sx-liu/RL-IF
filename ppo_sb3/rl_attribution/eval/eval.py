import argparse
import gymnasium as gym
import minigrid
import numpy as np
np.float_ = np.float64
import torch as th
from stable_baselines3 import PPO
from tqdm import tqdm

import os
import json

from rl_attribution.utils.envs import make_vec_env

from stable_baselines3.common.evaluation import evaluate_policy as sb3_evaluate

def simulate_trajectory(model, env):
    """
    Simulate one full episode in LunarLander-v2.
    """
    obs, info = env.reset()
    total_reward = 0.0
    done = False

    predict = model.predict
    step = env.step
    with th.inference_mode():
        while not done:
            action, _ = predict(obs, deterministic=False)

            if hasattr(env.action_space, "n"):
                action = int(action)
            else:
                action = np.array(action, dtype=np.float32)

            obs, reward, terminated, truncated, info = step(action)
            total_reward += reward
            done = terminated or truncated
    return float(total_reward)

def evaluate_policy(model, env_name, num_episodes, n_envs=8):
    """
    Evaluate the policy on the specified environment over a given number of episodes.
    """

    vec_env = make_vec_env(env_name, n_envs)

    returns = []
    pbar = tqdm(total=num_episodes, desc="Evaluating")

    obs = vec_env.reset()

    with th.inference_mode():
        while len(returns) < num_episodes:
            actions, _ = model.predict(obs, deterministic=False)

            if hasattr(vec_env.action_space, "n"):

                actions = np.asarray(actions).reshape(-1).astype(np.int64)
            else:
                actions = np.asarray(actions, dtype=np.float32)

            obs, rewards, dones, infos = vec_env.step(actions)

            for info in infos:
                if "episode" in info:
                    ep_r = float(info["episode"]["r"])
                    returns.append(ep_r)
                    pbar.update(1)
                    if len(returns) >= num_episodes:
                        break

    pbar.close()
    vec_env.close()

    return {
        "avg_return": np.mean(returns),
        "std_returns": np.std(returns),
        "returns": returns,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_path', type=str, default="models_lunarlander/ppo_lunarlander_run.zip",
                        help="Path to the trained LunarLander policy.")
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help="Number of evaluation episodes.")
    parser.add_argument('--env_name', type=str, default="LunarLander-v2",
                        help="Environment name (default: LunarLander-v2).")

    parser.add_argument('--save_dir', type=str, default="./eval_results")
    parser.add_argument('--model_name', type=str, default=None)

    args = parser.parse_args()

    th.set_num_threads(1)
    policy = PPO.load(args.policy_path, device="cpu")
    policy.policy.eval()
    print("Loaded policy from:", args.policy_path)

    avg_return, std_returns, returns = evaluate_policy(policy, args.env_name, args.num_episodes).values()

    print("Average return over {} episodes: {:.2f}".format(args.num_episodes, avg_return))
    print("Return standard deviation: {:.2f}".format(std_returns))
    print("Returns for each episode:", returns)

    model_name = args.model_name if args.model_name is not None else args.policy_path.split("/")[-2]
    policy_name = args.policy_path.split("/")[-1].split('.')[-2]
    save_dir = os.path.join(args.save_dir, model_name)
    save_file = os.path.join(save_dir, f"{policy_name}.json")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving to {save_file}.")
    with open(save_file, "w") as f:
        json.dump({
            "avg_return": avg_return,
            "std_returns": std_returns,
            "returns": returns,
        }, f)
