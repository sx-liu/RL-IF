import argparse
from stable_baselines3 import PPO
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
import os

import json

def compute_surrogate_validation_loss(model, sample, batch_size=2048):
    """
    Compute the surrogate validation loss using a fixed action target.

    The sample is a dictionary that contains:
      - 'observations': observations (e.g. image data)
      - 'actions': target actions (as integers) used to compute the advantage
      - 'advantages': advantage estimates computed by the oracle model

    This function computes the loss:
        L = -mean( advantage * log(pi(current_model)(target_action | observation)) )
    and backpropagates to compute gradients on the current model's policy.

    Returns:
      grad_dict: a dictionary mapping parameter names to their gradient tensors.
      loss_val: the computed scalar loss value.
    """
    device = model.device

    obs = torch.as_tensor(sample["observations"], dtype=torch.float32, device=device).squeeze(1)
    target_actions = torch.as_tensor(sample["actions"], device=device)
    advantages = torch.as_tensor(sample["advantages"], dtype=torch.float32, device=device).squeeze()

    if obs.ndim == len(model.policy.observation_space.shape):
        obs = obs.unsqueeze(0)

    if target_actions.ndim == 0:
        target_actions = target_actions.unsqueeze(0)

    if hasattr(model.action_space, "n"):
        target_actions = target_actions.long()
        if target_actions.ndim > 1:
            target_actions = target_actions.squeeze(-1)
            if target_actions.ndim > 1:
                target_actions = target_actions.flatten()
    else:
        target_actions = target_actions.float()

    model.policy.optimizer.zero_grad()

    dist = model.policy.get_distribution(obs)

    selected_log_probs = dist.log_prob(target_actions)

    if selected_log_probs.ndim > 1:
        selected_log_probs = selected_log_probs.sum(dim=-1)

    loss = -torch.mean(advantages * selected_log_probs)

    print(loss.item())
    return loss.item(), advantages.detach().cpu().numpy().tolist(), selected_log_probs.detach().cpu().numpy().tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_path', type=str, default='easy_to_hard_40k_ckpt_batch')
    parser.add_argument('--rollout_file', type=str, required=True)

    parser.add_argument('--save_dir', type=str, default="./eval_results")
    parser.add_argument('--model_name', type=str, default=None)

    args = parser.parse_args()

    rollouts = pickle.load(open(args.rollout_file, "rb"))

    sample = {
        "observations": rollouts.observations.squeeze(1),
        "actions": rollouts.actions,
        "advantages": rollouts.advantages,
    }

    print('rollout sample size =', sample['observations'].shape[0])

    model = PPO.load(args.policy_path)
    print('loaded model from', args.policy_path)
    loss, advantages, log_probs = compute_surrogate_validation_loss(model, sample)
    print('Surrogate validation loss:', loss)

    model_name = args.model_name if args.model_name is not None else args.policy_path.split("/")[-2]
    policy_name = args.policy_path.split("/")[-1].split('.')[-2]
    save_dir = os.path.join(args.save_dir, model_name)
    save_file = os.path.join(save_dir, f"{policy_name}_surrogate.json")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving to {save_file}.")
    with open(save_file, "w") as f:
        json.dump({
            "surrogate_val_loss": loss,
            "advantages": advantages,
            "log_probs": log_probs,
        }, f)
