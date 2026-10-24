import argparse
import pickle
import torch
from tqdm import tqdm
import os
from stable_baselines3 import PPO
import numpy as np
import torch.nn.functional as F

from rl_attribution.utils.gradients import (
    compute_surrogate_validation_loss_gradient,
    compute_gradients_for_train_batch,
    compute_ppo_policy_gradient,
    inner_product_gradients,
    compute_hvp,
)

def compute_log_prob_gradient(model, sample, progress, normalize_advantages=True):
    """Compute the gradient of log pi(a|s) w.r.t. model parameters for a single sample."""
    policy = model.policy
    device = model.device
    act_space = model.action_space

    if isinstance(sample, dict):
        obs = sample['observations']
        act = sample['actions']
        old_log_prob = sample['old_log_prob']
        advantages = sample['advantages']
    else:
        obs = sample.observations
        act = sample.actions
        old_log_prob = sample.old_log_prob
        advantages = sample.advantages

    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    old_log_prob = torch.as_tensor(old_log_prob, dtype=torch.float32, device=device)

    if obs.ndim == len(getattr(policy.observation_space, "shape", ())):
        obs = obs.unsqueeze(0)

    if hasattr(act_space, "n"):

        act = torch.as_tensor(act, dtype=torch.long, device=device).flatten()
    else:

        act = torch.as_tensor(act, dtype=torch.float32, device=device)
        if act.ndim == 1:
            act = act.unsqueeze(0)

        if act.shape[0] != obs.shape[0]:
            raise ValueError(f"actions batch {act.shape[0]} != obs batch {obs.shape[0]}")

    values, new_log_prob, entropy = policy.evaluate_actions(obs, act)

    new_log_prob = new_log_prob.view(-1)

    params = [p for n, p in policy.named_parameters() if "value_net" not in n and p.requires_grad]
    log_prob_grads = torch.autograd.grad(new_log_prob, params, create_graph=True, allow_unused=False)
    log_prob_grad_flat = torch.cat([g.reshape(-1) for g in log_prob_grads])

    return log_prob_grad_flat

def compute_sampling_influence_mvp(
    model_t,
    model_s,
    batch_t,
    progress,
    value_embedding,
):
    if isinstance(batch_t, dict):
        batch_size = batch_t['observations'].shape[0]
    else:
        batch_size = batch_t.observations.shape[0]
    grad_dim = sum(p.numel() for n, p in model_t.policy.named_parameters() if "value_net" not in n)
    mvp = torch.zeros(grad_dim, device=model_t.device)

    if isinstance(batch_t, dict):
        adjusted_advantages = (batch_t['advantages'] - batch_t['advantages'].mean()) / (batch_t['advantages'].std() + 1e-8)
    else:
        adjusted_advantages = (batch_t.advantages - batch_t.advantages.mean()) / (batch_t.advantages.std() + 1e-8)

    batch_grads = []
    for i in range(batch_size):
        if isinstance(batch_t, dict):
            sample = {
                'observations': batch_t['observations'][i].unsqueeze(0).to(model_t.device),
                'actions': batch_t['actions'][i].unsqueeze(0).to(model_t.device),
                'old_log_prob': batch_t['old_log_prob'][i].unsqueeze(0).to(model_t.device),
                'advantages': adjusted_advantages[i].unsqueeze(0).to(model_t.device),
            }
        else:
            sample = {
                'observations': batch_t.observations[i].unsqueeze(0).to(model_t.device),
                'actions': batch_t.actions[i].unsqueeze(0).to(model_t.device),
                'old_log_prob': batch_t.old_log_prob[i].unsqueeze(0).to(model_t.device),
                'advantages': adjusted_advantages[i].unsqueeze(0).to(model_t.device),
            }
        ppo_grad = compute_ppo_policy_gradient(model_t, sample, progress, normalize_advantages=False)
        batch_grads.append(ppo_grad)

    batch_grads = torch.stack(batch_grads)
    mean_grad = torch.mean(batch_grads, dim=0)

    for i in range(batch_size):
        if isinstance(batch_t, dict):
            sample = {
                'observations': batch_t['observations'][i].unsqueeze(0).to(model_t.device),
                'actions': batch_t['actions'][i].unsqueeze(0).to(model_t.device),
                'old_log_prob': batch_t['old_log_prob'][i].unsqueeze(0).to(model_t.device),
                'advantages': adjusted_advantages[i].unsqueeze(0).to(model_t.device),
            }
        else:
            sample = {
                'observations': batch_t.observations[i].unsqueeze(0).to(model_t.device),
                'actions': batch_t.actions[i].unsqueeze(0).to(model_t.device),
                'old_log_prob': batch_t.old_log_prob[i].unsqueeze(0).to(model_t.device),
                'advantages': adjusted_advantages[i].unsqueeze(0).to(model_t.device),
            }
        log_prob_grad = compute_log_prob_gradient(model_s, sample, progress, normalize_advantages=False)

        mvp += torch.dot(value_embedding, log_prob_grad) * batch_grads[i]

    return mvp

def compute_rlif_scores(
    target_model,
    target_rollout,
    n_batches,
    model_path,
    begin_step,
    end_step,
    total_updates,
    device="cpu",
):
    if isinstance(target_rollout, dict):
        sample = target_rollout
        sample["observations"] = sample["observations"].squeeze(1)
    else:
        sample = {
            "observations": target_rollout.observations.squeeze(1),
            "actions": target_rollout.actions,
            "advantages": target_rollout.advantages,
        }
    u_flat = compute_surrogate_validation_loss_gradient(target_model, sample)

    u_m = u_flat.clone()
    u_s = torch.zeros_like(u_m)
    influence_scores = {}

    sampling_model_t = 0
    sampling_steps = sampling_steps_in_range(begin_step, end_step, n_batches)
    sampling_steps = [step for step in sampling_steps if step > begin_step and step < end_step]
    sampling_steps = sorted(sampling_steps, reverse=True)
    if len(sampling_steps):
        model_s = PPO.load(f'{model_path}/policy_grad_{sampling_steps[sampling_model_t]}.zip', device=device)
        sampling_model_t += 1
    print('sampling steps:', sampling_steps)

    for t in tqdm(range(end_step - 1, begin_step - 1, -1), desc="Computing Sampling Influence"):

        model_t = PPO.load(f'{model_path}/policy_grad_{t}.zip', device="cpu")
        batch_t = pickle.load(open(f'{model_path}/batch_{t+1}.pkl', 'rb'))

        progress = 1 - (int(t / n_batches)+1)/(total_updates / n_batches)
        eta_t = model_t.lr_schedule(progress)
        if isinstance(batch_t, dict):
            batch_size = batch_t['observations'].shape[0]
        else:
            batch_size = batch_t.observations.shape[0]

        grad_train_list = compute_gradients_for_train_batch(model_t, batch_t, progress)

        for j, grad_train in enumerate(grad_train_list):
            curr_influence = (eta_t / batch_size) * inner_product_gradients(grad_train, u_m).item()

            influence_scores[(t, j)] = curr_influence

        if len(sampling_steps) and t >= sampling_steps[-1]:
            sampling_influence_mvp = compute_sampling_influence_mvp(
                model_t,
                model_s,
                batch_t,
                progress,
                u_m,
            ).detach()

            u_s -= (eta_t / batch_size) * sampling_influence_mvp

        hvp_flat = compute_hvp(model_t, batch_t, u_m, progress).detach()

        u_m = u_m - eta_t * hvp_flat

        if t in sampling_steps:

            u_m += u_s
            u_s = torch.zeros_like(u_m)

            if sampling_model_t < len(sampling_steps):
                model_s = PPO.load(f'{model_path}/policy_grad_{sampling_steps[sampling_model_t]}.zip', device="cpu")
                sampling_model_t += 1

    return {
        "influence_scores": influence_scores,
    }

def sampling_steps_in_range(
        begin_step,
        end_step,
        steps_per_rollout,
    ):
    """
    Generate a list of sampling steps based on the model path and other parameters.
    """
    sampling_steps = []
    for t in range(begin_step, end_step + 1):
        if t % steps_per_rollout == 0:
            sampling_steps.append(t)
    return sampling_steps
