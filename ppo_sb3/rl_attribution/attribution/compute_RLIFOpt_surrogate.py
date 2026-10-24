import argparse
import pickle
import torch
from tqdm import tqdm
import os
from stable_baselines3 import PPO
import numpy as np
import torch.nn.functional as F

from rl_attribution.utils.tensor import (
    flatten_ob_act,
    flatten_ob_act_prob_adv,
)

from rl_attribution.utils.gradients import (
    compute_surrogate_validation_loss_gradient,
    compute_gradients_for_train_batch,
    inner_product_gradients,
    compute_hvp,
)

def compute_rlif_opt_scores(
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

    if not isinstance(target_rollout, dict):
        target_rollout = {
            "observations": target_rollout.observations,
            "actions": target_rollout.actions,
            "advantages": target_rollout.advantages,
        }
    u_flat = compute_surrogate_validation_loss_gradient(target_model, target_rollout)

    for t in tqdm(range(end_step - 1, begin_step - 1, -1)):
        model_t = PPO.load(model_path / f'policy_grad_{t}.zip', device=device)
        batch_t = pickle.load(open(model_path / f'batch_{t+1}.pkl', 'rb'))

        if isinstance(batch_t, dict):
            batch_size = batch_t["observations"].shape[0]
        else:
            batch_size = batch_t.observations.shape[0]
        progress = 1 - (int(t / n_batches) + 1) / (total_updates / n_batches)
        eta_t = model_t.lr_schedule(progress)

        grad_train_list = compute_gradients_for_train_batch(model_t, batch_t, progress)

        for j, grad_train in enumerate(grad_train_list):
            curr_influence = (eta_t / batch_size) * inner_product_gradients(grad_train, u_flat).item()

            influence_scores[(t, j)] = curr_influence

        if t > begin_step:
            hvp_flat = compute_hvp(model_t, batch_t, u_flat, progress)
            u_flat = u_flat - eta_t * hvp_flat

    return {
        'influence_scores': influence_scores,
    }
