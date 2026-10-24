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
    compute_hessian,
)

def compute_rlif_opt_embeddings(
    target_model,
    target_rollout,
    n_batches,
    model_path,
    begin_step,
    end_step,
    total_updates,
    device,
):
    raw_grads = {}
    embeddings = {}
    hessian_ev = []

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
    i_hessian = torch.eye(u_flat.shape[0], device=device)

    for t in tqdm(range(end_step - 1, begin_step - 1, -1)):
        model_t = PPO.load(model_path / f'policy_grad_{t}.zip', device=device)
        batch_t = pickle.load(open(model_path / f'batch_{t+1}.pkl', 'rb'))

        batch_size = batch_t.observations.shape[0]
        progress = 1 - (int(t / n_batches) + 1) / (total_updates / n_batches)
        eta_t = model_t.lr_schedule(progress)

        grad_train_list = compute_gradients_for_train_batch(model_t, batch_t, progress)

        for j, grad_train in enumerate(grad_train_list):
            with torch.no_grad():
                curr_embedding = (eta_t / batch_size) * i_hessian @ grad_train
                curr_embedding = curr_embedding.detach().cpu().numpy()

            raw_grads[(t, j)] = grad_train.detach().cpu().numpy()
            embeddings[(t, j)] = curr_embedding

        if t > begin_step:
            hessian = compute_hessian(model_t, batch_t, progress)
            with torch.no_grad():
                lmbdas = torch.linalg.eigvalsh(hessian)
                hessian_ev.append(torch.max(torch.abs(lmbdas)).item())
            i_hessian -= eta_t * hessian

    return {
        "target_grad": u_flat,
        "raw_grads": raw_grads,
        'embeddings': embeddings,
        "hessian_ev": hessian_ev,
    }
