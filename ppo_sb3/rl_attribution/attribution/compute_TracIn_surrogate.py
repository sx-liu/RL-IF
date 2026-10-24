import argparse
from stable_baselines3 import PPO
import pickle
import numpy as np

import torch
from tqdm import tqdm
import torch.nn.functional as F
import os

from rl_attribution.utils.tensor import (
    flatten_ob_act,
    flatten_ob_act_prob_adv,
)

from rl_attribution.utils.gradients import (
    compute_surrogate_validation_loss_gradient,
    compute_gradients_for_train_batch,
    inner_product_gradients,
)

def compute_tracin_scores(
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

    grad_ref = compute_surrogate_validation_loss_gradient(target_model, target_rollout)

    for i in tqdm(range(begin_step, end_step)):

        batch_i = pickle.load(open(model_path / f'batch_{i+1}.pkl', 'rb'))

        progress = 1 - (int(i / n_batches) + 1) / (total_updates / n_batches)

        grad_train_list = compute_gradients_for_train_batch(target_model, batch_i, progress)

        for j, grad_train in enumerate(grad_train_list):
            curr_influence = inner_product_gradients(grad_train, grad_ref).item()
            influence_scores[(i, j)] = curr_influence

    return {
        'influence_scores': influence_scores,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='easy_to_hard_40k_ckpt_batch')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_updates', type=int, default=6400)
    parser.add_argument('--total_updates', type=int, default=6400)
    parser.add_argument('--begin_step', type=int, default=0)
    parser.add_argument('--rollout_file', type=str, required=True)
    parser.add_argument('--target_model_path', type=str, default=None)
    args = parser.parse_args()

    rollouts = pickle.load(open(args.rollout_file, "rb"))
    print('rollout sample size =', rollouts.observations.squeeze(1).shape[0])

    if args.target_model_path is None:
        args.target_model_path = f'./models/{args.model_path}/policy_grad_{args.num_updates}.zip'

    influence_scores = {}

    os.makedirs(args.output_dir, exist_ok=True)
    pickle.dump(influence_scores, open(os.path.join(args.output_dir, 'tracin_influence_scores.pkl'), 'wb'))
