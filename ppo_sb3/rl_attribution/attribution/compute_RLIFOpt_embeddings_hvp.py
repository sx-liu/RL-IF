import argparse
import pickle
import torch
import torch.nn as nn
from torch.func import functional_call, grad, jvp, vmap

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

class EvalWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self._model = model
        self.policy = model.policy
        self.device = model.device

    def forward(self, obs, act):

        return self.policy.evaluate_actions(obs, act)

def module_state_dicts(mod: nn.Module):

    params = {n: p.detach().clone() for n, p in mod.named_parameters() if "value_net" not in n and p.requires_grad}
    buffers = dict(mod.named_buffers())
    return params, buffers

def split_vector_to_param_tree(vec: torch.Tensor, params: dict):

    out, offset = {}, 0
    for k, p in params.items():
        n = p.numel()
        out[k] = vec[offset:offset+n].view_as(p)
        offset += n
    assert offset == vec.numel(), f"vec size mismatch: {offset} vs. {vec.numel()}"
    return out

def flatten_tree(tree: dict, params: dict):
    return torch.cat([tree[k].reshape(-1) for k in params.keys()])

def hvp_single_with_jvp(model, obs_i, act_i, old_logp_i, adv_i, progress, u_flat):
    device = model.device

    params, _ = module_state_dicts(model)

    def loss_fn(prm):

        values, new_log_prob, entropy = functional_call(
            model, prm, (obs_i, act_i.long())
        )

        ratio = torch.exp(new_log_prob - old_logp_i)
        clip = model._model.clip_range(progress)
        policy_loss = -torch.min(ratio * adv_i,
                                 torch.clamp(ratio, 1.0-clip, 1.0+clip) * adv_i)
        loss = policy_loss + model._model.ent_coef * (-entropy)
        return loss.sum()

    g = grad(loss_fn)

    u_flat = u_flat.to(device)
    v_tree = split_vector_to_param_tree(u_flat, params)

    for k, p in params.items():
        assert k in v_tree and p.shape == v_tree[k].shape, f"shape mismatch at {k}"
        assert p.dtype == v_tree[k].dtype, f"dtype mismatch at {k}"

    g_val, hvp_tree = jvp(g, (params,), (v_tree,))

    hvp_flat = flatten_tree(hvp_tree, params)
    return hvp_flat

def compute_hvp_with_jvp(model, batch, u_flat, progress):
    device = model.device
    u_flat = u_flat.to(device)

    adv = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

    batch_size = batch.observations.size(0)

    hvps = torch.zeros(u_flat.size(0), device=device)

    for i in range(batch_size):
        obs_i  = batch.observations[i].unsqueeze(0).to(device)
        act_i  = batch.actions[i].unsqueeze(0).to(device).flatten()
        oldp_i = batch.old_log_prob[i].unsqueeze(0).to(device)
        adv_i  = adv[i].unsqueeze(0).to(device)
        hvp_i  = hvp_single_with_jvp(
            model, obs_i, act_i, oldp_i, adv_i, progress, u_flat
        ).detach()

        hvps = hvps + hvp_i

    return hvps / batch_size

def hvp_with_jvp_vmap(model_wrapped: EvalWrapper, batch, U: torch.Tensor, progress: float):
    """
    U: [M, D] matrix (each row is a vector u). Returns H @ Uᵀ as [M, D].
    """

    def hvp_one(u_flat):
        return compute_hvp_with_jvp(model_wrapped, batch, u_flat, progress)

    hvp_vmapped = vmap(hvp_one, in_dims=0, out_dims=0)
    return hvp_vmapped(U)

def chunked_hvp_update_jvp(*, model_t, batch_t, V, progress, eta, chunk=512, device="cpu"):
    """
    V: [M, D] tensor of vectors (on device).
    Returns: V - eta * (H @ V), computed in chunks using JVP-based HVPs.
    """
    M = V.shape[0]

    out = V.detach().cpu()
    model_wrapper = EvalWrapper(model_t).to(device)

    for start in range(0, M, chunk):
        end = min(start + chunk, M)
        H_chunk = hvp_with_jvp_vmap(
            model_wrapper,
            batch_t,
            out[start:end],
            progress
        ).detach().cpu()

        out[start:end] = out[start:end] - eta * H_chunk

    return out

def compute_rlif_opt_embeddings_hvp(
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

    for t in tqdm(range(begin_step, end_step)):
        model_t = PPO.load(model_path / f'policy_grad_{t}.zip', device="cpu")
        batch_t = pickle.load(open(model_path / f'batch_{t+1}.pkl', 'rb'))

        progress = 1 - (int(t / n_batches) + 1)/(total_updates / n_batches)

        eta_t = float(model_t.lr_schedule(progress))
        batch_size = batch_t.observations.shape[0]

        grad_train_list = compute_gradients_for_train_batch(model_t, batch_t, progress)

        for j, grad_train in enumerate(grad_train_list):
            raw_grads[(t, j)] = grad_train.detach().cpu().numpy()
            embeddings[(t, j)] = (eta_t / batch_size) * grad_train.detach().cpu()

    for t in tqdm(range(begin_step, end_step)):

        model_t = PPO.load(model_path / f'policy_grad_{t}.zip', device=device)
        batch_t = pickle.load(open(model_path / f'batch_{t+1}.pkl', 'rb'))

        eta_t = float(model_t.lr_schedule(progress))
        batch_size = batch_t.observations.shape[0]
        progress = 1 - (int(t / n_batches) + 1) / (total_updates / n_batches)

        if t - begin_step > 0:

            Vs = []
            for s in range(begin_step, t):
                V_s = torch.stack([embeddings[(s, j)] for j in range(batch_size)], dim=0)
                Vs.append(V_s)
            V_all = torch.cat(Vs, dim=0).to(device)

            V_all_updated = chunked_hvp_update_jvp(

                model_t=model_t,
                batch_t=batch_t,
                V=V_all,
                progress=progress,
                eta=eta_t,
                chunk=40960,
                device=device,
            ).detach().cpu()

            with torch.no_grad():
                offset = 0
                for s in range(begin_step, t):
                    V_s_updated = V_all_updated[offset:offset + batch_size]
                    offset += batch_size
                    for j in range(batch_size):
                        embeddings[(s, j)] = V_s_updated[j]

    return {
        "target_grad": u_flat,
        "raw_grads": raw_grads,
        'embeddings': {k: v.detach().cpu().numpy() for k, v in embeddings.items()},
    }
