import pickle
import weakref
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from stable_baselines3 import PPO

from rl_attribution.utils.gradients import (
    compute_surrogate_validation_loss_gradient,
)

def _batch_get(batch, key):
    if isinstance(batch, dict):
        return batch[key]
    return getattr(batch, key)

def _slice_batch(batch, start, end):
    if isinstance(batch, dict):
        return {
            "observations": batch["observations"][start:end],
            "actions": batch["actions"][start:end],
            "old_log_prob": batch["old_log_prob"][start:end],
            "advantages": batch["advantages"][start:end],
        }
    return {
        "observations": batch.observations[start:end],
        "actions": batch.actions[start:end],
        "old_log_prob": batch.old_log_prob[start:end],
        "advantages": batch.advantages[start:end],
    }

def _maybe_squeeze_obs(obs):
    if hasattr(obs, "ndim") and obs.ndim > 1 and obs.shape[1] == 1:
        return obs.squeeze(1)
    return obs

def _prepare_obs_actions(model, batch):
    policy = model.policy
    device = model.device
    act_space = model.action_space

    obs = torch.as_tensor(_batch_get(batch, "observations"), dtype=torch.float32, device=device)
    if obs.ndim == len(getattr(policy.observation_space, "shape", ())):
        obs = obs.unsqueeze(0)

    raw_actions = _batch_get(batch, "actions")
    if hasattr(act_space, "n"):
        act = torch.as_tensor(raw_actions, dtype=torch.long, device=device).flatten()
    else:
        act = torch.as_tensor(raw_actions, dtype=torch.float32, device=device)
        if act.ndim == 1:
            act = act.unsqueeze(0)
        if act.shape[0] != obs.shape[0]:
            raise ValueError(f"actions batch {act.shape[0]} != obs batch {obs.shape[0]}")

    return obs, act

def _compute_ppo_per_sample_losses(model, batch, progress):
    policy = model.policy
    device = model.device

    old_log_prob = torch.as_tensor(
        _batch_get(batch, "old_log_prob"), dtype=torch.float32, device=device
    ).view(-1)
    advantages = torch.as_tensor(
        _batch_get(batch, "advantages"), dtype=torch.float32, device=device
    ).view(-1)

    obs, act = _prepare_obs_actions(model, batch)
    _, new_log_prob, entropy = policy.evaluate_actions(obs, act)

    new_log_prob = new_log_prob.view(-1)
    entropy = entropy.view(-1)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    if old_log_prob.numel() == 1:
        old_log_prob = old_log_prob.expand_as(new_log_prob)

    if not (advantages.shape == new_log_prob.shape == old_log_prob.shape):
        raise ValueError(
            f"Shape mismatch: adv {advantages.shape}, new_lp {new_log_prob.shape}, old_lp {old_log_prob.shape}"
        )

    ratio = torch.exp(new_log_prob - old_log_prob)
    clip_range = torch.as_tensor(model.clip_range(progress), dtype=torch.float32, device=device)

    policy_loss_1 = ratio * advantages
    policy_loss_2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    policy_loss = -torch.min(policy_loss_1, policy_loss_2)

    entropy_loss = -entropy
    log_ratio = new_log_prob - old_log_prob
    approx_kl_loss = (torch.exp(log_ratio) - 1) - log_ratio

    losses = policy_loss + model.ent_coef * entropy_loss + model.kl_coef * approx_kl_loss
    return losses

class GhostPolicy:
    def __init__(self, model):
        self.model = model
        self.conv_params = {}
        self._xs = {}
        self._gs = {}
        self._hooks_enabled = True
        self._hook_handles = []
        self._name_to_slice = {}
        self._register_hooks()
        self._build_param_slices()

    def _register_hooks(self):
        weak_self = weakref.ref(self)
        for name, module in self.model.policy.named_modules():
            if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                continue
            if "value_net" in name:
                continue

            if isinstance(module, torch.nn.Conv2d):
                self.conv_params[name] = {
                    "kernel_size": module.kernel_size,
                    "stride": module.stride,
                    "padding": module.padding,
                }

            self._xs[name] = None
            self._gs[name] = None

            def fwd(_module, inp, _out, nm=name):
                inst = weak_self()
                if inst is None or not inst._hooks_enabled:
                    return
                x, = inp
                inst._xs[nm] = x.detach()

            def bwd(_module, _grad_in, grad_out, nm=name):
                inst = weak_self()
                if inst is None or not inst._hooks_enabled:
                    return
                g, = grad_out
                inst._gs[nm] = g.detach()

            self._hook_handles.append(module.register_forward_hook(fwd))
            self._hook_handles.append(module.register_full_backward_hook(bwd))

    def close(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        for k in self._xs:
            self._xs[k] = None
            self._gs[k] = None
        self._xs = {}
        self._gs = {}
        self._name_to_slice = {}
        self.conv_params = {}
        self.model = None

    def _build_param_slices(self):
        offset = 0
        for name, p in self.model.policy.named_parameters():
            if not p.requires_grad or "value_net" in name:
                continue
            size = p.numel()
            self._name_to_slice[name] = (offset, offset + size, p.shape, p.device, p.dtype)
            offset += size
        self._param_dim = offset

    def _snapshot_hooks(self):
        xs = {k: v for k, v in self._xs.items()}
        gs = {k: v for k, v in self._gs.items()}
        for k in self._xs:
            self._xs[k] = None
            self._gs[k] = None
        return xs, gs

    def _split_u_by_param(self, u_flat):
        if u_flat.numel() != self._param_dim:
            raise ValueError(f"u_flat dim {u_flat.numel()} != actor param dim {self._param_dim}")

        u_by_name = {}
        for name, (start, end, shape, device, dtype) in self._name_to_slice.items():
            u_by_name[name] = u_flat[start:end].to(device=device, dtype=dtype).view(shape)
        return u_by_name

    def _actor_params(self):
        return [
            p for n, p in self.model.policy.named_parameters()
            if p.requires_grad and "value_net" not in n
        ]

    def _flatten_actor_grads(self):
        grads = []
        for p in self._actor_params():
            if p.grad is None:
                grads.append(torch.zeros_like(p).reshape(-1))
            else:
                grads.append(p.grad.reshape(-1))
        return torch.cat(grads)

    def compute_train_scores_for_batch(self, batch, progress, u_flat, chunk_size=None):
        batch_size = _batch_get(batch, "observations").shape[0]
        if chunk_size is None or chunk_size <= 0:
            chunk_size = batch_size

        u_by_name = self._split_u_by_param(u_flat)
        scores_all = []

        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            batch_chunk = _slice_batch(batch, start, end)

            self.model.policy.optimizer.zero_grad(set_to_none=True)
            losses = _compute_ppo_per_sample_losses(self.model, batch_chunk, progress)
            losses.sum().backward()

            train_xs, train_gs = self._snapshot_hooks()
            scores_chunk = compute_sample_ip_with_u_vec(train_xs, train_gs, self.conv_params, u_by_name)
            scores_all.append(scores_chunk)

        return torch.cat(scores_all, dim=0)

    def compute_hvp_batch(self, batch, progress, u_flat, chunk_size=None):
        self._hooks_enabled = False
        params = self._actor_params()
        if not params:
            raise RuntimeError("No actor parameters found (excluding value_net).")

        batch_size = _batch_get(batch, "observations").shape[0]
        if chunk_size is None or chunk_size <= 0:
            chunk_size = batch_size

        hvp_accum = None
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            batch_chunk = _slice_batch(batch, start, end)

            losses = _compute_ppo_per_sample_losses(self.model, batch_chunk, progress)
            loss = losses.mean()

            grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=False)
            grad_flat = torch.cat([g.reshape(-1) for g in grads])

            if grad_flat.numel() != u_flat.numel():
                raise ValueError(f"u_flat dim {u_flat.numel()} != grad dim {grad_flat.numel()}")

            dot_product = torch.dot(grad_flat, u_flat.to(device=grad_flat.device, dtype=grad_flat.dtype))
            hvp = torch.autograd.grad(dot_product, params, allow_unused=False)
            hvp_flat_chunk = torch.cat([h.reshape(-1) for h in hvp])

            weight = (end - start) / batch_size
            if hvp_accum is None:
                hvp_accum = hvp_flat_chunk * weight
            else:
                hvp_accum = hvp_accum + hvp_flat_chunk * weight

        self._hooks_enabled = True
        return hvp_accum

    def compute_log_prob_projection_scores(self, batch, u_flat):
        self.model.policy.optimizer.zero_grad(set_to_none=True)
        obs, act = _prepare_obs_actions(self.model, batch)
        _, new_log_prob, _ = self.model.policy.evaluate_actions(obs, act)
        new_log_prob = new_log_prob.view(-1)

        new_log_prob.sum().backward()
        train_xs, train_gs = self._snapshot_hooks()
        u_by_name = self._split_u_by_param(u_flat)
        return compute_sample_ip_with_u_vec(train_xs, train_gs, self.conv_params, u_by_name)

    def compute_weighted_policy_grad(self, batch, progress, weights):
        self.model.policy.optimizer.zero_grad(set_to_none=True)
        losses = _compute_ppo_per_sample_losses(self.model, batch, progress)
        weights = torch.as_tensor(weights, dtype=losses.dtype, device=losses.device).view(-1)
        if weights.shape[0] != losses.shape[0]:
            raise ValueError(f"weights shape {weights.shape} incompatible with losses {losses.shape}")
        (losses * weights).sum().backward()
        return self._flatten_actor_grads()

def compute_sample_ip_with_u_vec(train_xs, train_gs, conv_params, u_by_name):
    layers = list(train_xs.keys())
    if not layers:
        raise RuntimeError("No hooked layers found. Did the policy architecture change?")

    first_x = next(iter(train_xs.values()))
    if first_x is None:
        raise RuntimeError("Hook activations are empty. Ensure a backward pass happened before scoring.")

    n_train = first_x.shape[0]
    sample_ip = torch.zeros(n_train, dtype=first_x.dtype, device=first_x.device)

    for layer in layers:
        x_t = train_xs[layer]
        g_t = train_gs[layer]

        if x_t is None or g_t is None:
            raise RuntimeError(
                f"Missing hook tensors at layer '{layer}' (got X_t={x_t is not None}, G_t={g_t is not None})."
            )

        w_name = f"{layer}.weight" if layer else "weight"
        b_name = f"{layer}.bias" if layer else "bias"

        if w_name not in u_by_name:
            continue

        u_w = u_by_name[w_name]
        u_b = u_by_name.get(b_name, None)

        if x_t.dim() == 2:
            q = g_t.unsqueeze(2) * x_t.unsqueeze(1)
            sample_ip += (q * u_w.unsqueeze(0)).sum(dim=(1, 2))
            if u_b is not None:
                sample_ip += (g_t * u_b.unsqueeze(0)).sum(dim=1)
        else:
            p = conv_params[layer]
            patches_t = F.unfold(
                x_t,
                kernel_size=p["kernel_size"],
                stride=p["stride"],
                padding=p["padding"],
            )
            c_out = g_t.shape[1]
            gt_flat = g_t.view(g_t.shape[0], c_out, -1)
            wg_t = torch.matmul(gt_flat, patches_t.transpose(1, 2))
            sample_ip += (wg_t * u_w.view(c_out, -1).unsqueeze(0)).sum(dim=(1, 2))
            if u_b is not None:
                tr_bias = g_t.sum(dim=(2, 3))
                sample_ip += (tr_bias * u_b.unsqueeze(0)).sum(dim=1)

    return sample_ip

def compute_sampling_influence_mvp_ghost(
    ghost_t,
    ghost_s,
    batch_t,
    progress,
    value_embedding,
    chunk_size=None,
):
    batch_size = _batch_get(batch_t, "observations").shape[0]
    if chunk_size is None or chunk_size <= 0:
        chunk_size = batch_size

    mvp = None
    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        batch_chunk = _slice_batch(batch_t, start, end)

        alpha = ghost_s.compute_log_prob_projection_scores(batch_chunk, value_embedding).detach()
        chunk_mvp = ghost_t.compute_weighted_policy_grad(batch_chunk, progress, alpha)

        if mvp is None:
            mvp = chunk_mvp
        else:
            mvp = mvp + chunk_mvp

    return mvp

def sampling_steps_in_range(begin_step, end_step, steps_per_rollout):
    sampling_steps = []
    for t in range(begin_step, end_step + 1):
        if t % steps_per_rollout == 0:
            sampling_steps.append(t)
    return sampling_steps

def compute_rlif_scores(
    target_model,
    target_rollout,
    n_batches,
    model_path,
    begin_step,
    end_step,
    total_updates,
    chunk_size=256,
    device="cpu",
):
    model_path = Path(model_path)

    if isinstance(target_rollout, dict):
        sample = {
            "observations": _maybe_squeeze_obs(target_rollout["observations"]),
            "actions": target_rollout["actions"],
            "advantages": target_rollout["advantages"],
        }
    else:
        sample = {
            "observations": _maybe_squeeze_obs(target_rollout.observations),
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

    model_s = None
    ghost_s = None
    if len(sampling_steps):
        model_s = PPO.load(model_path / f"policy_grad_{sampling_steps[sampling_model_t]}.zip", device=device)
        ghost_s = GhostPolicy(model_s)
        sampling_model_t += 1

    print("sampling steps:", sampling_steps)

    for t in tqdm(range(end_step - 1, begin_step - 1, -1), desc="Computing Sampling Influence"):
        model_t = PPO.load(model_path / f"policy_grad_{t}.zip", device=device)
        ghost_t = GhostPolicy(model_t)

        with open(model_path / f"batch_{t + 1}.pkl", "rb") as f:
            batch_t = pickle.load(f)

        progress = 1 - (int(t / n_batches) + 1) / (total_updates / n_batches)
        eta_t = model_t.lr_schedule(progress)
        batch_size = _batch_get(batch_t, "observations").shape[0]

        scores_t = ghost_t.compute_train_scores_for_batch(batch_t, progress, u_m, chunk_size=chunk_size)
        if scores_t.shape[0] != batch_size:
            raise RuntimeError(f"scores_t shape {scores_t.shape} != batch_size {batch_size}")

        scale = eta_t / batch_size
        for j, score in enumerate(scores_t):
            influence_scores[(t, j)] = (scale * score).item()

        if ghost_s is not None and t >= sampling_steps[-1]:
            sampling_influence_mvp = compute_sampling_influence_mvp_ghost(
                ghost_t,
                ghost_s,
                batch_t,
                progress,
                u_m,
                chunk_size=chunk_size,
            ).detach()
            u_s -= (eta_t / batch_size) * sampling_influence_mvp

        hvp_flat = ghost_t.compute_hvp_batch(batch_t, progress, u_m, chunk_size=chunk_size).detach()
        u_m = u_m - eta_t * hvp_flat

        if t in sampling_steps:
            u_m = u_m + u_s
            u_s = torch.zeros_like(u_m)

            if sampling_model_t < len(sampling_steps):
                if ghost_s is not None:
                    ghost_s.close()
                    del ghost_s
                if model_s is not None:
                    del model_s

                model_s = PPO.load(model_path / f"policy_grad_{sampling_steps[sampling_model_t]}.zip", device=device)
                ghost_s = GhostPolicy(model_s)
                sampling_model_t += 1

        ghost_t.close()
        del ghost_t
        del model_t
        del batch_t

    if ghost_s is not None:
        ghost_s.close()
        del ghost_s
    if model_s is not None:
        del model_s

    return {
        "influence_scores": influence_scores,
    }
