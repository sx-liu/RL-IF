import argparse
from stable_baselines3 import PPO
import pickle
import numpy as np
import weakref

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

def _batch_get(batch, key):
	if isinstance(batch, dict):
		return batch[key]
	return getattr(batch, key)

def _maybe_squeeze_obs(obs):
    if hasattr(obs, "ndim") and obs.ndim > 1 and obs.shape[1] == 1:
        return obs.squeeze(1)
    return obs

class GhostTracIn:
    """
    Implements the 'ghost' technique: compute per-sample train-vs-validation gradient inner products
    without per-sample backward passes, by caching activations and output-gradients at each layer.
    """
    def __init__(self, model):
        self.model = model
        self.train_xs = {}
        self.train_gs = {}
        self.val_xs = {}
        self.val_gs = {}
        self.conv_params = {}
        self._hooks_enabled = True
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        self._xs = {}
        self._gs = {}
        self._conv_params = {}
        weak_self = weakref.ref(self)

        for name, module in self.model.policy.named_modules():
            if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                continue

            if 'value_net' in name:
                continue

            if isinstance(module, torch.nn.Conv2d):
                self._conv_params[name] = {
                    'kernel_size': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding
                }

            self._xs[name] = None
            self._gs[name] = None

            def fwd(m, inp, out, nm=name):
                inst = weak_self()
                if inst is None or not inst._hooks_enabled:
                    return
                x, = inp
                inst._xs[nm] = x.detach()

            def bwd(m, grad_in, grad_out, nm=name):
                inst = weak_self()
                if inst is None or not inst._hooks_enabled:
                    return
                g, = grad_out
                inst._gs[nm] = g.detach()

            self._hook_handles.append(module.register_forward_hook(fwd))
            self._hook_handles.append(module.register_full_backward_hook(bwd))

        self.conv_params = self._conv_params

    def close(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        for k in self._xs:
            self._xs[k] = None
            self._gs[k] = None
        self._xs = {}
        self._gs = {}
        self._conv_params = {}
        self.train_xs = {}
        self.train_gs = {}
        self.val_xs = {}
        self.val_gs = {}
        self.conv_params = {}
        self.model = None

    def _snapshot_hooks(self):
        xs = {k: v for k, v in self._xs.items()}
        gs = {k: v for k, v in self._gs.items()}

        for k in self._xs:
            self._xs[k] = None
            self._gs[k] = None
        return xs, gs

    def compute_val_cache(self, sample):
        """
        Compute and cache (val_xs, val_gs) for the surrogate validation loss:
            L_val = -mean( adv * log pi(a|s) )
        """
        self.model.policy.optimizer.zero_grad()

        device = self.model.device
        policy = self.model.policy
        act_space = self.model.action_space

        obs = torch.as_tensor(sample["observations"], dtype=torch.float32, device=device)
        adv = torch.as_tensor(sample["advantages"], dtype=torch.float32, device=device).flatten()

        if obs.ndim == len(getattr(policy.observation_space, "shape", ())):
            obs = obs.unsqueeze(0)

        if hasattr(act_space, "n"):
            acts = torch.as_tensor(sample["actions"], dtype=torch.long, device=device).flatten()
        else:
            acts = torch.as_tensor(sample["actions"], dtype=torch.float32, device=device)
            if acts.ndim == 1:
                acts = acts.unsqueeze(0)

            if acts.shape[0] != obs.shape[0]:
                raise ValueError(f"actions batch {acts.shape[0]} != obs batch {obs.shape[0]}")

        for p in policy.parameters():
            p.grad = None

        dist = policy.get_distribution(obs)
        log_prob = dist.log_prob(acts)

        if log_prob.ndim > 1:

            log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=-1)

        if adv.numel() == 1:
            adv = adv.expand_as(log_prob)
        else:
            adv = adv.reshape(-1)
            if adv.shape[0] != log_prob.shape[0]:
                raise ValueError(f"advantages shape {adv.shape} incompatible with log_prob {log_prob.shape}")

        loss = -(adv * log_prob).mean()
        loss.backward()

        self.val_xs, self.val_gs = self._snapshot_hooks()

    def compute_train_scores_for_batch(self, batch, progress):
        """
        Compute per-sample TracIn scores for one training batch, using PPO clipped loss per-sample:
            L_i = -min(r_i * adv_i, clamp(r_i)*adv_i) + ent_coef * (-entropy_i)
        and return a 1d tensor [B] with <grad_i, grad_val>.
        """
        self.model.policy.optimizer.zero_grad()

        policy = self.model.policy
        device = self.model.device
        act_space = self.model.action_space

        obs = _batch_get(batch, "observations").to(self.model.device)

        old_log_prob = _batch_get(batch, "old_log_prob").to(self.model.device).flatten()
        advantages = _batch_get(batch, "advantages").to(self.model.device).flatten()

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

        values, new_log_prob, entropy = policy.evaluate_actions(obs, act)

        new_log_prob = new_log_prob.view(-1)
        old_log_prob = old_log_prob.view(-1)
        advantages = advantages.view(-1)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if old_log_prob.numel() == 1:
            old_log_prob = old_log_prob.expand_as(new_log_prob)

        if not (advantages.shape == new_log_prob.shape == old_log_prob.shape):
            raise ValueError(
                f"Shape mismatch: adv {advantages.shape}, new_lp {new_log_prob.shape}, old_lp {old_log_prob.shape}"
            )

        ratio = torch.exp(new_log_prob - old_log_prob)

        clip_range = self.model.clip_range(progress)
        clip_range = torch.as_tensor(clip_range, dtype=torch.float32, device=device)

        policy_loss_1 = ratio * advantages
        policy_loss_2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
        policy_loss = -torch.min(policy_loss_1, policy_loss_2)

        entropy_loss = -entropy
        losses = policy_loss + self.model.ent_coef * entropy_loss

        loss = losses.sum()
        loss.backward()

        self.train_xs, self.train_gs = self._snapshot_hooks()
        scores = compute_sample_ip_layers_vec(self.train_xs, self.train_gs, self.val_xs, self.val_gs, self.conv_params)

        return scores

def compute_sample_ip_layers_vec(train_xs, train_gs, val_xs, val_gs, conv_params):
    """
    Compute per-train-sample influence inner-product (ghost technique) for Linear and Conv2d layers.

    train_xs, train_gs: dict layer_name -> Tensor of train activations and grads
    val_xs,   val_gs:   dict layer_name -> Tensor of val activations and grads
    conv_params: dict layer_name -> {'kernel_size','stride','padding'}
    """
    layers = list(train_xs.keys())
    if not layers:
        raise RuntimeError("No hooked layers found. Did the policy architecture change?")

    n_train = next(iter(train_xs.values())).shape[0]
    sample_ip = torch.zeros(
        n_train,
        dtype=next(iter(train_xs.values())).dtype,
        device=next(iter(train_xs.values())).device
    )

    for layer in layers:
        X_t = train_xs[layer]
        G_t = train_gs[layer]
        X_v = val_xs[layer]
        G_v = val_gs[layer]

        if X_t is None or G_t is None or X_v is None or G_v is None:
            raise RuntimeError(f"Missing hook tensors at layer '{layer}' (got X_t={X_t is not None}, G_t={G_t is not None}, "
                               f"X_v={X_v is not None}, G_v={G_v is not None}).")

        if X_t.dim() == 2:

            P = G_v.transpose(0, 1) @ X_v

            Q = G_t.unsqueeze(2) * X_t.unsqueeze(1)
            sample_ip += (Q * P.unsqueeze(0)).sum(dim=(1, 2))

            bias_sum = G_v.sum(dim=0)
            sample_ip += (G_t * bias_sum.unsqueeze(0)).sum(dim=1)

        else:
            p = conv_params[layer]

            patches_v = F.unfold(X_v, kernel_size=p['kernel_size'], stride=p['stride'], padding=p['padding'])
            n_val, C_out, H1, W1 = G_v.shape
            Gv_flat = G_v.view(n_val, C_out, -1)
            wg_v = torch.matmul(Gv_flat, patches_v.transpose(1, 2))
            P = wg_v.sum(dim=0)

            patches_t = F.unfold(X_t, kernel_size=p['kernel_size'], stride=p['stride'], padding=p['padding'])
            n_tr = G_t.shape[0]
            Gt_flat = G_t.view(n_tr, C_out, -1)
            wg_t = torch.matmul(Gt_flat, patches_t.transpose(1, 2))
            sample_ip += (wg_t * P.unsqueeze(0)).sum(dim=(1, 2))

            bias_sum = G_v.sum(dim=(0, 2, 3))
            tr_bias = G_t.sum(dim=(2, 3))
            sample_ip += (tr_bias * bias_sum.unsqueeze(0)).sum(dim=1)

    return sample_ip

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
    influence_scores_ob_act_prob_adv = {}
    ghost = GhostTracIn(target_model)

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

    ghost.compute_val_cache(sample)

    for i in tqdm(range(begin_step, end_step)):

        batch_i = pickle.load(open(model_path / f'batch_{i+1}.pkl', 'rb'))

        batch_size = _batch_get(batch_i, "observations").shape[0]
        progress = 1 - (int(i / n_batches) + 1) / (total_updates / n_batches)

        scores_t = ghost.compute_train_scores_for_batch(batch_i, progress)
        assert scores_t.shape[0] == batch_size, f"scores_t shape {scores_t.shape} != batch_size {batch_size}"

        for j, curr_influence in enumerate(scores_t):
            influence_scores[(i, j)] = curr_influence.item()

    ghost.close()
    del ghost

    return {
        'influence_scores': influence_scores,
        'influence_scores_ob_act_prob_adv': influence_scores_ob_act_prob_adv,
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
