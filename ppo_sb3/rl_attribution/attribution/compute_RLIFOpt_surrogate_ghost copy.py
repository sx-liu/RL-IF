import pickle
import torch
from tqdm import tqdm
from stable_baselines3 import PPO
import torch.nn.functional as F

from rl_attribution.utils.gradients import (
	compute_surrogate_validation_loss_gradient,
)

def _batch_get(batch, key):
	if isinstance(batch, dict):
		return batch[key]
	return getattr(batch, key)

def _compute_ppo_per_sample_losses(model, batch, progress):
	policy = model.policy
	device = model.device
	act_space = model.action_space

	obs = torch.as_tensor(_batch_get(batch, "observations"), dtype=torch.float32, device=device)
	old_log_prob = torch.as_tensor(_batch_get(batch, "old_log_prob"), dtype=torch.float32, device=device).view(-1)
	advantages = torch.as_tensor(_batch_get(batch, "advantages"), dtype=torch.float32, device=device).view(-1)

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
	approx_kl_loss = old_log_prob - new_log_prob

	losses = policy_loss + model.ent_coef * entropy_loss + model.kl_coef * approx_kl_loss
	return losses

class GhostRLIFOpt:
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
				if not self._hooks_enabled:
					return
				x, = inp
				self._xs[nm] = x.detach()

			def bwd(_module, _grad_in, grad_out, nm=name):
				if not self._hooks_enabled:
					return
				g, = grad_out
				self._gs[nm] = g.detach()

			self._hook_handles.append(module.register_forward_hook(fwd))
			self._hook_handles.append(module.register_full_backward_hook(bwd))

	def close(self):
		for handle in self._hook_handles:
			handle.remove()
		self._hook_handles = []
		for k in self._xs:
			self._xs[k] = None
			self._gs[k] = None

	def __del__(self):
		self.close()

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

	def compute_train_scores_for_batch(self, batch, progress, u_flat):
		self.model.policy.optimizer.zero_grad(set_to_none=True)
		losses = _compute_ppo_per_sample_losses(self.model, batch, progress)
		losses.sum().backward()

		train_xs, train_gs = self._snapshot_hooks()
		u_by_name = self._split_u_by_param(u_flat)
		return compute_sample_ip_with_u_vec(train_xs, train_gs, self.conv_params, u_by_name)

	def compute_hvp_batch(self, batch, progress, u_flat):
		self._hooks_enabled = False
		params = [
			p for n, p in self.model.policy.named_parameters()
			if p.requires_grad and "value_net" not in n
		]
		if not params:
			raise RuntimeError("No actor parameters found (excluding value_net).")

		losses = _compute_ppo_per_sample_losses(self.model, batch, progress)
		loss = losses.mean()

		grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=False)
		grad_flat = torch.cat([g.reshape(-1) for g in grads])

		if grad_flat.numel() != u_flat.numel():
			raise ValueError(f"u_flat dim {u_flat.numel()} != grad dim {grad_flat.numel()}")

		dot_product = torch.dot(grad_flat, u_flat.to(device=grad_flat.device, dtype=grad_flat.dtype))
		hvp = torch.autograd.grad(dot_product, params, allow_unused=False)
		hvp_flat = torch.cat([h.reshape(-1) for h in hvp])
		self._hooks_enabled = True
		return hvp_flat

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
		X_t = train_xs[layer]
		G_t = train_gs[layer]

		if X_t is None or G_t is None:
			raise RuntimeError(
				f"Missing hook tensors at layer '{layer}' (got X_t={X_t is not None}, G_t={G_t is not None})."
			)

		w_name = f"{layer}.weight" if layer else "weight"
		b_name = f"{layer}.bias" if layer else "bias"

		if w_name not in u_by_name:
			continue

		u_w = u_by_name[w_name]
		u_b = u_by_name.get(b_name, None)

		if X_t.dim() == 2:
			Q = G_t.unsqueeze(2) * X_t.unsqueeze(1)
			sample_ip += (Q * u_w.unsqueeze(0)).sum(dim=(1, 2))
			if u_b is not None:
				sample_ip += (G_t * u_b.unsqueeze(0)).sum(dim=1)
		else:
			p = conv_params[layer]
			patches_t = F.unfold(
				X_t,
				kernel_size=p["kernel_size"],
				stride=p["stride"],
				padding=p["padding"],
			)
			c_out = G_t.shape[1]
			Gt_flat = G_t.view(G_t.shape[0], c_out, -1)
			wg_t = torch.matmul(Gt_flat, patches_t.transpose(1, 2))
			sample_ip += (wg_t * u_w.view(c_out, -1).unsqueeze(0)).sum(dim=(1, 2))
			if u_b is not None:
				tr_bias = G_t.sum(dim=(2, 3))
				sample_ip += (tr_bias * u_b.unsqueeze(0)).sum(dim=1)

	return sample_ip

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
		model_t = PPO.load(model_path / f"policy_grad_{t}.zip", device=device)
		ghost = GhostRLIFOpt(model_t)

		with open(model_path / f"batch_{t + 1}.pkl", "rb") as f:
			batch_t = pickle.load(f)

		batch_size = _batch_get(batch_t, "observations").shape[0]
		progress = 1 - (int(t / n_batches) + 1) / (total_updates / n_batches)
		eta_t = model_t.lr_schedule(progress)

		scores_t = ghost.compute_train_scores_for_batch(batch_t, progress, u_flat)
		if scores_t.shape[0] != batch_size:
			raise RuntimeError(f"scores_t shape {scores_t.shape} != batch_size {batch_size}")

		scale = eta_t / batch_size
		for j, score in enumerate(scores_t):
			influence_scores[(t, j)] = (scale * score).item()

		if t > begin_step:
			hvp_flat = ghost.compute_hvp_batch(batch_t, progress, u_flat)
			u_flat = u_flat - eta_t * hvp_flat

		ghost.close()
		del ghost
		del model_t
		del batch_t

	return {
		"influence_scores": influence_scores,
	}
