import torch

def compute_surrogate_validation_loss_gradient(model, sample, batch_size=2048):
    """
    Works for discrete and continuous (Box) action spaces.
    Computes: L = -mean( advantage * log pi(a_target | s) )

    Args:
        model: SB3 model
        sample: dict containing observations, actions, advantages
        batch_size: mini-batch size for gradient computation (default 2048)

    Returns:
        grad_vec (1D tensor)
    """
    device = model.device
    policy = model.policy
    act_space = model.action_space

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

    N = obs.shape[0]

    for p in policy.parameters():
        p.grad = None

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        obs_b = obs[start:end]
        adv_b = adv[start:end]

        if hasattr(act_space, "n"):
            acts_b = acts[start:end]
        else:
            acts_b = acts[start:end]

        dist = policy.get_distribution(obs_b)
        log_prob = dist.log_prob(acts_b)

        if log_prob.ndim > 1:
            log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=-1)

        if adv_b.numel() == 1:
            adv_b = adv_b.expand_as(log_prob)

        loss = -(adv_b * log_prob).mean()

        scaled_loss = loss * (end - start) / N
        scaled_loss.backward()

    grads = []
    for name, p in policy.named_parameters():
        if p.requires_grad and p.grad is not None and "value_net" not in name:
            grads.append(p.grad.detach().reshape(-1))

    grad_vec = torch.cat(grads) if grads else torch.tensor([], device=device)

    return grad_vec

def compute_ppo_policy_gradient(model, sample, progress, normalize_advantages=True):
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
    old_log_prob = old_log_prob.view(-1)
    advantages = advantages.view(-1)

    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    if old_log_prob.numel() == 1:
        old_log_prob = old_log_prob.expand_as(new_log_prob)

    if not (advantages.shape == new_log_prob.shape == old_log_prob.shape):
        raise ValueError(
            f"Shape mismatch: adv {advantages.shape}, new_lp {new_log_prob.shape}, old_lp {old_log_prob.shape}"
        )

    ratio = torch.exp(new_log_prob - old_log_prob)

    clip_range = model.clip_range(progress)
    clip_range = torch.as_tensor(clip_range, dtype=torch.float32, device=device)

    policy_loss_1 = ratio * advantages
    policy_loss_2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

    entropy_loss = -entropy.mean()

    log_ratio = new_log_prob - old_log_prob
    approx_kl_loss = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)

    loss = policy_loss + model.ent_coef * entropy_loss + model.kl_coef * approx_kl_loss

    params = [p for n, p in policy.named_parameters() if "value_net" not in n and p.requires_grad]

    grads = torch.autograd.grad(loss, params, allow_unused=False)
    grad_flat = torch.cat([g.reshape(-1) for g in grads if g is not None])

    return grad_flat

def compute_gradients_for_train_batch(model, train_batch, progress):
    grad_list_train = []
    if not isinstance(train_batch, dict):
        train_batch = {
            "observations": train_batch.observations,
            "actions": train_batch.actions,
            "old_log_prob": train_batch.old_log_prob,
            "advantages": train_batch.advantages,
        }

    batch_size = train_batch["observations"].shape[0]
    adjusted_advantages = (train_batch["advantages"] - train_batch["advantages"].mean()) / (train_batch["advantages"].std() + 1e-8)
    for i in range(batch_size):
        sample = {
            'observations': train_batch["observations"][i].unsqueeze(0).to(model.device),
            'actions': train_batch["actions"][i].unsqueeze(0).to(model.device),
            'old_log_prob': train_batch["old_log_prob"][i].unsqueeze(0).to(model.device),
            'advantages': adjusted_advantages[i].unsqueeze(0).to(model.device),
        }
        grad = compute_ppo_policy_gradient(model, sample, progress, normalize_advantages=False)
        grad_list_train.append(grad)
    return grad_list_train

def flatten_gradients(grad_dict):
    return torch.cat([g.view(-1) for g in grad_dict.values()])

def inner_product_gradients(grad1, grad2):
    """
    Compute the inner product between two flattened gradient tensors.
    """
    return torch.dot(grad1, grad2)

def compute_hvp(model, batch, u_flat, progress):
    """
    Compute the Hessian-vector product (HVP) for the mini-batch, following the paper's method.
    Returns a flattened HVP tensor, averaged over the mini-batch samples.
    """
    model.policy.optimizer.zero_grad()
    batch_size = batch["observations"].shape[0] if isinstance(batch, dict) else batch.observations.shape[0]
    hvp_list = []

    if isinstance(batch, dict):
        adjusted_advantages = (batch['advantages'] - batch['advantages'].mean()) / (batch['advantages'].std() + 1e-8)
    else:
        adjusted_advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

    for i in range(batch_size):

        if isinstance(batch, dict):
            sample = {
                'observations': batch['observations'][i].unsqueeze(0).to(model.device),
                'actions': batch['actions'][i].unsqueeze(0).to(model.device),
                'old_log_prob': batch['old_log_prob'][i].unsqueeze(0).to(model.device),
                'advantages': adjusted_advantages[i].unsqueeze(0).to(model.device)
            }
        else:
            sample = {
                'observations': batch.observations[i].unsqueeze(0).to(model.device),
                'actions': batch.actions[i].unsqueeze(0).to(model.device),
                'old_log_prob': batch.old_log_prob[i].unsqueeze(0).to(model.device),
                'advantages': adjusted_advantages[i].unsqueeze(0).to(model.device)
            }

        obs = sample['observations']
        act = sample['actions'].flatten()
        old_log_prob = sample['old_log_prob']
        advantages = sample['advantages']

        values, new_log_prob, entropy = model.policy.evaluate_actions(obs, act.long())
        ratio = torch.exp(new_log_prob - old_log_prob)
        clip_range = model.clip_range(progress)
        policy_loss = -torch.min(ratio * advantages, torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages)
        entropy_loss = -entropy

        log_ratio = new_log_prob - old_log_prob
        approx_kl_loss = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)

        loss = policy_loss + model.ent_coef * entropy_loss + model.kl_coef * approx_kl_loss

        grads = torch.autograd.grad(loss.sum(), [p for n, p in model.policy.named_parameters() if "value_net" not in n], create_graph=True)
        grad_flat = torch.cat([g.view(-1) for g in grads])

        dot_product = torch.dot(u_flat, grad_flat)

        hvp_flat = torch.autograd.grad(dot_product, [p for n, p in model.policy.named_parameters() if "value_net" not in n])

        hvp_flat = torch.cat([h.flatten() for h in hvp_flat])
        hvp_list.append(hvp_flat)

    return torch.stack(hvp_list, dim=0).mean(dim=0)

def compute_hessian(model, batch, progress, batch_cols=10240, symmetrize=True):
    """
    Full Hessian (w.r.t. policy params only) using column batching.
    Uses VJP on grad_flat via grad_outputs instead of dot-product trick.
    """
    device = model.device

    params = [p for n, p in model.policy.named_parameters()
              if ("value_net" not in n) and p.requires_grad]
    if not params:
        raise RuntimeError("No policy parameters found (excluding value_net).")

    with torch.enable_grad():
        model.policy.optimizer.zero_grad(set_to_none=True)

        obs = batch.observations.to(device)
        act = batch.actions.to(device).long()
        if act.dim() > 2:
            act = act.flatten(1, -1)
        old_log_prob = batch.old_log_prob.to(device)
        adv = batch.advantages.to(device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        values, new_log_prob, entropy = model.policy.evaluate_actions(obs, act)
        ratio = torch.exp(new_log_prob - old_log_prob)
        clip_range = model.clip_range(progress)

        policy_loss = -torch.min(
            ratio * adv,
            torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
        )
        loss = policy_loss
        loss = loss + model.ent_coef * (-entropy)
        loss = loss.mean()

        grads = torch.autograd.grad(loss, params, create_graph=True)
        grad_flat = torch.cat([g.reshape(-1) for g in grads])
        P = grad_flat.numel()

        if not grad_flat.requires_grad:
            raise RuntimeError("grad_flat does not require grad. "
                               "Check for a surrounding torch.no_grad() or detaches in your loss path.")

        def hvp_single(e_vec):

            hv = torch.autograd.grad(
                outputs=grad_flat, inputs=params,
                grad_outputs=e_vec,
                retain_graph=True, allow_unused=False
            )
            return torch.cat([h.reshape(-1) for h in hv])

        try:
            from torch.func import vmap
            batched_hvp = vmap(hvp_single)
            use_vmap = True
        except Exception:
            use_vmap = False

        H = torch.empty(P, P, dtype=grad_flat.dtype, device=grad_flat.device)

        def eye_chunk(start, end):
            B = end - start
            E = torch.zeros(B, P, device=grad_flat.device, dtype=grad_flat.dtype)
            E[torch.arange(B, device=E.device), torch.arange(start, end, device=E.device)] = 1
            return E

        col = 0
        while col < P:
            end = min(col + batch_cols, P)
            E = eye_chunk(col, end)

            if use_vmap:
                cols = batched_hvp(E)
            else:

                out = []
                for j in range(col, end):
                    e = torch.zeros(P, device=grad_flat.device, dtype=grad_flat.dtype)
                    e[j] = 1
                    out.append(hvp_single(e))
                cols = torch.stack(out, dim=0)

            H[:, col:end] = cols.T
            col = end

        if symmetrize:
            H = 0.5 * (H + H.T)

        return H
