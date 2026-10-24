"""
TracIn influence scoring for RLHF with LoRA transformer models.

TracIn (Pruthi et al., 2020) scores each training sample by the dot product
between its gradient and a validation gradient:

    score_i = <∇_θ loss_i, u>

where u is a "validation gradient" direction (here: self-reference, i.e. the
full mini-batch gradient, so u = Σ_j ∇_θ loss_j).

Ghost gradient trick: a single backward on sum(per_sample_losses) populates
hooks with per-sample activations [N,T,D], enabling O(N) score computation
without N separate backward passes.

NOTE: This is NOT RLIF. RLIF traces influence across the full gradient descent
trajectory via HVP backpropagation (see rl_attribution/attribution/). This
module only computes the single-checkpoint dot-product (TracIn).
"""

import weakref
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


def _is_v_head(name: str) -> bool:
    return "v_head" in name


def _has_trainable_weight(module: nn.Linear) -> bool:
    return hasattr(module, "weight") and module.weight is not None and module.weight.requires_grad


class RLHFGhostTracIn:
    """
    Ghost-gradient TracIn influence scorer for RLHF LoRA models.

    Hooks all trainable nn.Linear layers (requires_grad weight) that are NOT
    part of the value head. For LoRA models this captures lora_A and lora_B
    linear modules; frozen base-model weights are correctly skipped.

    Usage pattern:
        tracin = RLHFGhostTracIn(model)

        # 1. Compute per-sample scoring losses (pg + kl, no value head)
        per_sample_losses = ...
        per_sample_losses.sum().backward()              # hooks fire here

        # 2. Collect u_flat from accumulated param.grad
        u_flat = tracin.get_u_flat()

        # 3. Score each sample in the mini-batch
        scores = tracin.compute_scores(u_flat)          # [N]

        optimizer.zero_grad()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._xs: Dict[str, Optional[Tensor]] = {}
        self._gs: Dict[str, Optional[Tensor]] = {}
        self._hooks_enabled = True
        self._hook_handles = []
        self._name_to_slice: Dict[str, Tuple] = {}
        self._param_dim = 0

        self._register_hooks()
        self._build_param_slices()
        logger.info(
            "RLHFGhostTracIn: hooked %d layers, %d trainable policy params",
            len(self._xs),
            self._param_dim,
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _register_hooks(self):
        weak_self = weakref.ref(self)

        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if _is_v_head(name):
                continue
            if not _has_trainable_weight(module):
                continue

            self._xs[name] = None
            self._gs[name] = None

            def fwd(_module, inp, _out, nm=name):
                inst = weak_self()
                if inst is None or not inst._hooks_enabled:
                    return
                x = inp[0]
                inst._xs[nm] = x.detach()

            def bwd(_module, _grad_in, grad_out, nm=name):
                inst = weak_self()
                if inst is None or not inst._hooks_enabled:
                    return
                g = grad_out[0]
                if g is not None:
                    inst._gs[nm] = g.detach()

            self._hook_handles.append(module.register_forward_hook(fwd))
            self._hook_handles.append(module.register_full_backward_hook(bwd))

    def _build_param_slices(self):
        """Map each trainable policy parameter (excl. v_head) to a flat index range."""
        offset = 0
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if _is_v_head(name):
                continue
            size = p.numel()
            self._name_to_slice[name] = (offset, offset + size, p.shape, p.device, p.dtype)
            offset += size
        self._param_dim = offset

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _snapshot_and_clear_hooks(self) -> Tuple[Dict, Dict]:
        """Return current hook state and reset for next backward."""
        xs = {k: v for k, v in self._xs.items()}
        gs = {k: v for k, v in self._gs.items()}
        for k in self._xs:
            self._xs[k] = None
            self._gs[k] = None
        return xs, gs

    def _split_u_by_param(self, u_flat: Tensor) -> Dict[str, Tensor]:
        if u_flat.numel() != self._param_dim:
            raise ValueError(
                f"u_flat has {u_flat.numel()} elements but policy has {self._param_dim} trainable params"
            )
        u_by_name = {}
        for name, (start, end, shape, device, dtype) in self._name_to_slice.items():
            u_by_name[name] = u_flat[start:end].to(device=device, dtype=dtype).view(shape)
        return u_by_name

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get_u_flat(self) -> Tensor:
        """
        Collect current param.grad from all trainable policy params (excl. v_head)
        into a flat CPU tensor. Call immediately after .sum().backward().
        """
        parts = []
        for name, (start, end, shape, device, dtype) in self._name_to_slice.items():
            p = dict(self.model.named_parameters()).get(name)
            if p is None or p.grad is None:
                parts.append(torch.zeros(end - start, dtype=dtype))
            else:
                parts.append(p.grad.detach().reshape(-1).cpu().to(dtype))
        return torch.cat(parts)

    def compute_scores(self, u_flat: Tensor) -> Tensor:
        """
        Compute per-sample TracIn influence scores from the most recent backward pass.

        score_i = <∇_θ loss_i, u_flat>

        Must be called immediately after `per_sample_losses.sum().backward()` and
        before `optimizer.zero_grad()`.

        Args:
            u_flat: Flat gradient vector (policy params excl. v_head), shape [D].

        Returns:
            scores: Shape [N] — positive means helpful, negative means harmful.
        """
        xs, gs = self._snapshot_and_clear_hooks()

        if not xs:
            raise RuntimeError("No hooked layers found — did any linear layer receive a forward pass?")

        first_x = next(v for v in xs.values() if v is not None)
        if first_x is None:
            raise RuntimeError("Hook activations are empty — ensure backward was called first.")

        u_by_name = self._split_u_by_param(u_flat.to(device=first_x.device, dtype=first_x.dtype))
        return _compute_sample_ip_with_u_vec(xs, gs, u_by_name)

    def close(self):
        """Remove all hooks and free references."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        self._xs.clear()
        self._gs.clear()
        self._name_to_slice.clear()
        self.model = None


# ------------------------------------------------------------------
# Inner product computation (ghost gradient trick)
# ------------------------------------------------------------------

def _compute_sample_ip_with_u_vec(
    xs: Dict[str, Optional[Tensor]],
    gs: Dict[str, Optional[Tensor]],
    u_by_name: Dict[str, Tensor],
) -> Tensor:
    """
    Compute per-sample gradient inner products using the ghost gradient trick.

    For a linear layer with weight W (shape [D_out, D_in]):
      Per-sample weight gradient: dL_i/dW = G[i]^T @ X[i]
                                          (summed over sequence positions if 3D)
      Inner product with u_w:     score_i = (dL_i/dW * u_w).sum()
                                          = (X[i] * (G[i] @ u_w)).sum()

    Supports both 2D [N, D] and 3D [N, T, D] activations (transformers).
    """
    layers = [k for k in xs if xs[k] is not None and gs[k] is not None]
    if not layers:
        raise RuntimeError("All hook tensors are None — no valid layers to score.")

    first_x = xs[layers[0]]
    n = first_x.shape[0]
    scores = torch.zeros(n, dtype=first_x.dtype, device=first_x.device)

    for layer in layers:
        X = xs[layer]  # [N, D_in] or [N, T, D_in]
        G = gs[layer]  # [N, D_out] or [N, T, D_out]

        w_name = f"{layer}.weight"
        b_name = f"{layer}.bias"

        if w_name not in u_by_name:
            continue

        u_w = u_by_name[w_name].to(device=X.device, dtype=X.dtype)  # [D_out, D_in]
        u_b = u_by_name.get(b_name)

        if X.dim() == 2:
            proj = G @ u_w                          # [N, D_in]
            scores += (X * proj).sum(dim=1)
            if u_b is not None:
                u_b = u_b.to(device=G.device, dtype=G.dtype)
                scores += (G * u_b.unsqueeze(0)).sum(dim=1)

        elif X.dim() == 3:
            proj = torch.einsum("ntd,de->nte", G, u_w)  # [N, T, D_in]
            scores += (X * proj).sum(dim=(1, 2))
            if u_b is not None:
                u_b = u_b.to(device=G.device, dtype=G.dtype)
                scores += (G.sum(dim=1) * u_b.unsqueeze(0)).sum(dim=1)

        else:
            logger.warning("Skipping layer %s: unexpected activation dim %d", layer, X.dim())

    return scores
