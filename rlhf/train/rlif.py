"""
RLIF (Stochastic Gradient Descent Influence) scorer for RLHF with LoRA transformers.

Adapted from rl_attribution/attribution/compute_SGDI_surrogate_ghost.py.

RLIF vs TracIn
--------------
TracIn scores sample i at a single checkpoint as:
    score_i = <∇_θ loss_i, u>
where u is a fixed validation gradient.

RLIF propagates u backwards through the optimization trajectory via the
Hessian-vector product (HVP):
    u_{t-1} = u_t - η_t * H_t * u_t

This means u at step t represents the sensitivity of the validation loss
to a perturbation of the gradient at step t, accounting for all future
gradient steps. The score is still computed as a dot product at each step,
but against a u that has been "rolled back" through subsequent updates.

In the inline RLHF setting (no disk checkpoints), we approximate this
within a single rollout's PPO epochs:
  - u starts as the validation gradient computed on the full rollout batch
    (pre-PPO, so the model hasn't changed yet)
  - For each mini-batch update within PPO epochs, we:
      1. Score samples:  score_i = (η/N) * <∇_θ loss_i, u>
      2. Propagate u:    u ← u - η * H * u
  - This traces how a perturbation at the current mini-batch would affect
    the validation loss after all remaining mini-batch updates in this step

Components
----------
RLHFGhostRLIF:
  - Same hook infrastructure as RLHFGhostTracIn
  - compute_scores(u_flat): ghost inner product (same as TracIn, O(N) per layer)
  - compute_hvp(per_sample_losses, u_flat): H @ u via double backprop (second-order)
  - get_u_flat(): collect param.grad into flat vector after backward

compute_validation_gradient(model, ...):
  - Compute u = ∇_θ L_val on a validation mini-batch (the starting u for RLIF)
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


class RLHFGhostRLIF:
    """
    Ghost-gradient RLIF scorer for RLHF LoRA models.

    Extends the TracIn ghost-gradient scorer with:
      - compute_hvp(): Hessian-vector product for u propagation across steps
      - propagate_u(): convenience wrapper: u ← u - η * H * u

    Hook infrastructure is identical to RLHFGhostTracIn — hooks all trainable
    nn.Linear layers (requires_grad weight) excluding the value head.

    Typical usage within one PPO rollout:

        rlif = RLHFGhostRLIF(model)

        # Compute starting u from validation batch (pre-PPO-epoch model)
        u = compute_validation_gradient(model, val_batch, ...)

        for ppo_epoch in range(ppo_epochs):
            for mini_batch in mini_batches:
                # 1. Forward + backward to populate hooks and param.grad
                optimizer.zero_grad()
                per_sample_losses = compute_per_sample_losses(mini_batch)
                per_sample_losses.sum().backward()

                # 2. Score samples against current u
                scores = rlif.compute_scores(u)        # [N]

                # 3. Filter: keep positive-influence samples
                selected = scores > 0

                # 4. Full training update on selected subset
                optimizer.zero_grad()
                loss = compute_full_loss(mini_batch[selected])
                loss.backward()
                optimizer.step()

                # 5. Propagate u backwards through the gradient step
                optimizer.zero_grad()
                per_sample_losses = compute_per_sample_losses(mini_batch)
                u = rlif.propagate_u(per_sample_losses, u, lr)
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
            "RLHFGhostRLIF: hooked %d layers, %d trainable policy params (excl. v_head)",
            len(self._xs),
            self._param_dim,
        )

    # ------------------------------------------------------------------
    # Setup (identical to RLHFGhostTracIn)
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
                inst._xs[nm] = inp[0].detach()

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
        offset = 0
        for name, p in self.model.named_parameters():
            if not p.requires_grad or _is_v_head(name):
                continue
            size = p.numel()
            self._name_to_slice[name] = (offset, offset + size, p.shape, p.device, p.dtype)
            offset += size
        self._param_dim = offset

    def _snapshot_and_clear_hooks(self) -> Tuple[Dict, Dict]:
        xs = {k: v for k, v in self._xs.items()}
        gs = {k: v for k, v in self._gs.items()}
        for k in self._xs:
            self._xs[k] = None
            self._gs[k] = None
        return xs, gs

    def _snapshot_compressed_and_clear_hooks(self) -> Dict[str, Optional[Tensor]]:
        """
        Contract (xs, gs) over the token dimension on GPU and return
        per-sample weight-gradient matrices C[n, D_out, D_in] = Σ_t G[n,t]^T ⊗ X[n,t].

        ~6× smaller than raw (xs, gs) — eliminates large PCIe host transfers
        and keeps all snapshots on GPU throughout Phases 1–2.
        """
        compressed: Dict[str, Optional[Tensor]] = {}
        for k in list(self._xs.keys()):
            x = self._xs[k]; self._xs[k] = None
            g = self._gs[k]; self._gs[k] = None
            if x is None or g is None:
                compressed[k] = None
                continue
            if x.dim() == 3:           # [N, T, D_in], [N, T, D_out]
                c = torch.bmm(g.transpose(1, 2), x)   # [N, D_out, D_in]
            elif x.dim() == 2:         # [N, D_in], [N, D_out]
                c = g.unsqueeze(2) * x.unsqueeze(1)   # [N, D_out, D_in]
            else:
                compressed[k] = None
                continue
            compressed[k] = c          # stays on GPU
        return compressed

    def _split_u_by_param(self, u_flat: Tensor) -> Dict[str, Tensor]:
        if u_flat.numel() != self._param_dim:
            raise ValueError(
                f"u_flat dim {u_flat.numel()} != policy param dim {self._param_dim}"
            )
        u_by_name = {}
        for name, (start, end, shape, device, dtype) in self._name_to_slice.items():
            u_by_name[name] = u_flat[start:end].to(device=device, dtype=dtype).view(shape)
        return u_by_name

    def _get_policy_params(self):
        return [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and not _is_v_head(n)
        ]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get_u_flat(self) -> Tensor:
        """
        Collect param.grad from all trainable policy params (excl. v_head).
        Call immediately after per_sample_losses.sum().backward().
        """
        parts = []
        param_dict = dict(self.model.named_parameters())
        for name, (start, end, shape, device, dtype) in self._name_to_slice.items():
            p = param_dict.get(name)
            if p is None or p.grad is None:
                parts.append(torch.zeros(end - start, dtype=dtype))
            else:
                parts.append(p.grad.detach().reshape(-1).cpu().to(dtype))
        return torch.cat(parts)

    def compute_scores(self, u_flat: Tensor) -> Tensor:
        """
        Compute per-sample RLIF influence scores via the ghost gradient trick.

            score_i = <∇_θ loss_i, u_flat>

        Must be called after per_sample_losses.sum().backward() and before
        optimizer.zero_grad(). Clears hook state after reading.

        Args:
            u_flat: Validation gradient direction, shape [D].

        Returns:
            scores: [N] — positive = helpful, negative = harmful.
        """
        xs, gs = self._snapshot_and_clear_hooks()

        if not xs:
            raise RuntimeError("No hooked layers — did any linear layer run a forward pass?")

        first_x = next(v for v in xs.values() if v is not None)
        u_by_name = self._split_u_by_param(u_flat.to(device=first_x.device, dtype=first_x.dtype))
        return _compute_sample_ip_with_u_vec(xs, gs, u_by_name)

    def compute_hvp(self, per_sample_losses: Tensor, u_flat: Tensor) -> Tensor:
        """
        Compute the Hessian-vector product H @ u_flat.

        H is the Hessian of mean(per_sample_losses) w.r.t. policy params.
        Uses double backprop (create_graph=True), which requires a fresh
        forward pass through the model — caller must recompute per_sample_losses
        with the computation graph still attached (do NOT call .detach() on it).

        Hooks are disabled during this call to avoid polluting hook state.

        Args:
            per_sample_losses: [N] tensor with computation graph attached.
            u_flat: Vector to multiply H by, shape [D].

        Returns:
            hvp_flat: H @ u_flat, shape [D], on CPU.
        """
        self._hooks_enabled = False
        try:
            params = self._get_policy_params()
            if not params:
                raise RuntimeError("No trainable policy params found.")

            loss = per_sample_losses.mean()

            # First-order grads with graph retained for second differentiation
            grads = torch.autograd.grad(
                loss, params, create_graph=True, allow_unused=False
            )
            grad_flat = torch.cat([g.reshape(-1) for g in grads])

            u = u_flat.to(device=grad_flat.device, dtype=grad_flat.dtype)
            if grad_flat.numel() != u.numel():
                raise ValueError(
                    f"u_flat dim {u.numel()} != grad dim {grad_flat.numel()}"
                )

            # Dot product and second backward to get H @ u
            dot = torch.dot(grad_flat, u)
            hvp_grads = torch.autograd.grad(dot, params, allow_unused=False)
            hvp_flat = torch.cat([h.reshape(-1) for h in hvp_grads])

            return hvp_flat.detach().cpu()
        finally:
            self._hooks_enabled = True

    def propagate_u(
        self, per_sample_losses: Tensor, u_flat: Tensor, lr: float
    ) -> Tensor:
        """
        Propagate u backwards through one gradient step using the exact Hessian:
            u_new = u - lr * H @ u

        Requires a fresh forward pass with the computation graph attached
        (do NOT reuse the same losses tensor from compute_scores).
        Uses double backprop (create_graph=True) — more accurate but slower,
        and requires storing the full computation graph.

        Args:
            per_sample_losses: [N] losses with computation graph (fresh forward).
            u_flat: Current u vector, shape [D].
            lr: Learning rate used for this mini-batch update.

        Returns:
            u_new: Updated u_flat, shape [D], on CPU.
        """
        hvp = self.compute_hvp(per_sample_losses, u_flat)
        return u_flat.cpu() - lr * hvp

    def compute_fim_hvp(self, per_sample_losses: Tensor, scores: Tensor) -> Tensor:
        """
        Compute Fisher-vector product F_emp @ u using per-sample scores as weights.

        The empirical Fisher Information Matrix approximation (H ≈ F) gives:

            F @ u = (1/N) Σ_i (∇_θ loss_i · u) * ∇_θ loss_i
                  = ∇_θ [(1/N) Σ_i score_i * loss_i]   (score_i treated as constant)

        The inner term (∇_θ loss_i · u) is exactly the per-sample RLIF score already
        computed by compute_scores(), so no extra inner products are needed. The FVP
        reduces to a single score-weighted backward pass — no create_graph, no
        second-order differentiation, and no checkpoint storage required.

        Hooks are disabled during this call to avoid polluting hook state.

        Args:
            per_sample_losses: [N] losses with computation graph attached (fresh
                               forward pass — do not reuse the scoring backward's
                               tensor if the graph has already been freed).
            scores: [N] per-sample ghost scores from compute_scores(), detached.
                    These are (∇_θ loss_i · u) — the FIM weights.

        Returns:
            fvp: F_emp @ u, shape [D], on CPU.
        """
        self._hooks_enabled = False
        try:
            N = per_sample_losses.size(0)
            weights = scores.detach().to(
                device=per_sample_losses.device, dtype=per_sample_losses.dtype
            )
            weighted_loss = (weights * per_sample_losses).sum() / N

            # Zero out stale param.grad before the FIM backward
            for p in self._get_policy_params():
                p.grad = None

            weighted_loss.backward()

            # Collect param.grad → F @ u
            parts = []
            param_dict = dict(self.model.named_parameters())
            for name, (start, end, shape, device, dtype) in self._name_to_slice.items():
                p = param_dict.get(name)
                if p is None or p.grad is None:
                    parts.append(torch.zeros(end - start, dtype=dtype))
                else:
                    parts.append(p.grad.detach().reshape(-1).cpu().to(dtype))
            return torch.cat(parts)
        finally:
            self._hooks_enabled = True

    def propagate_u_fim(
        self, per_sample_losses: Tensor, scores: Tensor, u_flat: Tensor, lr: float
    ) -> Tensor:
        """
        Propagate u backwards through one gradient step using the FIM approximation:
            u_new = u - lr * F_emp @ u

        Preferred over propagate_u() for LLMs where:
          - Storing the full computation graph (create_graph=True) is memory-prohibitive
          - Historical model checkpoints are unavailable (inline RLHF setting)

        The FIM approximation replaces H with the empirical Fisher, which equals the
        outer product of gradients. The FVP is then a score-weighted gradient sum —
        one standard backward pass, no second-order differentiation.

        Args:
            per_sample_losses: [N] losses with computation graph (fresh forward).
            scores: [N] per-sample RLIF scores from compute_scores() on this batch.
            u_flat: Current u vector, shape [D].
            lr: Learning rate used for this mini-batch update.

        Returns:
            u_new: Updated u_flat, shape [D], on CPU.
        """
        fvp = self.compute_fim_hvp(per_sample_losses, scores)
        return u_flat.cpu() - lr * fvp

    # ------------------------------------------------------------------
    # Snapshot-based scoring (Phase 2 without model forward/backward)
    # ------------------------------------------------------------------

    def compute_scores_from_snapshot(
        self,
        xs: Dict[str, Optional[Tensor]],
        gs: Dict[str, Optional[Tensor]],
        u_flat: Tensor,
    ) -> Tensor:
        """
        Compute per-sample RLIF scores from a pre-captured (xs, gs) snapshot.

        Equivalent to compute_scores() but takes explicit xs/gs instead of reading
        from live hook state.  Allows Phase 2 scoring without model forward/backward.

        Args:
            xs: Dict mapping layer name → input tensor [N, ...] (may be on CPU).
            gs: Dict mapping layer name → output-grad tensor [N, ...] (may be on CPU).
            u_flat: Validation gradient direction, shape [D] (any device/dtype).

        Returns:
            scores: [N] — positive = helpful, negative = harmful (same device as xs).
        """
        layers = [k for k in xs if xs[k] is not None and gs[k] is not None]
        if not layers:
            raise RuntimeError("Snapshot has no valid layers.")
        first_x = xs[layers[0]]
        u_by_name = self._split_u_by_param(u_flat.to(device=first_x.device, dtype=first_x.dtype))
        return _compute_sample_ip_with_u_vec(xs, gs, u_by_name)

    def compute_fim_hvp_from_snapshot(
        self,
        xs: Dict[str, Optional[Tensor]],
        gs: Dict[str, Optional[Tensor]],
        scores: Tensor,
    ) -> Tensor:
        """
        Compute FIM-vector product F @ u from stored (xs, gs) snapshots.

        Uses the identity  F @ u = (1/N) Σ_i score_i * ∇_θ loss_i  together
        with the ghost-gradient expression for ∇_{W_l} loss_i:

            ∇_{W_l} loss_i = Σ_t G_{l,i,t}^T * X_{l,i,t}   (outer product, token sum)

        so that

            (F @ u)_{W_l} = (1/N) · sG_l.reshape(N*T, D_out).T @ X_l.reshape(N*T, D_in)

        where sG_l = scores[:,None,None] * G_l.  This is a single batched matmul per
        layer — no model operations, no second-order differentiation.

        Args:
            xs: Dict mapping layer name → input tensor (may be on CPU).
            gs: Dict mapping layer name → output-grad tensor (may be on CPU).
            scores: [N] per-sample scores from compute_scores_from_snapshot().

        Returns:
            fvp: F_emp @ u, shape [D], on CPU.
        """
        first_x = next((v for v in xs.values() if v is not None), None)
        if first_x is None:
            raise RuntimeError("Snapshot has no valid xs.")
        device = first_x.device
        dtype  = first_x.dtype
        N      = first_x.shape[0]
        sc     = scores.detach().to(device=device, dtype=dtype)  # [N]

        parts = []
        for name, (start, end, shape, param_device, param_dtype) in self._name_to_slice.items():
            if name.endswith(".weight"):
                layer_name = name[:-7]   # strip ".weight"
                is_weight  = True
            elif name.endswith(".bias"):
                layer_name = name[:-5]   # strip ".bias"
                is_weight  = False
            else:
                parts.append(torch.zeros(end - start, dtype=param_dtype))
                continue

            X = xs.get(layer_name)
            G = gs.get(layer_name)
            if X is None or G is None:
                parts.append(torch.zeros(end - start, dtype=param_dtype))
                continue

            X = X.to(device=device, dtype=dtype)
            G = G.to(device=device, dtype=dtype)

            if is_weight:
                # (F@u)_W = (1/N) Σ_i score_i * ∇_W loss_i
                #         = (1/N) (sc*G).reshape(N*T, D_out).T @ X.reshape(N*T, D_in)
                if X.dim() == 2:
                    fvp = (sc.unsqueeze(1) * G).T @ X / N          # [D_out, D_in]
                elif X.dim() == 3:
                    NT = X.shape[0] * X.shape[1]
                    sG = (sc.unsqueeze(1).unsqueeze(2) * G).reshape(NT, -1)
                    fvp = sG.T @ X.reshape(NT, -1) / N              # [D_out, D_in]
                else:
                    parts.append(torch.zeros(end - start, dtype=param_dtype))
                    continue
            else:
                # (F@u)_b = (1/N) Σ_i score_i * Σ_t G_{i,t}
                if G.dim() == 2:
                    fvp = (sc.unsqueeze(1) * G).sum(0) / N          # [D_out]
                elif G.dim() == 3:
                    fvp = (sc.unsqueeze(1).unsqueeze(2) * G).sum((0, 1)) / N  # [D_out]
                else:
                    parts.append(torch.zeros(end - start, dtype=param_dtype))
                    continue

            parts.append(fvp.reshape(-1).to(dtype=param_dtype))

        return torch.cat(parts)

    def compute_u_from_snapshot(
        self,
        xs: Dict[str, Optional[Tensor]],
        gs: Dict[str, Optional[Tensor]],
    ) -> Tensor:
        """
        Compute u = mean gradient  (1/N) Σ_i ∇_θ loss_i  from a stored snapshot.

        Equivalent to running a full forward+backward on this mini-batch and
        calling get_u_flat(), but uses only the already-captured (xs, gs) —
        no model operations needed.

        Mathematically: setting scores=ones in compute_fim_hvp_from_snapshot gives
            (1/N) Σ_i 1 * ∇_θ loss_i  =  mean gradient

        Returns:
            u_flat: Mean gradient vector, shape [D], on same device as xs.
        """
        first_x = next(v for v in xs.values() if v is not None)
        N = first_x.shape[0]
        ones = torch.ones(N, device=first_x.device, dtype=first_x.dtype)
        return self.compute_fim_hvp_from_snapshot(xs, gs, ones)

    def propagate_u_fim_from_snapshot(
        self,
        xs: Dict[str, Optional[Tensor]],
        gs: Dict[str, Optional[Tensor]],
        scores: Tensor,
        u_flat: Tensor,
        lr: float,
    ) -> Tensor:
        """
        Propagate u via FIM approximation using stored (xs, gs) snapshot.

            u_new = u - lr * F_emp @ u

        No model operations needed — all computation is from stored (xs, gs).

        Args:
            xs, gs: Gradient snapshot from Phase 1 (may be on CPU).
            scores: [N] per-sample scores from compute_scores_from_snapshot().
            u_flat: Current u vector, shape [D], on CPU.
            lr: Learning rate used for this step.

        Returns:
            u_new: Updated u_flat, shape [D], on CPU.
        """
        fvp = self.compute_fim_hvp_from_snapshot(xs, gs, scores)
        return u_flat.to(fvp.device, fvp.dtype) - lr * fvp

    # ------------------------------------------------------------------
    # Compressed-snapshot scoring (Phase 2 — all on GPU, no transfers)
    # ------------------------------------------------------------------

    def compute_scores_from_compressed(
        self,
        compressed: Dict[str, Optional[Tensor]],
        u_flat: Tensor,
    ) -> Tensor:
        """
        Per-sample RLIF scores from compressed gradient snapshots.

        score_i = Σ_layers (u_w * C[i]).sum()
                = Σ_layers einsum("de,nde->n", u_w, C)

        where C[n, D_out, D_in] = Σ_t G[n,t]^T ⊗ X[n,t] (from
        _snapshot_compressed_and_clear_hooks).

        Args:
            compressed: Dict layer_name → [N, D_out, D_in] (on GPU).
            u_flat: [D] validation gradient direction (any device/dtype).

        Returns:
            scores: [N] on the same device as compressed tensors.
        """
        layers = [k for k in compressed if compressed[k] is not None]
        if not layers:
            raise RuntimeError("Compressed snapshot has no valid layers.")
        first_c = compressed[layers[0]]
        device, dtype = first_c.device, first_c.dtype
        N = first_c.shape[0]
        u_by_name = self._split_u_by_param(u_flat.to(device=device, dtype=dtype))

        scores = torch.zeros(N, device=device, dtype=dtype)
        for layer in layers:
            C = compressed[layer]                          # [N, D_out, D_in]
            w_name = f"{layer}.weight"
            if w_name not in u_by_name:
                continue
            u_w = u_by_name[w_name].to(device=device, dtype=dtype)  # [D_out, D_in]
            scores += torch.einsum("de,nde->n", u_w, C)
        return scores

    def compute_fim_hvp_from_compressed(
        self,
        compressed: Dict[str, Optional[Tensor]],
        scores: Tensor,
    ) -> Tensor:
        """
        FIM-vector product from compressed gradient snapshots.

        (F@u)_W = (1/N) Σ_n scores[n] * C[n]
                = einsum("n,nde->de", scores, C) / N

        Args:
            compressed: Dict layer_name → [N, D_out, D_in] (on GPU).
            scores: [N] per-sample scores.

        Returns:
            fvp: F_emp @ u, shape [D], on same device as compressed tensors.
        """
        first_c = next((v for v in compressed.values() if v is not None), None)
        if first_c is None:
            raise RuntimeError("Compressed snapshot has no valid layers.")
        device, dtype = first_c.device, first_c.dtype
        N = first_c.shape[0]
        sc = scores.detach().to(device=device, dtype=dtype)

        parts = []
        for name, (start, end, _shape, _param_device, param_dtype) in self._name_to_slice.items():
            if not name.endswith(".weight"):
                # Bias: not captured in compressed (LoRA adapters have no bias)
                parts.append(torch.zeros(end - start, dtype=param_dtype, device=device))
                continue
            layer_name = name[:-7]
            C = compressed.get(layer_name)
            if C is None:
                parts.append(torch.zeros(end - start, dtype=param_dtype, device=device))
                continue
            C = C.to(device=device, dtype=dtype)
            fvp = torch.einsum("n,nde->de", sc, C) / N    # [D_out, D_in]
            parts.append(fvp.reshape(-1).to(dtype=param_dtype))

        return torch.cat(parts)

    def compute_u_from_compressed(
        self, compressed: Dict[str, Optional[Tensor]]
    ) -> Tensor:
        """u = mean gradient = (1/N) Σ_n C[n] per layer."""
        first_c = next(v for v in compressed.values() if v is not None)
        N = first_c.shape[0]
        ones = torch.ones(N, device=first_c.device, dtype=first_c.dtype)
        return self.compute_fim_hvp_from_compressed(compressed, ones)

    def propagate_u_fim_from_compressed(
        self,
        compressed: Dict[str, Optional[Tensor]],
        scores: Tensor,
        u_flat: Tensor,
        lr: float,
    ) -> Tensor:
        """u_new = u - lr * F_emp @ u, using compressed gradient snapshots."""
        fvp = self.compute_fim_hvp_from_compressed(compressed, scores)
        return u_flat.to(fvp.device, fvp.dtype) - lr * fvp

    def close(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        self._xs.clear()
        self._gs.clear()
        self._name_to_slice.clear()
        self.model = None


# ------------------------------------------------------------------
# Ghost gradient inner product (identical to tracin.py)
# ------------------------------------------------------------------

def _compute_sample_ip_with_u_vec(
    xs: Dict[str, Optional[Tensor]],
    gs: Dict[str, Optional[Tensor]],
    u_by_name: Dict[str, Tensor],
) -> Tensor:
    """
    Per-sample gradient inner product via ghost gradient trick.

    For weight W of shape [D_out, D_in]:
        score_i = (dL_i/dW * u_w).sum()
                = (X[i] * (G[i] @ u_w)).sum()   (sum over T if 3D)
    """
    layers = [k for k in xs if xs[k] is not None and gs[k] is not None]
    if not layers:
        raise RuntimeError("All hook tensors are None.")

    first_x = xs[layers[0]]
    n = first_x.shape[0]
    scores = torch.zeros(n, dtype=first_x.dtype, device=first_x.device)

    for layer in layers:
        X = xs[layer]
        G = gs[layer]

        w_name = f"{layer}.weight"
        b_name = f"{layer}.bias"
        if w_name not in u_by_name:
            continue

        u_w = u_by_name[w_name].to(device=X.device, dtype=X.dtype)
        u_b = u_by_name.get(b_name)

        if X.dim() == 2:
            proj = G @ u_w                              # [N, D_in]
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
            logger.warning("Skipping layer %s: unexpected dim %d", layer, X.dim())

    return scores


# ------------------------------------------------------------------
# Validation gradient computation (starting u for RLIF)
# ------------------------------------------------------------------

def compute_validation_gradient(
    model: nn.Module,
    per_sample_losses_val: Tensor,
) -> Tensor:
    """
    Compute u = ∇_θ L_val — the starting validation gradient for RLIF.

    This is the gradient of the mean validation loss w.r.t. all trainable
    policy params (excl. v_head). It defines the "direction we want training
    to move" that RLIF traces backwards through the optimization trajectory.

    Args:
        model: The policy model (hooks must NOT be registered for this call,
               or caller must zero_grad before).
        per_sample_losses_val: [N_val] validation losses with computation graph.

    Returns:
        u_flat: Flat gradient vector, shape [D], on CPU.
    """
    params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not _is_v_head(n)
    ]
    loss_val = per_sample_losses_val.mean()
    grads = torch.autograd.grad(loss_val, params, allow_unused=False)
    u_flat = torch.cat([g.detach().reshape(-1).cpu() for g in grads])
    return u_flat
