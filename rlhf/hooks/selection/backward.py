"""
Autograd functions for gradient-based data curation and compression.

This module provides three distinct autograd Functions for Linear layers:
- CompressedLinearBackward: Pure gradient compression (no curation)
- LayerwiseLinearBackward: Per-layer curation, single-pass
- SubsetLinearBackward: Score accumulation, two-pass

And one for non-linear layers in merged-batch one-pass mode:
- TrainOnlyRMSNormBackward: RMSNorm that computes grad_weight from train slice only

Each function routes to specific handlers based on CompressionMode:
- NONE: Full gradients for scoring and updates
- SCORE_ONLY: Compressed scoring, full gradient updates
- FULL: Compressed scoring and gradient updates (MeSO)
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Tuple
    from torch import Tensor
    from .state import LayerwiseState, SubsetState
    from ..hook import GradientHook
    from ..compressor import Compressor

import torch
import torch.nn.functional as F
from torch.autograd import Function

from ..compression_mode import CompressionMode
from .utils import (
    augment_input_for_bias,
    split_train_val_batch,
    compute_scores_and_similarity,
    compute_scores_full_ghost,
    compute_scores_direct_materialization,
    compute_selected_gradients,
    compute_total_gradient,
)


# =============================================================================
# Helper functions for backward passes
# =============================================================================

def _get_val_components(
    hook_manager: GradientHook,
    layer_idx: int
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """
    Get validation gradient components from hook manager.

    Returns:
        (val_grad_output, val_input, val_grad_total, val_bias_grad) tuple.
        Either (val_grad_output, val_input, None, None) for factorized mode,
        or (None, None, val_grad_total, val_bias_grad) for full gradient mode.
    """
    val_cache = hook_manager._val_cache

    # Try factorized mode first
    val_grad_output, val_input = val_cache.get_factorized(layer_idx)
    if val_grad_output is not None and val_input is not None:
        return val_grad_output, val_input, None, None

    # Fall back to full gradient mode
    val_grad_total = val_cache.get_full(layer_idx)
    val_bias_grad = val_cache.get_bias_grad(layer_idx)
    return None, None, val_grad_total, val_bias_grad


def _compute_scale_factor(state: LayerwiseState, selected_indices: Tensor) -> Tensor:
    """Compute token-based gradient scale factor for selected samples."""
    return state._compute_scale_factor(selected_indices)


# =============================================================================
# Shared helpers for Layerwise backward paths
# =============================================================================

def _dispatch_scoring(
    scoring_method: str,
    train_grad_output: "Tensor",
    train_input: "Tensor",
    val_grad_output: "Optional[Tensor]",
    val_input: "Optional[Tensor]",
    val_grad_total: "Optional[Tensor]",
    use_second_order: bool,
    direct_batch_size: int = 0,
    return_materialized: bool = False,
) -> "Tuple[Tensor, Optional[Tensor], Optional[Tensor]]":
    """Dispatch to the appropriate scoring function based on scoring_method.

    Returns:
        (scores, similarity, materialized_grads) where materialized_grads is
        the [B, O*I] per-sample weight gradient tensor when return_materialized=True
        and scoring_method="direct", otherwise None.
    """
    if scoring_method == "direct":
        scores, similarity, G_train = compute_scores_direct_materialization(
            train_grad_output, train_input, val_grad_output, val_input,
            val_grad_total, use_second_order, batch_size=direct_batch_size,
            return_materialized=return_materialized,
        )
        return scores, similarity, G_train
    elif scoring_method == "full_ghost":
        return (*compute_scores_full_ghost(
            train_grad_output, train_input, val_grad_output, val_input,
            val_grad_total, use_second_order
        ), None)
    else:  # "reduced_ghost" (default)
        return (*compute_scores_and_similarity(
            train_grad_output, train_input, val_grad_output, val_input,
            val_grad_total, use_second_order
        ), None)


def _add_bias_scores(
    scores: "Tensor",
    similarity: "Optional[Tensor]",
    train_grad_output: "Tensor",
    val_grad_output: "Optional[Tensor]",
    val_bias_grad: "Optional[Tensor]",
    has_bias: bool,
) -> "Tuple[Tensor, Optional[Tensor]]":
    """
    Add bias gradient contribution to influence scores and similarity.

    The standard scoring functions compute scores from weight gradients only
    (go ⊗ inp). This adds the bias gradient term: go_i · val_go.

    Args:
        scores: Weight-only influence scores [B]
        similarity: Weight-only similarity matrix [B, B] or None
        train_grad_output: Training grad_output [B, S, O] or [B, O]
        val_grad_output: Validation grad_output [V, S, O] or None
        val_bias_grad: Precomputed validation bias gradient [O] or None
        has_bias: Whether the layer has bias

    Returns:
        (scores, similarity) with bias contribution added
    """
    if not has_bias:
        return scores, similarity

    # Compute train bias gradients: sum over sequence dim
    if train_grad_output.dim() == 3:
        train_bias = train_grad_output.sum(dim=1)  # [B, O]
    else:
        train_bias = train_grad_output  # [B, O]

    # Compute val bias gradient from factorized components or cache
    if val_grad_output is not None:
        if val_grad_output.dim() == 3:
            val_bias = val_grad_output.sum(dim=(0, 1))  # [O]
        else:
            val_bias = val_grad_output.sum(dim=0)  # [O]
    elif val_bias_grad is not None:
        val_bias = val_bias_grad  # [O] from cache
    else:
        return scores, similarity

    # Add bias score: train_bias_i · val_bias
    scores = scores + train_bias @ val_bias.to(train_bias.dtype)

    # Add bias similarity: train_bias_i · train_bias_j
    if similarity is not None:
        similarity = similarity + train_bias @ train_bias.T

    return scores, similarity


def _do_selection(
    state: "LayerwiseState",
    layer_idx: int,
    scores: Tensor,
    similarity: Optional[Tensor],
) -> Tensor:
    """Run per-layer curation: pick indices, record stats."""
    selected_indices = state._select_indices(scores, similarity)
    selected_indices = selected_indices.sort()[0]
    state._last_selected_indices = selected_indices
    state.num_selected = selected_indices.shape[0]

    if hasattr(state, '_layer_selections'):
        state._layer_selections.append((layer_idx, state.num_selected))
    if state._record_selections:
        state._selection_records.append({
            'layer_idx': layer_idx,
            'selected_indices': selected_indices.tolist(),
            'scores': scores.detach().float().cpu().tolist(),
        })
    return selected_indices


def _produce_gradient_update(
    hook_manager: "GradientHook",
    update_compressor: Optional["Compressor"],
    state: "LayerwiseState",
    layer_idx: int,
    train_grad_output: Tensor,
    train_input: Tensor,
    selected_indices: Tensor,
    has_bias: bool,
    materialized_grads: Optional[Tensor] = None,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """After curation, produce the gradient update.

    If update_compressor exists: compress selected gradients → store for MeSO → return (None, None).
    Otherwise: compute full gradients for selected samples → return (grad_weight, grad_bias).

    When materialized_grads (G_train [B, O*I]) is provided from direct scoring,
    reuses it for w.grad instead of recomputing from go/inp — avoids holding both
    the raw activations and the gradient simultaneously.
    """
    scale_factor = _compute_scale_factor(state, selected_indices)

    if update_compressor is not None:
        sel_go = train_grad_output[selected_indices]
        sel_inp = augment_input_for_bias(train_input[selected_indices], has_bias)
        update_compressed = update_compressor.forward((sel_go, sel_inp))
        reduced_grad = update_compressed.mean(dim=0, keepdim=True) * scale_factor
        hook_manager._store_compressed_grad(layer_idx, reduced_grad)
        return None, None

    if materialized_grads is not None:
        # Reuse G_train from direct scoring: G_train[i] = go_i^T @ inp_i flattened
        # Note: materialized_grads contains weight grads only (no bias augmentation).
        O = train_grad_output.shape[-1]
        grad_weight = materialized_grads[selected_indices].sum(dim=0).reshape(O, -1) * scale_factor
        grad_bias = None
        if has_bias:
            sel_go = train_grad_output[selected_indices]
            if sel_go.dim() == 3:
                grad_bias = sel_go.sum(dim=(0, 1)) * scale_factor
            else:
                grad_bias = sel_go.sum(dim=0) * scale_factor
        return grad_weight, grad_bias

    grad_weight, grad_bias = compute_selected_gradients(
        train_grad_output, train_input, selected_indices, has_bias, scale_factor
    )
    return grad_weight, grad_bias


def _produce_gradient_update_with_val(
    state: "LayerwiseState",
    selected_indices: Tensor,
    train_grad_output: Tensor,
    train_input: Tensor,
    val_grad_output: Tensor,
    val_input: Tensor,
    has_bias: bool,
) -> "Tuple[Tensor, Optional[Tensor]]":
    """Compute gradient from selected train + val samples in a single einsum.

    Concatenates selected train and val activations, then computes the gradient
    with a unified scale factor: batch_total_tokens / (selected_train_tokens + val_tokens).
    When selection_frac=1.0, scale=1.0 (matching baseline).
    """
    # Concatenate selected train + all val for unified gradient computation
    all_go = torch.cat([train_grad_output[selected_indices], val_grad_output], dim=0)
    all_inp = torch.cat([train_input[selected_indices], val_input], dim=0)

    # Unified scale factor accounting for all contributing tokens
    scale = state._compute_scale_factor_with_val(selected_indices)

    # Single einsum over all contributing samples
    grad_weight = compute_total_gradient(all_go, all_inp) * scale

    grad_bias = None
    if has_bias:
        if all_go.dim() == 3:
            grad_bias = all_go.sum(dim=(0, 1)) * scale
        else:
            grad_bias = all_go.sum(dim=0) * scale

    return grad_weight, grad_bias


def _store_update_grad(
    hook_manager: "GradientHook",
    update_compressor: "Optional[Compressor]",
    score_compressor: "Optional[Compressor]",
    layer_idx: int,
    grad_output: Tensor,
    input_aug: Tensor,
    has_bias: bool,
    score_compressed_reduced: Tensor,
) -> None:
    """Store compressed gradient for MeSO when no curation is active.

    Reuses score-compressed grad if compressors are shared; otherwise re-compresses.
    """
    if update_compressor is None:
        return
    if update_compressor is score_compressor:
        hook_manager._store_compressed_grad(layer_idx, score_compressed_reduced)
    else:
        update_compressed = update_compressor.forward((grad_output, input_aug))
        hook_manager._store_compressed_grad(layer_idx, update_compressed.sum(dim=0, keepdim=True))


# =============================================================================
# Autograd Functions
# =============================================================================

class CompressedLinearBackward(Function):
    """
    Autograd Function for pure gradient compression (no data curation).

    Used when compression is enabled (MeSO optimizer) but no data curation is active.
    """

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        hook_manager: "GradientHook",
        layer_idx: int
    ) -> Tensor:
        """Forward pass: standard linear transformation."""
        input_compute = input.to(weight.dtype) if input.dtype != weight.dtype else input
        ctx.save_for_backward(input_compute, weight, bias)
        ctx.hook_manager_ref = weakref.ref(hook_manager)
        ctx.layer_idx = layer_idx
        return F.linear(input_compute, weight, bias)

    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor
    ) -> Tuple[Tensor, None, None, None, None]:
        """Backward pass: compress gradients and store for MeSO optimizer."""
        input, weight, bias = ctx.saved_tensors
        layer_idx = ctx.layer_idx

        hook_manager = ctx.hook_manager_ref()
        if hook_manager is None:
            raise RuntimeError("Hook manager was garbage collected before backward pass")

        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)

        grad_input = grad_output @ weight.to(grad_output.dtype)

        compressor = hook_manager.update_compressors[layer_idx]

        with torch.no_grad():
            input_aug = augment_input_for_bias(input, bias is not None)
            compressed_grad = compressor.forward((grad_output, input_aug))
            compressed_grad = compressed_grad.sum(dim=0, keepdim=True)
            hook_manager._store_compressed_grad(layer_idx, compressed_grad)

        return grad_input, None, None, None, None


class LayerwiseLinearBackward(Function):
    """
    Autograd Function for layer-wise descent (per-layer curation).

    Single-pass: At each layer, computes scores, selects samples,
    and aggregates gradients immediately.
    """

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        hook_manager: "GradientHook",
        layer_idx: int
    ) -> Tensor:
        """Forward pass: standard linear transformation."""
        input_compute = input.to(weight.dtype) if input.dtype != weight.dtype else input
        ctx.save_for_backward(input_compute, weight, bias)
        ctx.hook_manager_ref = weakref.ref(hook_manager)
        ctx.layer_idx = layer_idx
        return F.linear(input_compute, weight, bias)

    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], None, None]:
        """Backward pass with per-layer curation."""
        input, weight, bias = ctx.saved_tensors
        layer_idx = ctx.layer_idx

        hook_manager = ctx.hook_manager_ref()
        if hook_manager is None:
            raise RuntimeError("Hook manager was garbage collected before backward pass")

        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)

        grad_input = grad_output @ weight.to(grad_output.dtype)

        score_compressor = hook_manager.score_compressors[layer_idx]
        update_compressor = hook_manager.update_compressors[layer_idx]
        state: Optional[LayerwiseState] = hook_manager.selection_state
        capture_val_mode = hook_manager.capture_val_mode
        use_stored_val = (
            state is not None and
            getattr(state, '_use_stored_val', False)
        )

        # Determine scoring path: use compressed only when scoring_method="compress"
        # (or during val capture when val cache is in compressed mode)
        use_compressed_scoring = False
        if score_compressor is not None:
            if capture_val_mode:
                # Val capture: use compression only if val cache is in compressed mode
                use_compressed_scoring = hook_manager._val_cache.is_compressed
            elif state is not None:
                use_compressed_scoring = getattr(state, 'scoring_method', 'reduced_ghost') == 'compress'

        with torch.no_grad():
            if use_compressed_scoring:
                grad_weight, grad_bias = LayerwiseLinearBackward._backward_compressed(
                    hook_manager, score_compressor, update_compressor, state, layer_idx,
                    input, grad_output, bias, capture_val_mode, use_stored_val
                )
            else:
                grad_weight, grad_bias = LayerwiseLinearBackward._backward_full(
                    hook_manager, update_compressor, state, layer_idx,
                    input, bias, grad_output,
                    capture_val_mode, use_stored_val
                )

        return grad_input, grad_weight, grad_bias, None, None

    @staticmethod
    def _backward_compressed(
        hook_manager: "GradientHook",
        score_compressor: "Compressor",
        update_compressor: Optional["Compressor"],
        state: Optional["LayerwiseState"],
        layer_idx: int,
        input: Tensor,
        grad_output: Tensor,
        bias: Optional[Tensor],
        capture_val_mode: bool,
        use_stored_val: bool
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Layerwise backward with score compression.

        Score computation uses score_compressor. Gradient updates are either:
        - Compressed via update_compressor (MeSO): returns (None, None)
        - Full gradients (no MeSO): returns (grad_weight, grad_bias)
        When update_compressor is score_compressor, reuses already-compressed grads.
        """
        has_bias = bias is not None
        input_aug = augment_input_for_bias(input, has_bias)

        # --- Compress ---

        score_compressed = score_compressor.forward((grad_output, input_aug))


        # --- Validation capture: store compressed val gradient ---
        if capture_val_mode:
            total_grad = score_compressed.sum(dim=0)
            val_cache = hook_manager.val_cache
            if val_cache._compressed[layer_idx] is None:
                val_cache._compressed[layer_idx] = total_grad
            else:
                val_cache._compressed[layer_idx] = val_cache._compressed[layer_idx] + total_grad
            return None, None

        # --- No curation state: just store MeSO update if needed ---
        if state is None:
            _store_update_grad(hook_manager, update_compressor, score_compressor,
                               layer_idx, grad_output, input_aug, has_bias,
                               score_compressed.sum(dim=0, keepdim=True))
            return None, None

        # --- Compute scores from score-compressed gradients ---

        if use_stored_val:
            train_grads = score_compressed
            val_grad = hook_manager._val_cache.get_compressed(layer_idx)
            score_correction = None
        else:
            train_grads, val_grads = split_train_val_batch(score_compressed, state.train_batch_size)
            val_grad = val_grads.sum(dim=0)
            score_correction = state.score_correction

        if val_grad is None:
    
            _store_update_grad(hook_manager, update_compressor, score_compressor,
                               layer_idx, grad_output, input_aug, has_bias,
                               score_compressed.mean(dim=0, keepdim=True))
            return None, None

        scores = train_grads @ val_grad
        if score_correction is not None:
            scores = scores * score_correction

        similarity = None
        if state.use_second_order:
            similarity = train_grads @ train_grads.T
            if score_correction is not None:
                similarity = similarity * (score_correction ** 2)


        # --- Shared compressors: delegate to state for efficient select+reduce ---
        if update_compressor is not None and update_compressor is score_compressor:

            reduced_grad, _ = state.process_layer_gradients(
                train_grads, val_grad, layer_idx, score_correction
            )
            hook_manager._store_compressed_grad(layer_idx, reduced_grad)

            return None, None

        # --- Select ---

        selected_indices = _do_selection(state, layer_idx, scores, similarity)


        # --- w.grad ---

        if use_stored_val:
            train_grad_output, train_input = grad_output, input
        else:
            train_grad_output, val_grad_output_raw = split_train_val_batch(grad_output, state.train_batch_size)
            train_input, val_input_raw = split_train_val_batch(input, state.train_batch_size)

        # LayerwiseWithVal: unified gradient from selected train + val in one einsum
        if (getattr(state, 'include_val_in_update', False)
                and not use_stored_val):
            return _produce_gradient_update_with_val(
                state, selected_indices,
                train_grad_output, train_input,
                val_grad_output_raw, val_input_raw, has_bias,
            )

        return _produce_gradient_update(
            hook_manager, update_compressor, state, layer_idx,
            train_grad_output, train_input, selected_indices, has_bias
        )

    @staticmethod
    def _backward_full(
        hook_manager: "GradientHook",
        update_compressor: Optional["Compressor"],
        state: Optional["LayerwiseState"],
        layer_idx: int,
        input: Tensor,
        bias: Optional[Tensor],
        grad_output: Tensor,
        capture_val_mode: bool,
        use_stored_val: bool
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Layerwise backward without score compression (full gradient scoring).

        If update_compressor is set, compresses selected gradients for MeSO.
        """
        has_bias = bias is not None

        # --- Validation capture: store full gradients ---
        if capture_val_mode:
            hook_manager.val_cache.store_layer(
                layer_idx=layer_idx,
                grad_output=grad_output.detach(),
                input=input.detach(),
                compressor=None
            )
            return None, None

        # --- No curation state: just store MeSO update if needed ---
        if state is None:
            if update_compressor is not None:
                input_aug = augment_input_for_bias(input, has_bias)
                update_compressed = update_compressor.forward((grad_output, input_aug))
                hook_manager._store_compressed_grad(layer_idx, update_compressed.sum(dim=0, keepdim=True))
            return None, None

        # --- Compute scores from full gradients ---
        if use_stored_val:
            train_grad_output, train_input = grad_output, input
            val_grad_output, val_input, val_grad_total, val_bias_grad = _get_val_components(hook_manager, layer_idx)
            if val_grad_output is None and val_grad_total is None:
                return None, None
            score_correction = None
        else:
            train_grad_output, val_grad_output = split_train_val_batch(grad_output, state.train_batch_size)
            train_input, val_input = split_train_val_batch(input, state.train_batch_size)
            val_grad_total = None
            val_bias_grad = None
            score_correction = state.score_correction

        scoring_method = getattr(state, 'scoring_method', 'reduced_ghost')
        # Request materialized G_train for direct scoring so we can reuse it
        # for w.grad without keeping train_input alive.
        include_val = (getattr(state, 'include_val_in_update', False)
                       and not use_stored_val)
        want_materialized = (scoring_method == "direct"
                             and update_compressor is None
                             and not include_val)
        scores, similarity, materialized_grads = _dispatch_scoring(
            scoring_method, train_grad_output, train_input,
            val_grad_output, val_input, val_grad_total,
            state.use_second_order,
            direct_batch_size=getattr(state, 'direct_batch_size', 0),
            return_materialized=want_materialized,
        )

        # Add bias gradient contribution to scores
        scores, similarity = _add_bias_scores(
            scores, similarity, train_grad_output,
            val_grad_output, val_bias_grad, has_bias
        )

        # Free raw activations early when we have materialized grads for w.grad.
        # Use None assignment (not del) since train_input is still passed by name below.
        if materialized_grads is not None:
            train_input = val_input = val_grad_total = val_grad_output = None

        if score_correction is not None:
            scores = scores * score_correction
            if similarity is not None:
                similarity = similarity * (score_correction ** 2)

        # --- Select, then produce gradient update ---
        selected_indices = _do_selection(state, layer_idx, scores, similarity)

        # LayerwiseWithVal: unified gradient from selected train + val in one einsum
        if include_val:
            return _produce_gradient_update_with_val(
                state, selected_indices,
                train_grad_output, train_input,
                val_grad_output, val_input, has_bias,
            )

        return _produce_gradient_update(
            hook_manager, update_compressor, state, layer_idx,
            train_grad_output, train_input, selected_indices, has_bias,
            materialized_grads=materialized_grads,
        )


class SubsetLinearBackward(Function):
    """
    Autograd Function for Subset method (global curation).

    Pass 1: Accumulates scores across all layers (no gradient output).
    Pass 2: Forward/backward on selected samples only.
            Without MeSO: hooks disabled, standard autograd gradients.
            With MeSO: hooks stay enabled, CompressedLinearBackward stores
            compressed gradients for the optimizer.
    """

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        hook_manager: "GradientHook",
        layer_idx: int
    ) -> Tensor:
        """Forward pass: standard linear transformation."""
        input_compute = input.to(weight.dtype) if input.dtype != weight.dtype else input
        ctx.save_for_backward(input_compute, weight, bias)
        ctx.hook_manager_ref = weakref.ref(hook_manager)
        ctx.layer_idx = layer_idx
        return F.linear(input_compute, weight, bias)

    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor
    ) -> Tuple[Tensor, None, None, None, None]:
        """Backward pass: accumulate scores only."""
        input, weight, bias = ctx.saved_tensors
        layer_idx = ctx.layer_idx

        hook_manager = ctx.hook_manager_ref()
        if hook_manager is None:
            raise RuntimeError("Hook manager was garbage collected before backward pass")

        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)

        grad_input = grad_output @ weight.to(grad_output.dtype)

        score_compressor = hook_manager.score_compressors[layer_idx]
        state: Optional[SubsetState] = hook_manager.selection_state

        if state is None:
            return grad_input, None, None, None, None

        use_stored_val = getattr(state, '_use_stored_val', False)

        # Use compressed scoring only when scoring_method="compress"
        use_compressed_scoring = (
            score_compressor is not None
            and getattr(state, 'scoring_method', 'reduced_ghost') == 'compress'
        )

        with torch.no_grad():
            if use_compressed_scoring:
                SubsetLinearBackward._accumulate_compressed(
                    hook_manager, score_compressor, state, layer_idx,
                    input, grad_output, bias, use_stored_val
                )
            else:
                SubsetLinearBackward._accumulate_full(
                    hook_manager, state, layer_idx,
                    input, grad_output, bias is not None, use_stored_val
                )

            # One-pass mode: retain (grad_output, input) for post-hoc gradient assembly
            if state.one_pass:
                if use_stored_val:
                    # SeparateBatch: entire batch is train
                    hook_manager.retain_layer_data(layer_idx, grad_output, input)
                else:
                    # MergedBatch: extract train portion only
                    train_go, _ = split_train_val_batch(grad_output, state.train_batch_size)
                    train_inp, _ = split_train_val_batch(input, state.train_batch_size)
                    hook_manager.retain_layer_data(layer_idx, train_go, train_inp)

        return grad_input, None, None, None, None

    @staticmethod
    def _accumulate_compressed(
        hook_manager: "GradientHook",
        compressor: "Compressor",
        state: "SubsetState",
        layer_idx: int,
        input: Tensor,
        grad_output: Tensor,
        bias: Optional[Tensor],
        use_stored_val: bool
    ) -> None:
        """Accumulate scores from compressed gradients."""
        input_aug = augment_input_for_bias(input, bias is not None)
        compressed_grad = compressor.forward((grad_output, input_aug))

        if use_stored_val:
            train_grads = compressed_grad
            val_grad = hook_manager._val_cache.get_compressed(layer_idx)
            # Cached mode: gradients already correctly scaled
            score_correction = None
        else:
            train_grads, val_grads = split_train_val_batch(compressed_grad, state.train_batch_size)
            val_grad = val_grads.sum(dim=0)  # Sum, not mean, for token-weighted semantics
            # Joint batch needs correction: T_total²/(T_train × T_val)
            score_correction = state.score_correction  # Tensor

        if val_grad is not None:
            state.process_layer_gradients(train_grads, val_grad, layer_idx, score_correction)

    @staticmethod
    def _accumulate_full(
        hook_manager: "GradientHook",
        state: "SubsetState",
        layer_idx: int,
        input: Tensor,
        grad_output: Tensor,
        has_bias: bool,
        use_stored_val: bool
    ) -> None:
        """Accumulate scores from full gradients."""
        if use_stored_val:
            train_grad_output, train_input = grad_output, input
            val_go, val_inp, val_grad_total, val_bias_grad = _get_val_components(hook_manager, layer_idx)
            if val_go is None and val_grad_total is None:
                return
            # Cached mode: gradients already correctly scaled, no correction needed
            score_correction = None
        else:
            # Joint batch mode: split merged batch
            train_grad_output, val_go = split_train_val_batch(grad_output, state.train_batch_size)
            train_input, val_inp = split_train_val_batch(input, state.train_batch_size)
            val_grad_total = None
            val_bias_grad = None
            # Joint batch needs correction: T_total²/(T_train × T_val)
            score_correction = state.score_correction  # Tensor

        # Route to scoring method based on state configuration
        scores, similarity, _ = _dispatch_scoring(
            state.scoring_method, train_grad_output, train_input,
            val_go, val_inp, val_grad_total, state.use_second_order,
            direct_batch_size=getattr(state, 'direct_batch_size', 0)
        )

        # Add bias gradient contribution to scores
        scores, similarity = _add_bias_scores(
            scores, similarity, train_grad_output,
            val_go, val_bias_grad, has_bias
        )

        # Accumulate scores using the state method (handles correction internally)
        state.accumulate_precomputed_scores(scores, similarity, score_correction)


# =============================================================================
# Non-linear layer backward for merged-batch one-pass mode
# =============================================================================

class TrainOnlyRMSNormBackward(Function):
    """
    Custom autograd Function for RMSNorm in merged-batch one-pass mode.

    Computes grad_input for the full merged batch (needed for chain rule
    propagation to earlier layers), but computes grad_weight from only the
    train slice of the batch. This prevents validation gradient leakage
    into non-linear layer parameter updates.

    Matches the HuggingFace RMSNorm forward:
        hidden = hidden.to(float32)
        variance = hidden.pow(2).mean(-1, keepdim=True)
        hidden = hidden * rsqrt(variance + eps)
        return weight * hidden.to(input_dtype)
    """

    @staticmethod
    def forward(
        ctx,
        input: "Tensor",
        weight: "Tensor",
        eps: float,
        hook_manager: "GradientHook",
    ) -> "Tensor":
        """Forward pass: standard RMSNorm."""
        input_dtype = input.dtype
        input_f32 = input.to(torch.float32)
        variance = input_f32.pow(2).mean(-1, keepdim=True)
        rsqrt_var = torch.rsqrt(variance + eps)
        normalized = input_f32 * rsqrt_var
        output = weight * normalized.to(input_dtype)

        ctx.save_for_backward(input_f32, weight.to(torch.float32), rsqrt_var, normalized)
        ctx.hook_manager_ref = weakref.ref(hook_manager)
        ctx.input_dtype = input_dtype
        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: "Tensor",
    ) -> "Tuple[Tensor, Tensor, None, None]":
        """
        Backward pass: full grad_input, train-only grad_weight.

        grad_input must cover the full merged batch for correct chain rule
        propagation. grad_weight is computed from the train slice only.
        """
        input_f32, weight_f32, rsqrt_var, normalized = ctx.saved_tensors

        hook_manager = ctx.hook_manager_ref()
        state = hook_manager.selection_state if hook_manager is not None else None
        train_bs = state.train_batch_size if state is not None else None

        D = input_f32.shape[-1]
        grad_output_f32 = grad_output.to(torch.float32)

        # grad_input: FULL batch (chain rule to earlier layers)
        # d(RMSNorm)/dx = rsqrt * (I - x_hat * x_hat^T / D) * diag(weight)
        grad_normalized = grad_output_f32 * weight_f32
        grad_input = rsqrt_var * (
            grad_normalized
            - normalized * (grad_normalized * normalized).sum(-1, keepdim=True) / D
        )
        grad_input = grad_input.to(ctx.input_dtype)

        # grad_weight: TRAIN-ONLY slice with normalization correction.
        # grad_output carries 1/batch_total_tokens from the merged-batch loss.
        # RMSNorm effectively "selects all train samples", so we rescale by
        # batch_total_tokens / train_total_tokens to match a train-only loss
        # (i.e., what you'd get from forward/backward on just the train samples).
        if train_bs is not None and train_bs < grad_output.shape[0]:
            train_go = grad_output_f32[:train_bs]
            train_norm = normalized[:train_bs]
            grad_weight = (train_go * train_norm).flatten(0, -2).sum(0)
            if (state is not None
                    and state.batch_total_tokens_tensor is not None
                    and state.train_total_tokens_tensor is not None):
                scale = state.batch_total_tokens_tensor / state.train_total_tokens_tensor
                grad_weight = grad_weight * scale
        else:
            # Fallback: full batch (separate batch mode or no state)
            grad_weight = (grad_output_f32 * normalized).flatten(0, -2).sum(0)

        return grad_input, grad_weight, None, None


class SubsetEmbeddingBackward(Function):
    """
    Autograd Function for Embedding in Subset method (global curation).

    Accumulates per-sample influence scores using gather-dot-sum
    (the embedding analogue of the reduced ghost inner product).
    In one-pass mode, retains (grad_output, input_ids) for post-hoc assembly.
    """

    @staticmethod
    def forward(
        ctx,
        input_ids: "Tensor",
        weight: "Tensor",
        hook_manager: "GradientHook",
        layer_idx: int,
        padding_idx: int,
    ) -> "Tensor":
        """Forward pass: standard embedding lookup."""
        ctx.save_for_backward(input_ids, weight)
        ctx.hook_manager_ref = weakref.ref(hook_manager)
        ctx.layer_idx = layer_idx
        ctx.padding_idx = padding_idx
        return F.embedding(input_ids, weight, padding_idx=padding_idx if padding_idx >= 0 else None)

    @staticmethod
    def backward(
        ctx,
        grad_output: "Tensor",
    ) -> "Tuple[None, None, None, None, None]":
        """Backward pass: accumulate scores, retain data. No weight gradient."""
        from .utils import (
            compute_embedding_scores,
            compute_embedding_val_gradient,
            split_train_val_batch,
        )

        input_ids, weight = ctx.saved_tensors
        layer_idx = ctx.layer_idx

        hook_manager = ctx.hook_manager_ref()
        if hook_manager is None:
            return None, None, None, None, None

        state = hook_manager.selection_state
        if state is None:
            return None, None, None, None, None

        use_stored_val = getattr(state, '_use_stored_val', False)

        with torch.no_grad():
            if use_stored_val:
                # Separate batch: entire batch is train, val grad from cache
                train_go = grad_output
                train_ids = input_ids
                val_grad_weight = hook_manager.val_cache.get_full(layer_idx)
                if val_grad_weight is None:
                    # No val data for this layer — skip scoring
                    if state.one_pass:
                        hook_manager.retain_layer_data(layer_idx, grad_output, input_ids)
                    return None, None, None, None, None
                score_correction = None
            else:
                # Merged batch: split into train/val
                train_go, val_go = split_train_val_batch(grad_output, state.train_batch_size)
                train_ids, val_ids = split_train_val_batch(input_ids, state.train_batch_size)
                # Compute val gradient by scattering val grad_output
                V, D = weight.shape
                val_grad_weight = compute_embedding_val_gradient(val_go, val_ids, V, D)
                score_correction = state.score_correction

            # Compute per-sample scores
            scores = compute_embedding_scores(train_go, train_ids, val_grad_weight)

            if score_correction is not None:
                scores = scores * score_correction

            # Accumulate into SubsetState (no similarity for embedding)
            state.accumulate_precomputed_scores(scores, None, None)

            # One-pass: retain data for post-hoc gradient assembly
            if state.one_pass:
                if use_stored_val:
                    hook_manager.retain_layer_data(layer_idx, grad_output, input_ids)
                else:
                    # Merged batch: retain train portion only
                    hook_manager.retain_layer_data(layer_idx, train_go, train_ids)

        return None, None, None, None, None


class LayerwiseEmbeddingBackward(Function):
    """
    Autograd Function for Embedding in Layerwise method (per-layer curation).

    Computes scores, selects samples, and produces the curated gradient
    for the embedding layer in a single pass.
    """

    @staticmethod
    def forward(
        ctx,
        input_ids: "Tensor",
        weight: "Tensor",
        hook_manager: "GradientHook",
        layer_idx: int,
        padding_idx: int,
    ) -> "Tensor":
        """Forward pass: standard embedding lookup."""
        ctx.save_for_backward(input_ids, weight)
        ctx.hook_manager_ref = weakref.ref(hook_manager)
        ctx.layer_idx = layer_idx
        ctx.padding_idx = padding_idx
        return F.embedding(input_ids, weight, padding_idx=padding_idx if padding_idx >= 0 else None)

    @staticmethod
    def backward(
        ctx,
        grad_output: "Tensor",
    ) -> "Tuple[None, Optional[Tensor], None, None, None]":
        """Backward pass: score, select, produce curated grad_weight."""
        from .utils import (
            compute_embedding_scores,
            compute_embedding_val_gradient,
            compute_embedding_selected_gradients,
            split_train_val_batch,
        )

        input_ids, weight = ctx.saved_tensors
        layer_idx = ctx.layer_idx
        padding_idx = ctx.padding_idx
        V, D = weight.shape

        hook_manager = ctx.hook_manager_ref()
        if hook_manager is None:
            return None, None, None, None, None

        state = hook_manager.selection_state
        capture_val_mode = hook_manager.capture_val_mode
        use_stored_val = (
            state is not None and
            getattr(state, '_use_stored_val', False)
        )

        with torch.no_grad():
            # --- Val capture mode: store embedding val gradient ---
            if capture_val_mode:
                val_grad = compute_embedding_val_gradient(grad_output, input_ids, V, D)
                hook_manager.val_cache.store_precomputed(layer_idx, val_grad)
                return None, None, None, None, None

            # --- No state: no curation ---
            if state is None:
                return None, None, None, None, None

            # --- Get val gradient and train data ---
            if use_stored_val:
                train_go = grad_output
                train_ids = input_ids
                val_grad_weight = hook_manager.val_cache.get_full(layer_idx)
                if val_grad_weight is None:
                    return None, None, None, None, None
                score_correction = None
            else:
                train_go, val_go = split_train_val_batch(grad_output, state.train_batch_size)
                train_ids, val_ids = split_train_val_batch(input_ids, state.train_batch_size)
                val_grad_weight = compute_embedding_val_gradient(val_go, val_ids, V, D)
                score_correction = state.score_correction

            # --- Compute scores ---
            scores = compute_embedding_scores(train_go, train_ids, val_grad_weight)
            if score_correction is not None:
                scores = scores * score_correction

            # --- Selection ---
            selected_indices = _do_selection(state, layer_idx, scores, None)

            # --- Gradient update for selected samples ---
            scale_factor = state._compute_scale_factor(selected_indices)
            grad_weight = compute_embedding_selected_gradients(
                train_go, train_ids, selected_indices, scale_factor,
                V, D, padding_idx,
            )
            grad_weight = grad_weight.to(weight.dtype)

        return None, grad_weight, None, None, None
