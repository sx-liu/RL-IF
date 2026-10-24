"""
Pure gradient computation functions for data selection.

This module provides stateless, reusable functions for:
- Score computation (compressed and full gradient modes)
- Gradient aggregation for selected samples
- Validation gradient capture

These functions are used by the backward autograd functions but can also
be tested in isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ..compressor import Compressor

from .utils import augment_input_for_bias


# =============================================================================
# Score Computation Functions
# =============================================================================

def compute_scores_compressed(
    compressor: "Compressor",
    grad_output: Tensor,
    input: Tensor,
    val_grad_compressed: Tensor,
    has_bias: bool,
) -> Tensor:
    """
    Compute selection scores using compressed gradients.

    Scores are computed as: score[b] = <compressed_grad[b], val_grad_compressed>

    Args:
        compressor: Compressor for gradient compression.
        grad_output: Training grad_output [B, S, O] or [B, O].
        input: Training input [B, S, I] or [B, I].
        val_grad_compressed: Compressed validation gradient [k].
        has_bias: Whether to augment input for bias.

    Returns:
        Per-sample scores [B].
    """
    input_aug = augment_input_for_bias(input, has_bias)
    train_compressed = compressor.forward((grad_output, input_aug))
    # Score = dot product with validation gradient
    return train_compressed @ val_grad_compressed


def compute_similarity_compressed(
    train_compressed: Tensor,
) -> Tensor:
    """
    Compute similarity matrix from compressed gradients.

    Similarity[i,j] = <compressed_grad[i], compressed_grad[j]>

    Args:
        train_compressed: Compressed training gradients [B, k].

    Returns:
        Similarity matrix [B, B].
    """
    return train_compressed @ train_compressed.T


def compute_scores_factorized(
    grad_output: Tensor,
    input: Tensor,
    val_grad_output: Tensor,
    val_input: Tensor,
) -> Tensor:
    """
    Compute selection scores using factorized full gradients.

    This computes scores without materializing the full [B, O, I] gradient tensor.
    Uses the Ghost Inner Product optimization:
    score[b] = <grad_b, val_grad> = Σ_s Σ_o Σ_i grad_output[b,s,o] × input[b,s,i] × val_grad[o,i]

    Args:
        grad_output: Training grad_output [B, S, O] or [B, O].
        input: Training input [B, S, I] or [B, I].
        val_grad_output: Validation grad_output [V, S, O] or [V, O].
        val_input: Validation input [V, S, I] or [V, I].

    Returns:
        Per-sample scores [B].
    """
    # First compute total validation gradient
    if val_grad_output.dim() == 3:
        val_grad_total = torch.einsum('vto,vti->oi', val_grad_output, val_input)
    else:
        val_grad_total = torch.einsum('vo,vi->oi', val_grad_output, val_input)

    return compute_scores_with_total_grad(grad_output, input, val_grad_total)


def compute_scores_with_total_grad(
    grad_output: Tensor,
    input: Tensor,
    val_grad_total: Tensor,
) -> Tensor:
    """
    Compute selection scores using precomputed total validation gradient.

    This is the efficient path when validation gradient is already summed.

    Args:
        grad_output: Training grad_output [B, S, O] or [B, O].
        input: Training input [B, S, I] or [B, I].
        val_grad_total: Total validation gradient [O, I].

    Returns:
        Per-sample scores [B].
    """
    # Ghost inner product: score[b] = Σ (grad_output × (input @ val_grad.T))
    if grad_output.dim() == 3:
        temp = input @ val_grad_total.T  # [B, S, O]
        scores = (grad_output * temp).sum(dim=(1, 2))  # [B]
    else:
        temp = input @ val_grad_total.T  # [B, O]
        scores = (grad_output * temp).sum(dim=1)  # [B]

    return scores


@torch.compile
def compute_similarity_factorized(
    grad_output: Tensor,
    input: Tensor,
) -> Tensor:
    """
    Compute similarity matrix from factorized gradients.

    Similarity[i,j] = <grad_i, grad_j> without materializing [B, O, I].

    Args:
        grad_output: Training grad_output [B, S, O] or [B, O].
        input: Training input [B, S, I] or [B, I].

    Returns:
        Similarity matrix [B, B].
    """
    if grad_output.dim() == 3:
        # 3D: contract over sequence and features
        # grad_i = Σ_s grad_output[i,s,:] ⊗ input[i,s,:]
        # We need <grad_i, grad_j> = Σ_s1,s2 (go_i[s1,:] · go_j[s2,:]) × (in_i[s1,:] · in_j[s2,:])
        contracted = torch.bmm(
            grad_output.permute(0, 2, 1),  # [B, O, S]
            input  # [B, S, I]
        ).flatten(start_dim=1)  # [B, O*I]
        similarity = torch.matmul(contracted, contracted.T)
    else:
        # 2D: Use Hadamard product of dot products
        # <grad_i, grad_j> = (go_i · go_j) × (in_i · in_j)
        dot_g = torch.matmul(grad_output, grad_output.T)
        dot_x = torch.matmul(input, input.T)
        similarity = dot_g * dot_x

    return similarity


# =============================================================================
# Gradient Aggregation Functions
# =============================================================================

def compute_selected_full_gradients(
    grad_output: Tensor,
    input: Tensor,
    selected_indices: Tensor,
    has_bias: bool,
    scale_factor: Tensor,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Compute aggregated full gradients for selected samples.

    Args:
        grad_output: Training grad_output [B, S, O] or [B, O].
        input: Training input [B, S, I] or [B, I].
        selected_indices: Indices of selected samples [K].
        has_bias: Whether to compute bias gradient.
        scale_factor: Scaling factor to normalize for selection.

    Returns:
        (grad_weight, grad_bias) tuple where grad_bias is None if not has_bias.
    """
    selected_grad_output = grad_output[selected_indices]
    selected_input = input[selected_indices]

    if selected_grad_output.dim() == 3:
        # 3D case: [K, S, O] x [K, S, I] -> [O, I]
        grad_weight = torch.einsum(
            'kso,ksi->oi',
            selected_grad_output,
            selected_input
        ) * scale_factor

        grad_bias = None
        if has_bias:
            grad_bias = selected_grad_output.sum(dim=(0, 1)) * scale_factor
    else:
        # 2D case: [K, O] x [K, I] -> [O, I]
        grad_weight = torch.einsum(
            'ko,ki->oi',
            selected_grad_output,
            selected_input
        ) * scale_factor

        grad_bias = None
        if has_bias:
            grad_bias = selected_grad_output.sum(dim=0) * scale_factor

    return grad_weight, grad_bias


def compute_selected_compressed_gradients(
    compressor: "Compressor",
    grad_output: Tensor,
    input: Tensor,
    selected_indices: Tensor,
    has_bias: bool,
    scale_factor: Tensor,
) -> Tensor:
    """
    Compute aggregated compressed gradients for selected samples.

    Args:
        compressor: Compressor for gradient compression.
        grad_output: Training grad_output [B, S, O] or [B, O].
        input: Training input [B, S, I] or [B, I].
        selected_indices: Indices of selected samples [K].
        has_bias: Whether to augment input for bias.
        scale_factor: Scaling factor to normalize for selection.

    Returns:
        Aggregated compressed gradient [1, k].
    """
    selected_grad_output = grad_output[selected_indices]
    selected_input = input[selected_indices]

    input_aug = augment_input_for_bias(selected_input, has_bias)
    compressed = compressor.forward((selected_grad_output, input_aug))

    # Sum and scale
    return compressed.sum(dim=0, keepdim=True) * scale_factor


def compute_all_compressed_gradients(
    compressor: "Compressor",
    grad_output: Tensor,
    input: Tensor,
    has_bias: bool,
) -> Tensor:
    """
    Compute compressed gradients for all samples (no selection).

    Args:
        compressor: Compressor for gradient compression.
        grad_output: Training grad_output [B, S, O] or [B, O].
        input: Training input [B, S, I] or [B, I].
        has_bias: Whether to augment input for bias.

    Returns:
        Aggregated compressed gradient [1, k].
    """
    input_aug = augment_input_for_bias(input, has_bias)
    compressed = compressor.forward((grad_output, input_aug))
    return compressed.sum(dim=0, keepdim=True)


# =============================================================================
# Validation Gradient Capture Functions
# =============================================================================

def capture_val_grad_factorized(
    grad_output: Tensor,
    input: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Capture validation gradients in factorized form.

    Args:
        grad_output: Validation grad_output [V, S, O] or [V, O].
        input: Validation input [V, S, I] or [V, I].

    Returns:
        (grad_output, input) tuple with detached tensors.
    """
    return grad_output.detach(), input.detach()


def capture_val_grad_full(
    grad_output: Tensor,
    input: Tensor,
) -> Tensor:
    """
    Capture validation gradients as total gradient [O, I].

    Args:
        grad_output: Validation grad_output [V, S, O] or [V, O].
        input: Validation input [V, S, I] or [V, I].

    Returns:
        Total validation gradient [O, I].
    """
    if grad_output.dim() == 3:
        return torch.einsum('vso,vsi->oi', grad_output, input)
    else:
        return torch.einsum('vo,vi->oi', grad_output, input)


def capture_val_grad_compressed(
    compressor: "Compressor",
    grad_output: Tensor,
    input: Tensor,
    has_bias: bool,
) -> Tensor:
    """
    Capture validation gradients in compressed form.

    Args:
        compressor: Compressor for gradient compression.
        grad_output: Validation grad_output [V, S, O] or [V, O].
        input: Validation input [V, S, I] or [V, I].
        has_bias: Whether to augment input for bias.

    Returns:
        Compressed validation gradient [k].
    """
    input_aug = augment_input_for_bias(input, has_bias)
    compressed = compressor.forward((grad_output, input_aug))
    # Sum over batch to get total compressed gradient
    return compressed.sum(dim=0)
