"""
Utility functions for gradient-based data curation.

This module provides low-level helper functions used by the backward autograd functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Tuple
    from torch import Tensor

import torch


def augment_input_for_bias(input: Tensor, has_bias: bool) -> Tensor:
    """
    Augment input tensor with ones column for bias gradient computation.

    When computing gradients for a linear layer with bias, we can treat
    bias as an extra weight column by appending ones to the input.

    Args:
        input: Input tensor [B, S, I] or [B, I]
        has_bias: Whether the layer has bias

    Returns:
        Augmented input [B, S, I+1] or [B, I+1] if has_bias, else unchanged input
    """
    if not has_bias:
        return input

    batch_size = input.shape[0]
    if input.dim() == 3:
        seq_length = input.shape[1]
        ones = torch.ones(batch_size, seq_length, 1, device=input.device, dtype=input.dtype)
    else:
        ones = torch.ones(batch_size, 1, device=input.device, dtype=input.dtype)
    return torch.cat([input, ones], dim=-1)


def split_train_val_batch(
    tensor: Tensor,
    train_batch_size: int
) -> Tuple[Tensor, Tensor]:
    """
    Split a merged batch tensor into train and validation portions.

    Args:
        tensor: Merged tensor with train samples first, then val samples
        train_batch_size: Number of training samples (first N in batch)

    Returns:
        (train_portion, val_portion) tuple
    """
    train_portion = tensor[:train_batch_size]
    val_portion = tensor[train_batch_size:]
    return train_portion, val_portion


def compute_total_gradient(
    grad_output: Tensor,
    input: Tensor
) -> Tensor:
    """
    Compute total (summed) gradient [O, I] from grad_output and input.

    This computes the sum of gradients across samples: Σ_b grad_b
    where grad_b[o,i] = Σ_s grad_output[b,s,o] × input[b,s,i].

    Note on scaling:
    - The grad_output already has loss function scaling (1/total_tokens for
      token-averaged loss).
    - We sum (not average) to be consistent with token-weighted loss semantics.
    - Samples with more tokens naturally contribute more through gradient magnitude.

    Args:
        grad_output: Gradient of output [B, S, O] or [B, O]
        input: Input tensor [B, S, I] or [B, I]

    Returns:
        Total gradient [O, I]
    """
    if grad_output.dim() == 3:
        return torch.einsum('bso,bsi->oi', grad_output, input)
    else:
        return torch.einsum('bo,bi->oi', grad_output, input)


@torch.compile
def compute_scores_and_similarity(
    train_grad_output: Tensor,
    train_input: Tensor,
    val_grad_output: Optional[Tensor],
    val_input: Optional[Tensor],
    val_grad_total: Optional[Tensor],
    use_second_order: bool,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Ghost inner product scoring via collapse-first approach.

    Precomputes G_val = Σ_v g_v as [O, I], then scores each train sample via:
      temp = inp @ G_val.T → [B, S, O]
      s_i = (go_i * temp_i).sum()

    FLOPs: B × S × O × I (+ V × S × O × I for G_val if factorized val).
    Memory: O(B × S × O) for temp — same size as grad_output (already in memory).
    Never materializes per-sample gradients [B, O, I].

    Scales well with V (val batch size) since G_val precomputation amortizes
    over all train samples. For small V (m=1-2), full_ghost may be faster
    due to its O(B × V × S² × (O+I)) complexity when S < min(O, I).

    Args:
        train_grad_output: Training grad_output [B, S, O] or [B, O]
        train_input: Training input [B, S, I] or [B, I]
        val_grad_output: Validation grad_output [V, S, O] or None
        val_input: Validation input [V, S, I] or None
        val_grad_total: Total validation gradient [O, I] or None
        use_second_order: Whether to compute similarity matrix

    Returns:
        (scores, similarity) tuple where similarity is None if not use_second_order
    """
    target_dtype = train_grad_output.dtype
    if val_grad_output is not None:
        val_grad_output = val_grad_output.to(target_dtype)
    if val_input is not None:
        val_input = val_input.to(target_dtype)
    if val_grad_total is not None:
        val_grad_total = val_grad_total.to(target_dtype)

    if val_grad_output is not None and val_input is not None:
        if train_grad_output.dim() == 3:
            val_grad_total = torch.einsum('vto,vti->oi', val_grad_output, val_input)
            temp = train_input @ val_grad_total.T
            scores = (train_grad_output * temp).sum(dim=(1, 2))
        else:
            val_grad_total = torch.einsum('vo,vi->oi', val_grad_output, val_input)
            temp = train_input @ val_grad_total.T
            scores = (train_grad_output * temp).sum(dim=1)
    elif val_grad_total is not None:
        if train_grad_output.dim() == 3:
            temp = train_input @ val_grad_total.T
            scores = (train_grad_output * temp).sum(dim=(1, 2))
        else:
            temp = train_input @ val_grad_total.T
            scores = (train_grad_output * temp).sum(dim=1)
    else:
        raise ValueError("Must provide either (val_grad_output, val_input) or val_grad_total")

    similarity = None
    if use_second_order:
        if train_grad_output.dim() == 3:
            contracted = torch.bmm(
                train_grad_output.permute(0, 2, 1),
                train_input
            ).flatten(start_dim=1)
            similarity = torch.matmul(contracted, contracted.T)
        else:
            dot_g = torch.matmul(train_grad_output, train_grad_output.T)
            dot_x = torch.matmul(train_input, train_input.T)
            similarity = dot_g * dot_x

    return scores, similarity


def compute_scores_full_ghost(
    train_grad_output: Tensor,
    train_input: Tensor,
    val_grad_output: Optional[Tensor],
    val_input: Optional[Tensor],
    val_grad_total: Optional[Tensor],
    use_second_order: bool,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    GREATS-style ghost inner product scoring (true ghost, no materialization).

    Uses pairwise dot products to avoid materializing per-sample gradients:
      <g_i, g_v> = Σ_{s1,s2} (go_i[s1]·go_v[s2]) × (inp_i[s1]·inp_v[s2])

    This is the ghost identity for Kronecker products applied pairwise across
    the sequence dimension. Based on the GREATS reference implementation's
    alternative branch (see utils_ghost_dot_prod.py in the GREATS repo).

    Memory:
    - **2D**: O(B × V) — Hadamard of dot products
    - **3D**: O(B × V × T²) — pairwise dot products across sequence positions.
      Better than direct materialization O(B × O × I) when V × T² < O × I,
      which holds for typical LLM settings (V=1, T=512, O=2048, I=5632).

    When only val_grad_total is available (no factored components), falls back to
    our ghost approach (precompute total, then score).

    Args:
        train_grad_output: Training grad_output [B, S, O] or [B, O]
        train_input: Training input [B, S, I] or [B, I]
        val_grad_output: Validation grad_output [V, S, O] or None
        val_input: Validation input [V, S, I] or None
        val_grad_total: Total validation gradient [O, I] or None
        use_second_order: Whether to compute similarity matrix

    Returns:
        (scores, similarity) tuple where similarity is None if not use_second_order
    """
    target_dtype = train_grad_output.dtype
    if val_grad_output is not None:
        val_grad_output = val_grad_output.to(target_dtype)
    if val_input is not None:
        val_input = val_input.to(target_dtype)
    if val_grad_total is not None:
        val_grad_total = val_grad_total.to(target_dtype)

    if train_grad_output.dim() == 3:
        if val_grad_output is not None and val_input is not None:
            # 3D ghost via pairwise dot products across sequence positions:
            # <g_i, g_v> = Σ_{s1,s2} (go_i[s1]·go_v[s2]) * (inp_i[s1]·inp_v[s2])
            #
            # go_dotprod[i,v,s1,s2] = go_i[s1,:] · go_v[s2,:]
            # inp_dotprod[i,v,s1,s2] = inp_i[s1,:] · inp_v[s2,:]
            # score[i,v] = Σ_{s1,s2} go_dotprod * inp_dotprod
            #
            # Computed efficiently via einsum:
            # go_dotprod = train_go @ val_go^T → [B, S, V, S'] via einsum
            # But we use matmul with broadcasting for better perf.

            B, S, O = train_grad_output.shape
            V = val_grad_output.shape[0]
            I = train_input.shape[2]

            # [B, S, O] @ [V, O, S]^T → use einsum for clarity
            # go_dot[i, s1, v, s2] = go_i[s1,:] · go_v[s2,:]
            go_dot = torch.einsum('bso,vto->bvst', train_grad_output, val_grad_output)  # [B, V, S, S]
            inp_dot = torch.einsum('bsi,vti->bvst', train_input, val_input)               # [B, V, S, S]
            # score[i,v] = Σ_{s1,s2} go_dot * inp_dot, then sum over v
            scores = (go_dot * inp_dot).sum(dim=(2, 3)).sum(dim=1)  # [B]
            del go_dot, inp_dot
        elif val_grad_total is not None:
            # Fall back to ghost's collapse approach — pairwise not possible
            # without factorized val components. This happens in SeparateBatch
            # mode with use_factorized=False (e.g., RLHF with large val batch).
            temp = train_input @ val_grad_total.T  # [B, S, O]
            scores = (train_grad_output * temp).sum(dim=(1, 2))  # [B]
        else:
            raise ValueError("Must provide either (val_grad_output, val_input) or val_grad_total")

        # Similarity: <g_i, g_j> via same pairwise identity
        similarity = None
        if use_second_order:
            # For train-train similarity, reuse the contracted form:
            # g_i = Σ_s go_i[s] ⊗ inp_i[s], flatten to [O*I]
            contracted = torch.bmm(
                train_grad_output.permute(0, 2, 1), train_input
            ).flatten(start_dim=1)  # [B, O*I]
            similarity = torch.matmul(contracted, contracted.T)
    else:
        # 2D case: Hadamard of dot products (true ghost, efficient)
        if val_grad_output is not None and val_input is not None:
            dot_go = torch.matmul(train_grad_output, val_grad_output.T)  # [B, V]
            dot_inp = torch.matmul(train_input, val_input.T)             # [B, V]
            scores = (dot_go * dot_inp).sum(dim=1)                       # [B]
        elif val_grad_total is not None:
            temp = train_input @ val_grad_total.T
            scores = (train_grad_output * temp).sum(dim=1)
        else:
            raise ValueError("Must provide either (val_grad_output, val_input) or val_grad_total")

        similarity = None
        if use_second_order:
            dot_g = torch.matmul(train_grad_output, train_grad_output.T)
            dot_x = torch.matmul(train_input, train_input.T)
            similarity = dot_g * dot_x

    return scores, similarity


def compute_scores_direct_materialization(
    train_grad_output: Tensor,
    train_input: Tensor,
    val_grad_output: Optional[Tensor],
    val_input: Optional[Tensor],
    val_grad_total: Optional[Tensor],
    use_second_order: bool,
    batch_size: int = 0,
    return_materialized: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """
    Score computation via batch materialization of per-sample gradients.

    Materializes per-sample weight gradients and computes scores:
      G_train[i] = bmm(go_i^T, inp_i) → [O, I], flattened to [O*I]
      s = G_train @ G_val.flatten()

    When batch_size > 0, processes samples in chunks to reduce peak memory
    from O(B × O × I) to O(batch_size × O × I). This enables direct scoring
    at long sequence lengths where the full materialization would OOM.

    Args:
        train_grad_output: Training grad_output [B, S, O] or [B, O]
        train_input: Training input [B, S, I] or [B, I]
        val_grad_output: Validation grad_output [V, S, O] or None
        val_input: Validation input [V, S, I] or None
        val_grad_total: Total validation gradient [O, I] or None
        use_second_order: Whether to compute similarity matrix
        batch_size: Chunk size for batched processing. 0 = all at once.
        return_materialized: If True, return G_train [B, O*I] so the caller
            can reuse it for w.grad (avoids re-materializing from go/inp).

    Returns:
        (scores, similarity, G_train) tuple where similarity is None if not
        use_second_order, and G_train is None unless return_materialized=True
        with full (non-chunked) materialization.
    """
    target_dtype = train_grad_output.dtype
    if val_grad_output is not None:
        val_grad_output = val_grad_output.to(target_dtype)
    if val_input is not None:
        val_input = val_input.to(target_dtype)
    if val_grad_total is not None:
        val_grad_total = val_grad_total.to(target_dtype)

    # Step 1: Compute validation gradient total [O, I]
    if val_grad_total is None:
        if val_grad_output is not None and val_input is not None:
            if val_grad_output.dim() == 3:
                val_grad_total = torch.einsum('vso,vsi->oi', val_grad_output, val_input)
            else:
                val_grad_total = torch.einsum('vo,vi->oi', val_grad_output, val_input)
        else:
            raise ValueError("Must provide either (val_grad_output, val_input) or val_grad_total")

    B = train_grad_output.shape[0]
    val_flat = val_grad_total.flatten()  # [O*I]

    G_out = None  # optionally returned for w.grad reuse

    if batch_size <= 0 or batch_size >= B:
        # Full materialization: all B samples at once
        G_train = _materialize_gradients(train_grad_output, train_input)
        scores = torch.matmul(G_train, val_flat)
        similarity = torch.matmul(G_train, G_train.T) if use_second_order else None
        if return_materialized:
            G_out = G_train
    else:
        # Chunked: materialize batch_size samples at a time
        scores = torch.empty(B, device=train_grad_output.device, dtype=target_dtype)
        if use_second_order:
            # Need full G_train for similarity — fall back to sequential accumulation
            chunks = []
            for start in range(0, B, batch_size):
                end = min(start + batch_size, B)
                G_chunk = _materialize_gradients(
                    train_grad_output[start:end], train_input[start:end]
                )
                scores[start:end] = torch.matmul(G_chunk, val_flat)
                chunks.append(G_chunk)
            G_train = torch.cat(chunks, dim=0)
            similarity = torch.matmul(G_train, G_train.T)
            if return_materialized:
                G_out = G_train
        else:
            similarity = None
            for start in range(0, B, batch_size):
                end = min(start + batch_size, B)
                G_chunk = _materialize_gradients(
                    train_grad_output[start:end], train_input[start:end]
                )
                scores[start:end] = torch.matmul(G_chunk, val_flat)
                # G_chunk is freed here — peak memory is batch_size × O × I

    return scores, similarity, G_out


def _materialize_gradients(grad_output: Tensor, inp: Tensor) -> Tensor:
    """Materialize per-sample weight gradients: G[i] = go_i^T @ inp_i → [B, O*I]."""
    if grad_output.dim() == 3:
        return torch.bmm(
            grad_output.permute(0, 2, 1),  # [B, O, S]
            inp                             # [B, S, I]
        ).flatten(start_dim=1)              # [B, O*I]
    else:
        return (grad_output.unsqueeze(2) * inp.unsqueeze(1)).flatten(start_dim=1)


def compute_selected_gradients(
    train_grad_output: Tensor,
    train_input: Tensor,
    selected_indices: Tensor,
    has_bias: bool,
    scale_factor: Tensor
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Compute aggregated gradients for selected samples.

    Args:
        train_grad_output: Training grad_output [B, S, O] or [B, O]
        train_input: Training input [B, S, I] or [B, I]
        selected_indices: Indices of selected samples [K]
        has_bias: Whether to compute bias gradient
        scale_factor: Scaling factor to normalize for curation (scalar Tensor)

    Returns:
        (grad_weight, grad_bias) tuple where grad_bias is None if not has_bias
    """
    selected_grad_output = train_grad_output[selected_indices]
    selected_input = train_input[selected_indices]

    if selected_grad_output.dim() == 3:
        # 3D case: [K, S, O] x [K, S, I] -> [O, I]
        grad_weight = torch.einsum('kso,ksi->oi', selected_grad_output, selected_input) * scale_factor
        grad_bias = selected_grad_output.sum(dim=(0, 1)) * scale_factor if has_bias else None
    else:
        # 2D case: [K, O] x [K, I] -> [O, I]
        grad_weight = torch.einsum('ko,ki->oi', selected_grad_output, selected_input) * scale_factor
        grad_bias = selected_grad_output.sum(dim=0) * scale_factor if has_bias else None

    return grad_weight, grad_bias


# =============================================================================
# Embedding-specific utilities
# =============================================================================

def compute_embedding_scores(
    train_go: Tensor,
    train_ids: Tensor,
    val_grad_weight: Tensor,
) -> Tensor:
    """
    Compute per-sample influence scores for an Embedding layer.

    score_b = Σ_s train_go[b, s, :] · val_grad_weight[train_ids[b, s], :]

    This is the embedding analogue of the reduced ghost inner product — we
    never materialize the full per-sample [V, D] gradient.

    Args:
        train_go: Training grad_output [B, S, D] or [B, D]
        train_ids: Training token IDs [B, S] or [B]
        val_grad_weight: Validation embedding gradient [V, D]

    Returns:
        scores: Per-sample scores [B]
    """
    # Gather val gradient rows for each position
    val_at_positions = val_grad_weight[train_ids]  # [B, S, D] or [B, D]
    # Per-sample score = sum of elementwise products across position and dim
    if train_go.dim() == 3:
        scores = (train_go * val_at_positions).sum(dim=(-1, -2))  # [B]
    else:
        scores = (train_go * val_at_positions).sum(dim=-1)  # [B]
    return scores


def compute_embedding_val_gradient(
    grad_output: Tensor,
    input_ids: Tensor,
    num_embeddings: int,
    dim: int,
) -> Tensor:
    """
    Compute the total embedding gradient via scatter.

    grad_weight[v] = Σ_{(b,s): input_ids[b,s]=v} grad_output[b, s, :]

    Args:
        grad_output: [B, S, D] or [B, D]
        input_ids: [B, S] or [B]
        num_embeddings: Vocabulary size V
        dim: Embedding dimension D

    Returns:
        grad_weight: [V, D]
    """
    grad_weight = torch.zeros(num_embeddings, dim, device=grad_output.device, dtype=grad_output.dtype)
    grad_weight.index_add_(0, input_ids.reshape(-1), grad_output.reshape(-1, dim))
    return grad_weight


def compute_embedding_selected_gradients(
    grad_output: Tensor,
    input_ids: Tensor,
    selected_indices: Tensor,
    scale_factor: Tensor,
    num_embeddings: int,
    dim: int,
    padding_idx: int = -1,
) -> Tensor:
    """
    Compute embedding gradient from selected samples only.

    Args:
        grad_output: [B, S, D] or [B, D]
        input_ids: [B, S] or [B]
        selected_indices: [K] indices of selected samples
        scale_factor: Scalar scaling tensor
        num_embeddings: Vocabulary size V
        dim: Embedding dimension D
        padding_idx: Padding index to zero out (-1 for none)

    Returns:
        grad_weight: [V, D]
    """
    sel_go = grad_output[selected_indices]
    sel_ids = input_ids[selected_indices]

    grad_weight = torch.zeros(num_embeddings, dim, device=grad_output.device, dtype=grad_output.dtype)
    grad_weight.index_add_(0, sel_ids.reshape(-1), sel_go.reshape(-1, dim))
    grad_weight.mul_(scale_factor)

    if padding_idx >= 0:
        grad_weight[padding_idx].zero_()

    return grad_weight
