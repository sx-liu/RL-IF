"""
Curation strategies for gradient-based data curation in trainers.

This module provides two families of strategies based on how validation
gradients are obtained:

1. **MergedBatch Strategies**:
   - Train and val samples are merged into a single batch
   - Val gradients computed during the same forward/backward pass
   - Factory: create_merged_batch_strategy()
   - Note: Has padding overhead when val/train have different sequence lengths

2. **SeparateBatch Strategies**:
   - Val gradients are pre-captured and cached before training
   - Training uses cached val gradients for curation scoring
   - Factory: create_separate_batch_strategy()
   - Val storage mode is derived from scoring_method in start_val_capture():
       * reduced_ghost/direct: Stores total gradient [O, I] per layer.
       Better when validation batch is large (e.g., self-reference validation in RLHF).
       * full_ghost: Stores [V, S, O] and [V, S, I] components (for pairwise scoring).
       More memory-efficient during training as it avoids materializing [B_train, O, I].
       Better when validation batch is small (e.g., external validation set in SFT).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, Optional, Callable, Tuple
    from ..hook import GradientHook
    from torch import Tensor

import torch
import torch.nn as nn
from .state import LayerwiseState, SubsetState

logger = logging.getLogger(__name__)


# ============================================================
# JOINT BATCH STRATEGIES
# Val gradients computed from merged train+val batch
# ============================================================

class MergedBatchStrategy(ABC):
    """
    Abstract strategy for merged-batch data curation.

    Used when train and val samples are merged into a single batch,
    and val gradients are computed during the same forward/backward pass.

    Note: Has padding overhead when val/train have different sequence lengths.
    """

    def __init__(
        self,
        grad_hook: Optional[GradientHook],
        frac: float,
        use_second_order: bool = False,
        selection_mode: str = "topk",
        record_selections: bool = False,
        scoring_method: str = "reduced_ghost",
    ):
        """
        Initialize curation strategy.

        Args:
            grad_hook: GradientHook instance (can be None for NoSelection)
            frac: Curation fraction / filter fraction
            use_second_order: Use greedy curation with second-order
            selection_mode: "topk" (select top frac) or "filtering" (drop bottom frac of negative)
            record_selections: If True, record curation data for case study analysis
            scoring_method: Scoring method for influence scores ("reduced_ghost", "full_ghost", "direct")
        """
        self.grad_hook = grad_hook
        self.frac = frac
        self.use_second_order = use_second_order
        self.selection_mode = selection_mode
        self.record_selections = record_selections
        self.scoring_method = scoring_method
        self.last_selection_record = None

    @property
    def has_update_compression(self) -> bool:
        """Check if update compression (MeSO) is enabled.

        When True, hooks stay enabled during Subset pass 2 so that
        CompressedLinearBackward stores compressed gradients for MeSO.
        """
        if self.grad_hook is None:
            return False
        return self.grad_hook.compression_mode.uses_compressed_updates

    def _extract_selection_records(self):
        """Extract curation records from state before cleanup."""
        if self.grad_hook is None or self.grad_hook.selection_state is None:
            self.last_selection_record = None
            return
        state = self.grad_hook.selection_state
        if state._record_selections and state._selection_records:
            self.last_selection_record = list(state._selection_records)
        else:
            self.last_selection_record = None

    @abstractmethod
    def execute_training_step(
        self,
        model: nn.Module,
        merged_batch: Dict[str, Tensor],
        train_batch_size: int,
        compute_loss_fn: Callable[[nn.Module, Dict[str, Tensor]], Tensor],
        **kwargs
    ) -> Tensor:
        """
        Execute a complete training step with curation.

        Args:
            model: The model to train
            merged_batch: Merged train+val batch
            train_batch_size: Number of train samples in merged batch
            compute_loss_fn: Function to compute loss
            **kwargs: Additional arguments (lr, batch_train, etc.)

        Returns:
            Detached loss tensor
        """
        pass


class MergedBatchNoSelectionStrategy(MergedBatchStrategy):
    """
    Baseline strategy: no data curation, standard training.

    Uses all training samples without curation.
    """

    def execute_training_step(
        self,
        model: nn.Module,
        merged_batch: Dict[str, Tensor],
        train_batch_size: int,
        compute_loss_fn: Callable[[nn.Module, Dict[str, Tensor]], Tensor],
        **kwargs
    ) -> Tensor:
        """Standard training step without curation."""
        # Disable hooks if present (use standard gradient computation)
        if self.grad_hook is not None and not self.has_update_compression:
            self.grad_hook.disable_hooks()

        # Extract train-only portion (no need for val in baseline)
        train_batch = {k: v[:train_batch_size] for k, v in merged_batch.items()}

        model.zero_grad()
        loss = compute_loss_fn(model, train_batch)
        loss.backward()

        # Re-enable hooks
        if self.grad_hook is not None and not self.has_update_compression:
            self.grad_hook.enable_hooks()

        return loss.detach()


class MergedBatchLayerwiseStrategy(MergedBatchStrategy):
    """
    Layer-wise strategy with merged batch: single-pass, per-layer curation.

    Curation and gradient aggregation happen layer-by-layer
    during the backward pass.
    """

    def execute_training_step(
        self,
        model: nn.Module,
        merged_batch: Dict[str, Tensor],
        train_batch_size: int,
        compute_loss_fn: Callable[[nn.Module, Dict[str, Tensor]], Tensor],
        **kwargs
    ) -> Tensor:
        """Training step with per-layer curation."""
        lr = kwargs.get('lr', 1e-4)

        # Set up streaming state
        self._setup_state(train_batch_size, lr)

        # Set token counts for proper gradient scaling
        if 'labels' in merged_batch:
            self.grad_hook.set_token_counts(merged_batch['labels'], train_batch_size)

        model.zero_grad()
        loss = compute_loss_fn(model, merged_batch)
        loss.backward()  # Per-layer curation happens in backward hooks

        # Extract curation records before cleanup
        self._extract_selection_records()

        # Cleanup
        self._cleanup()

        return loss.detach()

    def _setup_state(self, train_batch_size: int, lr: float) -> None:
        """Set up LayerwiseState for this step."""
        dtype = next(self.grad_hook.model.parameters()).dtype

        state = LayerwiseState(
            train_batch_size=train_batch_size,
            num_layers=len(self.grad_hook.layer_names),
            frac=self.frac,
            lr=lr,
            device=self.grad_hook.device,
            dtype=dtype,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            scoring_method=self.scoring_method,
        )

        self.grad_hook.selection_state = state

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()
        self.grad_hook.clear_token_counts()


class MergedBatchLayerwiseWithValStrategy(MergedBatchLayerwiseStrategy):
    """
    Layer-wise strategy with merged batch that also trains on validation data.

    Same as MergedBatchLayerwiseStrategy but includes validation gradients
    in the parameter update, effectively training on both selected train
    samples and validation samples.
    """

    def _setup_state(self, train_batch_size: int, lr: float) -> None:
        """Set up LayerwiseState with include_val_in_update=True."""
        dtype = next(self.grad_hook.model.parameters()).dtype

        state = LayerwiseState(
            train_batch_size=train_batch_size,
            num_layers=len(self.grad_hook.layer_names),
            frac=self.frac,
            lr=lr,
            device=self.grad_hook.device,
            dtype=dtype,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            scoring_method=self.scoring_method,
            include_val_in_update=True,
        )

        self.grad_hook.selection_state = state


class MergedBatchSubsetStrategy(MergedBatchStrategy):
    """
    Subset strategy with merged batch: two-pass, global curation.

    Pass 1: Compute curation scores across all layers
    Pass 2: Forward/backward only on globally selected samples
    """

    def execute_training_step(
        self,
        model: nn.Module,
        merged_batch: Dict[str, Tensor],
        train_batch_size: int,
        compute_loss_fn: Callable[[nn.Module, Dict[str, Tensor]], Tensor],
        **kwargs
    ) -> Tensor:
        """Training step with global curation."""
        lr = kwargs.get('lr', 1e-4)
        batch_train = kwargs.get('batch_train')  # Original train batch for pass 2

        # === PASS 1: Score Accumulation ===
        self._setup_state(train_batch_size, lr)

        if 'labels' in merged_batch:
            self.grad_hook.set_token_counts(merged_batch['labels'], train_batch_size)

        model.zero_grad()
        loss_for_scoring = compute_loss_fn(model, merged_batch)
        loss_for_scoring.backward()

        # Get globally selected indices
        state: SubsetState = self.grad_hook.selection_state
        selected_indices = state.get_final_selection()
        # Sort indices for sequential memory access (better cache locality)
        selected_indices = selected_indices.sort()[0]
        n_selected = len(selected_indices)

        # Extract curation records before cleanup
        self._extract_selection_records()

        self._cleanup()

        # Handle empty curation: skip pass 2 and return zero loss
        if n_selected == 0:
            # Re-enable hooks for next step
            if not self.has_update_compression:
                self.grad_hook.enable_hooks()
            else:
                self.grad_hook.clear_token_counts()

            import torch
            zero_loss = torch.tensor(0.0, device=next(model.parameters()).device)
            return zero_loss

        # === PASS 2: Gradient Computation on Selected ===
        if batch_train is None:
            # Fall back to extracting from merged batch
            batch_train = {k: v[:train_batch_size] for k, v in merged_batch.items()}

        filtered_inputs = {
            'input_ids': batch_train['input_ids'][selected_indices],
            'attention_mask': batch_train['attention_mask'][selected_indices],
            'labels': batch_train['labels'][selected_indices]
        }

        # For pass 2, disable hooks if no compression (we want full gradients for selected samples)
        if not self.has_update_compression:
            self.grad_hook.disable_hooks()
        else:
            # For MeSO, set token counts for selected batch
            self.grad_hook.set_token_counts(filtered_inputs['labels'])

        model.zero_grad()
        loss = compute_loss_fn(model, filtered_inputs)
        loss.backward()

        # Re-enable hooks / clear token counts
        if not self.has_update_compression:
            self.grad_hook.enable_hooks()
        else:
            self.grad_hook.clear_token_counts()

        return loss.detach()

    def _setup_state(self, train_batch_size: int, lr: float) -> None:
        """Set up SubsetState for this step."""
        dtype = next(self.grad_hook.model.parameters()).dtype

        state = SubsetState(
            train_batch_size=train_batch_size,
            num_layers=len(self.grad_hook.layer_names),
            frac=self.frac,
            lr=lr,
            device=self.grad_hook.device,
            dtype=dtype,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            scoring_method=self.scoring_method,
        )

        self.grad_hook.selection_state = state

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()
        self.grad_hook.clear_token_counts()


class MergedBatchSubsetOnePassStrategy(MergedBatchStrategy):
    """
    One-pass subset strategy with merged batch (Algorithm 4.2).

    Single forward+backward pass: scoring and data retention happen during backward,
    then post-hoc gradient assembly from retained (grad_output, input) per layer.
    Saves the second forward+backward at the cost of higher peak memory.

    Non-linear layers (RMSNorm) are wrapped with TrainOnlyRMSNormBackward so
    their grad_weight is computed from the train slice only, preventing
    validation gradient leakage.
    """

    def execute_training_step(
        self,
        model: nn.Module,
        merged_batch: Dict[str, Tensor],
        train_batch_size: int,
        compute_loss_fn: Callable[[nn.Module, Dict[str, Tensor]], Tensor],
        **kwargs
    ) -> Tensor:
        """Training step with one-pass global curation."""
        lr = kwargs.get('lr', 1e-4)

        # Set up SubsetState with one_pass=True
        self._setup_state(train_batch_size, lr)

        if 'labels' in merged_batch:
            self.grad_hook.set_token_counts(merged_batch['labels'], train_batch_size)

        model.zero_grad()
        loss = compute_loss_fn(model, merged_batch)
        loss.backward()
        # After backward:
        # - Scores accumulated in state, layer data retained in hook
        # - Hooked linear layers: .grad is None (SubsetLinearBackward returns None)
        # - RMSNorm layers: .grad contains train-only gradient (TrainOnlyRMSNormBackward)

        # Get globally selected indices
        state: SubsetState = self.grad_hook.selection_state
        selected_indices = state.get_final_selection()
        selected_indices = selected_indices.sort()[0]
        n_selected = len(selected_indices)

        # Extract curation records before cleanup
        self._extract_selection_records()

        if n_selected == 0:
            self.grad_hook.clear_retained_data()
            self._cleanup()
            import torch
            return torch.tensor(0.0, device=next(model.parameters()).device)

        # Compute scale factor for exact parity with two-pass
        scale_factor = state._compute_scale_factor_for_assembly(selected_indices)

        # Post-hoc gradient assembly for linear layers.
        # No model.zero_grad() here — non-linear layers retain their train-only
        # gradients from backward, and hooked linear layers have None grad
        # (SubsetLinearBackward suppresses weight/bias grads).
        self.grad_hook.assemble_gradients_from_retained(selected_indices, scale_factor)

        self._cleanup()
        return loss.detach()

    def _setup_state(self, train_batch_size: int, lr: float) -> None:
        """Set up SubsetState with one_pass=True."""
        dtype = next(self.grad_hook.model.parameters()).dtype

        state = SubsetState(
            train_batch_size=train_batch_size,
            num_layers=len(self.grad_hook.layer_names),
            frac=self.frac,
            lr=lr,
            device=self.grad_hook.device,
            dtype=dtype,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            scoring_method=self.scoring_method,
            one_pass=True,
        )

        self.grad_hook.selection_state = state

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()
        self.grad_hook.clear_token_counts()
        self.grad_hook.clear_retained_data()  # Safety net


def create_merged_batch_strategy(
    method: str,
    grad_hook: Optional[GradientHook],
    frac: float = 0.5,
    use_second_order: bool = False,
    selection_mode: str = "topk",
    record_selections: bool = False,
    scoring_method: str = "reduced_ghost",
    subset_mode: str = "one_pass",
) -> MergedBatchStrategy:
    """
    Factory function to create merged-batch curation strategy.

    Note: Has padding overhead when val/train have different sequence lengths.

    Args:
        method: Curation method ("NA", "Layerwise", "Subset")
        grad_hook: GradientHook instance
        frac: Curation/filter fraction
        use_second_order: Use greedy curation with second-order
        selection_mode: "topk" (select top frac) or "filtering" (drop bottom frac of negative)
        record_selections: If True, record curation data for case study analysis
        scoring_method: Scoring method ("reduced_ghost", "full_ghost", "direct", "compress")
        subset_mode: For Subset method: "one_pass" (Algorithm 4.2) or "two_pass" (Algorithm 4.3)

    Returns:
        Appropriate MergedBatchStrategy instance
    """
    kwargs = dict(grad_hook=grad_hook, frac=frac, use_second_order=use_second_order,
                  selection_mode=selection_mode, record_selections=record_selections,
                  scoring_method=scoring_method)

    if method == "NA":
        return MergedBatchNoSelectionStrategy(**kwargs)

    if method == "Layerwise":
        strategy = MergedBatchLayerwiseStrategy(**kwargs)
        # Wrap non-linear layers so their grad_weight comes from train slice only.
        grad_hook.wrap_nonlinear_layers()
        return strategy

    if method == "LayerwiseWithVal":
        strategy = MergedBatchLayerwiseWithValStrategy(**kwargs)
        grad_hook.wrap_nonlinear_layers()
        return strategy

    if method == "Subset":
        if subset_mode == "one_pass":
            strategy = MergedBatchSubsetOnePassStrategy(**kwargs)
            grad_hook.wrap_nonlinear_layers()
            return strategy
        else:
            return MergedBatchSubsetStrategy(**kwargs)

    raise ValueError(f"Unknown curation method: {method}")




# ============================================================
# CACHED VAL STRATEGIES
# Val gradients pre-captured and cached before training
# Avoids padding overhead when val/train have different seq lengths
# ============================================================

class SeparateBatchStrategy(ABC):
    """
    Abstract strategy for separate-batch data curation.

    Used when val gradients are pre-captured and cached before training,
    rather than computed from a merged batch during the same forward pass.

    Val storage mode is derived from scoring_method in start_val_capture():
    - reduced_ghost/direct: Stores total gradient [O, I] per layer.
      Better when validation batch is large (e.g., self-reference validation in RLHF).
    - full_ghost: Stores [V, S, O] and [V, S, I] components (for pairwise scoring).
      More memory-efficient during training. Better when validation batch is small.
    """

    def __init__(
        self,
        grad_hook: Optional[GradientHook],
        frac: float,
        use_second_order: bool = False,
        selection_mode: str = "topk",
        record_selections: bool = False,
        scoring_method: str = "reduced_ghost",
    ):
        """
        Initialize stored-val curation strategy.

        Args:
            grad_hook: GradientHook instance (can be None for NoSelection)
            frac: Fraction parameter. Meaning depends on selection_mode:
                  - "topk": Fraction of samples to select (top frac by score)
                  - "filtering": Fraction of negative-influence samples to DROP
            use_second_order: Use greedy curation with second-order
            selection_mode: "topk" (select top frac) or "filtering" (drop bottom frac of negative)
            record_selections: If True, record curation data for case study analysis
            scoring_method: Scoring method ("reduced_ghost", "full_ghost", "direct", "compress")
        """
        self.grad_hook = grad_hook
        self.frac = frac
        self.use_second_order = use_second_order
        self.selection_mode = selection_mode
        self.record_selections = record_selections
        self.scoring_method = scoring_method
        self.last_selection_record = None

    @property
    def has_update_compression(self) -> bool:
        """Check if update compression (MeSO) is enabled.

        When True, hooks stay enabled during Subset pass 2 so that
        CompressedLinearBackward stores compressed gradients for MeSO.
        """
        if self.grad_hook is None:
            return False
        return self.grad_hook.compression_mode.uses_compressed_updates

    def _extract_selection_records(self):
        """Extract curation records from state before cleanup."""
        if self.grad_hook is None or self.grad_hook.selection_state is None:
            self.last_selection_record = None
            return
        state = self.grad_hook.selection_state
        if state._record_selections and state._selection_records:
            self.last_selection_record = list(state._selection_records)
        else:
            self.last_selection_record = None

    @abstractmethod
    def execute_training_step(
        self,
        model: nn.Module,
        batch_size: int,
        compute_loss_fn: Callable[[], Tuple[Tensor, Dict]],
        lr: float,
        **kwargs
    ) -> Tuple[Tensor, Dict]:
        """
        Execute a complete training step with curation.

        Args:
            model: The model to train
            batch_size: Number of samples in the batch
            compute_loss_fn: Zero-arg function that computes loss and returns (loss, stats)
            lr: Learning rate for score scaling
            **kwargs: Additional arguments (filter_batch_fn for Subset)

        Returns:
            Tuple of (loss, stats_dict) where stats includes curation metrics
        """
        pass


class SeparateBatchNoSelectionStrategy(SeparateBatchStrategy):
    """
    Baseline strategy: no data curation, standard training.
    """

    def execute_training_step(
        self,
        model: nn.Module,
        batch_size: int,
        compute_loss_fn: Callable[[], Tuple[Tensor, Dict]],
        lr: float,
        **kwargs
    ) -> Tuple[Tensor, Dict]:
        """Standard training step without curation."""
        # Disable hooks for baseline (use standard gradient computation)
        if self.grad_hook is not None:
            self.grad_hook.disable_hooks()

        model.zero_grad()
        loss, stats = compute_loss_fn()
        loss.backward()

        # Re-enable hooks
        if self.grad_hook is not None:
            self.grad_hook.enable_hooks()

        return loss.detach(), stats


class SeparateBatchLayerwiseStrategy(SeparateBatchStrategy):
    """
    Layer-wise strategy with cached val: per-layer curation.

    Curation and gradient aggregation happen layer-by-layer during backward,
    using pre-captured validation gradients.
    """

    def execute_training_step(
        self,
        model: nn.Module,
        batch_size: int,
        compute_loss_fn: Callable[[], Tuple[Tensor, Dict]],
        lr: float,
        **kwargs
    ) -> Tuple[Tensor, Dict]:
        """Training step with per-layer curation using stored val grads.

        Args:
            model: The model to train
            batch_size: Number of samples in the batch
            compute_loss_fn: Zero-arg function that computes loss and returns (loss, stats)
            lr: Learning rate for score scaling
            **kwargs:
                labels: Optional label tensor for token-based gradient scaling.
                        If provided, enables proper score scaling across modes.
        """
        # Set up streaming state with stored validation gradients
        self._setup_state(batch_size, lr)

        # Set token counts for gradient scaling (if labels provided)
        # In SeparateBatch mode, entire batch is train, so pass batch_size as train_batch_size
        labels = kwargs.get('labels')
        if labels is not None:
            self.grad_hook.set_token_counts(labels, batch_size)

        model.zero_grad()
        loss, stats = compute_loss_fn()
        loss.backward()  # Per-layer curation happens in backward hooks

        # Add curation stats from streaming state
        sel_state = self.grad_hook.selection_state
        if hasattr(sel_state, '_layer_selections') and sel_state._layer_selections:
            n_selected_list = [n for _, n in sel_state._layer_selections]
            stats["selection/mean_selected"] = sum(n_selected_list) / len(n_selected_list)
            stats["selection/min_selected"] = min(n_selected_list)
            # Use min for n_selected check - if any layer had 0, flag it
            stats["selection/n_selected"] = min(n_selected_list)

        # Extract curation records before cleanup
        self._extract_selection_records()

        # Cleanup
        self._cleanup()

        return loss.detach(), stats

    def _setup_state(self, batch_size: int, lr: float) -> None:
        """Set up LayerwiseState with stored val gradients."""
        self.grad_hook.setup_selection_with_stored_val(
            train_batch_size=batch_size,
            selection_method="Layerwise",
            frac=self.frac,
            lr=lr,
            compute_scores_only=False,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            scoring_method=self.scoring_method,
        )

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()


class SeparateBatchSubsetStrategy(SeparateBatchStrategy):
    """
    Subset strategy with cached val: global curation.

    Pass 1: Compute curation scores across all layers
    Pass 2: Forward/backward only on globally selected samples
    """

    def execute_training_step(
        self,
        model: nn.Module,
        batch_size: int,
        compute_loss_fn: Callable[[], Tuple[Tensor, Dict]],
        lr: float,
        **kwargs
    ) -> Tuple[Tensor, Dict]:
        """Training step with global curation using stored val grads."""
        # filter_batch_fn: Callable[[Tensor], Callable] that takes selected_indices
        # and returns a new compute_loss_fn for the filtered batch
        filter_batch_fn = kwargs.get('filter_batch_fn')
        if filter_batch_fn is None:
            raise ValueError("Subset strategy requires 'filter_batch_fn' in kwargs")

        # === PASS 1: Score Accumulation ===
        self._setup_state(batch_size, lr)

        model.zero_grad()
        loss_for_scoring, _ = compute_loss_fn()
        loss_for_scoring.backward()

        # Get globally selected indices
        selected_indices = self.grad_hook.selection_state.get_final_selection()
        # Sort indices for sequential memory access (better cache locality)
        selected_indices = selected_indices.sort()[0]
        n_selected = len(selected_indices)

        # Extract curation records before cleanup
        self._extract_selection_records()

        self._cleanup()

        # Handle empty curation: skip pass 2 and return zero loss
        if n_selected == 0:
            # Re-enable hooks for next step
            self.grad_hook.enable_hooks()

            # Return zero loss and stats indicating batch was skipped
            # Include placeholder loss stats for consistent logging
            import torch
            zero_loss = torch.tensor(0.0, device=next(model.parameters()).device)
            stats = {
                "loss/total": 0.0,
                "loss/policy": 0.0,
                "loss/value": 0.0,
                "policy/approx_kl": 0.0,
                "policy/clipfrac": 0.0,
                "policy/ratio_mean": 1.0,
                "values/mean": 0.0,
                "selection/n_selected": 0,
            }
            return zero_loss, stats

        # === PASS 2: Gradient Computation on Selected ===
        # Disable hooks only if no update compression (standard optimizer for selected samples).
        # With MeSO, keep hooks enabled so CompressedLinearBackward stores
        # compressed gradients for the optimizer.
        if not self.has_update_compression:
            self.grad_hook.disable_hooks()

        model.zero_grad()

        # Get filtered compute_loss_fn for selected samples
        filtered_compute_loss_fn = filter_batch_fn(selected_indices)
        loss, stats = filtered_compute_loss_fn()
        loss.backward()

        # Re-enable hooks if we disabled them
        if not self.has_update_compression:
            self.grad_hook.enable_hooks()

        # Add curation stats
        stats["selection/n_selected"] = n_selected

        return loss.detach(), stats

    def _setup_state(self, batch_size: int, lr: float) -> None:
        """Set up SubsetState with stored val gradients."""
        self.grad_hook.setup_selection_with_stored_val(
            train_batch_size=batch_size,
            selection_method="Subset",
            frac=self.frac,
            lr=lr,
            compute_scores_only=True,  # Only accumulate scores in pass 1
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            scoring_method=self.scoring_method,
        )

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()


class SeparateBatchSubsetOnePassStrategy(SeparateBatchStrategy):
    """
    One-pass subset strategy with separate val batch (Algorithm 4.2).

    Recommended one-pass mode: exact scale factor parity with two-pass since
    batch_total_tokens == train_total_tokens in separate batch mode.
    """

    def execute_training_step(
        self,
        model: nn.Module,
        batch_size: int,
        compute_loss_fn: Callable[[], Tuple[Tensor, Dict]],
        lr: float,
        **kwargs
    ) -> Tuple[Tensor, Dict]:
        """Training step with one-pass global curation using stored val grads."""
        labels = kwargs.get('labels')

        # Set up SubsetState with one_pass=True and stored val
        self._setup_state(batch_size, lr)

        if labels is not None:
            self.grad_hook.set_token_counts(labels, batch_size)

        model.zero_grad()
        loss, stats = compute_loss_fn()
        loss.backward()
        # After backward: scores accumulated, layer data retained

        # Get globally selected indices
        selected_indices = self.grad_hook.selection_state.get_final_selection()
        selected_indices = selected_indices.sort()[0]
        n_selected = len(selected_indices)

        # Extract curation records before cleanup
        self._extract_selection_records()

        if n_selected == 0:
            self.grad_hook.clear_retained_data()
            self._cleanup()
            import torch
            zero_loss = torch.tensor(0.0, device=next(model.parameters()).device)
            stats["selection/n_selected"] = 0
            return zero_loss, stats

        # Compute scale factor for exact parity with two-pass
        state = self.grad_hook.selection_state
        scale_factor = state._compute_scale_factor_for_assembly(selected_indices)

        # Post-hoc gradient assembly for linear layers.
        # Non-linear layers (LayerNorm, embeddings, etc.) retain their autograd
        # gradients from backward — SubsetLinearBackward returns None for
        # weight/bias, so hooked linear layers have no stale grad to clear.
        self.grad_hook.assemble_gradients_from_retained(selected_indices, scale_factor)

        stats["selection/n_selected"] = n_selected
        self._cleanup()
        return loss.detach(), stats

    def _setup_state(self, batch_size: int, lr: float) -> None:
        """Set up SubsetState with one_pass=True and stored val gradients."""
        self.grad_hook.setup_selection_with_stored_val(
            train_batch_size=batch_size,
            selection_method="Subset",
            frac=self.frac,
            lr=lr,
            compute_scores_only=True,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            scoring_method=self.scoring_method,
            one_pass=True,
        )

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()
        self.grad_hook.clear_retained_data()  # Safety net


def create_separate_batch_strategy(
    method: str,
    grad_hook: Optional[GradientHook],
    frac: float = 0.5,
    use_second_order: bool = False,
    selection_mode: str = "topk",
    record_selections: bool = False,
    scoring_method: str = "reduced_ghost",
    subset_mode: str = "one_pass",
) -> SeparateBatchStrategy:
    """
    Factory function to create separate-batch curation strategy.

    Avoids padding overhead when val/train have different sequence lengths.

    Args:
        method: Curation method ("NA", "Layerwise", "Subset")
        grad_hook: GradientHook instance
        frac: Fraction parameter. Meaning depends on selection_mode:
              - "topk": Fraction of samples to select (top frac by score)
              - "filtering": Fraction of negative-influence samples to DROP
        use_second_order: Use greedy curation with second-order
        selection_mode: "topk" (select top frac) or "filtering" (drop bottom frac of negative)
        record_selections: If True, record curation data for case study analysis
        scoring_method: Scoring method ("reduced_ghost", "full_ghost", "direct", "compress")
        subset_mode: For Subset method: "one_pass" (Algorithm 4.2) or "two_pass" (Algorithm 4.3)

    Returns:
        Appropriate SeparateBatchStrategy instance
    """
    kwargs = dict(grad_hook=grad_hook, frac=frac, use_second_order=use_second_order,
                  selection_mode=selection_mode, record_selections=record_selections,
                  scoring_method=scoring_method)

    if method == "NA":
        return SeparateBatchNoSelectionStrategy(**kwargs)

    if method == "Layerwise":
        return SeparateBatchLayerwiseStrategy(**kwargs)

    if method == "Subset":
        if subset_mode == "one_pass":
            strategy = SeparateBatchSubsetOnePassStrategy(**kwargs)
            # Safety check: warn about trainable params not covered by hooks.
            # No non-linear wrapping needed (no val contamination in separate batch),
            # but unhooked params get full-batch train grad instead of curated.
            grad_hook.check_unhooked_trainable_params()
            return strategy
        else:
            return SeparateBatchSubsetStrategy(**kwargs)

    raise ValueError(f"Unknown curation method: {method}")
