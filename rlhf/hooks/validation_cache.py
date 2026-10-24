"""
Validation gradient cache for separate-batch strategies.

This module provides a unified interface for storing and retrieving validation
gradients during the separate-batch training workflow.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .compressor import Compressor


class ValidationStorageMode(Enum):
    """
    Storage mode for validation gradients.

    - FACTORIZED: Store (grad_output, input) components separately.
      Memory: O(V * S * (O + I)) per layer. Efficient for small validation batches.

    - FULL: Store total gradient [O, I] per layer.
      Memory: O(O * I) per layer. Efficient for large validation batches.

    - COMPRESSED: Store compressed gradient [k] per layer.
      Memory: O(k) per layer. Used when compression is enabled.
    """

    FACTORIZED = "factorized"
    FULL = "full"
    COMPRESSED = "compressed"


class ValidationCache:
    """
    Manages cached validation gradients for separate-batch strategies.

    This class provides three storage modes for validation gradients:
    - factorized: Store (grad_output, input) components separately
    - full: Store total gradient [O, I] per layer
    - compressed: Store compressed gradient [k] per layer

    Usage:
        # Start capture phase
        cache.start_capture(mode="factorized")

        # During backward pass, store gradients for each layer
        cache.store_layer(layer_idx, grad_output, input)

        # End capture and optionally record token count
        cache.end_capture(total_tokens=1024)

        # During training, retrieve for scoring
        val_data = cache.get_for_scoring(layer_idx)

        # Clear after training step
        cache.clear()
    """

    def __init__(self, num_layers: int):
        """
        Initialize validation cache.

        Args:
            num_layers: Number of layers to cache gradients for.
        """
        self.num_layers = num_layers
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset all internal state."""
        self.storage_mode: ValidationStorageMode = ValidationStorageMode.FACTORIZED
        self.capturing: bool = False
        self.total_tokens: Optional[int] = None

        # Storage buffers (only one set is used depending on mode)
        self._factorized: List[Optional[Tuple[Tensor, Tensor]]] = [None] * self.num_layers
        self._full: List[Optional[Tensor]] = [None] * self.num_layers
        self._bias_grad: List[Optional[Tensor]] = [None] * self.num_layers
        self._compressed: List[Optional[Tensor]] = [None] * self.num_layers

    def start_capture(self, mode: str = "factorized") -> None:
        """
        Begin validation gradient capture.

        Args:
            mode: Storage mode - one of "factorized", "full", or "compressed".
                - "factorized": Store (grad_output, input) components.
                  Better for small validation batches.
                - "full": Store total gradient [O, I] per layer.
                  Better for large validation batches.
                - "compressed": Store compressed gradient [k] per layer.
                  Used when compression is enabled.

        Raises:
            ValueError: If mode is invalid.
        """
        try:
            self.storage_mode = ValidationStorageMode(mode)
        except ValueError:
            valid_modes = [m.value for m in ValidationStorageMode]
            raise ValueError(
                f"Invalid storage mode: {mode}. Must be one of {valid_modes}"
            )

        self.capturing = True
        self.total_tokens = None

        # Clear all buffers
        self._factorized = [None] * self.num_layers
        self._full = [None] * self.num_layers
        self._bias_grad = [None] * self.num_layers
        self._compressed = [None] * self.num_layers

    def store_layer(
        self,
        layer_idx: int,
        grad_output: Tensor,
        input: Tensor,
        compressor: Optional["Compressor"] = None,
    ) -> None:
        """
        Store validation gradient for a layer.

        The storage format depends on the current storage_mode:
        - FACTORIZED: Stores (grad_output, input) tuple
        - FULL: Computes and stores total gradient [O, I]
        - COMPRESSED: Compresses and stores [k] vector

        Args:
            layer_idx: Index of the layer.
            grad_output: Gradient of output [V, S, O] or [V, O].
            input: Input tensor [V, S, I] or [V, I].
            compressor: Optional compressor for COMPRESSED mode.

        Raises:
            RuntimeError: If not in capture mode.
            ValueError: If COMPRESSED mode but no compressor provided.
        """
        if not self.capturing:
            raise RuntimeError(
                "Cannot store validation gradient: not in capture mode. "
                "Call start_capture() first."
            )

        if self.storage_mode == ValidationStorageMode.FACTORIZED:
            self._factorized[layer_idx] = (grad_output.detach(), input.detach())

        elif self.storage_mode == ValidationStorageMode.FULL:
            # Compute total gradient: Σ_v Σ_s grad_output[v,s,o] × input[v,s,i]
            val_grad_total = self._compute_total_gradient(grad_output, input)

            # Accumulate if already exists (for mini-batch capture)
            if self._full[layer_idx] is None:
                self._full[layer_idx] = val_grad_total
            else:
                self._full[layer_idx] = self._full[layer_idx] + val_grad_total

            # Store bias gradient: Σ_v Σ_s grad_output[v,s,o] → [O]
            if grad_output.dim() == 3:
                bias_grad = grad_output.detach().sum(dim=(0, 1))
            else:
                bias_grad = grad_output.detach().sum(dim=0)
            if self._bias_grad[layer_idx] is None:
                self._bias_grad[layer_idx] = bias_grad
            else:
                self._bias_grad[layer_idx] = self._bias_grad[layer_idx] + bias_grad

        elif self.storage_mode == ValidationStorageMode.COMPRESSED:
            if compressor is None:
                raise ValueError(
                    "Compressor required for COMPRESSED storage mode. "
                    "Pass compressor argument or use a different mode."
                )

            # Import here to avoid circular dependency
            from .selection.utils import augment_input_for_bias

            # Determine if bias exists from compressor's layer info
            # For now, assume bias handling is done externally
            input_aug = input  # Caller should handle bias augmentation

            compressed = compressor.forward((grad_output, input_aug))
            # Sum over batch dimension to get total compressed gradient
            compressed_total = compressed.sum(dim=0)

            # Accumulate if already exists (for mini-batch capture)
            if self._compressed[layer_idx] is None:
                self._compressed[layer_idx] = compressed_total
            else:
                self._compressed[layer_idx] = self._compressed[layer_idx] + compressed_total

    def store_precomputed(
        self,
        layer_idx: int,
        grad_weight: Tensor,
    ) -> None:
        """
        Store a pre-computed gradient directly (e.g., for Embedding layers).

        Unlike store_layer which computes the gradient from (grad_output, input),
        this accepts the already-materialized gradient and stores it in FULL mode.

        Args:
            layer_idx: Index of the layer.
            grad_weight: Pre-computed gradient tensor (e.g., [V, D] for Embedding).
        """
        if not self.capturing:
            raise RuntimeError(
                "Cannot store validation gradient: not in capture mode. "
                "Call start_capture() first."
            )
        if self._full[layer_idx] is None:
            self._full[layer_idx] = grad_weight.detach()
        else:
            self._full[layer_idx] = self._full[layer_idx] + grad_weight.detach()

    def _compute_total_gradient(self, grad_output: Tensor, input: Tensor) -> Tensor:
        """Compute total gradient [O, I] from grad_output and input."""
        if grad_output.dim() == 3:
            return torch.einsum('bso,bsi->oi', grad_output, input)
        else:
            return torch.einsum('bo,bi->oi', grad_output, input)

    def end_capture(self, total_tokens: Optional[int] = None) -> None:
        """
        End validation gradient capture mode.

        Args:
            total_tokens: Total valid tokens in validation batch.
                         Used for score scaling in separate-batch mode.
        """
        self.capturing = False
        self.total_tokens = total_tokens

    def get_for_scoring(
        self,
        layer_idx: int,
    ) -> Union[Tuple[Tensor, Tensor], Tensor, None]:
        """
        Get validation gradient in format needed for score computation.

        Returns:
            - FACTORIZED mode: (grad_output, input) tuple
            - FULL mode: total gradient [O, I]
            - COMPRESSED mode: compressed gradient [k]
            - None if no gradient stored for this layer
        """
        if self.storage_mode == ValidationStorageMode.FACTORIZED:
            return self._factorized[layer_idx]
        elif self.storage_mode == ValidationStorageMode.FULL:
            return self._full[layer_idx]
        elif self.storage_mode == ValidationStorageMode.COMPRESSED:
            return self._compressed[layer_idx]
        return None

    def has_data(self, layer_idx: int) -> bool:
        """Check if validation gradient is stored for a layer."""
        if self.storage_mode == ValidationStorageMode.FACTORIZED:
            return self._factorized[layer_idx] is not None
        elif self.storage_mode == ValidationStorageMode.FULL:
            return self._full[layer_idx] is not None
        elif self.storage_mode == ValidationStorageMode.COMPRESSED:
            return self._compressed[layer_idx] is not None
        return False

    def get_num_captured(self) -> int:
        """Get number of layers with captured validation gradients."""
        if self.storage_mode == ValidationStorageMode.FACTORIZED:
            return sum(1 for g in self._factorized if g is not None)
        elif self.storage_mode == ValidationStorageMode.FULL:
            return sum(1 for g in self._full if g is not None)
        elif self.storage_mode == ValidationStorageMode.COMPRESSED:
            return sum(1 for g in self._compressed if g is not None)
        return 0

    def clear(self) -> None:
        """Clear all cached validation gradients."""
        self._factorized = [None] * self.num_layers
        self._full = [None] * self.num_layers
        self._bias_grad = [None] * self.num_layers
        self._compressed = [None] * self.num_layers
        self.total_tokens = None
        self.capturing = False

    # =========================================================================
    # Storage mode helpers
    # =========================================================================

    @property
    def is_factorized(self) -> bool:
        """Check if using factorized storage mode."""
        return self.storage_mode == ValidationStorageMode.FACTORIZED

    @property
    def is_compressed(self) -> bool:
        """Check if using compressed storage mode."""
        return self.storage_mode == ValidationStorageMode.COMPRESSED

    def get_factorized(self, layer_idx: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Get factorized components for a layer.

        Returns:
            (grad_output, input) tuple, or (None, None) if not stored.
        """
        data = self._factorized[layer_idx]
        if data is not None:
            return data
        return (None, None)

    def get_full(self, layer_idx: int) -> Optional[Tensor]:
        """Get full gradient for a layer."""
        return self._full[layer_idx]

    def get_bias_grad(self, layer_idx: int) -> Optional[Tensor]:
        """Get cached bias gradient [O] for a layer (FULL mode only)."""
        return self._bias_grad[layer_idx]

    def get_compressed(self, layer_idx: int) -> Optional[Tensor]:
        """Get compressed gradient for a layer."""
        return self._compressed[layer_idx]
