"""
Hook manager with monkey-patching and custom autograd Functions.

The hook supports two distinct curation methods via the selection module:
- Layerwise: Per-layer curation, single-pass (LayerwiseLinearBackward)
- Subset: Global curation, two-pass (SubsetLinearBackward)

Compression is configured independently for two purposes:
- Score compression: compresses gradients for influence score computation
- Update compression: compresses gradients for MeSO optimizer updates

See CompressionMode for the full mode matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Tuple
    from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F
import functools
import logging

# Compression mode configuration
from .compression_mode import CompressionMode

# Validation gradient cache
from .validation_cache import ValidationCache

# Curation module
from .selection.state import SelectionState, LayerwiseState, SubsetState
from .selection.backward import (
    CompressedLinearBackward,
    LayerwiseLinearBackward,
    SubsetLinearBackward,
    TrainOnlyRMSNormBackward,
    SubsetEmbeddingBackward,
    LayerwiseEmbeddingBackward,
)

logger = logging.getLogger(__name__)


class GradientHook:
    """
    Hook manager for custom gradient computation.

    This class manages:
    1. Monkey-patching Linear layers for custom backward passes
    2. Compression configuration (via CompressionMode)
    3. Curation state management (Layerwise/Subset)
    4. Validation gradient caching (for separate-batch strategies)
    5. Token count tracking for proper gradient scaling
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        device: str = 'cpu',
    ) -> None:
        """
        Initialize the hook manager.

        Args:
            model: The model to hook
            layer_names: Names of layers to hook (only Linear layers supported)
            device: Device for synchronization
        """
        self.model: nn.Module = model
        self.layer_names: List[str] = layer_names
        self.device: str = device

        # Create mapping from layer name to index
        self.layer_name_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(layer_names)}

        # Create mapping from layer name to module
        self.layer_name_to_module: Dict[str, nn.Module] = {}

        # Centralized storage arrays
        self.forward_hooks: List[Optional[Any]] = [None] * len(layer_names)

        # Separate compressor lists for score computation and optimizer updates.
        # When both use the same config, they share the same Compressor objects.
        self.score_compressors: List[Optional[Compressor]] = [None] * len(layer_names)
        self.update_compressors: List[Optional[Compressor]] = [None] * len(layer_names)

        # Track hook registration status
        self.hooks_registered: bool = False
        self.hooks_enabled: bool = True

        # Curation state (set by trainer before forward/backward)
        self.selection_state: Optional[SelectionState] = None

        # Validation gradient cache (consolidated from three separate buffers)
        self._val_cache: ValidationCache = ValidationCache(len(layer_names))

        # Token count tracking for proper gradient scaling
        # Kept as Tensors to avoid D2H memory copies during backward
        self.total_tokens: Optional[Tensor] = None
        self.tokens_per_sample: Optional[Tensor] = None

        # Retained layer data for one-pass subset descent (Algorithm 4.2)
        # During backward, SubsetLinearBackward stores (grad_output, input) per layer
        # so we can assemble gradients post-hoc after global selection
        self._retained_data: Dict[int, Tuple[Tensor, Tensor]] = {}

        # Track which layer indices are Embedding (vs Linear)
        self._embedding_layer_indices: set = set()

        # Non-linear layer wrapping for merged-batch one-pass mode.
        # When active, RMSNorm layers compute grad_weight from train slice only,
        # preventing validation gradient leakage.
        self._nonlinear_wrapped: bool = False
        self._nonlinear_modules: Dict[str, nn.Module] = {}

        # Register hooks
        self._register_hooks()

        logger.info(f"Initialized GradientHook with {len(layer_names)} layers")

    # =========================================================================
    # Compression Mode Configuration
    # =========================================================================

    @property
    def compression_mode(self) -> CompressionMode:
        """Derive compression mode from which compressor lists are populated."""
        has_score = any(c is not None for c in self.score_compressors)
        has_update = any(c is not None for c in self.update_compressors)
        if has_score and has_update:
            return CompressionMode.FULL
        elif has_score:
            return CompressionMode.SCORE_ONLY
        elif has_update:
            return CompressionMode.UPDATE_ONLY
        else:
            return CompressionMode.NONE

    # =========================================================================
    # Hook Registration
    # =========================================================================

    def _register_hooks(self):
        """Monkey-patch Linear and Embedding layers to use custom Functions."""
        if self.hooks_registered:
            logger.warning("Hooks already registered, skipping")
            return

        n_linear = 0
        n_embedding = 0

        for name, module in self.model.named_modules():
            if name in self.layer_names:
                idx = self.layer_name_to_idx[name]

                # Cache the module
                self.layer_name_to_module[name] = module

                # Save original forward method
                module._original_forward = module.forward

                if isinstance(module, nn.Linear):
                    module.forward = functools.partial(
                        self._custom_linear_forward, module, idx
                    )
                    n_linear += 1
                elif isinstance(module, nn.Embedding):
                    self._embedding_layer_indices.add(idx)
                    module.forward = functools.partial(
                        self._custom_embedding_forward, module, idx
                    )
                    n_embedding += 1
                else:
                    logger.warning(f"Layer {name} is neither nn.Linear nor nn.Embedding, skipping")
                    continue

        self.hooks_registered = True
        logger.info(f"Successfully wrapped {len(self.layer_names)} layers ({n_linear} Linear, {n_embedding} Embedding)")

    def _custom_linear_forward(self, module: nn.Linear, idx: int, input: Tensor) -> Tensor:
        """
        Replacement forward method that uses our custom autograd Function.

        Routing logic:
        1. If hooks disabled -> call original forward method
        2. If Subset state -> SubsetLinearBackward (score accumulation)
        3. If Layerwise state -> LayerwiseLinearBackward (per-layer curation)
        4. If capture_val_mode -> LayerwiseLinearBackward (val gradient capture)
        5. If compressor present -> CompressedLinearBackward (compression only)
        6. Otherwise -> call original forward method
        """
        if not self.hooks_enabled:
            # Use original forward method to preserve any layer-specific behavior
            # This is important for LoRA layers where we hook lora_A/lora_B Linear modules
            return module._original_forward(input)

        # Route based on selection state type
        state = self.selection_state

        if isinstance(state, SubsetState):
            return SubsetLinearBackward.apply(
                input, module.weight, module.bias, self, idx
            )
        elif isinstance(state, LayerwiseState):
            return LayerwiseLinearBackward.apply(
                input, module.weight, module.bias, self, idx
            )
        elif self.capture_val_mode:
            # Val capture mode: use LayerwiseLinearBackward
            return LayerwiseLinearBackward.apply(
                input, module.weight, module.bias, self, idx
            )
        elif self.update_compressors[idx] is not None:
            # Update compression only, no data curation (MeSO without curation)
            return CompressedLinearBackward.apply(
                input, module.weight, module.bias, self, idx
            )
        else:
            # No curation and no compression: use original forward
            return module._original_forward(input)

    def _custom_embedding_forward(self, module: nn.Embedding, idx: int, input_ids: Tensor) -> Tensor:
        """
        Replacement forward method for Embedding layers.

        Routing logic:
        1. If hooks disabled -> call original forward
        2. If Subset state -> SubsetEmbeddingBackward (score accumulation)
        3. If Layerwise state -> LayerwiseEmbeddingBackward (per-layer curation)
        4. If capture_val_mode -> LayerwiseEmbeddingBackward (val gradient capture)
        5. Otherwise -> call original forward
        """
        if not self.hooks_enabled:
            return module._original_forward(input_ids)

        padding_idx = module.padding_idx if module.padding_idx is not None else -1
        state = self.selection_state

        if isinstance(state, SubsetState):
            return SubsetEmbeddingBackward.apply(
                input_ids, module.weight, self, idx, padding_idx
            )
        elif isinstance(state, LayerwiseState):
            return LayerwiseEmbeddingBackward.apply(
                input_ids, module.weight, self, idx, padding_idx
            )
        elif self.capture_val_mode:
            return LayerwiseEmbeddingBackward.apply(
                input_ids, module.weight, self, idx, padding_idx
            )
        else:
            return module._original_forward(input_ids)

    def set_score_compressors(self, compressors: List[Optional[Compressor]]) -> None:
        """Set compressors for influence score computation."""
        self.score_compressors = compressors

    def set_update_compressors(self, compressors: List[Optional[Compressor]]) -> None:
        """Set compressors for MeSO optimizer updates."""
        self.update_compressors = compressors

    def set_compressors(self, compressors: List[Optional[Compressor]]) -> None:
        """Set same compressors for both scoring and updates (shared objects)."""
        self.score_compressors = compressors
        self.update_compressors = compressors

    def enable_hooks(self) -> None:
        """Enable hooks to compute custom gradients."""
        self.hooks_enabled = True

    def disable_hooks(self) -> None:
        """Disable hooks to allow standard gradient computation."""
        self.hooks_enabled = False

    # =========================================================================
    # Curation State Management
    # =========================================================================

    def setup_selection(
        self,
        train_batch_size: int,
        selection_method: str,
        frac: float,
        lr: float,
        compute_scores_only: bool = False,
        use_second_order: bool = False,
        selection_mode: str = "topk",
        scoring_method: str = "reduced_ghost",
        one_pass: bool = False,
        direct_batch_size: int = 0,
    ) -> None:
        """
        Set up curation state for Dr. Post-Training.

        Args:
            train_batch_size: Number of training samples
            selection_method: Curation method ("Layerwise", "Subset", or "Regular")
            frac: Curation fraction (topk) or filter fraction (filtering)
            lr: Learning rate for score scaling
            compute_scores_only: If True, only compute scores (Subset pass 1)
            use_second_order: If True, use greedy curation with similarity matrix
            selection_mode: "topk" (select top frac) or "filtering" (drop bottom frac of negative)
            scoring_method: For Subset: "reduced_ghost" (factored inner product) or "direct"
                           (explicit per-sample gradient materialization, Algorithm 4.4)
            direct_batch_size: Chunk size for batched direct scoring. 0 = all at once.
                              Set to 1 for minimal memory at long sequences.
        """
        if selection_method == "Regular":
            self.selection_state = None
            logger.debug("Set up baseline mode (no curation)")
            return

        # Validate scoring_method vs compressor state
        has_score_comp = any(c is not None for c in self.score_compressors)
        if scoring_method == "compress" and not has_score_comp:
            raise ValueError(
                "scoring_method='compress' requires score compressors to be configured. "
                "Set score_compression in your config or call set_score_compressors()."
            )
        if has_score_comp and scoring_method != "compress":
            logger.warning(
                f"Score compressors are configured but scoring_method='{scoring_method}'. "
                f"Compressors will be ignored for scoring. "
                f"Set scoring_method='compress' to use compressed scoring."
            )

        dtype = next(self.model.parameters()).dtype
        num_layers = len(self.layer_names)

        if selection_method == "Subset":
            self.selection_state = SubsetState(
                train_batch_size=train_batch_size,
                num_layers=num_layers,
                frac=frac,
                lr=lr,
                device=self.device,
                dtype=dtype,
                use_second_order=use_second_order,
                selection_mode=selection_mode,
                scoring_method=scoring_method,
                one_pass=one_pass,
                direct_batch_size=direct_batch_size,
            )
        elif selection_method == "Layerwise":
            self.selection_state = LayerwiseState(
                train_batch_size=train_batch_size,
                num_layers=num_layers,
                frac=frac,
                lr=lr,
                device=self.device,
                dtype=dtype,
                use_second_order=use_second_order,
                selection_mode=selection_mode,
                scoring_method=scoring_method,
                direct_batch_size=direct_batch_size,
            )
        else:
            raise ValueError(
                f"Unknown selection_method: {selection_method}. "
                f"Use 'Layerwise', 'Subset', or 'Regular'."
            )

        logger.debug(
            f"Set up {selection_method} state: {train_batch_size} train, "
            f"scores_only={compute_scores_only}, use_second_order={use_second_order}, "
            f"selection_mode={selection_mode}, frac={frac}"
        )

    def clear_selection(self) -> None:
        """Clear curation state after forward/backward."""
        self.selection_state = None

    # =========================================================================
    # Token Count Tracking
    # =========================================================================

    def set_token_counts(
        self,
        labels: Tensor,
        train_batch_size: Optional[int] = None,
    ) -> None:
        """
        Set token counts for proper gradient scaling.

        Args:
            labels: Label tensor [batch_size, seq_length] with -100 for ignored positions.
                    Used for gradient scaling (only response tokens count).
            train_batch_size: If provided, only count tokens for first train_batch_size samples
        """
        valid_mask = (labels != -100)
        tokens_per_sample = valid_mask.sum(dim=1)

        # Keep as Tensor to avoid D2H memory copy
        self.total_tokens = tokens_per_sample.sum()
        self.tokens_per_sample = tokens_per_sample

        if train_batch_size is not None and self.selection_state is not None:
            train_tokens = tokens_per_sample[:train_batch_size]
            total_train_tokens = train_tokens.sum()
            self.selection_state.set_token_counts(
                train_tokens, total_train_tokens, self.total_tokens
            )

        logger.debug(f"Set token counts: total={self.total_tokens}")

    def clear_token_counts(self) -> None:
        """Clear token counts after forward/backward."""
        self.total_tokens = None
        self.tokens_per_sample = None

    # =========================================================================
    # Validation Gradient Management
    # =========================================================================

    @property
    def val_cache(self) -> ValidationCache:
        """Get the validation gradient cache."""
        return self._val_cache

    @property
    def capture_val_mode(self) -> bool:
        """Check if in validation gradient capture mode."""
        return self._val_cache.capturing

    @property
    def use_factorized_val(self) -> bool:
        """Check if using factorized validation gradient storage."""
        return self._val_cache.is_factorized

    @property
    def val_total_tokens(self) -> Optional[int]:
        """Get total tokens in validation batch."""
        return self._val_cache.total_tokens

    def start_val_capture(self, scoring_method: str = "reduced_ghost") -> None:
        """
        Start capturing validation gradients.

        The storage mode is derived from the scoring method:
        - compress      → compressed storage [k] per layer
        - full_ghost    → factorized storage (val_go, val_inp) per layer
                          (needed for pairwise dot products across val samples)
        - reduced_ghost/direct → full storage G_val [O, I] per layer
                          (only the summed gradient is needed)

        Args:
            scoring_method: Scoring method for upcoming training.
        """
        if scoring_method == "compress" and self.compression_mode.uses_compressed_scoring:
            mode = "compressed"
        elif scoring_method == "full_ghost":
            mode = "factorized"
        else:
            # reduced_ghost, direct: only need the summed val gradient
            mode = "full"

        self._val_cache.start_capture(mode=mode)
        logger.debug(f"Started validation gradient capture mode (mode={mode}, scoring={scoring_method})")

    def end_val_capture(self, val_total_tokens: Optional[int] = None) -> None:
        """
        End validation gradient capture mode.

        Args:
            val_total_tokens: Total valid tokens in validation batch.
        """
        if val_total_tokens is None:
            val_total_tokens = self.total_tokens

        self._val_cache.end_capture(total_tokens=val_total_tokens)

        num_captured = self._val_cache.get_num_captured()
        logger.debug(
            f"Ended validation gradient capture, captured {num_captured} layers, "
            f"val_tokens={self._val_cache.total_tokens}"
        )

    def clear_val_buffer(self) -> None:
        """Clear all validation gradient buffers."""
        self._val_cache.clear()

    def setup_selection_with_stored_val(
        self,
        train_batch_size: int,
        selection_method: str,
        frac: float,
        lr: float,
        compute_scores_only: bool = False,
        use_second_order: bool = False,
        selection_mode: str = "topk",
        record_selections: bool = False,
        scoring_method: str = "reduced_ghost",
        one_pass: bool = False,
        direct_batch_size: int = 0,
    ) -> None:
        """
        Set up curation state using pre-captured validation gradients.

        Args:
            train_batch_size: Number of training samples
            selection_method: Curation method ("Layerwise" or "Subset")
            frac: Curation/filter fraction
            lr: Learning rate for score scaling
            compute_scores_only: If True, only compute scores (Subset pass 1)
            use_second_order: If True, use greedy curation
            selection_mode: "topk" or "filtering"
            record_selections: If True, record selected indices/scores for case study
            scoring_method: Scoring method ("reduced_ghost", "full_ghost", "direct", "compress")
            one_pass: If True, enable one-pass subset mode (retain layer data for post-hoc assembly)
            direct_batch_size: Chunk size for batched direct scoring. 0 = all at once.
        """
        num_captured = self._val_cache.get_num_captured()
        if num_captured == 0:
            raise RuntimeError(
                "No validation gradients captured. Call start_val_capture(), "
                "run forward/backward on validation data, then end_val_capture() first."
            )

        # Validate scoring_method vs compressor state
        has_score_comp = any(c is not None for c in self.score_compressors)
        if scoring_method == "compress" and not has_score_comp:
            raise ValueError(
                "scoring_method='compress' requires score compressors to be configured."
            )
        if has_score_comp and scoring_method != "compress":
            logger.warning(
                f"Score compressors are configured but scoring_method='{scoring_method}'. "
                f"Compressors will be ignored for scoring."
            )

        dtype = next(self.model.parameters()).dtype
        num_layers = len(self.layer_names)

        if selection_method == "Subset":
            self.selection_state = SubsetState(
                train_batch_size=train_batch_size,
                num_layers=num_layers,
                frac=frac,
                lr=lr,
                device=self.device,
                dtype=dtype,
                use_second_order=use_second_order,
                selection_mode=selection_mode,
                record_selections=record_selections,
                scoring_method=scoring_method,
                one_pass=one_pass,
                direct_batch_size=direct_batch_size,
            )
        elif selection_method == "Layerwise":
            self.selection_state = LayerwiseState(
                train_batch_size=train_batch_size,
                num_layers=num_layers,
                frac=frac,
                lr=lr,
                device=self.device,
                dtype=dtype,
                use_second_order=use_second_order,
                selection_mode=selection_mode,
                record_selections=record_selections,
                scoring_method=scoring_method,
                direct_batch_size=direct_batch_size,
            )
        else:
            raise ValueError(
                f"Unknown selection_method: {selection_method}. "
                f"Use 'Layerwise' or 'Subset'."
            )

        # Mark that we're using stored validation gradients
        self.selection_state._use_stored_val = True

        logger.debug(
            f"Set up curation with stored val gradients: {train_batch_size} train samples, "
            f"{num_captured} layers with val gradients, selection_mode={selection_mode}, frac={frac}"
        )

    # =========================================================================
    # Compressed Gradient Storage (for MeSO optimizer)
    # =========================================================================

    def _get_layer_name_from_idx(self, layer_idx: int) -> str:
        """Get layer name from layer index."""
        return self.layer_names[layer_idx]

    def _get_module_from_idx(self, layer_idx: int) -> nn.Module:
        """Get module from layer index."""
        layer_name = self._get_layer_name_from_idx(layer_idx)
        return self.layer_name_to_module[layer_name]

    def _store_compressed_grad(self, layer_idx: int, compressed_grad: Tensor) -> None:
        """Store compressed gradient on the weight parameter."""
        module = self._get_module_from_idx(layer_idx)
        module.weight._compressed_grad = compressed_grad

    def _get_compressed_grad(self, layer_idx: int) -> Optional[Tensor]:
        """Get compressed gradient from the weight parameter."""
        module = self._get_module_from_idx(layer_idx)
        return getattr(module.weight, '_compressed_grad', None)

    def _clear_compressed_grad(self, layer_idx: int) -> None:
        """Clear compressed gradient from the weight parameter."""
        module = self._get_module_from_idx(layer_idx)
        if hasattr(module.weight, '_compressed_grad'):
            module.weight._compressed_grad = None

    def clear_all_compressed_grads(self) -> None:
        """Clear all compressed gradients from all layer weight parameters."""
        for layer_idx in range(len(self.layer_names)):
            self._clear_compressed_grad(layer_idx)

    def get_compressed_grads(self) -> List[Optional[Tensor]]:
        """Get all captured compressed gradients."""
        return [self._get_compressed_grad(idx) for idx in range(len(self.layer_names))]

    # =========================================================================
    # Retained Layer Data (One-Pass Subset Descent)
    # =========================================================================

    def retain_layer_data(self, layer_idx: int, grad_output: Tensor, input: Tensor) -> None:
        """Store grad_output and input for post-hoc gradient assembly (one-pass subset)."""
        self._retained_data[layer_idx] = (grad_output.detach(), input.detach())

    def clear_retained_data(self) -> None:
        """Release all retained layer data."""
        self._retained_data.clear()

    def has_retained_data(self) -> bool:
        """Check if any retained data is stored."""
        return len(self._retained_data) > 0

    def assemble_gradients_from_retained(
        self,
        selected_indices: Tensor,
        scale_factor: Tensor,
    ) -> None:
        """
        Assemble weight gradients for selected samples using retained layer data.

        This is the post-hoc gradient assembly step in one-pass subset descent
        (Algorithm 4.2). For each layer, computes:
            g_l = scale_factor * Σ_{i∈S} grad_output_i ⊗ input_i
        and assigns to param.grad.

        For MeSO (update compression), compresses selected gradients instead.

        Args:
            selected_indices: Globally selected sample indices [K]
            scale_factor: Scaling factor (batch_total_tokens / selected_tokens)
        """
        import torch
        from .selection.utils import (
            compute_selected_gradients,
            compute_embedding_selected_gradients,
            augment_input_for_bias,
        )

        for layer_idx in range(len(self.layer_names)):
            retained = self._retained_data.get(layer_idx)
            if retained is None:
                continue
            grad_output, input_tensor = retained
            module = self._get_module_from_idx(layer_idx)

            is_embedding = layer_idx in self._embedding_layer_indices

            if is_embedding:
                # Embedding layer: scatter selected gradients
                padding_idx = module.padding_idx if module.padding_idx is not None else -1
                V, D = module.weight.shape
                grad_weight = compute_embedding_selected_gradients(
                    grad_output, input_tensor, selected_indices, scale_factor,
                    V, D, padding_idx,
                )
                grad_weight = grad_weight.to(module.weight.dtype)
                if module.weight.grad is None:
                    module.weight.grad = grad_weight
                else:
                    module.weight.grad.add_(grad_weight)
            else:
                # Linear layer
                has_bias = module.bias is not None
                update_compressor = self.update_compressors[layer_idx]
                if update_compressor is not None:
                    # MeSO mode: compress selected gradients and store for optimizer
                    sel_go = grad_output[selected_indices]
                    sel_inp = augment_input_for_bias(input_tensor[selected_indices], has_bias)
                    compressed = update_compressor.forward((sel_go, sel_inp))
                    reduced = compressed.mean(dim=0, keepdim=True) * scale_factor
                    self._store_compressed_grad(layer_idx, reduced)
                else:
                    # Standard mode: compute full weight gradient
                    grad_weight, grad_bias = compute_selected_gradients(
                        grad_output, input_tensor,
                        selected_indices, has_bias, scale_factor
                    )
                    if grad_weight is not None:
                        grad_weight = grad_weight.to(module.weight.dtype)
                        if module.weight.grad is None:
                            module.weight.grad = grad_weight
                        else:
                            module.weight.grad.add_(grad_weight)
                    if grad_bias is not None and module.bias is not None:
                        grad_bias = grad_bias.to(module.bias.dtype)
                        if module.bias.grad is None:
                            module.bias.grad = grad_bias
                        else:
                            module.bias.grad.add_(grad_bias)

        self.clear_retained_data()

    # =========================================================================
    # Compressor Management
    # =========================================================================

    def refresh_compressors(self, step: int) -> Tuple[int, List[Optional[Any]]]:
        """
        Refresh all compressors if needed.

        Refreshes update_compressors (used by MeSO optimizer for state transformation).
        Score compressors that share the same objects are refreshed implicitly.
        Score-only compressors (not shared with update) are refreshed separately.

        Args:
            step: Current training step

        Returns:
            Tuple of (num_refreshed, old_update_compressors) for optimizer state transform
        """
        num_refreshed = 0
        old_update_compressors = []

        # Refresh update compressors (MeSO optimizer needs old state for transform)
        for idx, compressor in enumerate(self.update_compressors):
            if compressor is not None:
                old_container = compressor.refresh(step)
                old_update_compressors.append(old_container)
                if old_container is not None:
                    num_refreshed += 1
            else:
                old_update_compressors.append(None)

        # Refresh score-only compressors (those not shared with update)
        for idx, compressor in enumerate(self.score_compressors):
            if compressor is not None and compressor is not self.update_compressors[idx]:
                compressor.refresh(step)

        return num_refreshed, old_update_compressors

    def remove_hooks(self) -> None:
        """Restore original forward methods for all wrapped layers."""
        for name, module in self.layer_name_to_module.items():
            if hasattr(module, '_original_forward'):
                module.forward = module._original_forward
                delattr(module, '_original_forward')

        self.forward_hooks = [None] * len(self.layer_names)
        self.hooks_registered = False

        # Also unwrap non-linear layers if wrapped
        self.unwrap_nonlinear_layers()

        logger.info("Restored original forward methods for all layers")

    # =========================================================================
    # Non-linear layer wrapping (merged-batch one-pass)
    # =========================================================================

    @staticmethod
    def _is_rmsnorm(module: nn.Module) -> bool:
        """Check if module is an RMSNorm variant (HuggingFace or PyTorch)."""
        cls_name = type(module).__name__
        return 'RMSNorm' in cls_name and hasattr(module, 'weight')

    def wrap_nonlinear_layers(self) -> None:
        """
        Wrap trainable non-linear layers for train-only grad_weight.

        Supported layer types:
        - RMSNorm (LlamaRMSNorm, Qwen2RMSNorm, etc.) -> TrainOnlyRMSNormBackward

        Note: Embedding layers are NOT wrapped here — they are registered as
        hooked layers (like Linear) and participate in scoring/selection directly.

        Call once at setup (not per-step). The wrapped forward reads
        train_batch_size from self.selection_state during backward, so it
        adapts per-step automatically.

        Also checks for trainable parameters that are neither hooked
        nor wrapped, and warns about them.
        """
        if self._nonlinear_wrapped:
            return

        n_rmsnorm = 0

        for name, module in self.model.named_modules():
            # Skip hooked layers (Linear and Embedding)
            if name in self.layer_name_to_idx:
                continue

            if self._is_rmsnorm(module):
                if not module.weight.requires_grad:
                    continue
                self._nonlinear_modules[name] = module
                module._original_forward = module.forward

                eps = getattr(module, 'variance_epsilon', None) or getattr(module, 'eps', 1e-6)

                def make_rmsnorm_wrapped(mod, mod_eps):
                    def wrapped_forward(hidden_states):
                        return TrainOnlyRMSNormBackward.apply(
                            hidden_states, mod.weight, mod_eps, self
                        )
                    return wrapped_forward

                module.forward = make_rmsnorm_wrapped(module, eps)
                n_rmsnorm += 1

        self._nonlinear_wrapped = True
        if self._nonlinear_modules:
            logger.info(
                f"Wrapped {n_rmsnorm} RMSNorm layers for train-only grad_weight"
            )

        # Safety check: warn about trainable params not covered by any hook
        self.check_unhooked_trainable_params()

    def unwrap_nonlinear_layers(self) -> None:
        """Restore original forward methods for wrapped non-linear layers."""
        if not self._nonlinear_wrapped:
            return

        for name, module in self._nonlinear_modules.items():
            if hasattr(module, '_original_forward'):
                module.forward = module._original_forward
                delattr(module, '_original_forward')

        self._nonlinear_modules.clear()
        self._nonlinear_wrapped = False

    def check_unhooked_trainable_params(self) -> None:
        """
        Check that all trainable parameters belong to either a hooked Linear
        layer or a wrapped non-linear layer.

        Logs warnings for any trainable parameters that fall through the cracks.
        In one-pass mode, such parameters would get incorrect gradients:
        - Merged batch: grad from full merged batch (val + train) instead of train-only
        - Separate batch: full-batch train grad instead of curated subset grad
          (same approximation as non-linear layers, but worth flagging if unexpected)
        """
        hooked_names = set(self.layer_names)
        wrapped_names = set(self._nonlinear_modules.keys())

        uncovered = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Check if this param belongs to a hooked or wrapped module.
            # param name is like "model.layers.0.self_attn.q_proj.weight"
            # module name is like "model.layers.0.self_attn.q_proj"
            # Strip the param suffix (.weight, .bias) to get the module name.
            parts = name.rsplit('.', 1)
            if len(parts) < 2:
                continue
            module_name = parts[0]

            # Check all ancestor modules too (e.g. param "model.layers.0.input_layernorm.weight"
            # has module_name "model.layers.0.input_layernorm")
            covered = False
            candidate = module_name
            while candidate:
                if candidate in hooked_names or candidate in wrapped_names:
                    covered = True
                    break
                # Go up one level
                if '.' in candidate:
                    candidate = candidate.rsplit('.', 1)[0]
                else:
                    break

            if not covered:
                uncovered.append(name)

        if uncovered:
            logger.warning(
                f"Found {len(uncovered)} trainable parameter(s) not covered by "
                f"any hook or non-linear wrapper. In one-pass mode, these will "
                f"receive full-batch gradients (not curated):"
            )
            for name in uncovered[:10]:
                logger.warning(f"  - {name}")
            if len(uncovered) > 10:
                logger.warning(f"  ... and {len(uncovered) - 10} more")
