"""
Gradient hook and data-selection utilities.

  GradientHook (hook.py)
    Monkey-patches Linear layers with custom autograd Functions.
    Captures per-sample gradients for influence scoring during the backward pass.

  Curation (selection/)
    Layerwise: per-layer curation in a single backward pass.
    Subset: global curation — scoring pass, then gradient pass on selected subset.
    Scoring methods: reduced_ghost (default), full_ghost, direct.

  CompressionMode (compression_mode.py)
    Derived from which compressor lists are populated:
      NONE, SCORE_ONLY, UPDATE_ONLY, FULL.

  ValidationCache (validation_cache.py)
    Stores validation gradients in factorized or full form.
"""

from .hook import GradientHook
from .compression_mode import CompressionMode
from .validation_cache import ValidationCache, ValidationStorageMode

# Curation module exports (gradient-based)
from .selection import (
    SelectionState,
    LayerwiseState,
    SubsetState,
    # SeparateBatch strategies
    SeparateBatchStrategy,
    SeparateBatchNoSelectionStrategy,
    SeparateBatchLayerwiseStrategy,
    SeparateBatchSubsetStrategy,
    SeparateBatchSubsetOnePassStrategy,
    create_separate_batch_strategy,
)

__all__ = [
    # Core components
    "GradientHook",
    # Compression configuration
    "CompressionMode",
    # Validation cache
    "ValidationCache",
    "ValidationStorageMode",
    # Curation state classes
    "SelectionState",
    "LayerwiseState",
    "SubsetState",
    # SeparateBatch strategy classes
    "SeparateBatchStrategy",
    "SeparateBatchNoSelectionStrategy",
    "SeparateBatchLayerwiseStrategy",
    "SeparateBatchSubsetStrategy",
    "SeparateBatchSubsetOnePassStrategy",
    "create_separate_batch_strategy",
]
