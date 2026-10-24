"""
Compression mode configuration for gradient compression.

Defines the CompressionMode enum which specifies how gradients are compressed
for influence score computation and optimizer updates independently.
"""

from enum import Enum


class CompressionMode(Enum):
    """
    Defines how gradients are compressed for scoring and updates.

    Score compression and update compression are configured independently:
      - Score compression: compresses gradients for influence score computation
        (used by Layerwise/Subset curation methods)
      - Update compression: compresses gradients for memory-efficient optimizer
        updates (MeSO optimizer)

    Modes:
      - NONE: No compression. Full gradients everywhere.
      - SCORE_ONLY: Compressed scoring, full gradient updates.
      - UPDATE_ONLY: Full gradient scoring, compressed optimizer updates (MeSO).
      - FULL: Compressed scoring AND optimizer updates (may share compressors).
    """

    NONE = "none"
    """No compression. Full gradients for scoring and model updates."""

    SCORE_ONLY = "score"
    """Compress for scoring only. Full gradients for model updates."""

    UPDATE_ONLY = "update"
    """Compress for model updates only (MeSO). Full gradients for scoring."""

    FULL = "full"
    """Compressed gradients for both scoring and model updates."""

    @property
    def uses_compression(self) -> bool:
        """Check if this mode uses any compression."""
        return self != CompressionMode.NONE

    @property
    def uses_compressed_updates(self) -> bool:
        """Check if this mode uses compressed gradients for model updates."""
        return self in (CompressionMode.UPDATE_ONLY, CompressionMode.FULL)

    @property
    def uses_compressed_scoring(self) -> bool:
        """Check if this mode uses compressed gradients for scoring."""
        return self in (CompressionMode.SCORE_ONLY, CompressionMode.FULL)
