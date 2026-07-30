"""Activation-source abstraction.

An ``ActivationSource`` presents pre-extracted activations as ``(batch, d)`` fp32
rows, decoupled from where they came from (a frozen vision encoder or an LLM
residual stream). The trainer builds the ``DataLoader`` around ``dataset()`` so it
can inject a ``DistributedSampler`` (map-style sources) or rely on in-dataset
sharding (iterable sources).

Two normalization conventions live behind the same interface:
  - vision: rows are pre-normalised in the notebook -> ``stats() == (None, 1.0)``.
  - captures: ``stats()`` returns ``(mean:(d,), scale)`` so ``(x - mean) * scale``
    gives ``mean‖x‖² ≈ d`` (the convention the featurizers expect).
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ActivationSource(ABC):
    d: int              # activation dimension
    is_iterable: bool   # True -> dataset() is an IterableDataset (self-sharding)

    @abstractmethod
    def dataset(self):
        """A torch ``Dataset`` (map-style) or ``IterableDataset`` yielding (d,) rows."""

    def stats(self):
        """(mean:(d,) tensor | None, scale: float) applied as ``(x - mean) * scale``."""
        return None, 1.0

    def num_rows(self):
        """Total rows if known (for ``steps_per_epoch``), else ``None``."""
        return None
