"""Vision activation source -- wraps an in-memory activation matrix.

Adapts the existing DINOv3 path (`bsf.data`): activations are extracted and
normalised once in the notebook, then handed in as a single ``(N, d)`` tensor.
This is the map-style source; the trainer wraps it with a ``DistributedSampler``
under DDP. ``stats()`` is a no-op because the notebook already normalises.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .base import ActivationSource


class _RowDataset(Dataset):
    """Yields bare ``(d,)`` rows (not ``(row,)`` tuples) so the collated batch is
    ``(batch, d)`` -- matching the iterable captures source."""

    def __init__(self, x):
        self.x = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i]


class VisionSource(ActivationSource):
    is_iterable = False

    def __init__(self, x):
        # x: (N, d) already centered + scaled per the notebook convention.
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.d = int(self.x.shape[1])

    def dataset(self):
        return _RowDataset(self.x)

    def num_rows(self):
        return int(self.x.shape[0])
