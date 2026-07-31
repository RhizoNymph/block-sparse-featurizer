"""Captures activation source -- streams offline LLM activations from disk.

Reads a vLLM filesystem-capture tree (`bsf.capture_format`) for one
``(layer, hook)`` point and streams normalised ``(d,)`` rows. Design points:

  - **Sharding**: the global read-unit list is split across the combined
    ``(DDP rank, DataLoader worker)`` grid so every row is read exactly once.
    Rank/world come from the environment (worker-safe; no collectives).
  - **Shuffle buffer**: rows within a capture file are request-ordered (adjacent
    tokens are correlated); a reservoir buffer decorrelates them. RNG is seeded
    per ``(seed, epoch, rank, worker)``.
  - **fp32 upcast**: bf16 captures come back as uint16 and are bit-reinterpreted
    via torch before ``.float()``.
  - **Normalization**: ``(row - mean) * scale`` applied here.
"""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .. import capture_format as cf
from ..distributed import env_rank_world
from .base import ActivationSource


def _to_fp32(arr, logical_dtype_is_bf16):
    """(rows, d) numpy -> owned (rows, d) fp32 torch, reinterpreting bf16-as-uint16.

    ``np.frombuffer`` yields a read-only array; we always return an owned,
    writable fp32 tensor (``.float()`` copies for bf16; ``copy=True`` for fp32)."""
    # np.array copies -> an owned, writable, C-contiguous array (no torch
    # read-only warning); the copy we would pay anyway on upcast.
    t = torch.from_numpy(np.array(arr))
    if logical_dtype_is_bf16:
        return t.view(torch.bfloat16).float()
    return t.to(torch.float32)


class CapturesDataset(IterableDataset):
    def __init__(self, root, layer, hook, *, shuffle_buffer=1 << 16,
                 mean=None, scale=1.0, seed=0, drop_first=0):
        self.root = root
        self.layer = layer
        self.hook = hook
        self.shuffle_buffer = int(shuffle_buffer)
        if int(drop_first) < 0:
            raise ValueError(f'drop_first must be >= 0, got {drop_first}')
        # One read unit holds one request, in sequence order, so this drops the
        # first `drop_first` sequence positions. Position 0 is an attention sink:
        # on layer 32 of pile25m its residual norm is ~4.5x the rest, putting
        # 0.17% of rows into 17% of the top-1% norm tail.
        self.drop_first = int(drop_first)
        self.mean = mean if mean is None else torch.as_tensor(mean, dtype=torch.float32)
        self.scale = float(scale)
        self.seed = int(seed)
        self.epoch = 0
        # deterministic global unit list (sorted paths) -> identical shard split
        # on every rank/worker.
        self.units = cf.discover_units(root, layer, hook)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _my_units(self):
        rank, world = env_rank_world()
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        nw = wi.num_workers if wi is not None else 1
        gid, gtot = rank * nw + wid, world * nw
        return self.units[gid::gtot], (rank, wid)

    def _normalize(self, t):
        if self.mean is not None:
            t = t - self.mean
        if self.scale != 1.0:
            t = t * self.scale
        return t

    def __iter__(self):
        my_units, (rank, wid) = self._my_units()
        rng = random.Random((self.seed, self.epoch, rank, wid).__hash__())
        buf = []
        cap = self.shuffle_buffer
        for unit in my_units:
            arr = unit.read(self.layer, self.hook)
            if arr.shape[0] <= self.drop_first:
                continue
            arr = arr[self.drop_first:]
            rows = self._normalize(_to_fp32(arr, arr.dtype == np.uint16))
            for r in rows:
                if len(buf) < cap:
                    buf.append(r)
                else:
                    # reservoir swap: emit a random slot, refill it with r.
                    j = rng.randrange(cap)
                    yield buf[j]
                    buf[j] = r
        rng.shuffle(buf)
        yield from buf


class CapturesSource(ActivationSource):
    is_iterable = True

    def __init__(self, root, layer, hook, *, shuffle_buffer=1 << 16,
                 mean=None, scale=1.0, seed=0, drop_first=0):
        self.root = root
        self.layer = layer
        self.hook = hook
        self.shuffle_buffer = shuffle_buffer
        self._mean = mean
        self._scale = scale
        self.seed = seed
        self.drop_first = int(drop_first)
        # infer d from the first non-empty unit's metadata via a cheap read.
        self._ds = CapturesDataset(root, layer, hook, shuffle_buffer=shuffle_buffer,
                                   mean=mean, scale=scale, seed=seed,
                                   drop_first=drop_first)
        self.d = self._infer_d()

    def _infer_d(self):
        for unit in self._ds.units:
            arr = unit.read(self.layer, self.hook)
            if arr.shape[0] > 0:
                return int(arr.shape[1])
        raise ValueError(
            f'no captures for (layer={self.layer}, hook={self.hook!r}) under {self.root}')

    def dataset(self):
        return self._ds

    def stats(self):
        return self._mean, self._scale

    def num_rows(self):
        # must match what __iter__ yields: steps_per_epoch is derived from this.
        return sum(max(u.count(self.layer, self.hook) - self.drop_first, 0)
                   for u in self._ds.units)
