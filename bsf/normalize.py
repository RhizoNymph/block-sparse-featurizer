"""Activation normalization stats for LLM captures.

The featurizers expect activations centered and scaled so the mean squared norm
is ``d`` (``‖x‖ ≈ sqrt(d)``) -- the same convention `bsf.data` documents for the
vision path. For LLM captures we compute a single global mean vector ``(d,)`` and
a scalar scale in one streaming pass, then cache them to a JSON next to the
capture root so training runs (and all DDP ranks) share identical stats.

Given centered energy ``E = mean_row ‖x - mean‖²``, the scale is ``sqrt(d / E)``
so that ``mean_row ‖scale·(x - mean)‖² = d``. Using
``E = mean‖x‖² - ‖mean‖²`` keeps it to one pass.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from . import capture_format as cf


def _stats_path(root, layer, hook):
    return pathlib.Path(root) / f'norm_stats_l{layer}_{hook}.json'


def compute_stats(root, layer, hook, *, max_rows=None):
    """One streaming pass -> (mean:(d,) float32, scale:float). Not cached here."""
    units = cf.discover_units(root, layer, hook)
    d = None
    n = 0
    sum_x = None
    sum_sq = 0.0
    for unit in units:
        arr = unit.read(layer, hook)
        if arr.shape[0] == 0:
            continue
        x = arr.astype(np.float64) if arr.dtype != np.uint16 else \
            _bf16_to_f64(arr)
        if d is None:
            d = x.shape[1]
            sum_x = np.zeros(d, dtype=np.float64)
        sum_x += x.sum(0)
        sum_sq += float((x * x).sum())
        n += x.shape[0]
        if max_rows is not None and n >= max_rows:
            break
    if n == 0:
        raise ValueError(f'no rows for (layer={layer}, hook={hook!r}) under {root}')
    mean = sum_x / n
    mean_sq_norm = sum_sq / n              # mean ‖x‖²
    centered_energy = mean_sq_norm - float(mean @ mean)
    scale = float(np.sqrt(d / max(centered_energy, 1e-12)))
    return mean.astype(np.float32), scale


def _bf16_to_f64(arr_u16):
    """uint16 bf16 bits -> float64 (bf16 = top 16 bits of a float32)."""
    u32 = arr_u16.astype(np.uint32) << 16
    return u32.view(np.float32).astype(np.float64)


def load_or_compute(root, layer, hook, *, max_rows=None, recompute=False):
    """Return cached ``(mean, scale)`` or compute + cache them. Rank-0-safe:
    callers should compute on rank 0 and broadcast / re-load elsewhere."""
    path = _stats_path(root, layer, hook)
    if path.exists() and not recompute:
        blob = json.loads(path.read_text())
        return np.asarray(blob['mean'], dtype=np.float32), float(blob['scale'])
    mean, scale = compute_stats(root, layer, hook, max_rows=max_rows)
    blob = json.dumps({'layer': layer, 'hook': hook, 'd': int(mean.shape[0]),
                       'scale': scale, 'mean': mean.tolist()})
    # atomic write: several DDP ranks may compute identical stats concurrently;
    # tmp + replace means the final file is never a partial write.
    tmp = path.with_suffix(f'.json.tmp.{__import__("os").getpid()}')
    tmp.write_text(blob)
    tmp.replace(path)
    return mean, scale
