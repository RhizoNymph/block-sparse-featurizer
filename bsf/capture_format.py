"""Reader for vLLM filesystem-captured activations (vendored, NumPy-only).

Mirrors vLLM's ``vllm/v1/capture/consumers/filesystem/reader.py`` so the
training cluster can read captures without a vLLM install. The consumer writes
raw row-major ``.bin`` files + ``.json`` sidecars in one of three layouts:

``per_file``  one file per (layer, hook)::
    {root}/{tag}/{request}/{layer}_{hook}.bin    raw bytes, residual dtype
    {root}/{tag}/{request}/{layer}_{hook}.json   {shape:[rows,d], dtype, layer, hook, ...}

``packed``    one file per request, all (layer, hook) tensors concatenated::
    {root}/{tag}/{request}/packed.bin
    {root}/{tag}/{request}/packed.json           {layout:"packed", dtype,
                                                  entries:[{layer,hook,offset,nbytes,shape}]}

``sharded``   many requests share per-tag shard files (no request dir)::
    {root}/{tag}/shard-NNN-SSSSSS.bin
    {root}/{tag}/shard-NNN-SSSSSS.json           {layout:"sharded", dtype, seq,
                                                  entries:[{request_id,layer,hook,offset,nbytes,shape}]}

Bytes are native-endian, row-major, in the model's residual dtype; ``bfloat16``
is stored as raw ``uint16`` (NumPy has no native bf16) and returned as ``uint16``
here -- recover with ``torch.from_numpy(a).view(torch.bfloat16)``. A single
(layer, hook) may span several chunk entries (one per decode step); they are
concatenated in byte-offset order.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np

# On-disk filename conventions (must match the vLLM filesystem consumer).
PACKED_INDEX_NAME = 'packed.json'
PACKED_INDEX_GLOB = 'packed*.json'
SHARD_INDEX_GLOB = 'shard-*.json'

# Logical dtype string (as recorded in the sidecar) -> NumPy dtype used to
# interpret the on-disk bytes. bfloat16 has no NumPy equivalent, so its bytes
# are read as uint16 (their on-disk representation).
_DTYPE_TO_NUMPY = {
    'float64': 'float64', 'float32': 'float32', 'float16': 'float16',
    'bfloat16': 'uint16', 'uint16': 'uint16',
    'int8': 'int8', 'uint8': 'uint8',
    'int16': 'int16', 'int32': 'int32', 'int64': 'int64',
}
# Used when a per_file sidecar predates the self-describing ``dtype`` field.
_DEFAULT_DTYPE = 'float32'


@dataclass
class CaptureEntry:
    """One captured ``(layer, hook)`` tensor, decoded to NumPy."""
    layer: int
    hook: str
    array: np.ndarray  # (rows, d); bf16 captures come back uint16
    dtype: str         # logical dtype string from the sidecar (e.g. "bfloat16")


def _np_dtype(logical):
    try:
        return np.dtype(_DTYPE_TO_NUMPY[logical])
    except KeyError as exc:
        raise ValueError(
            f'unknown capture dtype {logical!r}; known: {sorted(_DTYPE_TO_NUMPY)}'
        ) from exc


def _decode(buf, shape, logical_dtype):
    return np.frombuffer(buf, dtype=_np_dtype(logical_dtype)).reshape(shape)


def _concat_offset_parts(parts):
    """parts: list[(offset, array)] -> single array concatenated in offset order."""
    parts.sort(key=lambda p: p[0])
    arrays = [a for _, a in parts]
    return arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)


def read_per_file(bin_path):
    """Read a single ``per_file`` capture (``{layer}_{hook}.bin`` + sidecar)."""
    bin_path = pathlib.Path(bin_path)
    sidecar = json.loads(bin_path.with_suffix('.json').read_text())
    dtype = sidecar.get('dtype', _DEFAULT_DTYPE)
    array = _decode(bin_path.read_bytes(), sidecar['shape'], dtype)
    return CaptureEntry(int(sidecar['layer']), str(sidecar['hook']), array, dtype)


def _read_one_packed_index(index_path):
    """Decode a single packed index (``packed*.json``) + its ``.bin``."""
    index = json.loads(index_path.read_text())
    dtype = index['dtype']
    raw = index_path.with_suffix('.bin').read_bytes()
    grouped = {}
    for entry in index['entries']:
        offset, nbytes = int(entry['offset']), int(entry['nbytes'])
        chunk = raw[offset:offset + nbytes]
        if len(chunk) != nbytes:
            raise ValueError(
                f'packed bin {index_path.with_suffix(".bin")} truncated: entry '
                f'({entry["layer"]}, {entry["hook"]}) wants [{offset}:{offset + nbytes}] '
                f'but file is {len(raw)} bytes')
        key = (int(entry['layer']), str(entry['hook']))
        grouped.setdefault(key, []).append((offset, _decode(chunk, entry['shape'], dtype)))
    return {
        (layer, hook): CaptureEntry(layer, hook, _concat_offset_parts(parts), dtype)
        for (layer, hook), parts in grouped.items()
    }


def read_packed(path):
    """Read a ``packed`` capture keyed by ``(layer, hook)``.

    ``path`` may be a packed index, a packed ``.bin``, or the request directory
    (which merges any per-pipeline-stage ``packed-pp{rank}.json`` files).
    """
    path = pathlib.Path(path)
    if path.is_dir():
        index_paths = sorted(path.glob(PACKED_INDEX_GLOB)) or [path / PACKED_INDEX_NAME]
    elif path.suffix == '.bin':
        index_paths = [path.with_suffix('.json')]
    else:
        index_paths = [path]
    out = {}
    for index_path in index_paths:
        out.update(_read_one_packed_index(index_path))
    return out


def read_request(request_dir):
    """Read every capture for a request, auto-detecting per_file vs packed."""
    request_dir = pathlib.Path(request_dir)
    if any(request_dir.glob(PACKED_INDEX_GLOB)):
        return read_packed(request_dir)
    out = {}
    for bin_path in sorted(request_dir.glob('*.bin')):
        if bin_path.name.startswith('packed'):
            continue
        entry = read_per_file(bin_path)
        out[(entry.layer, entry.hook)] = entry
    return out


def read_sharded(tag_dir):
    """Read every capture in a tag's sealed shard files, grouped by request.

    Returns ``{request_id: {(layer, hook): CaptureEntry}}``. Only sealed shards
    (those with a ``.json``) are visible.
    """
    tag_dir = pathlib.Path(tag_dir)
    grouped = {}  # (rid, layer, hook) -> (dtype, list[((seq, offset), array)])
    for index_path in sorted(tag_dir.glob(SHARD_INDEX_GLOB)):
        index = json.loads(index_path.read_text())
        dtype = index['dtype']
        seq = int(index.get('seq', 0))
        raw = index_path.with_suffix('.bin').read_bytes()
        for entry in index['entries']:
            offset, nbytes = int(entry['offset']), int(entry['nbytes'])
            chunk = raw[offset:offset + nbytes]
            if len(chunk) != nbytes:
                raise ValueError(
                    f'shard bin {index_path.with_suffix(".bin")} truncated: entry '
                    f'({entry["request_id"]}, {entry["layer"]}, {entry["hook"]}) wants '
                    f'[{offset}:{offset + nbytes}] but file is {len(raw)} bytes')
            key = (str(entry['request_id']), int(entry['layer']), str(entry['hook']))
            grouped.setdefault(key, (dtype, []))[1].append(
                ((seq, offset), _decode(chunk, entry['shape'], dtype)))
    out = {}
    for (rid, layer, hook), (dtype, parts) in grouped.items():
        parts.sort(key=lambda p: p[0])
        arrays = [a for _, a in parts]
        array = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
        out.setdefault(rid, {})[(layer, hook)] = CaptureEntry(layer, hook, array, dtype)
    return out


# ----------------------------------------------------------------------------
# Discovery -- enumerate independent read units under a capture root so the
# dataset can shard them across DDP ranks and DataLoader workers.
# ----------------------------------------------------------------------------
@dataclass
class ReadUnit:
    """An independently-readable file yielding rows for one ``(layer, hook)``.

    ``kind`` is 'per_file' | 'packed' | 'sharded'; ``path`` is the index/bin to
    read. ``read(layer, hook)`` returns an ``(n, d)`` array (possibly empty) of
    the rows this unit holds for that hook point.
    """
    kind: str
    path: pathlib.Path

    def read(self, layer, hook):
        if self.kind == 'per_file':
            return read_per_file(self.path).array
        if self.kind == 'packed':
            entry = read_packed(self.path).get((layer, hook))
            return entry.array if entry is not None else _empty()
        # sharded: read only THIS shard's rows for the hook (rows are treated
        # independently in training, so no cross-shard reassembly is needed).
        return _read_shard_hook(self.path, layer, hook)

    def count(self, layer, hook):
        """Rows this unit holds for ``(layer, hook)`` -- reads the sidecar only."""
        if self.kind == 'per_file':
            shape = json.loads(self.path.with_suffix('.json').read_text())['shape']
            return int(shape[0])
        index = json.loads(self.path.read_text())
        return sum(int(e['shape'][0]) for e in index['entries']
                   if int(e['layer']) == layer and str(e['hook']) == hook)


def _read_shard_hook(index_path, layer, hook):
    """All rows for one ``(layer, hook)`` in a single shard index file."""
    index = json.loads(index_path.read_text())
    dtype = index['dtype']
    raw = index_path.with_suffix('.bin').read_bytes()
    parts = []
    for entry in index['entries']:
        if int(entry['layer']) != layer or str(entry['hook']) != hook:
            continue
        offset, nbytes = int(entry['offset']), int(entry['nbytes'])
        chunk = raw[offset:offset + nbytes]
        if len(chunk) != nbytes:
            raise ValueError(f'shard bin {index_path.with_suffix(".bin")} truncated')
        parts.append((offset, _decode(chunk, entry['shape'], dtype)))
    return _concat_offset_parts(parts) if parts else _empty()


def _empty():
    return np.empty((0, 0), dtype=np.float32)


def discover_units(root, layer, hook):
    """List all read units under ``root`` that may hold ``(layer, hook)`` rows.

    Deterministically ordered (sorted paths) so DDP ranks shard the same global
    list identically. ``per_file`` matches ``{layer}_{hook}.bin`` exactly;
    ``packed`` and ``sharded`` index files are included unconditionally (they may
    or may not contain the hook -- ``ReadUnit.read`` returns empty if not).
    """
    root = pathlib.Path(root)
    units = []
    for p in sorted(root.rglob(f'{layer}_{hook}.bin')):
        units.append(ReadUnit('per_file', p))
    for p in sorted(root.rglob(PACKED_INDEX_GLOB)):
        units.append(ReadUnit('packed', p))
    # One unit per shard index; ReadUnit.read re-reads the whole tag dir, so we
    # deduplicate to a single unit per shard file but only read its own bin.
    for p in sorted(root.rglob(SHARD_INDEX_GLOB)):
        units.append(ReadUnit('sharded', p))
    return units


__all__ = [
    'CaptureEntry', 'ReadUnit',
    'read_per_file', 'read_packed', 'read_request', 'read_sharded',
    'discover_units',
]
