"""Reader tests against synthesized captures matching the vLLM on-disk schema.

Bytes are raw row-major (``ndarray.tobytes()``); bf16 is stored as uint16.
"""
import json

import numpy as np
import pytest

from bsf import capture_format as cf


def _write(path, arr):
    path.write_bytes(arr.tobytes())


def _sidecar(path, payload):
    path.write_text(json.dumps(payload))


def make_per_file(root, tag, req, layer, hook, arr):
    d = root / tag / req
    d.mkdir(parents=True, exist_ok=True)
    _write(d / f'{layer}_{hook}.bin', arr)
    _sidecar(d / f'{layer}_{hook}.json',
             {'layer': layer, 'hook': hook, 'shape': list(arr.shape),
              'dtype': str(arr.dtype) if arr.dtype != np.uint16 else 'bfloat16'})


def make_packed(root, tag, req, tensors):
    """tensors: list[(layer, hook, arr)] -> one packed.bin + packed.json."""
    d = root / tag / req
    d.mkdir(parents=True, exist_ok=True)
    blob, entries, offset = b'', [], 0
    dtype = None
    for layer, hook, arr in tensors:
        b = arr.tobytes()
        dtype = str(arr.dtype) if arr.dtype != np.uint16 else 'bfloat16'
        entries.append({'layer': layer, 'hook': hook, 'offset': offset,
                        'nbytes': len(b), 'shape': list(arr.shape)})
        blob += b
        offset += len(b)
    (d / 'packed.bin').write_bytes(blob)
    _sidecar(d / 'packed.json', {'layout': 'packed', 'dtype': dtype, 'entries': entries})


def make_shard(root, tag, shard_idx, seq, rows):
    """rows: list[(request_id, layer, hook, arr)] -> one shard bin + json."""
    d = root / tag
    d.mkdir(parents=True, exist_ok=True)
    blob, entries, offset = b'', [], 0
    dtype = None
    for rid, layer, hook, arr in rows:
        b = arr.tobytes()
        dtype = str(arr.dtype) if arr.dtype != np.uint16 else 'bfloat16'
        entries.append({'request_id': rid, 'layer': layer, 'hook': hook,
                        'offset': offset, 'nbytes': len(b), 'shape': list(arr.shape)})
        blob += b
        offset += len(b)
    stem = f'shard-{shard_idx:03d}-{seq:06d}'
    (d / f'{stem}.bin').write_bytes(blob)
    _sidecar(d / f'{stem}.json',
             {'layout': 'sharded', 'shard_idx': shard_idx, 'seq': seq,
              'dtype': dtype, 'entries': entries})


def test_per_file_roundtrip(tmp_path):
    arr = np.random.randn(7, 4).astype(np.float32)
    make_per_file(tmp_path, 'tg', 'r0', 12, 'post_block', arr)
    entry = cf.read_per_file(tmp_path / 'tg' / 'r0' / '12_post_block.bin')
    assert entry.layer == 12 and entry.hook == 'post_block'
    np.testing.assert_array_equal(entry.array, arr)


def test_packed_multichunk(tmp_path):
    # same (layer, hook) split across two chunks -> concatenated in offset order
    a1 = np.random.randn(3, 4).astype(np.float32)
    a2 = np.random.randn(5, 4).astype(np.float32)
    other = np.random.randn(2, 4).astype(np.float32)
    make_packed(tmp_path, 'tg', 'r0',
                [(12, 'post_block', a1), (13, 'post_block', other), (12, 'post_block', a2)])
    got = cf.read_packed(tmp_path / 'tg' / 'r0')
    np.testing.assert_array_equal(got[(12, 'post_block')].array, np.concatenate([a1, a2]))
    np.testing.assert_array_equal(got[(13, 'post_block')].array, other)


def test_sharded_grouping(tmp_path):
    a = np.random.randn(4, 4).astype(np.float32)
    b = np.random.randn(6, 4).astype(np.float32)
    make_shard(tmp_path, 'tg', 0, 0,
               [('rA', 12, 'post_block', a), ('rB', 12, 'post_block', b)])
    by_req = cf.read_sharded(tmp_path / 'tg')
    np.testing.assert_array_equal(by_req['rA'][(12, 'post_block')].array, a)
    np.testing.assert_array_equal(by_req['rB'][(12, 'post_block')].array, b)


def test_bfloat16_returns_uint16(tmp_path):
    raw = np.random.randint(0, 65535, size=(5, 4), dtype=np.uint16)
    make_per_file(tmp_path, 'tg', 'r0', 12, 'post_block', raw)
    entry = cf.read_per_file(tmp_path / 'tg' / 'r0' / '12_post_block.bin')
    assert entry.dtype == 'bfloat16'
    assert entry.array.dtype == np.uint16
    np.testing.assert_array_equal(entry.array, raw)


def test_discover_units_reads_target_hook(tmp_path):
    # one of each layout, all holding (12, post_block); a decoy hook is ignored.
    pf = np.random.randn(3, 4).astype(np.float32)
    make_per_file(tmp_path, 'tg', 'rp', 12, 'post_block', pf)
    make_per_file(tmp_path, 'tg', 'rp', 12, 'pre_attn', np.random.randn(3, 4).astype(np.float32))
    pk = np.random.randn(5, 4).astype(np.float32)
    make_packed(tmp_path, 'tg', 'rk', [(12, 'post_block', pk),
                                       (9, 'post_block', np.random.randn(1, 4).astype(np.float32))])
    sh = np.random.randn(6, 4).astype(np.float32)
    make_shard(tmp_path, 'tg2', 0, 0, [('rs', 12, 'post_block', sh)])

    units = cf.discover_units(tmp_path, 12, 'post_block')
    kinds = sorted(u.kind for u in units)
    assert kinds == ['packed', 'per_file', 'sharded']
    rows = np.concatenate([u.read(12, 'post_block') for u in units])
    assert rows.shape == (3 + 5 + 6, 4)
    # every target row is present (order-independent set check via sums)
    want = np.concatenate([pf, pk, sh])
    np.testing.assert_allclose(np.sort(rows.sum(1)), np.sort(want.sum(1)), rtol=1e-6)


def test_packed_truncation_raises(tmp_path):
    arr = np.random.randn(3, 4).astype(np.float32)
    make_packed(tmp_path, 'tg', 'r0', [(12, 'post_block', arr)])
    binp = tmp_path / 'tg' / 'r0' / 'packed.bin'
    binp.write_bytes(binp.read_bytes()[:-8])  # truncate
    with pytest.raises(ValueError, match='truncated'):
        cf.read_packed(tmp_path / 'tg' / 'r0')
