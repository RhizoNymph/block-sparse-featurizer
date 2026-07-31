"""Dropping the first ``drop_first`` positions of every captured request.

Why: the first position of a sequence is an attention sink, and its residual
stream is a large-norm outlier that is not a concept. Measured on layer 32 of
the pile25m capture (7002 tokens, 12 requests):

    per-token max block norm   position 0: median 166.5
                               everything else: median 37.4   (4.5x)
    position-0 rows are 0.17% of all rows but 17.1% of the top-1% norm tail
    -> ~100x over-represented

They inflate the global norm statistics, the cold-start theta quantile, and the
top-activating-token lists. One packed capture unit holds exactly one request,
so "position 0" is row 0 of a unit.
"""
import json

import numpy as np
import pytest
import torch

from bsf import CapturesSource, normalize


def _write_packed(root, tag, req, layer, hook, arr, token_ids=None):
    d = root / tag / req
    d.mkdir(parents=True, exist_ok=True)
    b = arr.astype(np.float32).tobytes()
    (d / 'packed.bin').write_bytes(b)
    payload = {
        'layout': 'packed', 'dtype': 'float32',
        'entries': [{'layer': layer, 'hook': hook, 'shape': list(arr.shape),
                     'offset': 0, 'nbytes': len(b)}],
    }
    if token_ids is not None:
        payload['prompt_token_ids'] = list(token_ids)
    (d / 'packed.json').write_text(json.dumps(payload))


def _tree(root, n_req=5, rows=8, d=6, spike=100.0):
    """Each request: row 0 is a large-norm 'sink', the rest are unit-ish."""
    rng = np.random.default_rng(0)
    for r in range(n_req):
        arr = rng.standard_normal((rows, d)).astype(np.float32)
        arr[0] *= spike
        _write_packed(root, 'tg', f'r{r}', 5, 'post_block', arr,
                      token_ids=list(range(rows)))
    return n_req, rows, d


def _drain(source):
    return torch.stack(list(iter(source.dataset())))


def test_drop_first_removes_exactly_one_row_per_request(tmp_path):
    n_req, rows, d = _tree(tmp_path)
    keep = CapturesSource(tmp_path, 5, 'post_block', shuffle_buffer=4)
    drop = CapturesSource(tmp_path, 5, 'post_block', shuffle_buffer=4, drop_first=1)
    assert _drain(keep).shape[0] == n_req * rows
    assert _drain(drop).shape[0] == n_req * (rows - 1)


def test_drop_first_removes_the_high_norm_rows(tmp_path):
    _tree(tmp_path)
    kept = _drain(CapturesSource(tmp_path, 5, 'post_block', shuffle_buffer=4,
                                 drop_first=1))
    # the spiked sinks were ~100x; nothing that large may survive
    assert float(kept.norm(dim=1).max()) < 20.0


def test_num_rows_matches_what_is_yielded(tmp_path):
    """steps_per_epoch is derived from num_rows(); a mismatch silently truncates."""
    n_req, rows, _ = _tree(tmp_path)
    src = CapturesSource(tmp_path, 5, 'post_block', shuffle_buffer=4, drop_first=2)
    assert src.num_rows() == n_req * (rows - 2)
    assert _drain(src).shape[0] == src.num_rows()


def test_drop_first_zero_is_the_old_behaviour(tmp_path):
    n_req, rows, _ = _tree(tmp_path)
    a = _drain(CapturesSource(tmp_path, 5, 'post_block', shuffle_buffer=4))
    b = _drain(CapturesSource(tmp_path, 5, 'post_block', shuffle_buffer=4,
                              drop_first=0))
    assert a.shape == b.shape == (n_req * rows, 6)


def test_drop_first_larger_than_a_request_yields_nothing(tmp_path):
    _tree(tmp_path, rows=4)
    src = CapturesSource(tmp_path, 5, 'post_block', shuffle_buffer=4, drop_first=4)
    assert src.num_rows() == 0
    assert list(iter(src.dataset())) == []


def test_negative_drop_first_is_rejected(tmp_path):
    _tree(tmp_path)
    with pytest.raises(ValueError):
        CapturesSource(tmp_path, 5, 'post_block', drop_first=-1)


# ---------------------------------------------------------------------------
# norm stats must use the same filter, or training sees a mis-scaled input
# ---------------------------------------------------------------------------
def test_norm_stats_respect_drop_first(tmp_path):
    _tree(tmp_path)
    _, scale_all = normalize.compute_stats(tmp_path, 5, 'post_block')
    _, scale_dropped = normalize.compute_stats(tmp_path, 5, 'post_block',
                                               drop_first=1)
    # sinks carry huge energy -> including them shrinks the scale a lot
    assert scale_dropped > scale_all * 5


def test_norm_stats_cache_key_separates_drop_first(tmp_path):
    """A cached file from a different drop_first must not be silently reused."""
    _tree(tmp_path)
    _, s0 = normalize.load_or_compute(tmp_path, 5, 'post_block', drop_first=0)
    _, s1 = normalize.load_or_compute(tmp_path, 5, 'post_block', drop_first=1)
    assert s0 != s1
    # and re-loading each returns its own value
    assert normalize.load_or_compute(tmp_path, 5, 'post_block', drop_first=0)[1] == s0
    assert normalize.load_or_compute(tmp_path, 5, 'post_block', drop_first=1)[1] == s1


def test_normalized_rows_hit_the_target_convention(tmp_path):
    """mean ||x||^2 ~= d after normalisation, with the filter applied."""
    _, _, d = _tree(tmp_path, n_req=40, rows=12)
    mean, scale = normalize.compute_stats(tmp_path, 5, 'post_block', drop_first=1)
    rows = _drain(CapturesSource(tmp_path, 5, 'post_block', shuffle_buffer=8,
                                 mean=mean, scale=scale, drop_first=1))
    assert float(rows.pow(2).sum(1).mean()) == pytest.approx(d, rel=0.05)
