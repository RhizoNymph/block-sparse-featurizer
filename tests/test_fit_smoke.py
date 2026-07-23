"""End-to-end smoke of the source-driven trainer (single process, CPU).

Exercises both `fit` paths: map-style (VisionSource) and iterable (CapturesSource
over a synthesized capture tree). Uses structured sparse-dictionary data so a
featurizer can actually raise R2 -- the assertion is that training improves it.
"""
import json

import numpy as np
import torch

from bsf import fit, train, recon_r2, VisionSource, CapturesSource
from bsf.group_lasso import GroupLassoBSF
from bsf.vanilla import VanillaBSF
from bsf.grassmannian import GrassmannianBSF


def _structured_data(n, d, k, l0, seed=0):
    """n rows built as sparse combinations of a random (k, d) dictionary."""
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(k, d, generator=g)
    D = D / D.norm(dim=1, keepdim=True)
    X = torch.zeros(n, d)
    for i in range(n):
        idx = torch.randperm(k, generator=g)[:l0]
        X[i] = (torch.randn(l0, 1, generator=g) * D[idx]).sum(0)
    # normalise to the convention: mean ‖x‖² ≈ d
    X = X - X.mean(0, keepdim=True)
    X = X * (d / X.pow(2).sum(1).mean()).sqrt()
    return X


def test_fit_vision_improves_r2():
    torch.manual_seed(0)
    d, n_groups, gs = 16, 8, 2
    X = _structured_data(3000, d, n_groups * gs, l0=3)
    model = GroupLassoBSF(d, n_groups, gs, coef=1e-3, target_l0=3)

    sub = X[:1000]
    pre = recon_r2(model, sub)
    fit(model, VisionSource(X), epochs=30, lr=5e-3, batch_size=64, snr=0.0,
        device='cpu', log_every=100, num_workers=0)
    post = recon_r2(model, sub)
    assert post > pre + 0.05, f'no learning: pre={pre:.3f} post={post:.3f}'


def _write_packed(root, tag, req, layer, hook, arr):
    d = root / tag / req
    d.mkdir(parents=True, exist_ok=True)
    b = arr.astype(np.float32).tobytes()
    (d / 'packed.bin').write_bytes(b)
    (d / 'packed.json').write_text(json.dumps({
        'layout': 'packed', 'dtype': 'float32',
        'entries': [{'layer': layer, 'hook': hook, 'offset': 0,
                     'nbytes': len(b), 'shape': list(arr.shape)}]}))


def test_fit_captures_runs(tmp_path):
    torch.manual_seed(0)
    d, n_groups, gs = 16, 8, 2
    X = _structured_data(2400, d, n_groups * gs, l0=3).numpy()
    # spread across several packed requests
    for r, s in enumerate(range(0, len(X), 400)):
        _write_packed(tmp_path, 'tg', f'r{r}', 5, 'post_block', X[s:s + 400])

    from bsf import normalize
    mean, scale = normalize.load_or_compute(tmp_path, 5, 'post_block')
    source = CapturesSource(tmp_path, 5, 'post_block', shuffle_buffer=256,
                            mean=mean, scale=scale)
    assert source.d == d
    assert source.num_rows() == 2400

    model = VanillaBSF(d, n_groups, gs, l0=3)
    fit(model, source, epochs=6, lr=5e-3, batch_size=64, snr=0.0, device='cpu',
        log_every=100, num_workers=0, steps_per_epoch=20)
    # trained model produces finite, non-degenerate reconstructions
    xb = torch.from_numpy(X[:200])
    xb = (xb - torch.as_tensor(mean)) * scale
    r2 = recon_r2(model, xb)
    assert np.isfinite(r2)


def test_fit_grassmannian_improves_r2():
    torch.manual_seed(0)
    d, n_groups, gs = 16, 8, 2
    X = _structured_data(3000, d, n_groups * gs, l0=3)
    model = GrassmannianBSF(d, n_groups, gs, l0=3)
    sub = X[:1000]
    pre = recon_r2(model, sub)
    fit(model, VisionSource(X), epochs=30, lr=5e-3, batch_size=64, snr=0.0,
        device='cpu', log_every=100, num_workers=0)
    assert recon_r2(model, sub) > pre + 0.05


def test_legacy_train_regression():
    """The original in-memory trainer still runs and learns (numerics path
    untouched by the DDP/compile changes)."""
    torch.manual_seed(0)
    d, n_groups, gs = 16, 8, 2
    X = _structured_data(2000, d, n_groups * gs, l0=3)
    model = GroupLassoBSF(d, n_groups, gs, coef=1e-3, target_l0=3)
    pre = recon_r2(model, X[:500])
    train(model, X, epochs=20, lr=5e-3, batch_size=64, snr=0.0, device='cpu',
          log_every=100)
    assert recon_r2(model, X[:500]) > pre + 0.05
