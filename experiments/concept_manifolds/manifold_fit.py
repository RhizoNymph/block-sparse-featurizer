"""Fit a gamfit sphere manifold to each BSF concept and render it.

The construction, mirroring gam's manifold-SAE figure but with BSF's native
geometry:

  1. a concept's firing tokens each carry a signed code z in the concept's own
     group_size-dim subspace (no manifold fit needed to GET coordinates -- a flat
     SAE has only a direction, so it must fit one);
  2. take the cloud's top-3 principal axes and normalise to unit length: the
     concept's *directions* then lie exactly on S^2;
  3. fit a gamfit `Sphere` smooth of activation magnitude over that sphere by
     REML -- an intrinsic manifold smooth, not a Euclidean one;
  4. decode the fitted surface on a lat/lon grid and draw the tokens at their
     positions on it.

The fit's held-out-free R^2 is reported per concept, so "is this concept a smooth
function on its own manifold?" gets a number rather than a vibe.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import gamfit
from gamfit.torch import fit as tfit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK = '#0a0f1e'
FG = '#cfd3da'
MONO = {'family': 'monospace'}


def sphere_coords(Z):
    """(n, k) codes -> (unit (n,3) directions, latlon (n,2) degrees)."""
    C = Z.astype(np.float64)
    C = C - C.mean(0)
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    P = C @ Vt[:3].T if Vt.shape[0] >= 3 else np.pad(C @ Vt.T, ((0, 0), (0, 3 - Vt.shape[0])))
    n = np.linalg.norm(P, axis=1, keepdims=True)
    U = P / np.maximum(n, 1e-12)
    lat = np.degrees(np.arcsin(np.clip(U[:, 2], -1, 1)))
    lon = np.degrees(np.arctan2(U[:, 1], U[:, 0]))
    return U, np.stack([lat, lon], 1), n[:, 0]


def fit_sphere(latlon, y, n_centers=18):
    """Fit activation magnitude as a smooth function over the concept's sphere.

    The `Sphere` basis carries NO intercept, so the response must be centred --
    otherwise the basis has to spend itself representing a constant offset that
    the penalty is simultaneously shrinking, and in-sample R^2 goes negative
    (median -5.5 before this fix). The mean is added back for display.
    """
    spec = gamfit.Sphere(n_centers=n_centers, penalty_order=2,
                         kernel='sobolev', radians=False)
    y = np.asarray(y, dtype=np.float64)
    y0 = float(y.mean())
    r = tfit(torch.as_tensor(latlon, dtype=torch.float64),
             torch.as_tensor(y - y0, dtype=torch.float64), spec)
    f = r.fitted.detach().cpu().numpy().reshape(-1) + y0
    ss = 1.0 - ((y - f) ** 2).sum() / max(((y - y0) ** 2).sum(), 1e-12)
    coef = r.coefficients.detach().cpu().numpy().reshape(-1)
    edf = float(np.sum(r.edf.detach().cpu().numpy())) if hasattr(r, 'edf') else float('nan')
    return dict(r2=float(ss), coef=coef, edf=edf, spec=spec, fitted=f,
                y0=y0)


def decode_surface(spec, coef, y0=0.0, n_lat=44, n_lon=88):
    """Evaluate the fitted smooth on a lat/lon grid -> (LAT, LON, VAL, XYZ)."""
    lat = np.linspace(-89.0, 89.0, n_lat)
    lon = np.linspace(-180.0, 180.0, n_lon)
    LA, LO = np.meshgrid(lat, lon, indexing='ij')
    pts = np.stack([LA.ravel(), LO.ravel()], 1)
    B, _ = gamfit.sphere_basis(pts, spec.n_centers, penalty_order=2,
                              kernel='sobolev', radians=False)
    B = np.asarray(B, dtype=np.float64)
    m = min(B.shape[1], coef.shape[0])
    V = (B[:, :m] @ coef[:m]).reshape(LA.shape) + y0
    la, lo = np.radians(LA), np.radians(LO)
    X = np.cos(la) * np.cos(lo)
    Y = np.cos(la) * np.sin(lo)
    Zc = np.sin(la)
    return LA, LO, V, (X, Y, Zc)


def render(ax, U, y, surf, tokens, title, sub, label_n=10, seed=0, s_pt=None):
    LA, LO, V, (X, Y, Zc) = surf
    ax.set_facecolor(INK)
    # graticule of the fitted sphere (the wireframe)
    step_i, step_j = max(LA.shape[0] // 11, 1), max(LA.shape[1] // 16, 1)
    for i in range(0, LA.shape[0], step_i):
        ax.plot(X[i], Y[i], Zc[i], color='#5566aa', lw=0.35, alpha=0.30, zorder=1)
    for j in range(0, LA.shape[1], step_j):
        ax.plot(X[:, j], Y[:, j], Zc[:, j], color='#5566aa', lw=0.35, alpha=0.30, zorder=1)
    # tokens at their positions on the manifold, coloured by fitted value
    # point size shrinks as the cloud gets denser, so a 9k-token concept reads
    # as a surface rather than a solid blob
    sp = s_pt if s_pt is not None else max(0.8, min(3.0, 4000.0 / max(len(U), 1)))
    ax.scatter(U[:, 0], U[:, 1], U[:, 2], c=y, cmap='magma', s=sp,
               edgecolors='none', alpha=0.85, zorder=3)
    # a few token labels, spread over the sphere
    rng = np.random.default_rng(seed)
    if tokens is not None and label_n:
        pick = rng.choice(len(U), min(label_n, len(U)), replace=False)
        for i in pick:
            t = tokens[i]
            if not t or not t.strip():
                continue
            ax.text(U[i, 0] * 1.14, U[i, 1] * 1.14, U[i, 2] * 1.14, t.strip()[:14],
                    color=FG, fontsize=7, ha='center', zorder=5, **MONO)
    ax.set_title(title, color=FG, fontsize=11, pad=6, **MONO)
    ax.text2D(0.5, 0.955, sub, transform=ax.transAxes, color='#8a93a5',
              fontsize=8, ha='center', **MONO)
    ax.set_axis_off()
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)
    ax.view_init(elev=16, azim=-60)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--codes', required=True)
    ap.add_argument('--min-firings', type=int, default=400)
    ap.add_argument('--n-fit', type=int, default=150, help='concepts to fit')
    ap.add_argument('--n-centers', type=int, default=18)
    ap.add_argument('--out-census', default='sphere_census.json')
    ap.add_argument('--out-fig', default='bsf_manifold.png')
    ap.add_argument('--gallery', type=int, default=6)
    args = ap.parse_args()

    z = np.load(args.codes, allow_pickle=True)
    codes, offs, counts = z['codes'], z['offsets'], z['counts']
    vocab = list(z['vocab'])
    tokids = z['token_ids']
    meta = json.loads(str(z['meta']))
    print(f'layer {meta["layer"]} · {meta["n_groups"]} concepts × '
          f'{meta["group_size"]}-dim · {meta["n_tokens"]:,} tokens')

    elig = np.nonzero(counts >= args.min_firings)[0]
    rng = np.random.default_rng(0)
    pick = rng.choice(elig, min(args.n_fit, len(elig)), replace=False)
    print(f'{len(elig)} concepts with >= {args.min_firings} firings; '
          f'fitting {len(pick)}\n')

    rows = []
    for g in pick:
        g = int(g)
        Z = codes[offs[g]:offs[g + 1]]
        U, latlon, mag = sphere_coords(Z)
        try:
            f = fit_sphere(latlon, mag, args.n_centers)
        except Exception as exc:
            print(f'  concept {g}: fit failed ({type(exc).__name__})')
            continue
        rows.append(dict(g=g, n=int(counts[g]), r2=f['r2'], edf=f['edf']))
    rows.sort(key=lambda r: -r['r2'])
    r2s = np.array([r['r2'] for r in rows])
    print(f'sphere-smooth R2 over {len(rows)} concepts:')
    for q in (10, 25, 50, 75, 90):
        print(f'   p{q:<3d} {np.percentile(r2s, q):.3f}')
    print(f'   best {r2s.max():.3f} (concept {rows[0]["g"]}) · '
          f'worst {r2s.min():.3f}')
    json.dump(rows, open(args.out_census, 'w'), indent=1)

    # ---- the beauty shot: best-fitting concept, full page
    best = rows[0]['g']
    Z = codes[offs[best]:offs[best + 1]]
    U, latlon, mag = sphere_coords(Z)
    f = fit_sphere(latlon, mag, args.n_centers)
    surf = decode_surface(f['spec'], f['coef'], f['y0'])
    toks = [vocab[t] if 0 <= t < len(vocab) else '' for t in
            tokids[offs[best]:offs[best + 1]]]
    toks = [t.replace('Ġ', ' ').replace('Ċ', '\\n') for t in toks]

    fig = plt.figure(figsize=(11, 11), facecolor=INK)
    ax = fig.add_subplot(111, projection='3d')
    render(ax, U, f['fitted'], surf, toks,
           'a BSF concept on its own manifold',
           f'concept {best} · layer {meta["layer"]} · {counts[best]:,} firing tokens · '
           f'gamfit Sphere smooth R2={f["r2"]:.3f}, edf={f["edf"]:.1f}')
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=140, facecolor=INK)
    print('\n->', args.out_fig)

    # ---- gallery of the next best
    gal = rows[:args.gallery]
    ncol = 3
    nrow = int(np.ceil(len(gal) / ncol))
    fig2 = plt.figure(figsize=(4.6 * ncol, 4.6 * nrow), facecolor=INK)
    for i, r in enumerate(gal, 1):
        g = r['g']
        Zg = codes[offs[g]:offs[g + 1]]
        Ug, ll, mg = sphere_coords(Zg)
        try:
            fg = fit_sphere(ll, mg, args.n_centers)
        except Exception:
            continue
        sg = decode_surface(fg['spec'], fg['coef'], fg['y0'], 30, 60)
        tk = [vocab[t] if 0 <= t < len(vocab) else '' for t in
              tokids[offs[g]:offs[g + 1]]]
        tk = [t.replace('Ġ', ' ').replace('Ċ', '\\n') for t in tk]
        ax = fig2.add_subplot(nrow, ncol, i, projection='3d')
        render(ax, Ug, fg['fitted'], sg, tk, f'concept {g}',
               f'{r["n"]:,} tokens · R2={fg["r2"]:.3f}', label_n=5, seed=i)
    fig2.tight_layout()
    out2 = args.out_fig.replace('.png', '_gallery.png')
    fig2.savefig(out2, dpi=130, facecolor=INK)
    print('->', out2)


if __name__ == '__main__':
    main()
