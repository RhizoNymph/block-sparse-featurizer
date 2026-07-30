"""What shape is each BSF concept's code cloud? (CPU only, reads the artifact.)

gam's manifold-SAE fits a parametrized manifold per atom. A BSF concept instead
*owns* a group_size-dim subspace, and every firing token has a signed code in it,
so the concept's "manifold" is the shape of that code cloud -- already stored in
the analysis artifact as manifold_xyz (3D PCA per concept).

Their README claims a nonnegative-gate dictionary shatters a circular feature into
up to four rectified half-atoms. BSF keeps the FULL SIGNED code, so a circular
feature should stay in one block as a ring. This classifies every concept's cloud:

  intrinsic_dim  participation ratio of the PCA eigenvalues (1 = line, 3 = ball)
  annularity     in the top-2 plane: mean(r)/std(r); high => points sit on a
                 ring rather than filling the disc
  hollow_2d      1 - (density in the inner 40% of the radius / uniform expectation)
  antipodal      cos-similarity of the cloud with its own negation, matched by
                 nearest neighbour -- high means the cloud is symmetric about 0,
                 which is what a SIGNED code buys you and a rectified gate destroys
"""
from __future__ import annotations

import argparse

import numpy as np

from bsf.analysis import Analysis


def shape_stats(P):
    """P: (n, 3) cloud -> dict of shape descriptors."""
    P = P[np.abs(P).sum(1) > 0]
    n = len(P)
    if n < 30:
        return None
    C = np.cov((P - P.mean(0)).T)
    w = np.sort(np.linalg.eigvalsh(C))[::-1].clip(0, None)
    if w.sum() <= 0:
        return None
    p = w / w.sum()
    # participation ratio: 1 for a line, 2 for a disc, 3 for a ball
    idim = 1.0 / (p ** 2).sum()

    xy = P[:, :2] - P[:, :2].mean(0)
    r = np.linalg.norm(xy, axis=1)
    rmax = np.percentile(r, 98) or 1.0
    annul = float(r.mean() / max(r.std(), 1e-9))
    inner = float((r < 0.4 * rmax).mean())
    # a uniform disc puts 16% of its mass inside 40% of the radius
    hollow = float(1.0 - inner / 0.16) if inner < 0.16 else 0.0

    # antipodal symmetry: for each point, distance to the nearest point of -P,
    # normalised by the cloud scale. Symmetric cloud -> small.
    m = min(n, 200)
    A = P[:m]
    d = np.linalg.norm(A[:, None] + A[None], axis=-1)   # |a - (-b)|
    scale = np.linalg.norm(A, axis=1).mean() or 1.0
    anti = float(1.0 - np.median(d.min(1)) / (2 * scale))
    return dict(n=n, idim=float(idim), annul=annul, hollow=hollow, anti=anti,
                ev=p[:3].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis', required=True)
    ap.add_argument('--out', default='bsf_manifolds.png')
    ap.add_argument('--n-show', type=int, default=12)
    args = ap.parse_args()

    an = Analysis.load(args.analysis)
    G = an.meta.n_groups
    rows = []
    for g in range(G):
        s = shape_stats(an.manifold_xyz[g])
        if s:
            s['g'] = g
            rows.append(s)
    print(f'{len(rows)}/{G} concepts with enough cloud points '
          f'(group_size={an.meta.group_size})\n')

    idim = np.array([r['idim'] for r in rows])
    annul = np.array([r['annul'] for r in rows])
    hollow = np.array([r['hollow'] for r in rows])
    anti = np.array([r['anti'] for r in rows])
    print('intrinsic dim (participation ratio):')
    for q in (5, 25, 50, 75, 95):
        print(f'   p{q:<3d} {np.percentile(idim, q):.2f}')
    print(f'\nring-like (annularity > 3, hollow > 0.3): '
          f'{int(((annul > 3) & (hollow > 0.3)).sum())}/{len(rows)}')
    print(f'antipodally symmetric (anti > 0.6):        '
          f'{int((anti > 0.6).sum())}/{len(rows)}')
    print(f'\nmedian annularity {np.median(annul):.2f} | '
          f'median hollow {np.median(hollow):.2f} | median anti {np.median(anti):.2f}')

    # gallery: the most ring-like clouds
    score = annul * (1 + hollow)
    order = np.argsort(-score)[:args.n_show]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ncol = 4
    nrow = int(np.ceil(len(order) / ncol))
    fig = plt.figure(figsize=(3.2 * ncol, 3.2 * nrow), facecolor='#0a0f1e')
    for i, oi in enumerate(order, 1):
        r = rows[oi]
        g = r['g']
        P = an.manifold_xyz[g]
        P = P[np.abs(P).sum(1) > 0]
        ax = fig.add_subplot(nrow, ncol, i, projection='3d')
        ax.set_facecolor('#0a0f1e')
        ang = np.arctan2(P[:, 1], P[:, 0])
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=ang, cmap='twilight', s=7,
                   edgecolors='none', alpha=0.9)
        tok = an.examples[g][0].token if an.examples[g] else ''
        ax.set_title(f'concept {g}  {tok!r}\nidim {r["idim"]:.2f} · ring '
                     f'{r["annul"]:.1f} · sym {r["anti"]:.2f}',
                     color='#cfd3da', fontsize=8, pad=1)
        ax.set_axis_off()
        ax.view_init(elev=20, azim=-60)
        lim = np.percentile(np.abs(P), 98) or 1.0
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
    fig.tight_layout()
    fig.savefig(args.out, dpi=125, facecolor='#0a0f1e')
    print('\n->', args.out)


if __name__ == '__main__':
    main()
