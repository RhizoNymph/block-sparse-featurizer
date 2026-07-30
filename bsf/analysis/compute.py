"""Computing the concept analysis: subspace geometry, statistics, examples.

Why chordal distance and not cosine similarity: a BSF concept is a whole
``group_size``-dim subspace of R^d (a ``(group_size, d)`` decoder block), and any
two bases spanning the same subspace describe the *same* concept. The chordal
distance between subspaces spanned by orthonormal ``Q_i, Q_j`` is

    dist(i, j)^2 = k - ||Q_i^T Q_j||_F^2 = sum_l sin^2(theta_l)

over principal angles ``theta_l``. It is invariant to within-block rotation,
which cosine-between-flattened-blocks is not.

Computing it for all pairs never needs per-pair SVDs: stack every orthonormal
basis into ``Qf`` (d, G*k), form ``Qf^T Qf`` and sum squares inside each (k, k)
tile. That is one matmul plus a reduction, chunked to bound memory.
"""
from __future__ import annotations

import numpy as np
import torch

from .types import AnalysisError


# ---------------------------------------------------------------------------
# subspace geometry
# ---------------------------------------------------------------------------
def orthonormal_bases(W_dec, n_groups, group_size):
    """(G*k, d) decoder -> (G, d, k) orthonormal bases of each block's row space.

    ``W_dec`` blocks are unit-Frobenius but not orthonormal, so the QR is what
    turns "a set of k atoms" into "a basis for the subspace they span".
    """
    blocks = W_dec.reshape(n_groups, group_size, -1)          # (G, k, d)
    # QR of the transpose: columns of Q span the same space as the block's rows.
    Q, _ = torch.linalg.qr(blocks.transpose(1, 2))            # (G, d, k)
    return Q


def chordal_distances(W_dec, n_groups, group_size, chunk=256, device=None,
                      dtype=torch.float64):
    """(G, G) chordal distances between concept subspaces. Symmetric, diag 0.

    ``dtype`` defaults to float64 deliberately: ``dist^2 = k - sum cos^2`` cancels
    catastrophically for near-identical subspaces (the interesting case -- close
    neighbours), and the subsequent sqrt amplifies the error. In float32 a
    zero distance comes back as ~1e-3; in float64 it is ~1e-8. The reduction is
    chunked, so the cost is bounded and paid once per analysis.
    """
    W = torch.as_tensor(W_dec).to(dtype)
    if device is not None:
        W = W.to(device)
    G, k = int(n_groups), int(group_size)
    Q = orthonormal_bases(W, G, k)                            # (G, d, k)
    d = Q.shape[1]
    Qf = Q.permute(1, 0, 2).reshape(d, G * k)                 # (d, G*k)

    out = torch.empty((G, G), dtype=dtype, device=Qf.device)
    for lo in range(0, G, max(int(chunk), 1)):
        hi = min(lo + max(int(chunk), 1), G)
        # (n*k, G*k) inner products, then squared Frobenius norm of each (k, k)
        # tile -> sum_l cos^2(theta_l) between block i and block j.
        M = Qf[:, lo * k:hi * k].t() @ Qf                     # (n*k, G*k)
        cos2 = M.reshape(hi - lo, k, G, k).pow(2).sum(dim=(1, 3))   # (n, G)
        out[lo:hi] = (float(k) - cos2).clamp_min(0.0).sqrt()
    # the closed form is symmetric up to float error; enforce it exactly
    out = 0.5 * (out + out.t())
    out.fill_diagonal_(0.0)
    return out


def nearest_neighbors(D, n=8):
    """Per row, the ``n`` closest OTHER indices and their distances."""
    D = torch.as_tensor(D).to(torch.float64).clone()
    G = D.shape[0]
    D.fill_diagonal_(float('inf'))
    n = min(int(n), G - 1)
    val, idx = torch.topk(D, n, dim=1, largest=False)
    return idx.cpu().numpy().astype(np.int32), val.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# co-activation
# ---------------------------------------------------------------------------
def coactivation(gate, n=8, chunk=512):
    """Top-``n`` co-firing partners per concept by Jaccard overlap.

    ``gate`` is (N, G) boolean: gate[t, g] iff concept g fires on token t.
    Jaccard = |A and B| / |A or B|; concepts that never fire get 0 (not nan).
    """
    g = torch.as_tensor(gate)
    gf = g.to(torch.float32)
    counts = gf.sum(0)                                        # (G,)
    G = gf.shape[1]
    out_idx = np.zeros((G, min(n, max(G - 1, 1))), dtype=np.int32)
    out_jac = np.zeros_like(out_idx, dtype=np.float32)
    nn = out_idx.shape[1]
    for lo in range(0, G, chunk):
        hi = min(lo + chunk, G)
        inter = gf[:, lo:hi].t() @ gf                         # (n, G)
        union = counts[lo:hi, None] + counts[None, :] - inter
        jac = torch.where(union > 0, inter / union.clamp_min(1e-9),
                          torch.zeros_like(inter))
        # never compare a concept with itself
        for r in range(hi - lo):
            jac[r, lo + r] = -1.0
        val, idx = torch.topk(jac, nn, dim=1)
        out_idx[lo:hi] = idx.cpu().numpy()
        out_jac[lo:hi] = val.clamp_min(0.0).cpu().numpy()
    return out_idx, out_jac


# ---------------------------------------------------------------------------
# co-activation graph + layout
#
# Measured on a trained 4096x4 dictionary at d=5120: pairwise chordal distances
# have median 1.9971 of a maximum 2.0 (spread 0.28% of the mean) and 2D captures
# only 0.63% of their variance -- 4096 4-planes in R^5120 are forced to be
# near-orthogonal, so *no* 2D projection of that metric is informative. Jaccard
# co-activation is likewise near-uniform globally (median distance 1.0) but its
# TAIL is sparse and real (~800 pairs above 0.3). So the honest global view is a
# graph of the strong relations, not an embedding of a near-constant metric.
# ---------------------------------------------------------------------------
def coactivation_graph(gate, threshold=0.1, max_edges=200_000, chunk=512):
    """Sparse strong co-firing graph: (E,2) edge indices + (E,) Jaccard weights.

    Only pairs with ``jaccard >= threshold`` become edges; ``i < j`` so each pair
    appears once. Edges are returned strongest-first and truncated to
    ``max_edges`` (the count kept vs. found is reported by the caller).
    """
    gf = torch.as_tensor(gate).to(torch.float32)
    counts = gf.sum(0)
    G = gf.shape[1]
    rows, cols, vals = [], [], []
    for lo in range(0, G, chunk):
        hi = min(lo + chunk, G)
        inter = gf[:, lo:hi].t() @ gf
        union = counts[lo:hi, None] + counts[None, :] - inter
        jac = torch.where(union > 0, inter / union.clamp_min(1e-9),
                          torch.zeros_like(inter))
        r, c = torch.where(jac >= threshold)
        keep = (r + lo) < c                                # upper triangle only
        r, c = r[keep], c[keep]
        if r.numel():
            rows.append(r + lo); cols.append(c); vals.append(jac[r, c])
    if not rows:
        return (np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.float32), 0)
    r = torch.cat(rows); c = torch.cat(cols); v = torch.cat(vals)
    order = torch.argsort(v, descending=True)
    found = int(v.numel())
    order = order[:max_edges]
    edges = torch.stack([r[order], c[order]], dim=1).cpu().numpy().astype(np.int32)
    return edges, v[order].cpu().numpy().astype(np.float32), found


def graph_layout(edges, weights, n_nodes, seed=0):
    """(n, 2) layout: spectral embedding of the graph's giant component, with
    everything else placed on an outer ring.

    Spectral (Laplacian eigenvector) layout is deterministic and needs only
    scipy, and unlike MDS it does not pretend a near-constant distance matrix has
    2D structure -- it lays out the *edges that exist*.
    """
    import numpy as _np
    from scipy.sparse import coo_matrix, csgraph

    pos = _np.zeros((n_nodes, 2), dtype=_np.float32)
    if len(edges) == 0:
        ang = _np.linspace(0, 2 * _np.pi, n_nodes, endpoint=False)
        return _np.stack([_np.cos(ang), _np.sin(ang)], 1).astype(_np.float32)

    w = _np.asarray(weights, dtype=_np.float64)
    A = coo_matrix((_np.concatenate([w, w]),
                    (_np.concatenate([edges[:, 0], edges[:, 1]]),
                     _np.concatenate([edges[:, 1], edges[:, 0]]))),
                   shape=(n_nodes, n_nodes)).tocsr()
    n_comp, labels = csgraph.connected_components(A, directed=False)
    sizes = _np.bincount(labels, minlength=n_comp)
    giant = int(_np.argmax(sizes))
    inside = _np.where(labels == giant)[0]

    if inside.size >= 3:
        sub = A[inside][:, inside]
        deg = _np.asarray(sub.sum(1)).ravel()
        deg[deg == 0] = 1.0
        # normalised Laplacian; take the 2nd/3rd smallest eigenvectors
        from scipy.sparse import diags
        dinv = diags(1.0 / _np.sqrt(deg))
        L = diags(_np.ones(inside.size)) - dinv @ sub @ dinv
        k = min(4, inside.size - 1)
        try:
            from scipy.sparse.linalg import eigsh
            vals, vecs = eigsh(L.tocsc(), k=k, sigma=-1e-5, which='LM')
        except Exception:                                  # pragma: no cover
            vals, vecs = _np.linalg.eigh(L.toarray())
        idx = _np.argsort(vals)[1:3]
        xy = vecs[:, idx] / _np.sqrt(deg)[:, None]
        # robust scaling so a few high-degree hubs do not flatten the rest
        for j in range(2):
            s = _np.percentile(_np.abs(xy[:, j]), 98) or 1.0
            xy[:, j] = _np.clip(xy[:, j] / s, -1.5, 1.5)
        pos[inside] = xy.astype(_np.float32)

    outside = _np.where(labels != giant)[0]
    if outside.size:
        rng = _np.random.default_rng(seed)
        ang = _np.sort(rng.random(outside.size)) * 2 * _np.pi
        rad = 2.0 + 0.25 * rng.random(outside.size)
        pos[outside] = _np.stack([rad * _np.cos(ang), rad * _np.sin(ang)],
                                 1).astype(_np.float32)
    return pos


def embedding_quality(D, n_sample=600, seed=0):
    """Fraction of classical-MDS variance the first 2 dims capture, and the number
    of dims needed for 50%. Lets the UI state honestly how much a 2D map can show."""
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if n > n_sample:
        sel = np.random.default_rng(seed).choice(n, n_sample, replace=False)
        D = D[np.ix_(sel, sel)]
        n = n_sample
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w = np.sort(np.linalg.eigvalsh(B))[::-1]
    pos = w[w > 0]
    if pos.size == 0:
        return 0.0, n
    frac2 = float(pos[:2].sum() / pos.sum())
    dims50 = int(np.searchsorted(np.cumsum(pos) / pos.sum(), 0.5) + 1)
    return frac2, dims50


# ---------------------------------------------------------------------------
# 2D embedding
# ---------------------------------------------------------------------------
def embed_distances(D, method='mds', seed=0):
    """(G, G) distances -> (G, 2) layout. ``method`` is 'mds' | 'tsne' | 'pca'."""
    D = np.asarray(D, dtype=np.float64)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    if method == 'mds':
        from sklearn.manifold import MDS
        # every default that sklearn 1.8 warns about is pinned explicitly so the
        # layout does not silently change under a library upgrade.
        m = MDS(n_components=2, metric='precomputed', random_state=seed,
                n_init=4, init='random', normalized_stress='auto')
        return np.asarray(m.fit_transform(D), dtype=np.float32)
    if method == 'tsne':
        from sklearn.manifold import TSNE
        per = float(min(30.0, max(5.0, (D.shape[0] - 1) / 3.0)))
        m = TSNE(n_components=2, metric='precomputed', init='random',
                 random_state=seed, perplexity=per)
        return np.asarray(m.fit_transform(D), dtype=np.float32)
    if method == 'pca':
        # classical MDS (PCA on the double-centred squared-distance matrix)
        n = D.shape[0]
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (D ** 2) @ J
        w, v = np.linalg.eigh(B)
        order = np.argsort(w)[::-1][:2]
        return np.asarray(v[:, order] * np.sqrt(np.maximum(w[order], 0)),
                          dtype=np.float32)
    raise AnalysisError(
        f'unknown embedding method {method!r}; expected mds | tsne | pca')


__all__ = ['orthonormal_bases', 'chordal_distances', 'nearest_neighbors',
           'coactivation', 'coactivation_graph', 'graph_layout',
           'embedding_quality', 'embed_distances']
