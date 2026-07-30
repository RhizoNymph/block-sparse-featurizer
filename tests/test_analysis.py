"""Tests for the concept-analysis artifact and its geometry.

The load-bearing claim is that concept relations are *subspace* relations: each
concept spans a ``group_size``-dim subspace of R^d, so the metric must be the
chordal distance between subspaces, invariant to any change of basis within a
block. These tests pin that down against a brute-force principal-angle
reference, plus the artifact's shape invariants and round-trip.
"""
import numpy as np
import pytest
import torch

from bsf.analysis import compute as C
from bsf.analysis.types import (
    Analysis, Meta, ConceptExample, ConceptBand, ARTIFACT_VERSION,
    ArtifactVersionError, ArtifactShapeError, AnalysisError,
)


# --------------------------------------------------------------------------
# chordal distance
# --------------------------------------------------------------------------
def _reference_chordal(blocks, i, j):
    """Brute-force chordal distance via principal angles (SVD of Qi^T Qj).

    Computed in float64: ``k - sum cos^2`` cancels badly, so a float32 oracle is
    itself only accurate to ~5e-4 and cannot validate a float64 implementation.
    """
    blocks = np.asarray(blocks, dtype=np.float64)
    Qi = np.linalg.qr(blocks[i].T)[0]        # (d, k) orthonormal basis
    Qj = np.linalg.qr(blocks[j].T)[0]
    s = np.linalg.svd(Qi.T @ Qj, compute_uv=False).clip(0, 1)
    k = blocks[i].shape[0]
    return float(np.sqrt(max(k - (s ** 2).sum(), 0.0)))


def test_chordal_matches_principal_angle_reference():
    torch.manual_seed(0)
    G, k, d = 12, 4, 32
    W = torch.randn(G * k, d)
    D = C.chordal_distances(W, G, k).numpy()
    blocks = W.reshape(G, k, d).numpy()
    for i, j in [(0, 1), (2, 7), (5, 5), (11, 3)]:
        assert D[i, j] == pytest.approx(_reference_chordal(blocks, i, j), abs=1e-4)


def test_chordal_self_distance_is_zero_and_symmetric():
    torch.manual_seed(1)
    G, k, d = 10, 3, 24
    D = C.chordal_distances(torch.randn(G * k, d), G, k)
    assert torch.allclose(D.diagonal(), torch.zeros(G, dtype=D.dtype), atol=1e-4)
    assert torch.allclose(D, D.t(), atol=1e-5)


def test_chordal_invariant_to_within_block_basis_change():
    """Rotating a block's rows spans the SAME subspace -> distances unchanged.

    This is the property cosine-similarity-between-vectors does NOT have, and
    the reason the dashboard uses a subspace metric.
    """
    torch.manual_seed(2)
    G, k, d = 8, 4, 20
    W = torch.randn(G * k, d)
    D0 = C.chordal_distances(W, G, k)

    blocks = W.reshape(G, k, d).clone()
    R = torch.linalg.qr(torch.randn(k, k))[0]          # orthogonal k x k
    blocks[3] = R @ blocks[3]                          # same row space
    D1 = C.chordal_distances(blocks.reshape(G * k, d), G, k)
    assert torch.allclose(D0, D1, atol=1e-4)


def test_chordal_orthogonal_subspaces_are_maximally_distant():
    k, d = 2, 8
    eye = torch.eye(d)
    W = torch.cat([eye[0:k], eye[k:2 * k]])            # two orthogonal 2-planes
    D = C.chordal_distances(W, 2, k)
    assert float(D[0, 1]) == pytest.approx(np.sqrt(k), abs=1e-4)


def test_chordal_identical_subspaces_are_zero_distance():
    k, d = 3, 12
    b = torch.randn(k, d)
    W = torch.cat([b, b * -2.0])                       # same span, scaled/flipped
    D = C.chordal_distances(W, 2, k)
    assert float(D[0, 1]) == pytest.approx(0.0, abs=1e-4)


def test_chordal_chunking_matches_unchunked():
    torch.manual_seed(3)
    G, k, d = 17, 3, 16
    W = torch.randn(G * k, d)
    assert torch.allclose(C.chordal_distances(W, G, k, chunk=4),
                          C.chordal_distances(W, G, k, chunk=G), atol=1e-5)


# --------------------------------------------------------------------------
# neighbours / co-activation
# --------------------------------------------------------------------------
def test_nearest_neighbours_exclude_self_and_are_sorted():
    torch.manual_seed(4)
    G, k, d = 20, 3, 16
    D = C.chordal_distances(torch.randn(G * k, d), G, k)
    idx, val = C.nearest_neighbors(D, n=5)
    assert idx.shape == (G, 5) and val.shape == (G, 5)
    for g in range(G):
        assert g not in idx[g].tolist()
        assert np.all(np.diff(val[g]) >= -1e-6)


def test_coactivation_jaccard_on_known_pattern():
    # concept 0 and 1 always fire together; 2 never fires with them
    gate = np.zeros((10, 3), dtype=bool)
    gate[:6, 0] = True
    gate[:6, 1] = True
    gate[6:, 2] = True
    idx, jac = C.coactivation(torch.from_numpy(gate), n=1)
    assert idx[0, 0] == 1 and jac[0, 0] == pytest.approx(1.0)
    assert idx[1, 0] == 0 and jac[1, 0] == pytest.approx(1.0)
    # concept 2 shares no token with either -> zero overlap
    assert jac[2, 0] == pytest.approx(0.0)


def test_coactivation_handles_never_firing_concept():
    gate = np.zeros((5, 3), dtype=bool)
    gate[:, 0] = True
    idx, jac = C.coactivation(torch.from_numpy(gate), n=2)
    assert np.all(np.isfinite(jac))          # no 0/0 -> nan
    assert jac.shape == (3, 2)


# --------------------------------------------------------------------------
# co-activation graph + layout
# --------------------------------------------------------------------------
def test_coactivation_graph_thresholds_and_dedupes():
    # 0-1 always together (J=1); 2 fires alone; 3 overlaps 0 weakly
    gate = np.zeros((20, 4), dtype=bool)
    gate[:10, 0] = True
    gate[:10, 1] = True
    gate[10:, 2] = True
    # concept 3 fires only on row 0, which BOTH 0 and 1 also cover, so
    # J(0,3) = J(1,3) = 1/10 = 0.1
    gate[:1, 3] = True
    edges, w, found = C.coactivation_graph(torch.from_numpy(gate), threshold=0.5)
    assert edges.tolist() == [[0, 1]] and w[0] == pytest.approx(1.0)
    assert found == 1
    # each undirected pair appears exactly once, i < j
    assert all(int(a) < int(b) for a, b in edges)
    # lower threshold admits both weak edges
    edges2, _, found2 = C.coactivation_graph(torch.from_numpy(gate), threshold=0.09)
    assert found2 == 3
    assert sorted(map(sorted, edges2.tolist())) == [[0, 1], [0, 3], [1, 3]]


def test_coactivation_graph_empty_when_nothing_overlaps():
    gate = np.zeros((6, 3), dtype=bool)
    gate[0, 0] = gate[1, 1] = gate[2, 2] = True
    edges, w, found = C.coactivation_graph(torch.from_numpy(gate), threshold=0.1)
    assert edges.shape == (0, 2) and w.shape == (0,) and found == 0


def test_coactivation_graph_respects_max_edges_and_reports_found():
    gate = np.ones((10, 6), dtype=bool)      # every pair J=1 -> 15 edges
    edges, w, found = C.coactivation_graph(torch.from_numpy(gate), threshold=0.5,
                                           max_edges=4)
    assert found == 15 and len(edges) == 4   # truncated but the true count survives
    assert np.all(np.diff(w) <= 1e-6)        # strongest first


def test_graph_layout_separates_two_clusters():
    """Spectral layout must place two disjoint cliques apart, which is exactly
    what an MDS of a near-constant distance matrix fails to do."""
    n = 8
    edges, ws = [], []
    for a in range(4):
        for b in range(a + 1, 4):
            edges.append((a, b)); ws.append(1.0)
            edges.append((a + 4, b + 4)); ws.append(1.0)
    edges.append((0, 4)); ws.append(0.01)            # one weak bridge
    pos = C.graph_layout(np.array(edges, dtype=np.int32), np.array(ws), n)
    assert pos.shape == (n, 2)
    c0, c1 = pos[:4].mean(0), pos[4:].mean(0)
    within = max(np.linalg.norm(pos[i] - c0) for i in range(4))
    assert np.linalg.norm(c0 - c1) > within           # clusters are resolved


def test_graph_layout_handles_no_edges_and_isolated_nodes():
    pos = C.graph_layout(np.zeros((0, 2), dtype=np.int32), np.zeros(0), 5)
    assert pos.shape == (5, 2) and np.all(np.isfinite(pos))
    # one connected pair + 3 isolated -> isolated pushed to an outer ring
    pos2 = C.graph_layout(np.array([[0, 1]], dtype=np.int32), np.array([1.0]), 5)
    assert np.all(np.isfinite(pos2))
    assert np.linalg.norm(pos2[4]) > 1.0


def test_embedding_quality_flags_near_orthogonal_metric():
    """A near-constant off-diagonal metric must report ~no 2D structure.

    This is the measurement that ruled out a chordal-distance map: real BSF
    subspaces sit at median distance 1.997/2.0 with 2D capturing <1%.
    """
    rng = np.random.default_rng(0)
    n = 80
    D = np.full((n, n), 2.0) + rng.normal(scale=1e-3, size=(n, n))
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    frac2, dims50 = C.embedding_quality(D)
    assert frac2 < 0.10
    assert dims50 > 5

    # a genuinely 2D metric must score high on the same measure
    pts = rng.normal(size=(n, 2))
    D2 = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    frac2b, dims50b = C.embedding_quality(D2)
    assert frac2b > 0.95 and dims50b <= 2


# --------------------------------------------------------------------------
# artifact round-trip + validation
# --------------------------------------------------------------------------
def _toy_analysis(G=4, k=2, P=3, n_neighbors=2):
    meta = Meta(version=ARTIFACT_VERSION, layer=32, hook='post_block', d=8,
                n_groups=G, group_size=k, n_tokens=100, n_units=5,
                checkpoint='toy.pt', model_kind='group_lasso',
                embedding_method='graph', seed=0,
                chordal_2d_variance=0.006, chordal_dims_for_half=276,
                coact_threshold=0.1, coact_edges_found=3)
    rng = np.random.default_rng(0)
    return Analysis(
        meta=meta,
        embedding=rng.normal(size=(G, 2)),
        edges=np.array([[0, 1]], dtype=np.int32),
        edge_weight=np.array([0.7], dtype=np.float32),
        neighbor_idx=np.tile(np.arange(n_neighbors), (G, 1)),
        neighbor_dist=np.abs(rng.normal(size=(G, n_neighbors))) * 0.1,
        coact_idx=np.tile(np.arange(n_neighbors), (G, 1)),
        coact_jaccard=rng.random((G, n_neighbors)),
        fire_rate=rng.random(G),
        mean_act=rng.random(G),
        max_act=rng.random(G),
        manifold_xyz=rng.normal(size=(G, P, 3)),
        manifold_token_ids=rng.integers(0, 3, size=(G, P)),
        act_quantiles=np.tile(np.array([1., 2., 3., 4., 5.]), (G, 1)),
        near_threshold_frac=np.full(G, 0.5),
        vocab=['a', 'b', 'c'],
        examples=[[ConceptExample(1.0, 3, 'tok', 'pre', 'post')] for _ in range(G)],
        bands=[[ConceptBand('top', 4.0, 5.0, 0, 2,
                            [ConceptExample(5.0, 1, 'hi', 'a', 'b')]),
                ConceptBand('p50', 2.0, 3.0, 10, 12,
                            [ConceptExample(2.5, 9, 'lo', 'c', 'd')])]
               for _ in range(G)],
    )


def test_artifact_roundtrip(tmp_path):
    a = _toy_analysis().validate()
    p = tmp_path / 'a.npz'
    a.save(p)
    b = Analysis.load(p)
    assert b.meta == a.meta
    assert b.vocab == a.vocab
    assert b.examples[0][0].token == 'tok'
    np.testing.assert_allclose(b.embedding, a.embedding, rtol=1e-5)
    np.testing.assert_array_equal(b.neighbor_idx, a.neighbor_idx)


def test_artifact_rejects_wrong_version(tmp_path):
    a = _toy_analysis()
    p = tmp_path / 'a.npz'
    a.save(p)
    # rewrite the JSON blob with a bogus version
    import json
    with np.load(p, allow_pickle=False) as z:
        arrays = {kk: z[kk] for kk in z.files}
    blob = json.loads(str(arrays['_json']))
    blob['meta']['version'] = ARTIFACT_VERSION + 99
    arrays['_json'] = np.array(json.dumps(blob))
    np.savez_compressed(p, **arrays)
    with pytest.raises(ArtifactVersionError):
        Analysis.load(p)


def test_validate_catches_shape_mismatch():
    a = _toy_analysis()
    a.fire_rate = np.zeros(a.meta.n_groups + 1)
    with pytest.raises(ArtifactShapeError):
        a.validate()


def test_validate_catches_out_of_range_distance():
    a = _toy_analysis()
    a.neighbor_dist = np.full_like(a.neighbor_dist, 99.0)
    with pytest.raises(AnalysisError):
        a.validate()


def test_validate_catches_manifold_token_mismatch():
    a = _toy_analysis()
    a.manifold_token_ids = np.zeros((a.meta.n_groups, 99), dtype=np.int32)
    with pytest.raises(ArtifactShapeError):
        a.validate()


def test_validate_catches_edge_weight_length_mismatch():
    a = _toy_analysis()
    a.edge_weight = np.zeros(a.edges.shape[0] + 3, dtype=np.float32)
    with pytest.raises(ArtifactShapeError):
        a.validate()


def test_validate_catches_out_of_range_edge_index():
    a = _toy_analysis()
    a.edges = np.array([[0, a.meta.n_groups + 5]], dtype=np.int32)
    a.edge_weight = np.array([0.5], dtype=np.float32)
    with pytest.raises(AnalysisError):
        a.validate()


# --------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------
def test_embedding_shape_and_determinism():
    torch.manual_seed(5)
    G, k, d = 30, 3, 16
    D = C.chordal_distances(torch.randn(G * k, d), G, k).numpy()
    e1 = C.embed_distances(D, method='mds', seed=0)
    e2 = C.embed_distances(D, method='mds', seed=0)
    assert e1.shape == (G, 2)
    np.testing.assert_allclose(e1, e2, rtol=1e-6)


def test_embedding_rejects_unknown_method():
    D = np.zeros((4, 4))
    with pytest.raises(AnalysisError):
        C.embed_distances(D, method='not-a-method', seed=0)


# --------------------------------------------------------------------------
# activation bands / quantiles
# --------------------------------------------------------------------------
def test_bands_survive_roundtrip(tmp_path):
    a = _toy_analysis().validate()
    p = tmp_path / 'b.npz'
    a.save(p)
    b = Analysis.load(p)
    assert len(b.bands) == a.meta.n_groups
    assert [x.label for x in b.bands[0]] == ['top', 'p50']
    assert b.bands[0][0].examples[0].token == 'hi'
    assert b.bands[0][1].rank_lo == 10
    np.testing.assert_allclose(b.act_quantiles, a.act_quantiles, rtol=1e-5)
    np.testing.assert_allclose(b.near_threshold_frac, a.near_threshold_frac, rtol=1e-5)


def test_validate_rejects_non_monotonic_quantiles():
    a = _toy_analysis()
    a.act_quantiles[1] = [5.0, 4.0, 3.0, 2.0, 1.0]      # descending: invalid
    with pytest.raises(AnalysisError):
        a.validate()


def test_validate_rejects_wrong_quantile_shape():
    a = _toy_analysis()
    a.act_quantiles = np.zeros((a.meta.n_groups, 3))
    with pytest.raises(ArtifactShapeError):
        a.validate()


def test_validate_rejects_bad_near_threshold_shape():
    a = _toy_analysis()
    a.near_threshold_frac = np.zeros(a.meta.n_groups + 2)
    with pytest.raises(ArtifactShapeError):
        a.validate()


# --------------------------------------------------------------------------
# concept manifold: the k-dim shortcut must be EXACTLY the d-dim PCA
# --------------------------------------------------------------------------
def test_cloud_to_3d_matches_full_dimensional_pca():
    """Projecting codes through the atoms' SVD must equal PCA of the d-dim cloud.

    build_analysis stores (k,)-dim codes instead of (d,)-dim contributions -- a
    ~1280x memory saving that is only legitimate if the resulting geometry is
    identical, up to per-axis sign (PCA axes have arbitrary orientation).
    """
    from bsf.analysis.build import _cloud_to_3d
    from bsf.viz import pca_fit
    rng = np.random.default_rng(0)
    n, k, d = 60, 4, 128
    A = rng.normal(size=(k, d))
    Z = rng.normal(size=(n, k))

    cheap = _cloud_to_3d(Z, A)
    C_full = Z @ A                                  # the (n, d) cloud, explicitly
    mean, comps = pca_fit(C_full, 3)
    ref = (C_full - mean) @ comps.T

    assert cheap.shape == ref.shape
    # pairwise distances are basis-independent -> must match exactly
    def pdist(X):
        return np.linalg.norm(X[:, None] - X[None], axis=-1)
    np.testing.assert_allclose(pdist(cheap), pdist(ref), rtol=1e-8, atol=1e-8)
    # and each axis matches up to sign
    for j in range(3):
        assert (np.allclose(cheap[:, j], ref[:, j], atol=1e-8)
                or np.allclose(cheap[:, j], -ref[:, j], atol=1e-8))


def test_cloud_to_3d_handles_rank_deficient_block():
    """A block whose atoms are linearly dependent must still project cleanly."""
    from bsf.analysis.build import _cloud_to_3d
    rng = np.random.default_rng(1)
    k, d = 4, 32
    A = rng.normal(size=(k, d))
    A[3] = A[0] * 2.0                                # rank 3, not 4
    out = _cloud_to_3d(rng.normal(size=(40, k)), A)
    assert out.shape == (40, 3) and np.all(np.isfinite(out))
