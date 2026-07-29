"""Smoke tests for the dashboard layer.

These do not drive a browser; they assert the parts that silently rot: that every
figure builder produces a figure for ordinary AND degenerate concepts (never
fired, no manifold points, no relations), and that the app wires up without
touching torch.
"""
import numpy as np
import pytest

from bsf.analysis.types import Analysis, Meta, ConceptExample, ARTIFACT_VERSION

pytest.importorskip('plotly', reason='dashboard extra not installed')


def _analysis(G=6, k=4, P=8, n_nb=3, dead_last=True):
    # k=4 matches the trained dictionaries; chordal distances are then bounded by
    # sqrt(4)=2, which the 1.99 neighbour distances below sit just inside.
    rng = np.random.default_rng(0)
    meta = Meta(version=ARTIFACT_VERSION, layer=32, hook='post_block', d=16,
                n_groups=G, group_size=k, n_tokens=500, n_units=4,
                checkpoint='toy.pt', model_kind='group_lasso',
                embedding_method='graph', seed=0,
                chordal_2d_variance=0.006, chordal_dims_for_half=276,
                coact_threshold=0.1, coact_edges_found=9)
    fire = rng.random(G)
    manifold = rng.normal(size=(G, P, 3))
    examples = [[ConceptExample(10.0 - i, i, f'tok{i}', 'before ', ' after')
                 for i in range(3)] for _ in range(G)]
    if dead_last:
        fire[-1] = 0.0                     # a concept that never fired
        manifold[-1] = 0.0                 # -> no manifold points
        examples[-1] = []                  # -> no examples
    return Analysis(
        meta=meta,
        embedding=rng.normal(size=(G, 2)),
        edges=np.array([[0, 1], [1, 2], [3, 4]], dtype=np.int32),
        edge_weight=np.array([0.9, 0.5, 0.2], dtype=np.float32),
        neighbor_idx=np.tile(np.arange(n_nb), (G, 1)).astype(np.int32),
        neighbor_dist=np.full((G, n_nb), 1.99, dtype=np.float32),
        coact_idx=np.tile(np.arange(n_nb), (G, 1)).astype(np.int32),
        coact_jaccard=rng.random((G, n_nb)).astype(np.float32),
        fire_rate=fire.astype(np.float32),
        mean_act=rng.random(G).astype(np.float32),
        max_act=rng.random(G).astype(np.float32),
        manifold_xyz=manifold.astype(np.float32),
        manifold_token_ids=rng.integers(0, 3, size=(G, P)).astype(np.int32),
        vocab=['', 'a', 'b'],
        examples=examples,
    ).validate()


@pytest.mark.parametrize('color_by', ['fire_rate', 'mean_act', 'max_act'])
def test_concept_map_builds_for_each_colouring(color_by):
    from bsf.dashboard import figures as F
    an = _analysis()
    fig = F.concept_map(an, selected=0, color_by=color_by,
                        highlight=an.neighbor_idx[0])
    assert len(fig.data) >= 2                       # edges + nodes at minimum
    assert 'co-activation graph' in fig.layout.title.text


def test_chordal_map_title_discloses_low_variance():
    """A chordal-distance map must say how little 2D actually captures."""
    from bsf.dashboard import figures as F
    an = _analysis()
    an.meta = Meta(**{**an.meta.__dict__, 'embedding_method': 'mds'})
    title = F.concept_map(an, selected=0).layout.title.text
    assert '0.6%' in title and 'suspicion' in title


def test_manifold_handles_empty_cloud():
    from bsf.dashboard import figures as F
    an = _analysis()
    fig = F.concept_manifold(an, an.meta.n_groups - 1)   # the dead concept
    assert 'too few firing tokens' in str(fig.layout.annotations)


def test_manifold_builds_for_live_concept():
    from bsf.dashboard import figures as F
    an = _analysis()
    fig = F.concept_manifold(an, 0)
    assert len(fig.data) == 1 and len(fig.data[0].x) > 0


@pytest.mark.parametrize('kind', ['subspace', 'coact'])
def test_neighbor_bars_build(kind):
    from bsf.dashboard import figures as F
    fig = F.neighbor_bars(_analysis(), 0, kind)
    assert len(fig.data) == 1


def test_stats_hist_reports_dead_count():
    from bsf.dashboard import figures as F
    fig = F.stats_hist(_analysis(), 0)
    assert '1 never fired' in fig.layout.title.text


def test_build_app_wires_up():
    pytest.importorskip('dash', reason='dash not installed')
    from bsf.dashboard import build_app
    app = build_app(_analysis())
    assert app.layout is not None
    # the four views must all be present as graph ids
    ids = str(app.layout)
    for gid in ('map', 'manifold', 'nbr-subspace', 'nbr-coact', 'hist', 'examples'):
        assert gid in ids


def test_token_search_finds_concept():
    from bsf.dashboard.app import _find_by_token
    an = _analysis()
    assert _find_by_token(an, 'tok1') == 0
    assert _find_by_token(an, 'definitely-not-present') is None
    assert _find_by_token(an, '') is None
