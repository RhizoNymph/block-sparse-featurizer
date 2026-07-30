"""Smoke tests for the dashboard layer.

These do not drive a browser; they assert the parts that silently rot: that every
figure builder produces a figure for ordinary AND degenerate concepts (never
fired, no manifold points, no relations), and that the app wires up without
touching torch.
"""
import numpy as np
import pytest

from bsf.analysis.types import (Analysis, Meta, ConceptExample, ConceptBand,
                                ARTIFACT_VERSION)

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
    # concept 0: coherent only at the top (max 100, median 20) -> mostly noise
    # concept 1: flat profile (max 30, median 28) -> uniformly meaningful
    quant = np.tile(np.array([15., 18., 20., 25., 100.]), (G, 1))
    quant[1] = [25., 26., 28., 29., 30.]
    near = np.full(G, 0.8, dtype=np.float32)
    near[1] = 0.0
    bands = [[ConceptBand('top', 90.0, 100.0, 0, 2,
                          [ConceptExample(100.0, 1, 'sleep', 'could not ', ' well')]),
              ConceptBand('p50', 19.0, 20.0, 50, 52,
                          [ConceptExample(20.0, 7, ' the', 'in ', ' room')])]
             for _ in range(G)]
    if dead_last:
        fire[-1] = 0.0                     # a concept that never fired
        manifold[-1] = 0.0                 # -> no manifold points
        examples[-1] = []                  # -> no examples
        bands[-1] = []
        quant[-1] = 0.0
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
        act_quantiles=quant.astype(np.float32),
        near_threshold_frac=near,
        vocab=['', 'a', 'b'],
        examples=examples,
        bands=bands,
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


# --------------------------------------------------------------------------
# activation-band view
# --------------------------------------------------------------------------
def test_act_profile_reports_skew():
    from bsf.dashboard import figures as F
    an = _analysis()
    # concept 0: median 20 of max 100 -> 20% of max, 80% near threshold
    t = F.act_profile(an, 0).layout.title.text
    assert 'median is 20% of max' in t
    assert '80% of firings below 40% of max' in t


def test_act_profile_flat_concept_reads_as_healthy():
    from bsf.dashboard import figures as F
    t = F.act_profile(_analysis(), 1).layout.title.text
    assert 'median is 93% of max' in t and '0% of firings below' in t


def test_act_profile_empty_for_dead_concept():
    from bsf.dashboard import figures as F
    an = _analysis()
    fig = F.act_profile(an, an.meta.n_groups - 1)
    assert 'never fired' in str(fig.layout.annotations)


def test_bands_block_flags_weak_bands():
    from bsf.dashboard.app import _bands_block
    an = _analysis()
    txt = str(_bands_block(an, 0))
    # the p50 band sits at 20% of max -> must be marked as likely noise
    assert 'likely noise' in txt
    assert 'MEDIAN' in txt and 'TOP' in txt


def test_bands_block_does_not_flag_healthy_concept():
    from bsf.dashboard.app import _bands_block
    txt = str(_bands_block(_analysis(), 1))
    assert 'likely noise' not in txt


def test_bands_block_handles_missing_bands():
    from bsf.dashboard.app import _bands_block
    an = _analysis()
    txt = str(_bands_block(an, an.meta.n_groups - 1))
    assert 're-run' in txt


def test_app_exposes_band_view_controls():
    pytest.importorskip('dash', reason='dash not installed')
    from bsf.dashboard import build_app
    ids = str(build_app(_analysis()).layout)
    assert 'act-profile' in ids and 'token-view' in ids


# --------------------------------------------------------------------------
# co-activation edge highlighting
# --------------------------------------------------------------------------
def test_connected_returns_partners_strongest_first():
    from bsf.dashboard.figures import connected
    an = _analysis()          # edges: (0,1,w=.9) (1,2,w=.5) (3,4,w=.2)
    p, w = connected(an, 1)
    assert p.tolist() == [0, 2]                 # both neighbours of 1
    assert w.tolist() == pytest.approx([0.9, 0.5])
    assert list(w) == sorted(w, reverse=True)


def test_connected_is_symmetric_over_edge_direction():
    from bsf.dashboard.figures import connected
    an = _analysis()
    assert connected(an, 0)[0].tolist() == [1]   # edge stored as (0,1)
    assert connected(an, 4)[0].tolist() == [3]   # edge stored as (3,4)


def test_connected_empty_for_unconnected_concept():
    from bsf.dashboard.figures import connected
    an = _analysis()
    p, w = connected(an, 5)                      # concept 5 has no edges
    assert p.size == 0 and w.size == 0


def test_map_splits_selected_edges_into_own_trace():
    from bsf.dashboard import figures as F
    an = _analysis()
    names = [t.name for t in F.concept_map(an, selected=1).data]
    assert 'co-firing' in names and 'connected' in names
    # the accent trace must carry only the 2 edges incident to concept 1
    conn = [t for t in F.concept_map(an, selected=1).data if t.name == 'connected'][0]
    assert len(conn.x) == 2 * 3                  # 2 edges x (a, b, None)


def test_map_omits_connected_trace_when_selection_has_no_edges():
    from bsf.dashboard import figures as F
    an = _analysis()
    fig = F.concept_map(an, selected=5)
    assert 'connected' not in [t.name for t in fig.data]
    assert 'has no edges' in fig.layout.title.text


def test_map_title_counts_connected_partners():
    from bsf.dashboard import figures as F
    t = F.concept_map(_analysis(), selected=1).layout.title.text
    assert 'concept 1: 2 connected' in t
