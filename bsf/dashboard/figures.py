"""Plotly figure builders for the concept dashboard.

Deliberately torch-free: everything here reads a loaded ``Analysis`` artifact, so
the dashboard runs on a laptop with only numpy + plotly + dash installed.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# One palette for the whole app so the four views read as one system.
BG = 'rgba(0,0,0,0)'
GRID = 'rgba(128,128,128,0.18)'
FG = '#8a8f98'
ACCENT = '#e8833a'
DIM = 'rgba(120,140,170,0.55)'


def _empty(msg):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(color=FG, size=13))
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=8, r=8, t=8, b=8))
    return fig


def _axes(fig, **kw):
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
                      font=dict(color=FG, size=11),
                      margin=dict(l=40, r=12, t=28, b=36),
                      showlegend=False, **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def connected(an, g):
    """Concepts sharing a co-activation edge with ``g``, strongest first.

    These are the partners the map's edges actually encode, so they -- not the
    chordal neighbours -- are what the map highlights.
    """
    if not len(an.edges):
        return np.zeros(0, dtype=int), np.zeros(0, dtype=np.float32)
    a, b = an.edges[:, 0], an.edges[:, 1]
    inc = (a == g) | (b == g)
    if not inc.any():
        return np.zeros(0, dtype=int), np.zeros(0, dtype=np.float32)
    partners = np.where(a[inc] == g, b[inc], a[inc])
    weights = np.asarray(an.edge_weight)[inc]
    order = np.argsort(-weights)
    return partners[order].astype(int), weights[order]


def _edge_xy(e, edges):
    """Flatten edges into None-separated polyline coordinates for one trace."""
    ex, ey = [], []
    for a, b in edges:
        ex += [e[a, 0], e[b, 0], None]
        ey += [e[a, 1], e[b, 1], None]
    return ex, ey


def concept_map(an, selected=None, color_by='fire_rate', highlight=()):
    """The global view: every concept placed by chordal subspace distance.

    Point colour encodes an activation statistic; ``highlight`` (neighbours of the
    selection) is drawn on top so relations are visible without a legend.
    """
    e = an.embedding
    stat = {'fire_rate': an.fire_rate, 'mean_act': an.mean_act,
            'max_act': an.max_act}.get(color_by, an.fire_rate)
    # log-scale firing rate: it spans orders of magnitude
    c = np.log10(np.maximum(stat, 1e-6)) if color_by == 'fire_rate' else stat
    label = {'fire_rate': 'log10 fire rate', 'mean_act': 'mean act',
             'max_act': 'max act'}.get(color_by, color_by)

    top = [(an.examples[g][0].token if an.examples[g] else '') for g in range(len(e))]
    hover = [f'concept {g}<br>fire {an.fire_rate[g]*100:.3f}%'
             f'<br>mean {an.mean_act[g]:.1f}<br>top: {t!r}'
             for g, t in enumerate(top)]

    fig = go.Figure()
    # Edges first so nodes draw over them. The selection's own edges are split into
    # their own trace and drawn in the accent colour, so "what connects to this
    # concept" is answerable at a glance rather than by reading the bar chart.
    partners = np.zeros(0, dtype=int)
    if an.meta.embedding_method == 'graph' and len(an.edges):
        inc = np.zeros(len(an.edges), dtype=bool)
        if selected is not None:
            a, b = an.edges[:, 0], an.edges[:, 1]
            inc = (a == selected) | (b == selected)
            partners, _ = connected(an, selected)
        bx, by = _edge_xy(e, an.edges[~inc])
        fig.add_trace(go.Scattergl(
            x=bx, y=by, mode='lines', hoverinfo='skip', name='co-firing',
            line=dict(color='rgba(120,140,170,0.18)', width=1)))
        if inc.any():
            ix, iy = _edge_xy(e, an.edges[inc])
            fig.add_trace(go.Scattergl(
                x=ix, y=iy, mode='lines', hoverinfo='skip', name='connected',
                line=dict(color='rgba(232,131,58,0.85)', width=2)))
    fig.add_trace(go.Scattergl(
        x=e[:, 0], y=e[:, 1], mode='markers', name='concepts',
        marker=dict(size=5, color=c, colorscale='Viridis', opacity=0.75,
                    colorbar=dict(title=dict(text=label, side='right'),
                                  thickness=10, len=0.7)),
        text=hover, hoverinfo='text', customdata=np.arange(len(e))))
    if len(highlight):
        h = np.asarray(list(highlight), dtype=int)
        fig.add_trace(go.Scattergl(
            x=e[h, 0], y=e[h, 1], mode='markers', name='neighbours',
            marker=dict(size=11, color='rgba(0,0,0,0)',
                        line=dict(color=ACCENT, width=2)),
            hoverinfo='skip'))
    if selected is not None:
        fig.add_trace(go.Scattergl(
            x=[e[selected, 0]], y=[e[selected, 1]], mode='markers',
            marker=dict(size=15, color=ACCENT,
                        line=dict(color='white', width=1.5)),
            hoverinfo='skip', name='selected'))
    m = an.meta
    if m.embedding_method == 'graph':
        sub = (f'co-activation graph · {len(an.edges)} of {m.coact_edges_found} '
               f'edges at Jaccard≥{m.coact_threshold:g}')
        if selected is not None:
            sub += (f'  ·  concept {selected}: {len(partners)} connected'
                    if len(partners) else
                    f'  ·  concept {selected} has no edges at this threshold')
    else:
        # be explicit that a chordal map is near-meaningless for this geometry
        sub = (f'{m.embedding_method} on chordal distance · 2D captures only '
               f'{m.chordal_2d_variance*100:.1f}% of variance '
               f'({m.chordal_dims_for_half} dims for 50%) — treat with suspicion')
    return _axes(fig, title=f'{len(e)} concepts — {sub}', hovermode='closest')


def concept_manifold(an, g):
    """The paper's signature view: one concept's firing cloud as a 3D manifold.

    Colour comes from each point's radial *direction* (as ``bsf.viz`` does) so hue
    says where on the manifold a token sits; opacity carries magnitude.
    """
    xyz = an.manifold_xyz[g]
    keep = np.abs(xyz).sum(1) > 0
    xyz = xyz[keep]
    if xyz.shape[0] < 4:
        return _empty('too few firing tokens for a manifold')
    toks = [an.vocab[i] if i < len(an.vocab) else ''
            for i in an.manifold_token_ids[g][keep]]

    n = np.linalg.norm(xyz, axis=1, keepdims=True)
    unit = xyz / np.maximum(n, 1e-8)
    rgb = (0.5 + 0.5 * unit).clip(0, 1)
    colors = [f'rgb({int(r*255)},{int(gg*255)},{int(b*255)})' for r, gg, b in rgb]
    inten = (n[:, 0] / max(n.max(), 1e-8)).clip(0.15, 1.0)

    fig = go.Figure(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers',
        marker=dict(size=4, color=colors, opacity=0.85),
        text=[f'{t!r}' for t in toks], hoverinfo='text'))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=FG, size=11),
        margin=dict(l=0, r=0, t=28, b=0), showlegend=False,
        title=f'concept {g} firing manifold ({xyz.shape[0]} tokens, '
              f'{an.meta.group_size}-dim subspace → PCA 3D)',
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False), bgcolor=BG))
    # opacity per point is not supported per-marker in Scatter3d; approximate by
    # sizing with magnitude so low-norm points recede.
    fig.data[0].marker.size = (3.0 + 4.0 * inten).tolist()
    return fig


def neighbor_bars(an, g, kind='subspace'):
    """Ranked relations for one concept: nearest subspaces, or co-firing partners."""
    if kind == 'subspace':
        idx, val = an.neighbor_idx[g], an.neighbor_dist[g]
        title, xlab = f'nearest subspaces to {g}', 'chordal distance (lower = closer)'
    else:
        idx, val = an.coact_idx[g], an.coact_jaccard[g]
        title, xlab = f'co-firing partners of {g}', 'Jaccard overlap'
    if not len(idx):
        return _empty('no relations')
    labels = [f'{int(i)}  {(an.examples[int(i)][0].token if an.examples[int(i)] else "")!r}'
              for i in idx]
    fig = go.Figure(go.Bar(x=list(val)[::-1], y=labels[::-1], orientation='h',
                           marker=dict(color=DIM, line=dict(color=ACCENT, width=1)),
                           hovertemplate='%{y}<br>%{x:.4f}<extra></extra>'))
    fig = _axes(fig, title=title, height=260)
    fig.update_xaxes(title=xlab)
    fig.update_yaxes(automargin=True)
    return fig


def stats_hist(an, g=None):
    """Corpus-level firing-rate distribution, with the selection marked."""
    fr = np.maximum(an.fire_rate, 1e-6)
    fig = go.Figure(go.Histogram(x=np.log10(fr), nbinsx=60,
                                 marker=dict(color=DIM,
                                             line=dict(color=GRID, width=1)),
                                 hovertemplate='log10 rate %{x:.2f}<br>'
                                               '%{y} concepts<extra></extra>'))
    if g is not None:
        fig.add_vline(x=float(np.log10(fr[g])), line=dict(color=ACCENT, width=2),
                      annotation_text=f'{g}', annotation_position='top')
    n_dead = int((an.fire_rate <= 0).sum())
    fig = _axes(fig, title=f'firing-rate distribution — {n_dead} never fired',
                height=220, bargap=0.02)
    fig.update_xaxes(title='log10 fire rate')
    fig.update_yaxes(title='concepts')
    return fig


def act_profile(an, g):
    """One concept's activation distribution, with the band cuts marked.

    The point of this chart: if the median sits far below the max, the concept's
    *typical* firing is near-threshold and carries none of its identity -- so the
    top-k examples above are unrepresentative of what the concept usually does.
    """
    q = an.act_quantiles[g]
    if not np.any(q):
        return _empty('never fired')
    lo, p25, p50, p75, hi = (float(v) for v in q)
    near = float(an.near_threshold_frac[g])

    fig = go.Figure()
    # the full firing range, with the interquartile span emphasised
    fig.add_trace(go.Scatter(x=[lo, hi], y=[0, 0], mode='lines',
                             line=dict(color=DIM, width=4), hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[p25, p75], y=[0, 0], mode='lines',
                             line=dict(color=ACCENT, width=12), hoverinfo='skip'))
    for x, lab in ((lo, 'min'), (p50, 'median'), (hi, 'max')):
        fig.add_trace(go.Scatter(
            x=[x], y=[0], mode='markers+text', text=[lab], textposition='top center',
            marker=dict(size=9, color='white', line=dict(color=ACCENT, width=2)),
            hovertemplate=f'{lab} {x:.1f}<extra></extra>'))
    # the 40%-of-max line: below it, tokens are empirically uninformative
    cut = 0.4 * hi
    fig.add_vline(x=cut, line=dict(color=FG, width=1, dash='dot'))
    fig = _axes(fig, height=170,
                title=f'activation profile — median is {100*p50/max(hi,1e-9):.0f}% '
                      f'of max · {near*100:.0f}% of firings below 40% of max')
    fig.update_yaxes(visible=False, range=[-1, 1])
    fig.update_xaxes(title='block activation', range=[0, hi * 1.08])
    return fig


__all__ = ['concept_map', 'concept_manifold', 'neighbor_bars', 'stats_hist',
           'act_profile']
