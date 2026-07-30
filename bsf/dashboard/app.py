"""The Dash application: layout + callbacks over a loaded ``Analysis``.

Interaction model -- one selected concept drives every panel:
  click a point on the map (or type an index, or search a token) -> the detail
  column, both relation charts, and the manifold all follow. Neighbours of the
  selection are ringed on the map, so "where does this concept sit and what is it
  near" is answerable in one glance.
"""
from __future__ import annotations

import numpy as np
from dash import Dash, dcc, html, Input, Output, State, no_update

from . import figures as F

CARD = {'background': 'rgba(127,127,127,0.06)', 'borderRadius': '8px',
        'padding': '6px 8px', 'border': '1px solid rgba(127,127,127,0.18)'}
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# Explicit tab styling: Dash's stock tab is a fixed 44px box with its own
# font/padding, which clipped these labels. `height: auto` + real padding lets the
# text size the tab instead of the reverse.
TAB = {'padding': '7px 14px', 'fontSize': '12px', 'height': 'auto',
       'lineHeight': '1.35', 'whiteSpace': 'nowrap', 'background': 'transparent',
       'border': 'none', 'borderBottom': '2px solid transparent'}
TAB_ON = {**TAB, 'fontWeight': '600', 'borderBottom': '2px solid #e8833a',
          'background': 'rgba(232,131,58,0.10)'}

# Grid, not flex-wrap, for the two rows.
#
# The earlier version used `flex: 1 1 52%` / `1 1 48%` with `flex-wrap: wrap`.
# Wrapping is decided on flex-BASIS before any shrinking, and 52% + 48% + a 12px
# gap exceeds 100% of the container -- so the right panel always wrapped onto its
# own line instead of sitting beside the left one. An explicit two-column grid
# cannot wrap, and a media query handles genuinely narrow viewports.
CSS = """
.bsf-split  { display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(0, 0.95fr);
              gap: 8px; align-items: stretch; }
.bsf-bottom { display: grid; gap: 8px; margin-top: 8px;
              grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr) minmax(0, 1fr); }
@media (max-width: 980px) {
  .bsf-split  { grid-template-columns: minmax(0, 1fr); }
  .bsf-bottom { grid-template-columns: minmax(0, 1fr); }
}
/* keep Dash's stock 44px tab box from clipping the labels */
.bsf-tabs .tab { height: auto !important; padding: 7px 14px !important;
                 line-height: 1.35 !important; }
"""

INDEX = """<!DOCTYPE html>
<html>
  <head>
    {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
    <style>__CSS__</style>
  </head>
  <body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>""".replace('__CSS__', CSS)


def _examples_block(an, g):
    """Top-activating tokens with the firing token marked inside its context."""
    rows = []
    for ex in an.examples[g]:
        rows.append(html.Div([
            html.Span(f'{ex.activation:8.2f}  ',
                      style={'color': F.ACCENT, 'fontFamily': MONO,
                             'whiteSpace': 'pre'}),
            html.Span(ex.context_before, style={'opacity': 0.65}),
            html.Mark(ex.token, style={'background': 'rgba(232,131,58,0.28)',
                                       'padding': '0 2px', 'borderRadius': '3px'}),
            html.Span(ex.context_after, style={'opacity': 0.65}),
        ], style={'fontFamily': MONO, 'fontSize': '12px', 'whiteSpace': 'pre-wrap',
                  'padding': '2px 0', 'borderBottom': '1px solid rgba(127,127,127,0.12)'}))
    if not rows:
        rows = [html.Em('this concept never fired on the scanned tokens',
                        style={'opacity': 0.6})]
    return rows


BAND_TITLE = {
    'top': 'TOP — strongest firings',
    'p75': '75th percentile',
    'p50': 'MEDIAN — the typical firing',
    'p10': '10th percentile — just above threshold',
}


def _example_row(ex, dim=False):
    op = 0.45 if dim else 0.65
    return html.Div([
        html.Span(f'{ex.activation:7.1f}  ',
                  style={'color': F.ACCENT if not dim else F.FG,
                         'fontFamily': MONO, 'whiteSpace': 'pre'}),
        html.Span(ex.context_before, style={'opacity': op}),
        html.Mark(ex.token, style={
            'background': f'rgba(232,131,58,{0.28 if not dim else 0.12})',
            'padding': '0 2px', 'borderRadius': '3px'}),
        html.Span(ex.context_after, style={'opacity': op}),
    ], style={'fontFamily': MONO, 'fontSize': '12px', 'whiteSpace': 'pre-wrap',
              'padding': '1px 0'})


def _bands_block(an, g):
    """Activation-stratified examples: the honest way to judge a concept.

    A concept coherent only in its 'top' band is one whose typical activity is
    meaningless, so the bands are shown together rather than behind a toggle.
    """
    per = an.bands[g] if g < len(an.bands) else []
    if not per:
        return [html.Em('no banded examples in this artifact — re-run `bsf analyze`',
                        style={'opacity': 0.6})]
    hi = float(an.act_quantiles[g][4]) or 1.0
    out = []
    for band in per:
        frac = band.hi_act / hi
        weak = frac < 0.4          # empirically the uninformative region
        out.append(html.Div([
            html.Div([
                html.Strong(BAND_TITLE.get(band.label, band.label),
                            style={'fontSize': '11px',
                                   'color': F.FG if weak else 'inherit'}),
                html.Span(f'   act {band.lo_act:.1f}–{band.hi_act:.1f}'
                          f'   ({100*frac:.0f}% of max)'
                          f'   ranks {band.rank_lo}–{band.rank_hi}'
                          + ('   ← likely noise' if weak else ''),
                          style={'opacity': 0.6, 'fontSize': '11px'}),
            ], style={'borderBottom': '1px solid rgba(127,127,127,0.25)',
                      'marginBottom': '3px', 'paddingBottom': '2px'}),
            *[_example_row(e, dim=weak) for e in band.examples],
        ], style={'marginBottom': '8px',
                  'opacity': 0.75 if weak else 1.0}))
    return out


def _find_by_token(an, needle):
    """First concept whose top example token contains ``needle`` (case-folded)."""
    needle = needle.strip().casefold()
    if not needle:
        return None
    for g in range(an.meta.n_groups):
        for ex in an.examples[g]:
            if needle in ex.token.casefold():
                return g
    return None


def build_app(an, title=None):
    """Wire an ``Analysis`` into a Dash app. Returns the app (call ``.run``)."""
    an.validate()
    m = an.meta
    app = Dash(__name__, title=title or f'BSF concepts — layer {m.layer}',
               index_string=INDEX)
    order = np.argsort(-an.fire_rate)
    default = int(order[0]) if len(order) else 0

    header = html.Div([
        html.Div([
            html.Strong(f'Block Sparse Featurizer Exploration - layer {m.layer} · {m.hook}'),
            html.Span(f'  {m.n_groups} concepts × {m.group_size}-dim  ·  '
                      f'{m.n_tokens:,} tokens  ·  {m.model_kind}',
                      style={'opacity': 0.7}),
        ]),
        html.Div([
            dcc.Input(id='concept-input', type='number', min=0,
                      max=m.n_groups - 1, step=1, value=default,
                      debounce=True, style={'width': '90px', 'marginRight': '8px'}),
            dcc.Input(id='token-search', type='text', debounce=True,
                      placeholder='find a token…',
                      style={'width': '160px', 'marginRight': '8px'}),
            dcc.Dropdown(id='color-by', clearable=False,
                         options=[{'label': 'colour: fire rate', 'value': 'fire_rate'},
                                  {'label': 'colour: mean act', 'value': 'mean_act'},
                                  {'label': 'colour: max act', 'value': 'max_act'}],
                         value='fire_rate',
                         style={'width': '190px', 'display': 'inline-block',
                                'verticalAlign': 'middle'}),
        ], style={'display': 'flex', 'alignItems': 'center'}),
    ], style={**CARD, 'display': 'flex', 'justifyContent': 'space-between',
              'alignItems': 'center', 'marginBottom': '8px'})

    # Top-left: where the concept sits (graph) + how its firing rate compares.
    top_left = html.Div([
        dcc.Graph(id='map', style={'height': '440px'},
                  config={'displaylogo': False}),
        dcc.Graph(id='hist', style={'height': '210px', 'marginTop': '8px'},
                  config={'displaylogo': False}),
    ], style={**CARD, 'minWidth': '0'})

    # Top-right: is this concept trustworthy (profile) and what does it fire on.
    # A column flexbox so `examples` absorbs the leftover height and scrolls
    # inside itself rather than overrunning the panel below it.
    top_right = html.Div([
        html.Div(id='detail-head', style={'marginBottom': '4px'}),
        dcc.Graph(id='act-profile', style={'height': '160px', 'flex': '0 0 auto'},
                  config={'displaylogo': False}),
        dcc.Tabs(id='token-view', value='bands',
                 children=[
                     dcc.Tab(label='by activation band', value='bands',
                             style=TAB, selected_style=TAB_ON),
                     dcc.Tab(label='top-k only', value='top',
                             style=TAB, selected_style=TAB_ON),
                 ],
                 # no fixed height: Dash's default 44px tab box was clipping the
                 # labels. The tabs render no content of their own (the examples
                 # div lives outside), so its content area is collapsed.
                 content_style={'display': 'none'},
                 className='bsf-tabs',
                 parent_style={'flex': '0 0 auto', 'marginBottom': '2px'},
                 style={'height': 'auto', 'borderBottom':
                        '1px solid rgba(127,127,127,0.25)'}),
        html.Div(id='examples',
                 style={'flex': '1 1 auto', 'overflowY': 'auto',
                        'minHeight': '260px', 'paddingTop': '8px'}),
    ], style={**CARD, 'minWidth': '0',
              'display': 'flex', 'flexDirection': 'column'})

    # Bottom, full width: the geometry views.
    bottom = html.Div([
        dcc.Graph(id='manifold', style={'height': '330px'},
                  config={'displaylogo': False}),
        dcc.Graph(id='nbr-subspace', style={'height': '330px'},
                  config={'displaylogo': False}),
        dcc.Graph(id='nbr-coact', style={'height': '330px'},
                  config={'displaylogo': False}),
    ], className='bsf-bottom', style={**CARD})

    app.layout = html.Div([
        header,
        html.Div([top_left, top_right], className='bsf-split'),
        bottom,
        dcc.Store(id='selected', data=default),
    ], style={'padding': '8px', 'fontFamily':
              'ui-sans-serif, system-ui, -apple-system, sans-serif',
              # full-bleed: no max-width cap, so the panels use the whole window
              'maxWidth': 'none', 'margin': '0'})

    # ---- selection: map click / index box / token search all converge here
    @app.callback(
        Output('selected', 'data'),
        Output('concept-input', 'value'),
        Input('map', 'clickData'),
        Input('concept-input', 'value'),
        Input('token-search', 'value'),
        State('selected', 'data'),
        prevent_initial_call=True,
    )
    def _select(click, typed, needle, current):
        from dash import ctx
        src = ctx.triggered_id
        if src == 'map' and click:
            pt = click['points'][0]
            g = int(pt.get('customdata', pt.get('pointIndex', current)))
            return g, g
        if src == 'concept-input' and typed is not None:
            g = int(np.clip(int(typed), 0, an.meta.n_groups - 1))
            return g, g
        if src == 'token-search':
            g = _find_by_token(an, needle or '')
            if g is None:
                return no_update, no_update
            return g, g
        return no_update, no_update

    @app.callback(
        Output('map', 'figure'),
        Input('selected', 'data'),
        Input('color-by', 'value'),
    )
    def _map(g, color_by):
        g = int(g)
        # ring the concepts the map's EDGES lead to (co-activation partners), not
        # the chordal neighbours -- the map encodes co-activation, and chordal
        # distance carries almost no signal at this dictionary's near-orthogonality.
        partners, _ = F.connected(an, g)
        return F.concept_map(an, selected=g, color_by=color_by,
                             highlight=partners.tolist())

    @app.callback(
        Output('detail-head', 'children'),
        Output('examples', 'children'),
        Output('act-profile', 'figure'),
        Output('manifold', 'figure'),
        Output('nbr-subspace', 'figure'),
        Output('nbr-coact', 'figure'),
        Output('hist', 'figure'),
        Input('selected', 'data'),
        Input('token-view', 'value'),
    )
    def _detail(g, view):
        g = int(g)
        n_fires = int(round(float(an.fire_rate[g]) * an.meta.n_tokens))
        head = html.Div([
            html.Strong(f'concept {g}', style={'fontSize': '15px'}),
            html.Span(f'   fires on {an.fire_rate[g]*100:.3f}% of tokens'
                      f' (~{n_fires:,})'
                      f'   ·   mean {an.mean_act[g]:.1f}'
                      f'   ·   max {an.max_act[g]:.1f}',
                      style={'opacity': 0.75, 'fontSize': '12px'}),
        ])
        body = _bands_block(an, g) if view == 'bands' else _examples_block(an, g)
        return (head, body,
                F.act_profile(an, g),
                F.concept_manifold(an, g),
                F.neighbor_bars(an, g, 'subspace'),
                F.neighbor_bars(an, g, 'coact'),
                F.stats_hist(an, g))

    return app


__all__ = ['build_app']
