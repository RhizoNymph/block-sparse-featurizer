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

CARD = {'background': 'rgba(127,127,127,0.06)', 'borderRadius': '10px',
        'padding': '10px 12px', 'border': '1px solid rgba(127,127,127,0.18)'}
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


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
    app = Dash(__name__, title=title or f'BSF concepts — layer {m.layer}')
    order = np.argsort(-an.fire_rate)
    default = int(order[0]) if len(order) else 0

    header = html.Div([
        html.Div([
            html.Strong(f'layer {m.layer} · {m.hook}'),
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
              'alignItems': 'center', 'marginBottom': '10px'})

    left = html.Div([
        dcc.Graph(id='map', style={'height': '460px'},
                  config={'displaylogo': False}),
        html.Div(dcc.Graph(id='hist', style={'height': '220px'},
                           config={'displaylogo': False}),
                 style={'marginTop': '10px'}),
    ], style={'flex': '1 1 58%', 'minWidth': '380px'})

    right = html.Div([
        html.Div(id='detail-head', style={'marginBottom': '6px'}),
        html.Div(id='examples', style={'maxHeight': '260px', 'overflowY': 'auto',
                                       'marginBottom': '10px'}),
        dcc.Graph(id='manifold', style={'height': '300px'},
                  config={'displaylogo': False}),
        html.Div([
            dcc.Graph(id='nbr-subspace', style={'height': '260px'},
                      config={'displaylogo': False}),
            dcc.Graph(id='nbr-coact', style={'height': '260px'},
                      config={'displaylogo': False}),
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr',
                  'gap': '8px', 'marginTop': '8px'}),
    ], style={'flex': '1 1 42%', 'minWidth': '340px'})

    app.layout = html.Div([
        header,
        html.Div([left, right],
                 style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap'}),
        dcc.Store(id='selected', data=default),
    ], style={'padding': '12px', 'fontFamily':
              'ui-sans-serif, system-ui, -apple-system, sans-serif',
              'maxWidth': '1600px', 'margin': '0 auto'})

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
        return F.concept_map(an, selected=g, color_by=color_by,
                             highlight=an.neighbor_idx[g].tolist())

    @app.callback(
        Output('detail-head', 'children'),
        Output('examples', 'children'),
        Output('manifold', 'figure'),
        Output('nbr-subspace', 'figure'),
        Output('nbr-coact', 'figure'),
        Output('hist', 'figure'),
        Input('selected', 'data'),
    )
    def _detail(g):
        g = int(g)
        head = html.Div([
            html.Strong(f'concept {g}', style={'fontSize': '15px'}),
            html.Span(f'   fires on {an.fire_rate[g]*100:.3f}% of tokens'
                      f'   ·   mean {an.mean_act[g]:.1f}'
                      f'   ·   max {an.max_act[g]:.1f}',
                      style={'opacity': 0.75, 'fontSize': '12px'}),
        ])
        return (head, _examples_block(an, g),
                F.concept_manifold(an, g),
                F.neighbor_bars(an, g, 'subspace'),
                F.neighbor_bars(an, g, 'coact'),
                F.stats_hist(an, g))

    return app


__all__ = ['build_app']
