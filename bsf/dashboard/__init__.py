"""Interactive Dash dashboard for exploring a BSF's learned concepts.

Reads an ``Analysis`` artifact produced by ``bsf analyze``. Imports of ``dash`` /
``plotly`` are deferred into ``build_app`` so importing ``bsf`` (or running
``bsf train``) never requires the dashboard extra to be installed.
"""
from __future__ import annotations


def build_app(analysis, title=None):
    """Build the Dash app for ``analysis``. Requires the ``dashboard`` extra."""
    try:
        from .app import build_app as _build
    except ImportError as exc:                      # pragma: no cover
        raise ImportError(
            'the dashboard needs the optional extra: '
            'pip install "bsf[dashboard]"  (dash, plotly)') from exc
    return _build(analysis, title=title)


__all__ = ['build_app']
