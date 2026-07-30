"""Typed analysis artifact: the contract between ``bsf analyze`` and the dashboard.

The artifact is deliberately self-describing and torch-free once written, so the
dashboard process needs only numpy (no torch, no GPU, no capture root). Anything
the UI needs is precomputed here; anything the UI cannot recompute is validated
on load rather than trusted.

Invariants (enforced by ``Analysis.validate``, raised as typed errors):
  - every per-concept array has leading dim ``n_groups``
  - ``embedding`` is (n_groups, 2); ``neighbor_*`` are (n_groups, n_neighbors)
  - ``manifold_xyz`` is (n_groups, n_manifold_points, 3) and
    ``manifold_token_ids`` indexes ``vocab``
  - ``neighbor_dist`` holds CHORDAL distances in [0, sqrt(group_size)]
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field, asdict

import numpy as np

# Bumped whenever the on-disk layout changes incompatibly.
ARTIFACT_VERSION = 3


class AnalysisError(Exception):
    """Base for all analysis/artifact failures."""


class ArtifactVersionError(AnalysisError):
    """On-disk artifact was written by an incompatible version."""

    def __init__(self, found, expected=ARTIFACT_VERSION):
        super().__init__(
            f'analysis artifact version {found} != expected {expected}; '
            f're-run `bsf analyze`')
        self.found = found
        self.expected = expected


class ArtifactShapeError(AnalysisError):
    """An artifact array has a shape inconsistent with the declared metadata."""

    def __init__(self, name, got, want):
        super().__init__(f'artifact field {name!r} has shape {got}, expected {want}')
        self.name = name
        self.got = got
        self.want = want


class CheckpointMismatchError(AnalysisError):
    """Checkpoint geometry disagrees with what was requested on the CLI."""

    def __init__(self, what, ckpt_value, requested):
        super().__init__(
            f'{what} in checkpoint is {ckpt_value} but {requested} was requested')
        self.what = what
        self.ckpt_value = ckpt_value
        self.requested = requested


@dataclass(frozen=True)
class ConceptExample:
    """One top-activating token occurrence for a concept.

    ``context_before``/``context_after`` are already-decoded strings so the
    dashboard needs no tokenizer.
    """
    activation: float
    position: int
    token: str
    context_before: str
    context_after: str


@dataclass(frozen=True)
class ConceptBand:
    """Examples drawn from one activation band of a concept's firing distribution.

    Top-k examples alone are the extreme tail of a concept's firings and make an
    incoherent concept look clean (and vice versa). Measured on a trained
    dictionary, the median concept's median firing sits at ~39% of its own max and
    ~53% of its firings fall below 40% of max -- i.e. about half of every
    concept's activity is near-threshold and carries none of its identity. These
    bands expose that directly.
    """
    label: str            # 'top' | 'p75' | 'p50' | 'p10'
    lo_act: float         # activation range covered by this band
    hi_act: float
    rank_lo: int          # rank window within the concept's sorted firings
    rank_hi: int
    examples: list        # list[ConceptExample]


@dataclass(frozen=True)
class Meta:
    """Everything needed to interpret the arrays, and to reproduce the run."""
    version: int
    layer: int
    hook: str
    d: int
    n_groups: int
    group_size: int
    n_tokens: int
    n_units: int
    checkpoint: str
    model_kind: str
    embedding_method: str
    seed: int
    # honesty fields: how much 2D structure the chordal metric actually has,
    # and how many strong co-activation edges existed vs were kept.
    chordal_2d_variance: float
    chordal_dims_for_half: int
    coact_threshold: float
    coact_edges_found: int


@dataclass
class Analysis:
    """A complete, self-contained concept analysis for one (layer, hook)."""
    meta: Meta
    # (G, 2) layout: spectral embedding of the strong co-activation graph
    embedding: np.ndarray
    # (E, 2) strong co-firing edges + (E,) Jaccard weights, drawn on the map
    edges: np.ndarray
    edge_weight: np.ndarray
    # (G, k) nearest concepts by chordal distance, and those distances
    neighbor_idx: np.ndarray
    neighbor_dist: np.ndarray
    # (G, k) strongest co-activation partners and their Jaccard overlap
    coact_idx: np.ndarray
    coact_jaccard: np.ndarray
    # (G,) activation statistics over the scanned tokens
    fire_rate: np.ndarray
    mean_act: np.ndarray
    max_act: np.ndarray
    # (G, P, 3) PCA of each concept's firing cloud + (G, P) token ids into vocab
    manifold_xyz: np.ndarray
    manifold_token_ids: np.ndarray
    # (G, 5) firing-activation quantiles [min, p25, p50, p75, max]; zeros if never
    # fired. Drives the skew readout: how much of a concept sits near threshold.
    act_quantiles: np.ndarray
    # (G,) fraction of a concept's firings below 40% of its own max
    near_threshold_frac: np.ndarray
    # shared decoded-token table; manifold_token_ids indexes this
    vocab: list = field(default_factory=list)
    # per-concept top-activating examples (the 'top' band, kept for the token view)
    examples: list = field(default_factory=list)
    # per-concept activation-stratified bands: list[list[ConceptBand]]
    bands: list = field(default_factory=list)

    # ---------------------------------------------------------------- validate
    def validate(self):
        """Raise ``ArtifactShapeError`` unless every array matches ``meta``."""
        G = self.meta.n_groups
        checks = [
            ('embedding', self.embedding.shape, (G, 2)),
            ('fire_rate', self.fire_rate.shape, (G,)),
            ('mean_act', self.mean_act.shape, (G,)),
            ('max_act', self.max_act.shape, (G,)),
        ]
        for name, got, want in checks:
            if got != want:
                raise ArtifactShapeError(name, got, want)
        if self.edges.ndim != 2 or (self.edges.size and self.edges.shape[1] != 2):
            raise ArtifactShapeError('edges', self.edges.shape, ('E', 2))
        if self.edge_weight.shape != (self.edges.shape[0],):
            raise ArtifactShapeError('edge_weight', self.edge_weight.shape,
                                     (self.edges.shape[0],))
        if self.edges.size and (int(self.edges.max()) >= G or int(self.edges.min()) < 0):
            raise AnalysisError(
                f'edges reference concepts outside [0, {G}): '
                f'[{int(self.edges.min())}, {int(self.edges.max())}]')
        for name, arr in (('neighbor_idx', self.neighbor_idx),
                          ('neighbor_dist', self.neighbor_dist),
                          ('coact_idx', self.coact_idx),
                          ('coact_jaccard', self.coact_jaccard)):
            if arr.ndim != 2 or arr.shape[0] != G:
                raise ArtifactShapeError(name, arr.shape, (G, '*'))
        if self.manifold_xyz.ndim != 3 or self.manifold_xyz.shape[0] != G \
                or self.manifold_xyz.shape[2] != 3:
            raise ArtifactShapeError('manifold_xyz', self.manifold_xyz.shape, (G, '*', 3))
        if self.manifold_token_ids.shape != self.manifold_xyz.shape[:2]:
            raise ArtifactShapeError('manifold_token_ids',
                                     self.manifold_token_ids.shape,
                                     self.manifold_xyz.shape[:2])
        if len(self.examples) != G:
            raise ArtifactShapeError('examples', (len(self.examples),), (G,))
        if self.act_quantiles.shape != (G, 5):
            raise ArtifactShapeError('act_quantiles', self.act_quantiles.shape, (G, 5))
        if self.near_threshold_frac.shape != (G,):
            raise ArtifactShapeError('near_threshold_frac',
                                     self.near_threshold_frac.shape, (G,))
        if self.bands and len(self.bands) != G:
            raise ArtifactShapeError('bands', (len(self.bands),), (G,))
        # quantiles must be non-decreasing across [min, p25, p50, p75, max]
        q = self.act_quantiles
        if q.size and np.any(np.diff(q, axis=1) < -1e-3):
            bad = int(np.argmax(np.any(np.diff(q, axis=1) < -1e-3, axis=1)))
            raise AnalysisError(
                f'act_quantiles not monotonic for concept {bad}: {q[bad].tolist()}')
        # chordal distance is bounded by sqrt(group_size)
        hi = float(np.sqrt(self.meta.group_size)) + 1e-3
        if self.neighbor_dist.size and (self.neighbor_dist.min() < -1e-6
                                        or self.neighbor_dist.max() > hi):
            raise AnalysisError(
                f'neighbor_dist out of range [0, {hi:.3f}]: '
                f'[{self.neighbor_dist.min():.4f}, {self.neighbor_dist.max():.4f}]')
        return self

    # -------------------------------------------------------------------- I/O
    def save(self, path):
        """Write a single ``.npz``: arrays natively, the rest as one JSON blob."""
        blob = {
            'meta': asdict(self.meta),
            'vocab': self.vocab,
            'examples': [[asdict(e) for e in per] for per in self.examples],
            'bands': [[asdict(b) for b in per] for per in self.bands],
        }
        np.savez_compressed(
            path,
            _json=np.array(json.dumps(blob)),
            embedding=self.embedding.astype(np.float32),
            neighbor_idx=self.neighbor_idx.astype(np.int32),
            neighbor_dist=self.neighbor_dist.astype(np.float32),
            coact_idx=self.coact_idx.astype(np.int32),
            coact_jaccard=self.coact_jaccard.astype(np.float32),
            edges=self.edges.astype(np.int32),
            edge_weight=self.edge_weight.astype(np.float32),
            fire_rate=self.fire_rate.astype(np.float32),
            mean_act=self.mean_act.astype(np.float32),
            max_act=self.max_act.astype(np.float32),
            manifold_xyz=self.manifold_xyz.astype(np.float16),
            manifold_token_ids=self.manifold_token_ids.astype(np.int32),
            act_quantiles=self.act_quantiles.astype(np.float32),
            near_threshold_frac=self.near_threshold_frac.astype(np.float32),
        )

    @classmethod
    def load(cls, path):
        """Read + validate an artifact. Raises ``AnalysisError`` subclasses."""
        with np.load(path, allow_pickle=False) as z:
            blob = json.loads(str(z['_json']))
            meta_d = blob['meta']
            found = int(meta_d.get('version', -1))
            if found != ARTIFACT_VERSION:
                raise ArtifactVersionError(found)
            obj = cls(
                meta=Meta(**meta_d),
                embedding=z['embedding'],
                neighbor_idx=z['neighbor_idx'],
                neighbor_dist=z['neighbor_dist'],
                coact_idx=z['coact_idx'],
                coact_jaccard=z['coact_jaccard'],
                edges=z['edges'],
                edge_weight=z['edge_weight'],
                fire_rate=z['fire_rate'],
                mean_act=z['mean_act'],
                max_act=z['max_act'],
                manifold_xyz=z['manifold_xyz'].astype(np.float32),
                manifold_token_ids=z['manifold_token_ids'],
                act_quantiles=z['act_quantiles'],
                near_threshold_frac=z['near_threshold_frac'],
                vocab=list(blob['vocab']),
                examples=[[ConceptExample(**e) for e in per]
                          for per in blob['examples']],
                bands=[[ConceptBand(**{**b, 'examples':
                                       [ConceptExample(**e) for e in b['examples']]})
                        for b in per] for per in blob.get('bands', [])],
            )
        return obj.validate()


__all__ = [
    'ARTIFACT_VERSION', 'Analysis', 'Meta', 'ConceptExample', 'ConceptBand',
    'AnalysisError', 'ArtifactVersionError', 'ArtifactShapeError',
    'CheckpointMismatchError',
]
