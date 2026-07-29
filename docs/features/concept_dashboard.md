# Concept Dashboard

## Scope
Turning a trained BSF into something a human can explore: `bsf analyze` builds a
self-contained artifact from a checkpoint plus a capture root, and
`bsf dashboard` serves an interactive Dash app over it.

Four views, all driven by a single selected concept:
1. **Concept map** — every concept placed by a spectral layout of the strong
   co-activation graph, with edges drawn and the selection's neighbours ringed.
2. **Top-activating tokens** — the tokens that fire the concept hardest, in
   context, with the firing token highlighted.
3. **Concept manifold** — the concept's firing cloud PCA'd to 3D, hue from radial
   direction (the same convention as `bsf.viz`), one point per firing token.
4. **Relations + stats** — ranked nearest subspaces (chordal) and co-firing
   partners (Jaccard), plus the corpus firing-rate distribution.

## Non-scope
- Training or capturing (see `distributed_training`, `capture_cli`).
- Live encoding of arbitrary user text: the artifact is precomputed, so the app
  needs no torch and no GPU. Probing new prompts would need a compute backend.
- Cross-layer comparison. The CLI is per-layer; the artifact carries `layer` in
  its metadata so a multi-artifact view is an additive change.

## Why the map is a graph, not an embedding of subspace distance
A BSF concept is a `group_size`-dim **subspace** of R^d, so the natural relation
is between subspaces: the chordal distance
`dist(i,j)^2 = k - ||Q_i^T Q_j||_F^2 = sum_l sin^2(theta_l)` over principal
angles, invariant to any change of basis within a block (which cosine between
flattened blocks is not). It is computed without per-pair SVDs by stacking all
orthonormal bases into `Qf` (d, G*k), forming `Qf^T Qf`, and summing squares
inside each (k, k) tile — one matmul plus a chunked reduction.

**Measured on the trained layer-32 dictionary (G=4096, k=4, d=5120):**

| quantity | value |
|---|---|
| median pairwise chordal distance | 1.9971 (max possible 2.0) |
| spread (std / mean) | 0.28% |
| variance captured by 2 dims | 0.63% |
| dims needed for 50% of variance | 276 |

With `G*k = 16384 > d = 5120` the dictionary is overcomplete, so the subspaces
are *forced* near-orthogonal: random 4-planes in R^5120 have expected
`cos^2 theta ~ 16/5120`, giving distance ~1.997 — exactly what is observed. Any
2D projection of that metric is a blob that merely *looks* like structure, so it
is not the default. Co-activation is likewise near-uniform globally (median
Jaccard distance 1.0) but its **tail is real**: ~8400 pairs above Jaccard 0.1 and
~800 above 0.3. A sparse graph of those strong relations does have community
structure, so `--embedding graph` (the default) lays out that graph spectrally.

The chordal **neighbour list** is still kept and shown: its tail is exactly where
near-duplicate concepts live (the layer-32 "document-initial" family 836 / 1565 /
3103 is such a cluster). `--embedding mds|tsne|pca` remain available, and when
used the map's title states the measured 2D variance so it cannot mislead.

## Data / control flow
1. `bsf analyze` loads the checkpoint (`_model_from_checkpoint` validates its
   geometry against `--n-groups/--group-size`, raising `CheckpointMismatchError`),
   plus cached normalization stats for the `(layer, hook)`.
2. One streaming pass over the first `--requests` capture units accumulates, per
   concept: fire count, activation sum/max, a bounded top-k heap of the strongest
   `(activation, unit, position)` triples, a subsample of firing contributions for
   the manifold, and the boolean gate matrix.
3. Geometry: chordal distances -> nearest neighbours + `embedding_quality`;
   gate matrix -> `coactivation` (top-k partners) and `coactivation_graph`
   (thresholded edges) -> `graph_layout` (spectral layout of the giant component,
   isolated concepts on an outer ring).
4. Token strings are decoded from the **GGUF vocabulary** (`--gguf`), so the
   artifact is viewable without a tokenizer. Row *i* of a request maps to token
   *i* of its `prompt_token_ids` sidecar, valid because captures used
   `positions='all_prompt'`.
5. `Analysis.save` writes one `.npz`: arrays natively, metadata/vocab/examples as
   a single JSON blob. `Analysis.load` re-validates every shape invariant.
6. `bsf dashboard` loads the artifact and builds the Dash app. Selection flows
   map-click / index box / token search -> `dcc.Store` -> every panel.

## Related files
- `bsf/analysis/types.py` — `Analysis`, `Meta`, `ConceptExample`, the typed error
  hierarchy (`AnalysisError`, `ArtifactVersionError`, `ArtifactShapeError`,
  `CheckpointMismatchError`), `save`/`load`/`validate`, `ARTIFACT_VERSION`.
- `bsf/analysis/compute.py` — `orthonormal_bases`, `chordal_distances`,
  `nearest_neighbors`, `coactivation`, `coactivation_graph`, `graph_layout`,
  `embedding_quality`, `embed_distances`.
- `bsf/analysis/build.py` — `build_analysis`, `load_gguf_vocab`, `clean_token`.
- `bsf/dashboard/figures.py` — `concept_map`, `concept_manifold`,
  `neighbor_bars`, `stats_hist` (plotly only, no torch).
- `bsf/dashboard/app.py` — `build_app`: layout + callbacks.
- `bsf/cli.py` — `_cmd_analyze`, `_cmd_dashboard`, `_add_analyze_args`,
  `_add_dashboard_args`.
- `tests/test_analysis.py` — chordal distance vs a float64 principal-angle
  oracle, basis-change invariance, graph thresholding/layout, artifact
  round-trip and validation.

## Invariants / constraints
- `chordal_distances` computes in **float64 by default**: `k - sum cos^2` cancels
  catastrophically for near-identical subspaces (precisely the interesting case),
  and the following `sqrt` amplifies it. In float32 a true zero returns ~1e-3.
- Chordal distances lie in `[0, sqrt(group_size)]`; `validate` enforces it.
- `coactivation_graph` emits each undirected pair once (`i < j`), strongest
  first, truncated to `max_edges` — and returns the true count found so the UI
  can report what was dropped rather than silently implying full coverage.
- `graph_layout` is deterministic given `seed`; the giant component is laid out
  spectrally and all other concepts are placed on a ring so a disconnected graph
  never degenerates.
- The dashboard imports **no torch**; `bsf/dashboard/__init__.py` defers the
  `dash`/`plotly` import so `bsf train` works without the extra installed.
- Artifacts are versioned; a mismatch raises `ArtifactVersionError` rather than
  mis-rendering.

## Usage
```
pip install "bsf[dashboard]"          # dash==4.4.1, plotly==6.9.0

bsf analyze --ckpt runs/gl_l32.pt --data /captures/pile25m \
    --layer 32 --hook post_block --n-groups 4096 --group-size 4 \
    --gguf ~/Models/Qwen3.6-27B-UD-Q4_K_XL.gguf \
    --requests 300 --out runs/analysis_l32.npz

bsf dashboard --analysis runs/analysis_l32.npz --port 8050
# remote:  ssh -L 8050:localhost:8050 <host>
```
