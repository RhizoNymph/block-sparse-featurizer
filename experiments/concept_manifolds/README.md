# Concept manifolds: BSF concepts as fitted manifolds

Comparing BSF's block geometry against the manifold-SAE approach in
[SauersML/gam](https://github.com/SauersML/gam) (`gamfit`), whose campaign fits an
explicitly parametrized manifold per SAE atom and plots the atom's routed tokens
at their *fitted coordinates on that manifold*.

**The point of the comparison.** A flat SAE atom is a single direction, so a
manifold has to be *fitted* before a token has any coordinate on it. A BSF concept
already owns a `group_size`-dim subspace and every firing token carries a signed
code inside it, so the coordinates are architectural. `gamfit` is used here only
for the *smooth over* that manifold, not to invent it.

## Scripts
- `extract_codes.py` — encode captures through a trained BSF and dump, per
  concept, a **uniform** sample of its firing codes plus token ids. Uniform, not
  top-k: the activation tail is unrepresentative (see
  `docs/features/concept_dashboard.md`).
- `manifold_fit.py` — normalise each concept's top-3 principal code directions to
  unit length (they then lie exactly on S²), fit a `gamfit.Sphere` smooth of
  activation magnitude over that sphere by REML, decode the fitted surface on a
  lat/lon grid, and render tokens on it.
- `manifold_shapes.py` — cheap shape census straight off a dashboard analysis
  artifact (intrinsic dimension, annularity, antipodal symmetry). No GPU.

## Findings (layer 32, 4096x4 dictionary, 2.5M tokens)
- Concept code clouds are **98.7% antipodally symmetric** (median 0.85) and have
  **median intrinsic dimension 2.55 of 3**. The symmetry is the signature of BSF's
  *signed* block code: gam's README notes a nonnegative-gate dictionary can
  shatter a circle into up to four rectified half-atoms, which a signed code
  cannot do.
- Direction clouds concentrate near a **great circle** (the third principal
  component is small), which is the `circle` topology gam's adjudicator races for.
- Sphere-smooth R² over 50 dense concepts (>=8000 firings each): median **0.145**,
  p90 0.229, best **0.581** (concept 262). Fits on sparse concepts score higher
  (0.833 at 413 points, edf 17) but that is overfitting -- prefer dense concepts.
- Interpretable examples where position on the manifold carries meaning: concept
  262 (`9`, `1`, `6`, `MW` -- digits vs units), 2168 (`its`, `our`, `ihrer` --
  possessives across English and German), 94 (`flat`, `yields`, `highest`,
  `highly` -- degree words).

## Requirements
`gamfit` is **not** a dependency of `bsf`; it is only needed by these scripts:

```
pip install -r experiments/concept_manifolds/requirements.txt
```

## Known upstream issues (gamfit 0.1.261)
- `adjudicate_atom_shape` never certifies for us: `k_ladder=[2]` is rejected as
  needing a ring-cluster order, and any order >=3 fails Gaussian-mixture or
  ring-of-clusters EM. It fails identically on a **clean synthetic 2-D Gaussian**
  (400 points), so it is not data-specific. This blocked the rigorous held-out
  shape verdict; the shape statistics in `manifold_shapes.py` are descriptive
  substitutes.
- One `double free or corruption` abort inside the Rust extension, not
  reproducible. Don't run these unattended.
- `gamfit.Sphere` carries **no intercept**. Fitting an uncentred response gives
  in-sample R² around -5; `manifold_fit.fit_sphere` centres the response and adds
  the mean back.
