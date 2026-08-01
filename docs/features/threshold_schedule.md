# Threshold schedule — keeping the block gate trainable

## Scope
The learned per-block threshold in `GroupLassoBSF`'s default (block-JumpReLU +
STE) variant: how it is parameterised, how its straight-through window stays
populated for a whole run, the diagnostics that surface it, and the capture-side
row filter that keeps outlier positions out of the statistics it depends on.

## Non-scope
- The `paper_version=True` soft-threshold variant (single scalar `log_theta`, no
  STE) — unaffected.
- `VanillaBSF` / `GrassmannianBSF`, which gate by per-sample block TopK and have
  no learned threshold.
- Anything outside the GroupLasso gate: the decoder, the encoder, and the
  reconstruction objective are untouched.

## The failure this fixes

`BlockJumpReLU`'s pseudo-derivative for theta is a rectangle kernel: gradient
flows only where `|‖a_g‖ − θ| ≤ bandwidth/2`. Previously `bandwidth` was an
absolute width set from the **first training batch** (`gn.std() * 0.5`) and then
frozen — deliberately, as a python float, so `torch.compile` saw no host sync.

Both the threshold and the block norms move over a run, so an absolute width
calibrated once stops straddling the distribution it gates. When the window
empties, theta's gradient is *identically* zero and the threshold is frozen
wherever it happened to be. Measured, both directions:

| | theta stranded ABOVE (shipped `gl_l32` checkpoint) | theta stranded BELOW (norms drift 5×) |
|---|---|---|
| block norm | median 4.42, p99 11.6 | median 0.90 → 9.32 |
| theta | median 14.94 (12.2–18.2) | pinned at 0.945 |
| `bandwidth` | 0.197 — **0.66% of theta** | — |
| pairs in-window | **0.011%** | **0.000%** |
| blocks with zero theta gradient | **3239 / 4096** | **16 / 16** |
| effect | L0 collapses to 7.9 against `--target-l0 32` | every block saturates (L0 15.96/16) |

Nothing in the logs said so: `L0` was printed every epoch but never compared to
the target, so a 4× miss read as a normal training curve.

## The fix: gate in scale-free units

`GroupLassoBSF` carries a `norm_scale` buffer — an EMA of the mean block norm —
and the gate compares `‖a_g‖ / norm_scale` to theta. Theta and the kernel width
`rel_bandwidth` both live in those units, so neither can be left behind by a
drifting encoder. The width is a constant there, needing no calibration.

```
init_theta(gn):    norm_scale ← mean(gn)
                   raw_theta  ← softplus⁻¹(quantile(gn / norm_scale, 1 − target_l0/G)) / gain
train step:        norm_scale ← m·norm_scale + (1−m)·mean(gn)      # m = scale_momentum
gate:              H(gn / norm_scale − θ),  STE window = ±rel_bandwidth/2
```

`norm_scale` is a buffer rather than a per-batch statistic so train and eval gate
identically and small eval batches cannot jitter the threshold.

## The second problem the fix exposed: `target_l0` was never a target

With a fixed `coef`, `target_l0` places the cold-start threshold and nothing
else; L0 then slides to wherever `coef` balances reconstruction. Freeing the
threshold made that slide *fast* instead of fixing it — the two layer-32 runs
reach the same operating point, one just gets there 5x sooner:

| epoch | baseline L0 | scale-free-threshold L0 |
|---|---|---|
| 1 | 261.7 | 51.5 |
| 4 | 82.6 | 12.2 |
| 6 | 48.8 | **8.6** |
| 30 | **8.6** | — |

Read together: the frozen window was an accidental brake on a mis-set sparsity
penalty. Both land at L0 ≈ 8.6 against `--target-l0 32`.

`l0_control > 0` closes the loop by dual ascent on the multiplier:

```
l0_ema   ← m·l0_ema + (1−m)·L0_batch
err      ← clip((l0_ema − target_l0) / target_l0, −1, +1)
log_adj  ← clip(log_adj + l0_control·err, ∓l0_control_clip)
penalty  ← coef · exp(log_adj) · L0
```

Too dense raises the penalty, too sparse lowers it, and `coef` becomes a scale
rather than the thing that decides sparsity. The error is clipped because L0 can
start at 8x the target, which would otherwise slam the multiplier into its bound
in a single step. Off by default (`l0_control=0.0`) so existing configs are
numerically unchanged.

## Data / control flow
1. `fit()` cold-starts theta once, before the DDP wrap and compile
   (`_warmup_theta`), from block norms all-gathered across ranks — `init_theta`
   now sets `norm_scale` from the same global tensor, so every rank starts equal.
2. Each training step, `gate_default` updates `norm_scale` under `no_grad`
   (`_track_scale`). No collective in the hot path, so nothing graph-breaks.
3. Every `scale_sync_every` steps (default 50) `_sync_norm_scale` all-reduces the
   buffer to its cross-rank mean, bounding per-rank drift to that interval.
   No-op when not distributed.
4. `loss()` applies `effective_coef()`; with the controller on it first takes one
   `_control_l0` step from the batch's realized L0.
5. On log epochs the trainer prints `win=` (fraction of token×block pairs inside
   the STE window), `frozen=` (blocks with an empty window), and `coef=` when the
   controller is on. It warns when realized L0 is more than 2× off `target_l0` or
   when more than half the blocks are frozen.

   **`frozen=` is a single-batch statistic.** A block with no in-window pair in
   one 2048-token batch still gets ~1220 more chances per epoch, so the printed
   count overstates how many thresholds are genuinely stuck. Treat it as a trend,
   not a census.

## What it bought, measured

Layer 32, 40k held-out tokens from capture units the analysis artifacts never
touched. `PI` is GroupLasso with `--l0-control 0.002`; `TopK` is `VanillaBSF`
(`--model vanilla --l0 32`), which fixes L0 by construction and needs none of
the machinery above.

| | baseline | PI | TopK |
|---|---|---|---|
| realized L0 (asked for 32) | 8.70 | 31.82 | 32.00 |
| R^2 | 0.5098 | **0.5685** | 0.5659 |
| concepts that ever fire | 4066 (99.3%) | 3036 (74.1%) | 3068 (74.9%) |
| concepts with >=400 firings | 67 | **674** | **1067** |

Reconstruction improves ~11% *while running at 3.7x the sparsity budget*, and
the number of concepts with enough firings to say anything statistically
meaningful about goes up 10-16x. The cost is real: ~25% of the dictionary goes
silent, measured over 40k tokens rather than the 10-batch eval. That cost is
addressed separately by the auxiliary revival loss
(`docs/features/revival.md`), which takes the live dictionary back to 100%.

Two other metrics moved (band coherence, within-block anisotropy) but are
**confounded by L0** — at 3.7x the firings per concept the added firings are the
marginal ones near threshold, which moves both regardless of concept quality.
They are not evidence either way and are not reported as results.

PI and TopK land in the same place on every measure. Prefer **TopK** for new
concept work: it holds the operating point by construction and has none of the
failure modes this document exists to describe. The fixes here still matter as a
correctness fix — `--target-l0` silently doing nothing is a bug whichever
featurizer you choose.

Dictionary size trades reconstruction against dead concepts on a steep curve
(block-TopK, l0=32, same data): G=1024 -> R^2 0.494 with 2.9% dead; G=2048 ->
0.521 with 11.8%; G=4096 -> 0.548 with 25.2%. None of that was visible before
the operating point was pinned.

## Capture-side row filter
Position 0 of a sequence is an attention sink; its residual-stream norm is a
large outlier that is not a concept. On layer 32 of `pile25m`:

| | position 0 | everything else |
|---|---|---|
| per-token max block norm (median) | **166.5** | 37.4 |
| share of all rows | 0.17% | 99.83% |
| share of the top-1% norm tail | **17.1%** | 82.9% |

`--drop-first N` skips the first N positions of every request. One packed capture
unit holds exactly one request in sequence order, so this is a slice of the unit.
It is applied in **both** `CapturesDataset.__iter__` and `normalize.compute_stats`,
and `drop_first` is part of the norm-stats cache filename — statistics computed
under a different filter describe a different distribution and must never be
silently reused. `CapturesSource.num_rows()` accounts for it, since
`steps_per_epoch` is derived from it.

## Related files
- `bsf/group_lasso.py` — `BlockJumpReLU` (scale-free kernel), `window_occupancy`,
  `GroupLassoBSF.{init_theta, _track_scale, _control_l0, effective_coef,
  gate_default, window_stats, window_stats_from_norms, load_state_dict}`; ctor
  args `rel_bandwidth`, `scale_momentum`, `l0_control`, `l0_control_clip`;
  buffers `norm_scale`, `log_coef_adj`, `l0_ema`.
- `bsf/train.py` — `l0_off_target`, `_sync_norm_scale`, `_window_report`, the
  `fit()` logging block; `fit(..., scale_sync_every=50)`.
- `bsf/sources/captures.py` — `CapturesDataset`/`CapturesSource` `drop_first`.
- `bsf/normalize.py` — `compute_stats`/`load_or_compute` `drop_first`, cache key.
- `bsf/cli.py` — `--rel-bandwidth`, `--scale-momentum`, `--l0-control`,
  `--drop-first`.
- `tests/test_threshold_schedule.py`, `tests/test_position_filter.py`.

## Invariants / constraints
- `rel_bandwidth > 0` — enforced in both the ctor and `BlockJumpReLU.forward`. A
  zero-width window passes no gradient, which is the bug this feature exists to
  prevent.
- `norm_scale` is only updated in training mode; `eval()` never moves it. The
  same holds for the controller state (`l0_ema`, `log_coef_adj`).
- `l0_control=0` must leave the loss numerically identical to the fixed-`coef`
  path, so existing configs are unaffected by the controller's existence.
- Whatever `gn` normalisation `gate_default` applies, `window_occupancy` must see
  the same — both go through `window_stats_from_norms`.
- The row filter used for norm statistics must equal the one used for training
  rows; the shared `drop_first` argument and the cache key enforce it.
- **Pre-fix checkpoints are not loadable.** They carry a `bandwidth` buffer and an
  absolute theta with no recoverable `norm_scale`, so `load_state_dict` raises
  rather than silently reinterpreting the gate.
