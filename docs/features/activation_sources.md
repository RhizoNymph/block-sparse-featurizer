# Activation sources

## Scope
Present pre-extracted activations as normalized `(batch, d)` fp32 rows to the
trainer, decoupled from origin (frozen vision encoder or LLM residual stream).
Includes the on-disk reader for vLLM filesystem captures and the normalization
stats computation.

## Non-scope
- Extracting activations from a live model (vision extraction lives in `bsf.data`;
  LLM extraction is done by vLLM servers + the `capture` CLI, not here).
- Model parallelism / training (see distributed_training).

## Data / control flow
1. `capture_format.discover_units(root, layer, hook)` walks a capture tree and
   returns a deterministic (sorted) list of `ReadUnit`s -- one per independently
   readable file (`per_file` bin, `packed*.json`, or `shard-*.json`).
2. `CapturesDataset.__iter__` (an `IterableDataset`):
   - shards `units` across the combined `(DDP rank, DataLoader worker)` grid:
     `gid = rank*num_workers + worker_id`, stride `world_size*num_workers`;
   - reads each unit's rows for `(layer, hook)`, upcasts to fp32 (bf16 stored as
     uint16 is bit-reinterpreted), applies `(x - mean) * scale`;
   - feeds a reservoir shuffle buffer (request-ordered rows are correlated),
     RNG seeded per `(seed, epoch, rank, worker)`; drained at epoch end.
3. `CapturesSource` wraps the dataset, exposes `d` (inferred), `stats()`, and
   `num_rows()` (metadata-only scan of sidecars, for `steps_per_epoch`).
4. `VisionSource` wraps an in-memory `(N, d)` tensor as a map-style dataset;
   `stats() == (None, 1.0)` (the notebook already normalizes).
5. `normalize.load_or_compute(root, layer, hook)` computes `(mean, scale)` in one
   streaming pass so `mean‖x‖² ≈ d`, caching to `{root}/norm_stats_l{L}_{hook}.json`
   with an atomic write (safe under concurrent DDP ranks).

## On-disk capture format (input contract)
Written by vLLM's filesystem capture consumer. Raw native-endian row-major bytes
in `.bin`; `.json` sidecar with `shape:[rows,d]` + logical `dtype`. Layouts:
- `per_file`: `{root}/{tag}/{req}/{layer}_{hook}.{bin,json}`
- `packed`:   `{root}/{tag}/{req}/packed.{bin,json}` (index `entries[]` with
  `offset/nbytes/shape`; a (layer,hook) may span several chunk entries)
- `sharded`:  `{root}/{tag}/shard-NNN-SSSSSS.{bin,json}` (no req dir; entries also
  carry `request_id`)
`bfloat16` is stored as raw `uint16` and returned as `uint16` (recovered via
`torch.from_numpy(a).view(torch.bfloat16)` inside `_to_fp32`).

## Related files
- `bsf/capture_format.py` -- reader mirroring vLLM's `reader.py` (no vLLM dep).
  Exports: `read_per_file`, `read_packed`, `read_request`, `read_sharded`,
  `discover_units`, `ReadUnit`, `CaptureEntry`.
- `bsf/sources/base.py` -- `ActivationSource` ABC (`d`, `is_iterable`,
  `dataset()`, `stats()`, `num_rows()`).
- `bsf/sources/vision.py` -- `VisionSource`, `_RowDataset` (yields bare `(d,)` rows).
- `bsf/sources/captures.py` -- `CapturesDataset`, `CapturesSource`, `_to_fp32`.
- `bsf/normalize.py` -- `compute_stats`, `load_or_compute`.
- `tests/test_capture_format.py`, `tests/test_fit_smoke.py`.

## Invariants / constraints
- Unit list is globally deterministic (sorted paths) so every rank/worker shards
  the identical list -> each row read exactly once, no duplication.
- Rank/world for sharding come from the environment (`RANK`/`WORLD_SIZE`), never
  from collectives -- safe inside DataLoader worker processes.
- Both source kinds yield bare `(d,)` rows so the collated batch is `(batch, d)`.
- `_to_fp32` returns an owned, writable fp32 tensor (no read-only aliasing).
