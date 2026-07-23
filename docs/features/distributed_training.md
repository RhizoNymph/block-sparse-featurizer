# Distributed training

## Scope
`fit()` -- a source-driven trainer adding DistributedDataParallel, streaming
activation sources, and torch.compile, plus the featurizer correctness fixes the
DDP move requires. Phase 1 target: fp32, d≈4096, ≤32k groups -> DDP only (a full
featurizer replica per GPU), no sharding.

## Non-scope
- Model/dictionary sharding (Phase 2 sketch below).
- The original `train(model, x, ...)` -- kept verbatim; single-GPU + in-memory
  vision dispatches to it so its numerics never change.

## Data / control flow (`fit`)
1. `init_dist(device)` -> `(DistInfo, device)`. Under torchrun (`WORLD_SIZE>1`)
   inits the process group (nccl on CUDA, else gloo); honors an explicit
   `--device cpu` so more ranks than GPUs don't set an invalid ordinal.
2. Build the DataLoader: map-style sources get a `DistributedSampler(shuffle,
   drop_last)`; iterable sources self-shard from the environment. `drop_last`
   fixes the batch shape (compile-friendly, no recompiles).
3. `steps_per_epoch = num_rows // world_size // batch_size` (or `--steps-per-epoch`)
   -- every rank runs the SAME number of steps and metric all-reduces, so uneven
   iterable shards can't deadlock a collective.
4. Eager theta warm-up (GroupLasso default variant only), BEFORE the DDP wrap and
   compile: gather the first batch's block norms across ranks
   (`all_gather_cat`) and call `init_theta` identically on every rank. `raw_theta`
   is a Parameter broadcast only at DDP construction, so a per-rank cold-start in
   the loop would diverge permanently.
5. Wrap `_LossModule(model)` in DDP. **DDP wraps the loss forward, not `forward`**
   -- the loop calls `loss()`, and DDP only all-reduces gradients for the pass
   through `DDP.forward`; routing the loss through the wrapper is what arms the
   sync. `broadcast_buffers=False`.
6. `torch.compile(core)` if `--compile` (default off). Order: build -> warm-up ->
   DDP -> compile, so Dynamo's DDPOptimizer can split at bucket boundaries.
7. Step: `loss,_ = step_model(xb_in, xb); backward; opt.step();
   model.normalize_decoder()` (eager, outside the graph, on the unwrapped model).
   `snr>0` corrupts the input and reconstructs the clean batch (denoising).
8. On log epochs, `_evaluate` reduces metric *parts* across ranks: R2 from summed
   ss_res/ss_tot, L0 from summed active/token counts, dead blocks from the OR
   (MAX) of per-rank fired masks. Never averages per-rank R2.
9. `barrier`; rank-0 checkpoint; `destroy_process_group`.

## Featurizer correctness fixes
- **STE theta-grad scale** (`BlockJumpReLU`, `group_lasso.py`): `grad_scale`
  threaded through, default **1.0**. The upstream grad already carries `1/N` from
  the `.mean()` loss, so `.sum(0)` is batch-normalized and DDP's cross-rank
  average reproduces the single-GPU global-batch gradient with no correction.
  Verified: `tests/ddp_grad_scale_check.py` shows `max|Δθ.grad| ≈ 3e-8` at
  `grad_scale=1.0` and exactly `×world_size` too large at `grad_scale=world`.
- **No host syncs in the compiled hot loop**: `loss()` returns 0-dim detached
  tensors in the info dict (no per-step `.item()`); `bandwidth` is mirrored to a
  python float (`_bandwidth_f`) so `float(bandwidth)` doesn't graph-break; the
  init branch reads a python bool (`_theta_inited`), not a GPU tensor.
  `load_state_dict` re-syncs both mirrors so a resumed model doesn't re-init.
- **Grassmannian** does `torch.linalg.qr` every forward -- a known, accepted
  compile graph break (left as-is in Phase 1).

## Launch modes
```
# single GPU (dispatches vision -> legacy train; captures -> fit, world=1)
bsf train --source captures --data <root> --layer L --hook post_block --n-groups N ...
# single node, N GPUs
torchrun --nproc_per_node=N -m bsf.cli train ...
# multi-node
torchrun --nnodes=M --node-rank=R --master-addr=.. --master-port=.. \
         --nproc_per_node=N -m bsf.cli train ...
```
DDP knobs come from torchrun env vars, never CLI flags.

## Related files
- `bsf/train.py` -- `train` (legacy, untouched), `fit`, `_LossModule`,
  `_warmup_theta`, `_build_loader`, `_steps_per_epoch`, `_evaluate`.
- `bsf/distributed.py` -- `init_dist`, `cleanup_dist`, `barrier`,
  `all_reduce_sum/max`, `broadcast`, `all_gather_cat`, `env_rank_world`.
- `bsf/group_lasso.py` -- `grad_scale`, `_bandwidth_f`/`_theta_inited` mirrors,
  `load_state_dict` sync.
- `tests/test_fit_smoke.py`, `tests/ddp_grad_scale_check.py` (run under torchrun).

## Invariants / constraints
- All ranks execute identical step and metric-reduce counts (fixed
  `steps_per_epoch`) -> collectives never deadlock.
- `normalize_decoder` runs eager, outside the graph; DDP keeps `W_dec` bit-identical
  across ranks so all ranks normalize identically.
- Single-GPU + in-memory vision path bypasses `fit` entirely (legacy numerics).

## Phase 2 (not built): group-sharded model parallelism
Shard `n_groups` across ranks; each owns a slice of `W_enc`/`W_dec`/gate, computes
a partial `(B,d)` reconstruction, and ranks all-reduce the reconstruction (~33MB
at B=2048,d=4096) instead of the full gradient (~1GB) -- ~30x less comm, ideal for
no-NVLink 3090s. GroupLasso's per-block gate is fully local; Vanilla/Grassmannian
need a small distributed top-k for the global block-TopK.
