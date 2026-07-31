# Overview

```yaml
Overview:
  description: >
    Block-Sparse Featurizers (BSF) -- block-structured sparse dictionaries/SAEs
    that carve a d-dim activation space into n_groups blocks ("concepts") of
    group_size latent dims. Three featurizers share one decoder/forward
    interface and one trainer. Activations come from a frozen upstream model:
    the original path uses DINOv3 vision patch tokens; the LLM path streams
    residual-stream activations captured offline by vLLM's filesystem consumer.
  subsystems:
    featurizers: >
      BSF base + VanillaBSF / GrassmannianBSF / GroupLassoBSF. Small models: two
      matrices (W_enc, W_dec) plus a gate. Fit on one 24GB GPU up to ~32k groups
      at d=4096 with fp32 Adam.
    activation_sources: >
      A common ActivationSource interface feeds either vision (in-memory tensor)
      or LLM captures (streamed from disk) to the trainer.
    trainer: >
      train() -- original single-GPU in-memory trainer (numerics frozen).
      fit() -- source-driven trainer adding DDP, streaming, torch.compile.
    capture: >
      An async client that drives already-running vLLM servers to collect
      activations to a shared filesystem root (read back by the captures source).
    analysis: >
      Turns a trained checkpoint + capture root into one self-contained, versioned
      .npz: chordal distances between concept subspaces, firing statistics,
      top-activating tokens with decoded context, per-concept firing manifolds,
      and the strong co-activation graph. Torch is needed to build an artifact,
      never to read one.
    dashboard: >
      A Dash/plotly app over an analysis artifact -- concept map, token examples,
      3D concept manifold, ranked subspace/co-firing relations -- all driven by one
      selected concept. Optional extra; its imports are deferred.
    cli: >
      `bsf train`, `bsf capture`, `bsf analyze`, `bsf dashboard`; train reads
      torchrun env vars so bare / single-node-multi-GPU / multi-node all use one
      code path.
  data_flow: >
    (capture) prompts -> vLLM servers (--capture-consumers filesystem:root=...)
    -> {root}/{tag}/... .bin/.json shards.
    (train) shards -> capture_format reader -> CapturesSource (shard/shuffle/
    normalize) -> DataLoader -> featurizer.loss under DDP -> Adam -> checkpoint.
    Vision path: DINOv3 activations (bsf.data) -> VisionSource -> same trainer.
```

## Features Index

```yaml
Features Index:
  activation_sources:
    description: Vision + LLM-capture activations behind one streaming interface.
    entry_points: [bsf.sources.ActivationSource, bsf.sources.VisionSource, bsf.sources.CapturesSource, bsf.capture_format, bsf.normalize]
    depends_on: []
    doc: docs/features/activation_sources.md
  distributed_training:
    description: DDP + torch.compile trainer (fit) with the featurizer correctness fixes.
    entry_points: [bsf.train.fit, bsf.distributed, bsf.group_lasso]
    depends_on: [activation_sources]
    doc: docs/features/distributed_training.md
  threshold_schedule:
    description: >
      Keeps GroupLasso's learned block threshold trainable for a whole run. The
      gate compares block norms to theta in units of a running norm scale, so the
      STE window cannot drift off the distribution and freeze (it did: 79% of
      blocks had zero threshold gradient, and L0 landed at 7.9 against a target
      of 32). Adds window/L0 diagnostics and a capture-side attention-sink filter.
    entry_points: [bsf.group_lasso.BlockJumpReLU, bsf.group_lasso.window_occupancy, bsf.train.l0_off_target, bsf.sources.CapturesSource, bsf.normalize]
    depends_on: [activation_sources, distributed_training]
    doc: docs/features/threshold_schedule.md
  concept_dashboard:
    description: >
      `bsf analyze` + `bsf dashboard` -- interactive exploration of learned
      concepts: co-activation-graph map, top-activating tokens, 3D concept
      manifolds, chordal-subspace neighbours.
    entry_points: [bsf.analysis.build.build_analysis, bsf.analysis.Analysis, bsf.dashboard.build_app]
    depends_on: [activation_sources, distributed_training]
    doc: docs/features/concept_dashboard.md
  capture_cli:
    description: CLI (`bsf train` / `bsf capture`) incl. driving vLLM servers to capture.
    entry_points: [bsf.cli, bsf.capture_client]
    depends_on: [activation_sources, distributed_training]
    doc: docs/features/capture_cli.md
```

## Notes / roadmap
- Phase 1 (implemented): DDP-only, fp32, d≈4096 / ≤32k groups.
- Phase 2 (sketched, not built): group-sharded model parallelism for dictionaries
  too large for one GPU -- shard n_groups across ranks and all-reduce the (B,d)
  reconstruction (~33MB) instead of the full gradient (~1GB); ideal for
  no-NVLink 3090s. See docs/features/distributed_training.md.
