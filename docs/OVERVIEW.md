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
    cli: >
      `bsf train` and `bsf capture`; train reads torchrun env vars so bare /
      single-node-multi-GPU / multi-node all use one code path.
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
