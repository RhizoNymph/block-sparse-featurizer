# Capture CLI

## Scope
The `bsf` command-line interface: `bsf train` (see distributed_training) and
`bsf capture`, an async client that drives already-running vLLM servers to
collect activations to their shared filesystem root.

## Non-scope
- Launching / configuring the vLLM servers (the operator does that separately).
- Reading captures back for training (that is the captures activation source).

## Server prerequisite
Launch each vLLM server with the filesystem capture consumer enabled; the
filesystem `root` is server-side config (not a per-request field):
```
vllm serve <MODEL> --capture-consumers filesystem:root=/shared/activations
```
Capture has no dedicated endpoint -- it rides on `/v1/completions`.

## Data / control flow (`bsf capture`)
1. `read_prompts(path)` -- one prompt per line; lines starting with `{` are JSON
   and the `prompt`/`text` field is used.
2. Build the per-request spec `capture.filesystem = {request_id, tag, hooks,
   positions, layout}`. `hooks = {hook: [layers]}` from `--hook` (repeatable) and
   `--layers` (comma list). `request_id = f"{tag}-{i:08d}"`.
3. `capture_client.capture(...)` POSTs each prompt to `/v1/completions`,
   round-robining across `--server` URLs with an async httpx client bounded by a
   semaphore (`--concurrency`), retrying transient errors. `capture_wait=True`
   holds the response until files are durable.
4. Collect `capture_results.filesystem.payload` (written paths); print a summary;
   exit non-zero if any request failed.

The training loader is then pointed at the servers' shared `root` (which holds
`{root}/{tag}/...`), selecting the same `(layer, hook)`.

## Arguments
- `--server URL` (repeatable), `--model`, `--prompts FILE`, `--tag`
- `--layers` (comma ints), `--hook` (repeatable): `pre_attn`/`post_attn`/
  `post_block`/`mlp_in`/`mlp_out`
- `--positions` (`last_prompt`/`all_prompt`/`all_generated`/`all`, default `all_prompt`)
- `--layout` (`per_file`/`packed`/`sharded`, default `packed`)
- `--max-tokens`, `--concurrency`, `--capture-wait/--no-capture-wait`

Example:
```
bsf capture --server http://gpu0:8000 --server http://gpu1:8000 \
    --model meta-llama/Llama-3-8B --prompts prompts.txt --tag run42 \
    --layers 12,16,20 --hook post_block --layout packed --concurrency 32
```

## Related files
- `bsf/cli.py` -- `main`, `build_parser`, `_cmd_train`, `_cmd_capture`,
  `_build_model`, `_build_source`.
- `bsf/capture_client.py` -- `read_prompts`, `build_capture_spec`, `capture`,
  `CaptureResult`.
- `bsf/__main__.py` -- enables `python -m bsf` / `torchrun ... -m bsf.cli`.

## Invariants / constraints
- The client sets only `tag`/`hooks`/`positions`/`layout`/`request_id`; the
  filesystem `root` is fixed at server launch.
- `mlp_in`/`mlp_out` hooks are only wired on some model families (gemma3/4, qwen3)
  and must be named explicitly; the residual hooks work everywhere.
- Exit code is 0 only if every request succeeded.
