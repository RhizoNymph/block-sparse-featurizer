"""``bsf`` command-line interface.

Two subcommands:

  ``bsf train``    train a featurizer on a vision or LLM-capture source. Reads
                   torchrun's env vars, so it runs bare (single GPU) or under
                   ``torchrun`` (single-node multi-GPU, multi-node) unchanged:
                       bsf train --source vision --data acts.npy ...
                       torchrun --nproc_per_node=4 -m bsf.cli train --source captures ...
                       torchrun --nnodes=2 --node-rank=R --master-addr=.. \\
                                --master-port=.. --nproc_per_node=4 -m bsf.cli train ...

  ``bsf capture``  drive already-running vLLM servers (launched with
                   ``--capture-consumers filesystem:root=...``) to collect
                   activations to their shared filesystem root.

  ``bsf analyze``  turn a trained checkpoint + capture root into a self-contained
                   concept-analysis artifact (subspace geometry, statistics,
                   top-activating tokens, per-concept manifolds).

  ``bsf dashboard``  serve an interactive Dash app over that artifact.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

import numpy as np
import torch

from . import normalize
from .distributed import env_rank_world
from .group_lasso import GroupLassoBSF
from .grassmannian import GrassmannianBSF
from .vanilla import VanillaBSF
from .sources import VisionSource, CapturesSource
from .train import train, fit


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def _build_model(args, d):
    if args.model == 'vanilla':
        return VanillaBSF(d, args.n_groups, args.group_size, l0=args.l0,
                          revival_alpha=args.revival_alpha, k_aux=args.k_aux,
                          dead_after=args.dead_after)
    if args.model == 'grassmannian':
        return GrassmannianBSF(d, args.n_groups, args.group_size, l0=args.l0)
    return GroupLassoBSF(d, args.n_groups, args.group_size, coef=args.coef,
                         target_l0=args.target_l0, gain=args.gain,
                         paper_version=args.paper_version, grad_scale=args.grad_scale,
                         rel_bandwidth=args.rel_bandwidth,
                         scale_momentum=args.scale_momentum,
                         l0_control=args.l0_control)


def _build_source(args):
    if args.source == 'vision':
        x = np.load(args.data)
        if isinstance(x, np.lib.npyio.NpzFile):
            x = x[x.files[0]]
        return VisionSource(x)
    mean, scale = normalize.load_or_compute(args.data, args.layer, args.hook,
                                            max_rows=args.stats_max_rows,
                                            drop_first=args.drop_first)
    return CapturesSource(args.data, args.layer, args.hook,
                          shuffle_buffer=args.shuffle_buffer, mean=mean, scale=scale,
                          seed=args.seed, drop_first=args.drop_first)


def _cmd_train(args):
    source = _build_source(args)
    model = _build_model(args, source.d)
    _, world = env_rank_world()

    # single-GPU + in-memory vision -> the original trainer, numerics unchanged.
    if isinstance(source, VisionSource) and world == 1:
        train(model, source.x, epochs=args.epochs, lr=args.lr,
              batch_size=args.batch_size, snr=args.snr, device=args.device,
              log_every=args.log_every)
        if args.out:
            torch.save(model.state_dict(), args.out)
        return 0

    fit(model, source, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        snr=args.snr, device=args.device, log_every=args.log_every,
        steps_per_epoch=args.steps_per_epoch, num_workers=args.num_workers,
        compile=args.compile, out=args.out, seed=args.seed)
    return 0


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------
def _cmd_capture(args):
    from .capture_client import read_prompts, capture

    prompts = read_prompts(args.prompts)
    layers = [int(v) for v in args.layers.split(',') if v.strip() != '']
    hooks = {h: layers for h in args.hook}

    def progress(done, total, r):
        if r.status != 'ok':
            print(f'  [{done}/{total}] {r.request_id} ERROR: {r.error}', file=sys.stderr)
        elif done % 50 == 0 or done == total:
            print(f'  [{done}/{total}] captured', flush=True)

    results = asyncio.run(capture(
        args.server, args.model, prompts, tag=args.tag, hooks=hooks,
        positions=args.positions, layout=args.layout, max_tokens=args.max_tokens,
        concurrency=args.concurrency, capture_wait=args.capture_wait,
        progress=progress))

    ok = sum(1 for r in results if r.status == 'ok')
    n_paths = sum(len(r.paths) for r in results)
    print(f'captured {ok}/{len(results)} requests, {n_paths} files written '
          f'under the servers\' filesystem root (tag={args.tag!r})')
    return 0 if ok == len(results) else 1


# ---------------------------------------------------------------------------
# analyze / dashboard
# ---------------------------------------------------------------------------
def _cmd_analyze(args):
    from .analysis.build import build_analysis

    def progress(done, total):
        if done % 50 == 0 or done == total:
            print(f'  [{done}/{total}] units scanned', flush=True)

    an = build_analysis(
        args.ckpt, args.data, args.layer, args.hook,
        n_groups=args.n_groups, group_size=args.group_size,
        model_kind=args.model, l0=args.l0, requests=args.requests,
        top_k=args.top_k,
        n_neighbors=args.neighbors, manifold_points=args.manifold_points,
        context=args.context, embedding_method=args.embedding,
        coact_threshold=args.coact_threshold, gguf=args.gguf,
        device=args.device, seed=args.seed, drop_first=args.drop_first,
        progress=progress)
    an.save(args.out)
    live = int((an.fire_rate > 0).sum())
    m = an.meta
    print(f'analysis -> {args.out}\n'
          f'  {m.n_groups} concepts, {live} fired, '
          f'{m.n_tokens:,} tokens over {m.n_units} units\n'
          f'  co-activation graph: {len(an.edges)} edges kept of '
          f'{m.coact_edges_found} at Jaccard>={m.coact_threshold:g}\n'
          f'  chordal 2D variance: {m.chordal_2d_variance*100:.2f}% '
          f'({m.chordal_dims_for_half} dims for 50%) -- why the map uses the graph')
    return 0


def _cmd_dashboard(args):
    from .analysis import Analysis
    from .dashboard import build_app

    an = Analysis.load(args.analysis)
    app = build_app(an)
    print(f'serving layer {an.meta.layer} ({an.meta.n_groups} concepts) on '
          f'http://{args.host}:{args.port}\n'
          f'  over SSH:  ssh -L {args.port}:localhost:{args.port} <host>',
          flush=True)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


# ---------------------------------------------------------------------------
# arg parsing
# ---------------------------------------------------------------------------
def _add_train_args(p):
    m = p.add_argument_group('model')
    m.add_argument('--model', choices=['vanilla', 'grassmannian', 'group_lasso'],
                   default='group_lasso')
    m.add_argument('--n-groups', type=int, required=True)
    m.add_argument('--group-size', type=int, default=3)
    m.add_argument('--l0', type=int, default=16, help='vanilla/grassmannian block TopK')
    m.add_argument('--revival-alpha', type=float, default=0.0,
                   help='vanilla: weight of the auxiliary revival loss, in which '
                        'the top --k-aux DEAD blocks reconstruct the residual. '
                        '0 (default) disables it; try 0.03 (1/32).')
    m.add_argument('--k-aux', type=int, default=None,
                   help='dead blocks recruited per token by the revival loss '
                        '(default: --l0)')
    m.add_argument('--dead-after', type=float, default=1_000_000,
                   help='tokens of silence before a block counts as dead')
    m.add_argument('--target-l0', type=int, default=16,
                   help='group_lasso COLD-START sparsity. It places the initial '
                        'threshold only; the steady-state L0 is set by --coef '
                        'against reconstruction. The trainer warns if the two '
                        'end up more than 2x apart.')
    m.add_argument('--coef', type=float, default=1e-2, help='group_lasso penalty')
    m.add_argument('--gain', type=float, default=10.0)
    m.add_argument('--paper-version', action='store_true')
    m.add_argument('--grad-scale', type=float, default=1.0,
                   help='group_lasso STE theta-grad scale (1.0 is correct for DDP)')
    m.add_argument('--rel-bandwidth', type=float, default=0.1,
                   help='group_lasso STE kernel width, in units of the running '
                        'block-norm scale')
    m.add_argument('--scale-momentum', type=float, default=0.99,
                   help='EMA momentum for the running block-norm scale')
    m.add_argument('--l0-control', type=float, default=0.0,
                   help='dual-ascent gain holding realized L0 at --target-l0. '
                        '0 (default) = fixed --coef, in which case --target-l0 '
                        'only cold-starts the threshold. Try 0.02.')

    s = p.add_argument_group('source')
    s.add_argument('--source', choices=['vision', 'captures'], default='captures')
    s.add_argument('--data', required=True,
                   help='vision: (N,d) .npy/.npz of normalised activations; '
                        'captures: the vLLM filesystem capture root')
    s.add_argument('--layer', type=int, default=0, help='captures: layer index')
    s.add_argument('--hook', default='post_block', help='captures: hook name')
    s.add_argument('--num-workers', type=int, default=4)
    s.add_argument('--shuffle-buffer', type=int, default=1 << 16)
    s.add_argument('--drop-first', type=int, default=0,
                   help='captures: skip the first N sequence positions of every '
                        'request. Position 0 is an attention sink whose residual '
                        'norm is ~4.5x the rest; --drop-first 1 removes it.')
    s.add_argument('--stats-max-rows', type=int, default=None,
                   help='captures: cap rows scanned when computing norm stats')

    t = p.add_argument_group('trainer')
    t.add_argument('--epochs', type=int, default=40)
    t.add_argument('--lr', type=float, default=4e-4)
    t.add_argument('--batch-size', type=int, default=2048)
    t.add_argument('--snr', type=float, default=0.1)
    t.add_argument('--log-every', type=int, default=5)
    t.add_argument('--steps-per-epoch', type=int, default=None)
    t.add_argument('--seed', type=int, default=0)

    y = p.add_argument_group('system')
    y.add_argument('--device', default=None)
    y.add_argument('--compile', action=argparse.BooleanOptionalAction, default=False)
    y.add_argument('--out', default=None, help='checkpoint path (rank 0)')


def _add_capture_args(p):
    p.add_argument('--server', action='append', required=True,
                   help='vLLM base URL, e.g. http://localhost:8000 (repeatable)')
    p.add_argument('--model', required=True, help='served model name')
    p.add_argument('--prompts', required=True, help='prompt file (line/JSONL)')
    p.add_argument('--tag', required=True, help='capture tag (server dir name)')
    p.add_argument('--layers', default='0', help='comma-separated layer indices')
    p.add_argument('--hook', action='append', default=None,
                   help='hook name (repeatable): pre_attn/post_attn/post_block/mlp_in/mlp_out')
    p.add_argument('--positions', default='all_prompt',
                   choices=['last_prompt', 'all_prompt', 'all_generated', 'all'])
    p.add_argument('--layout', default='packed', choices=['per_file', 'packed', 'sharded'])
    p.add_argument('--max-tokens', type=int, default=1)
    p.add_argument('--concurrency', type=int, default=16)
    p.add_argument('--capture-wait', action=argparse.BooleanOptionalAction, default=True)


def _add_analyze_args(p):
    p.add_argument('--ckpt', required=True, help='trained featurizer checkpoint')
    p.add_argument('--data', required=True, help='vLLM filesystem capture root')
    p.add_argument('--layer', type=int, required=True)
    p.add_argument('--hook', default='post_block')
    p.add_argument('--model', choices=['vanilla', 'grassmannian', 'group_lasso'],
                   default='group_lasso')
    p.add_argument('--n-groups', type=int, required=True)
    p.add_argument('--group-size', type=int, default=3)
    p.add_argument('--l0', type=int, default=None,
                   help='REQUIRED for --model vanilla/grassmannian: the block '
                        'TopK the checkpoint was trained with. Not stored in the '
                        'checkpoint, and a wrong value analyses the model at the '
                        'wrong sparsity.')
    p.add_argument('--out', required=True, help='output .npz artifact')
    p.add_argument('--requests', type=int, default=300,
                   help='capture units (requests) to scan')
    p.add_argument('--top-k', type=int, default=12,
                   help='top-activating examples kept per concept')
    p.add_argument('--neighbors', type=int, default=8,
                   help='nearest subspaces / co-firing partners kept per concept')
    p.add_argument('--manifold-points', type=int, default=200,
                   help='firing-cloud points kept per concept')
    p.add_argument('--context', type=int, default=4, help='context tokens each side')
    p.add_argument('--embedding', default='graph',
                   choices=['graph', 'mds', 'tsne', 'pca'],
                   help="map layout: 'graph' = spectral layout of the strong "
                        "co-activation graph (recommended; concept subspaces are "
                        "near-orthogonal so a chordal-distance map shows <1%% of "
                        "its variance). mds/tsne/pca embed chordal distance.")
    p.add_argument('--coact-threshold', type=float, default=0.1,
                   help='minimum Jaccard overlap for a co-activation edge')
    p.add_argument('--gguf', default=None,
                   help='GGUF model file to decode token strings from')
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--drop-first', type=int, default=0,
                   help='MUST match the value the checkpoint was trained with: '
                        'it selects the norm statistics and skips the same '
                        'leading sequence positions')


def _add_dashboard_args(p):
    p.add_argument('--analysis', required=True, help='.npz from `bsf analyze`')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8050)
    p.add_argument('--debug', action='store_true')


def build_parser():
    parser = argparse.ArgumentParser(prog='bsf', description='Block-sparse featurizers')
    sub = parser.add_subparsers(dest='cmd', required=True)
    _add_train_args(sub.add_parser('train', help='train a featurizer'))
    _add_capture_args(sub.add_parser('capture', help='drive vLLM servers to capture activations'))
    _add_analyze_args(sub.add_parser('analyze', help='build a concept-analysis artifact'))
    _add_dashboard_args(sub.add_parser('dashboard', help='serve the concept dashboard'))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == 'capture' and not args.hook:
        args.hook = ['post_block']
    handler = {'train': _cmd_train, 'capture': _cmd_capture,
               'analyze': _cmd_analyze, 'dashboard': _cmd_dashboard}[args.cmd]
    return handler(args)


if __name__ == '__main__':
    raise SystemExit(main())
