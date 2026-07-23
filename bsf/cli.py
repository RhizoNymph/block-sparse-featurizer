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
        return VanillaBSF(d, args.n_groups, args.group_size, l0=args.l0)
    if args.model == 'grassmannian':
        return GrassmannianBSF(d, args.n_groups, args.group_size, l0=args.l0)
    return GroupLassoBSF(d, args.n_groups, args.group_size, coef=args.coef,
                         target_l0=args.target_l0, gain=args.gain,
                         paper_version=args.paper_version, grad_scale=args.grad_scale)


def _build_source(args):
    if args.source == 'vision':
        x = np.load(args.data)
        if isinstance(x, np.lib.npyio.NpzFile):
            x = x[x.files[0]]
        return VisionSource(x)
    mean, scale = normalize.load_or_compute(args.data, args.layer, args.hook,
                                            max_rows=args.stats_max_rows)
    return CapturesSource(args.data, args.layer, args.hook,
                          shuffle_buffer=args.shuffle_buffer, mean=mean, scale=scale,
                          seed=args.seed)


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
# arg parsing
# ---------------------------------------------------------------------------
def _add_train_args(p):
    m = p.add_argument_group('model')
    m.add_argument('--model', choices=['vanilla', 'grassmannian', 'group_lasso'],
                   default='group_lasso')
    m.add_argument('--n-groups', type=int, required=True)
    m.add_argument('--group-size', type=int, default=3)
    m.add_argument('--l0', type=int, default=16, help='vanilla/grassmannian block TopK')
    m.add_argument('--target-l0', type=int, default=16, help='group_lasso init sparsity')
    m.add_argument('--coef', type=float, default=1e-2, help='group_lasso penalty')
    m.add_argument('--gain', type=float, default=10.0)
    m.add_argument('--paper-version', action='store_true')
    m.add_argument('--grad-scale', type=float, default=1.0,
                   help='group_lasso STE theta-grad scale (1.0 is correct for DDP)')

    s = p.add_argument_group('source')
    s.add_argument('--source', choices=['vision', 'captures'], default='captures')
    s.add_argument('--data', required=True,
                   help='vision: (N,d) .npy/.npz of normalised activations; '
                        'captures: the vLLM filesystem capture root')
    s.add_argument('--layer', type=int, default=0, help='captures: layer index')
    s.add_argument('--hook', default='post_block', help='captures: hook name')
    s.add_argument('--num-workers', type=int, default=4)
    s.add_argument('--shuffle-buffer', type=int, default=1 << 16)
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


def build_parser():
    parser = argparse.ArgumentParser(prog='bsf', description='Block-sparse featurizers')
    sub = parser.add_subparsers(dest='cmd', required=True)
    _add_train_args(sub.add_parser('train', help='train a featurizer'))
    _add_capture_args(sub.add_parser('capture', help='drive vLLM servers to capture activations'))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == 'capture' and not args.hook:
        args.hook = ['post_block']
    handler = {'train': _cmd_train, 'capture': _cmd_capture}[args.cmd]
    return handler(args)


if __name__ == '__main__':
    raise SystemExit(main())
