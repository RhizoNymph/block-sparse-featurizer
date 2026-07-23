"""Distributed (DDP) helpers: torchrun setup, teardown, collective reducers.

The trainer is launched either bare (single process -> world_size 1, no process
group) or under ``torchrun`` (which sets ``RANK`` / ``WORLD_SIZE`` /
``LOCAL_RANK`` / ``MASTER_ADDR`` / ``MASTER_PORT``). ``init_dist`` reads those and
initialises the process group when running distributed. All collective reducers
are no-ops when ``world_size == 1``.

Note: ``env_rank_world`` reads the environment (not the process group) so it is
safe to call inside DataLoader worker processes, which must NOT touch collectives
but do need their rank/world to shard the file list.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    distributed: bool

    @property
    def is_main(self):
        return self.rank == 0


def env_rank_world():
    """(rank, world_size) from the environment; (0, 1) if unset. Worker-safe."""
    return int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))


def init_dist(device=None):
    """Initialise the process group from torchrun env vars if world_size > 1.

    Returns (DistInfo, device). When bare (no ``WORLD_SIZE`` > 1) this is a no-op
    and returns a single-process DistInfo.
    """
    rank, world = env_rank_world()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    # Honour an explicit CPU request (e.g. more ranks than GPUs); otherwise use
    # CUDA when present. Deciding purely on availability would set an invalid
    # device ordinal when local_rank exceeds the visible GPU count.
    want_cpu = device is not None and str(device).startswith('cpu')
    use_cuda = torch.cuda.is_available() and not want_cpu

    if world > 1:
        backend = 'nccl' if use_cuda else 'gloo'
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        if use_cuda:
            torch.cuda.set_device(local_rank)
        resolved = device or (f'cuda:{local_rank}' if use_cuda else 'cpu')
        return DistInfo(rank, world, local_rank, True), resolved

    resolved = device or ('cuda' if use_cuda else 'cpu')
    return DistInfo(0, 1, 0, False), resolved


def cleanup_dist(info):
    if info.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier(info):
    if info.distributed and dist.is_initialized():
        dist.barrier()


def all_reduce_sum(t, info):
    """In-place SUM all-reduce; returns the tensor. No-op if not distributed."""
    if info.distributed and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def all_reduce_max(t, info):
    """In-place MAX all-reduce (used for the dead-block OR over fired masks)."""
    if info.distributed and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t


def broadcast(t, info, src=0):
    """In-place broadcast from ``src``; returns the tensor. No-op if not dist."""
    if info.distributed and dist.is_initialized():
        dist.broadcast(t, src=src)
    return t


def all_gather_cat(t, info):
    """All-gather ``t`` (same shape on every rank) and concatenate along dim 0."""
    if not (info.distributed and dist.is_initialized()):
        return t
    parts = [torch.empty_like(t) for _ in range(info.world_size)]
    dist.all_gather(parts, t.contiguous())
    return torch.cat(parts, dim=0)
