"""Settles the STE ``grad_scale`` default empirically (plan Verification #1).

Claim: with ``grad_scale=1.0`` the DDP theta-gradient (W ranks x B rows, DDP
averaging) equals the single-GPU global-batch (W*B rows) theta-gradient, needing
NO ``x world_size`` correction; ``grad_scale=W`` over-scales it by W.

Run single-process (prints the reference) or under torchrun to assert:
    torchrun --nproc_per_node=2 tests/ddp_grad_scale_check.py
"""
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from bsf.group_lasso import GroupLassoBSF
from bsf.train import _LossModule

D, G, GS, B = 16, 8, 2, 64


def build_model(grad_scale=1.0):
    torch.manual_seed(1234)
    return GroupLassoBSF(D, G, GS, coef=1e-3, target_l0=3, grad_scale=grad_scale)


def make_data(n):
    g = torch.Generator().manual_seed(999)
    return torch.randn(n, D, generator=g)


def grads_full_batch(model, X):
    """Single-process reference: theta/enc/dec grads over the full X."""
    model.init_theta(model.preact(X).norm(dim=-1))
    loss, _ = model.loss(X, X)
    model.zero_grad()
    loss.backward()
    return (model.raw_theta.grad.clone(), model.W_enc.grad.clone(),
            model.W_dec.grad.clone())


def main():
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    X = make_data(world * B)

    ref_theta, ref_enc, ref_dec = grads_full_batch(build_model(), X)

    if world == 1:
        print('reference theta.grad norm =', ref_theta.norm().item())
        print('run under torchrun --nproc_per_node=2 to assert DDP equivalence')
        return

    dist.init_process_group('gloo')
    local = X[rank * B:(rank + 1) * B].contiguous()

    # init theta from the GLOBAL batch, identically on every rank
    m = build_model()
    gn_local = m.preact(local).norm(dim=-1)
    parts = [torch.empty_like(gn_local) for _ in range(world)]
    dist.all_gather(parts, gn_local.contiguous())
    m.init_theta(torch.cat(parts, 0))

    ddp = DDP(_LossModule(m))
    loss, _ = ddp(local, local)
    ddp.zero_grad()
    loss.backward()

    d_theta = (m.raw_theta.grad - ref_theta).abs().max().item()
    d_enc = (m.W_enc.grad - ref_enc).abs().max().item()
    d_dec = (m.W_dec.grad - ref_dec).abs().max().item()

    # grad_scale=world control: BOTH ranks must run this DDP collective.
    m2 = build_model(grad_scale=world)
    m2.init_theta(torch.cat(parts, 0))
    ddp2 = DDP(_LossModule(m2))
    l2, _ = ddp2(local, local)
    ddp2.zero_grad()
    l2.backward()
    mask = ref_theta.abs() > 1e-9
    factor = (m2.raw_theta.grad[mask] / ref_theta[mask]).abs().mean().item() \
        if mask.any() else float('nan')

    if rank == 0:
        print(f'world={world}  max|dtheta|={d_theta:.2e}  max|denc|={d_enc:.2e}  '
              f'max|ddec|={d_dec:.2e}')
        assert d_theta < 1e-5, f'theta grad mismatch {d_theta}'
        assert d_enc < 1e-5 and d_dec < 1e-5, f'enc/dec mismatch {d_enc}, {d_dec}'
        print(f'grad_scale={world}: theta.grad / ref ~= {factor:.2f} (should be ~{world})')
        assert abs(factor - world) < 0.1, factor
        print('PASS: grad_scale=1.0 reproduces single-GPU grads exactly')

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
