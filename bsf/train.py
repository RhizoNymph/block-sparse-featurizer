"""One trainer for all three featurizers.

It is featurizer-agnostic: it only relies on `model.loss(x, target)` and
`model.normalize_decoder()`. Plain Adam, full passes over the activation matrix.

With `snr > 0` it trains as a denoiser: the input is corrupted with Gaussian
noise of std `snr * std(x)`, but the model still reconstructs the clean `x`.

Two entry points:
  - `train(model, x, ...)` -- the original single-GPU, in-memory trainer, kept
    verbatim so its numerics never change.
  - `fit(model, source, ...)` -- the source-driven trainer that adds DDP,
    streaming activation sources, and torch.compile.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from .distributed import (
    init_dist, cleanup_dist, barrier, all_reduce_sum, all_reduce_max,
    all_gather_cat,
)


@torch.no_grad()
def recon_r2(model, x):
    """Fraction of variance explained by the (clean) reconstruction."""
    x_hat = model(x)[0]
    ss_res = (x - x_hat).pow(2).sum()
    ss_tot = (x - x.mean(0, keepdim=True)).pow(2).sum()
    return float(1.0 - ss_res / ss_tot.clamp_min(1e-12))


def l0_off_target(realized, target, tol=2.0):
    """Is the realized mean L0 more than ``tol``x away from what was asked for?

    The shipped layer-32 run requested ``--target-l0 32`` and realized 7.9 while
    logging L0 every epoch: the number was printed but never compared, so a 4x
    miss looked like a normal training curve. ``None`` target -> never off.
    """
    if target is None or target <= 0:
        return False
    r = max(float(realized), 1e-9)
    return r > tol * target or r < target / tol


@torch.no_grad()
def l0_dead(model, x):
    """Mean active blocks per token, and number of blocks that never fire."""
    # (N, G)
    active = model.encode(x).norm(dim=-1) > 1e-6
    l0 = float(active.float().sum(1).mean())
    dead = int((~active.any(0)).sum())
    return l0, dead


def train(model, x, *, epochs=40, lr=4e-4, batch_size=2048, snr=0.1,
          device=None, log_every=5):
    """Train `model` on activations `x` (N, d). Returns the trained model."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    x = torch.as_tensor(x, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = x.shape[0]

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        running = 0.0
        n_batches = 0
        for i in range(0, n - batch_size + 1, batch_size):
            xb = x[perm[i:i + batch_size]].to(device)
            xb_in = xb if snr <= 0 else xb + snr * xb.std() * torch.randn_like(xb)
            # reconstruct clean xb
            loss, _ = model.loss(xb_in, target=xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.normalize_decoder()
            running += float(loss.item())
            n_batches += 1

        if ep == 1 or ep == epochs or ep % log_every == 0:
            model.eval()
            sub = x[torch.randperm(n)[:20_000]].to(device)
            r2 = recon_r2(model, sub)
            l0, dead = l0_dead(model, sub)
            print(f'epoch {ep:3d}/{epochs}   loss={running / n_batches:.4f}   '
                  f'R2={r2:.4f}   L0={l0:.1f}   dead={dead}/{model.n_groups}', flush=True)
    return model


# ---------------------------------------------------------------------------
# Source-driven DDP trainer
# ---------------------------------------------------------------------------
class _LossModule(nn.Module):
    """Wraps a featurizer so DistributedDataParallel's ``forward`` IS the loss.

    The trainer calls ``model.loss(x, target)``, but DDP only all-reduces
    gradients for the pass that goes through ``DDP.forward``. Routing the loss
    through this wrapper's ``forward`` is what arms the gradient sync -- calling
    ``model.loss`` on the inner module directly would silently skip it.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, target):
        return self.model.loss(x, target)


def _needs_theta_warmup(model):
    return (hasattr(model, 'init_theta') and hasattr(model, '_theta_inited')
            and not model._theta_inited and not getattr(model, 'paper_version', False))


@torch.no_grad()
def _warmup_theta(model, xb_in, info):
    """Cold-start the per-block threshold identically on every rank.

    ``raw_theta`` is a Parameter broadcast only at DDP construction, so a
    per-rank cold-start inside the loop would diverge forever. We gather the
    first batch's block norms across ranks and init from the same global tensor.
    """
    model.train()
    gn = model.preact(xb_in).norm(dim=-1)      # (N, G)
    model.init_theta(all_gather_cat(gn, info))  # identical on all ranks


def _build_loader(source, ds, *, batch_size, num_workers, info, seed):
    """Returns (loader, sampler_or_None). Map-style sources get a
    DistributedSampler; iterable sources self-shard from the environment."""
    if source.is_iterable:
        loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                            drop_last=True, pin_memory=torch.cuda.is_available())
        return loader, None
    sampler = None
    if info.distributed:
        sampler = DistributedSampler(ds, num_replicas=info.world_size, rank=info.rank,
                                     shuffle=True, drop_last=True, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                        shuffle=(sampler is None), drop_last=True,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return loader, sampler


def _steps_per_epoch(source, batch_size, info, override):
    if override is not None:
        return int(override)
    rows = source.num_rows()
    if rows is None:
        raise ValueError('steps_per_epoch is required when the source row count '
                         'is unknown')
    return max(1, rows // info.world_size // batch_size)


@torch.no_grad()
def _sync_norm_scale(model, info):
    """Make the running block-norm scale exactly equal on every rank.

    It is EMA'd from each rank's own batches inside the gate (no collective in
    the hot path, so torch.compile sees no graph break). Left alone the ranks
    would slowly diverge and gate differently; an all-reduced mean every
    ``scale_sync_every`` steps bounds the drift to that interval.
    """
    if not info.distributed or not hasattr(model, 'norm_scale'):
        return
    s = model.norm_scale.clone()
    all_reduce_sum(s, info)
    model.norm_scale.copy_(s / info.world_size)


@torch.no_grad()
def _window_report(model, xb):
    """STE-window diagnostics, or None for featurizers without a learned gate."""
    if not hasattr(model, 'window_stats'):
        return None
    return model.window_stats(xb)


@torch.no_grad()
def _evaluate(model, loader, device, info, eval_steps):
    """Reduced metrics across ranks: (R2, mean L0, dead-block count).

    Parts are reduced, never per-rank averages: R2 from summed ss_res/ss_tot,
    L0 from summed active/token counts, dead blocks from the OR of fired masks.
    """
    model.eval()
    ss_res = torch.zeros((), device=device)
    ss_tot = torch.zeros((), device=device)
    active = torch.zeros((), device=device)
    tokens = torch.zeros((), device=device)
    fired_any = torch.zeros(model.n_groups, device=device)
    for i, batch in enumerate(loader):
        if i >= eval_steps:
            break
        xb = batch.to(device, non_blocking=True)
        x_hat, z = model(xb)
        ss_res += (xb - x_hat).pow(2).sum()
        ss_tot += (xb - xb.mean(0, keepdim=True)).pow(2).sum()
        fired = z.norm(dim=-1) > 1e-6           # (N, G)
        active += fired.float().sum()
        tokens += xb.shape[0]
        fired_any = torch.maximum(fired_any, fired.any(0).float())
    all_reduce_sum(ss_res, info)
    all_reduce_sum(ss_tot, info)
    all_reduce_sum(active, info)
    all_reduce_sum(tokens, info)
    all_reduce_max(fired_any, info)
    r2 = float(1.0 - ss_res / ss_tot.clamp_min(1e-12))
    l0 = float(active / tokens.clamp_min(1))
    dead = int((fired_any == 0).sum())
    return r2, l0, dead


def fit(model, source, *, epochs=40, lr=4e-4, batch_size=2048, snr=0.1,
        device=None, log_every=5, steps_per_epoch=None, num_workers=4,
        compile=False, out=None, seed=0, eval_steps=10, scale_sync_every=50):
    """Train `model` on an `ActivationSource`. DDP-aware (single process when not
    launched under torchrun). Returns the trained model.

    The single-GPU + in-memory-vision case should use `train()` instead; this
    path is for streaming sources and/or multi-GPU DDP.
    """
    info, device = init_dist(device)
    model.to(device)
    ds = source.dataset()
    loader, sampler = _build_loader(source, ds, batch_size=batch_size,
                                    num_workers=num_workers, info=info, seed=seed)
    spe = _steps_per_epoch(source, batch_size, info, steps_per_epoch)

    # Eager theta warm-up BEFORE the DDP wrap / compile so the data-dependent
    # init (quantile, GPU-bool branch) never enters the traced graph and the
    # threshold is identical on every rank.
    if _needs_theta_warmup(model):
        first = next(iter(loader)).to(device, non_blocking=True)
        first_in = first if snr <= 0 else first + snr * first.std() * torch.randn_like(first)
        _warmup_theta(model, first_in, info)

    core = _LossModule(model)
    if info.distributed:
        device_ids = [info.local_rank] if device.startswith('cuda') else None
        core = DistributedDataParallel(core, device_ids=device_ids,
                                       broadcast_buffers=False)
    step_model = torch.compile(core) if compile else core
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        if sampler is not None:
            sampler.set_epoch(ep)
        elif source.is_iterable:
            ds.set_epoch(ep)

        running = torch.zeros((), device=device)
        nb = 0
        last_xb = None
        for step, batch in enumerate(loader):
            if step >= spe:
                break
            xb = batch.to(device, non_blocking=True)
            xb_in = xb if snr <= 0 else xb + snr * xb.std() * torch.randn_like(xb)
            loss, _ = step_model(xb_in, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.normalize_decoder()
            if scale_sync_every and step % scale_sync_every == 0:
                _sync_norm_scale(model, info)
            running += loss.detach()
            nb += 1
            last_xb = xb_in

        if ep == 1 or ep == epochs or ep % log_every == 0:
            r2, l0, dead = _evaluate(model, loader, device, info, eval_steps)
            win = _window_report(model, last_xb) if last_xb is not None else None
            if info.is_main:
                avg = float(running / max(nb, 1))
                extra = ''
                if win is not None:
                    extra = (f'   win={win["in_window"] * 100:.3f}%'
                             f'   frozen={win["dead_blocks"]}/{model.n_groups}')
                if getattr(model, 'l0_control', 0):
                    extra += f'   coef={float(model.effective_coef()):.3e}'
                print(f'epoch {ep:3d}/{epochs}   loss={avg:.4f}   R2={r2:.4f}   '
                      f'L0={l0:.1f}   dead={dead}/{model.n_groups}{extra}',
                      flush=True)
                target = getattr(model, 'target_l0', None)
                if l0_off_target(l0, target):
                    print(f'  WARNING: realized L0 {l0:.1f} is off target_l0='
                          f'{target} by more than 2x. `target_l0` only sets the '
                          f'cold-start threshold; the operating point is set by '
                          f'`coef` against reconstruction — retune it.',
                          flush=True)
                if win is not None and win['dead_blocks'] > 0.5 * model.n_groups:
                    print(f'  WARNING: {win["dead_blocks"]}/{model.n_groups} blocks '
                          f'have an empty STE window — their thresholds are '
                          f'frozen and will not recover.', flush=True)

    barrier(info)
    if info.is_main and out is not None:
        torch.save(model.state_dict(), out)
    cleanup_dist(info)
    return model
