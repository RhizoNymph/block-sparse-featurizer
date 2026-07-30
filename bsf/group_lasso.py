"""Group-lasso BSF — two variants selectable via `paper_version`.

Default (paper_version=False) — block JumpReLU with STE:
    After the paper's release we found this variant trains more reliably.
    A free linear encoder produces signed per-block codes (no ReLU). A block
    fires when its L2 norm clears a per-block threshold theta; the full *signed*
    code is kept (no magnitude shrinkage, which collapses under a norm-constrained
    decoder). theta is learned by a straight-through estimator (the block analogue
    of JumpReLU): a rectangle-kernel pseudo-derivative carries the gradient. An
    L0 (active-block) penalty `coef` sets the sparsity level. theta is
    initialised from the first training batch so that ~`target_l0` blocks fire.

Paper version (paper_version=True) — true group-lasso soft-threshold:
    Matches Eq. (3) of the BSF paper: sh_θ(a)_g = max(1 - θ/||a_g||, 0) * a_g,
    the proximal operator of the ℓ_{2,1} norm. Sparsity is induced by shrinkage
    rather than a hard gate, and an ℓ_{2,1} penalty `coef` is added to the loss.
    theta is a single scalar; target_l0 is ignored.

Set `paper_version=True` only to reproduce the paper's reported architecture.
For new experiments the default variant is recommended.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BSF, unit_blocks



class BlockJumpReLU(torch.autograd.Function):
    """Hard block gate H(||a_g|| - theta) with a straight-through pseudo-derivative
    for theta (rectangle kernel). Passes no gradient to ||a_g|| -- the encoder
    learns through the magnitude path z = a * gate.

    ``grad_scale`` multiplies the theta gradient. It is 1.0 by default: the
    upstream grad ``g`` already carries the ``1/N`` factor from the ``.mean()``
    loss, so the ``.sum(0)`` here is batch-normalised exactly like every other
    parameter's gradient, and DDP's cross-rank averaging over equal per-rank
    batches reproduces the single-GPU global-batch gradient with no correction.
    Exposed as a knob so the DDP-equivalence test can settle it empirically."""

    @staticmethod
    def forward(ctx, gn, theta, bandwidth, grad_scale):
        ctx.save_for_backward(gn, theta)
        ctx.bw = float(bandwidth)
        ctx.grad_scale = float(grad_scale)
        return (gn > theta).to(gn.dtype)

    @staticmethod
    def backward(ctx, g):
        gn, theta = ctx.saved_tensors
        K = ((gn - theta).abs() <= ctx.bw / 2).to(gn.dtype) / ctx.bw
        return None, -(g * K).sum(0) * ctx.grad_scale, None, None


def block_soft_threshold(a, theta):
    """sh_θ(a)_g = max(1 - θ/||a_g||_2, 0) * a_g  (proximal op of ℓ_{2,1})."""
    gn = a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (1.0 - theta / gn).clamp_min(0.0)
    return a * scale


class GroupLassoBSF(BSF):
    """Group-lasso block-sparse featurizer.

    Args:
        paper_version: If True, use the true group-lasso soft-threshold from the
            paper (Eq. 3).  If False (default), use the block JumpReLU + STE
            variant found to work better after the paper's release.
    """

    def __init__(
        self,
        d,
        n_groups,
        group_size=3,
        coef=1e-2,
        target_l0=16,
        gain=10.0,
        paper_version=False,
        grad_scale=1.0,
    ):
        super().__init__(d, n_groups, group_size)
        self.coef = coef
        self.target_l0 = target_l0
        self.gain = gain
        self.paper_version = paper_version
        # theta-gradient scale for the STE (see BlockJumpReLU); 1.0 is correct
        # for DDP -- kept configurable for the equivalence test.
        self.grad_scale = grad_scale
        # python mirrors of the `inited`/`bandwidth` buffers so the steady-state
        # (compiled) gate has no data-dependent branch or host sync on a GPU
        # tensor (float(bandwidth) every forward would break the graph).
        self._theta_inited = False
        self._bandwidth_f = 1.0

        W = unit_blocks(torch.randn(n_groups * group_size, d), n_groups, group_size)
        self.W_dec = nn.Parameter(W)
        self.W_enc = nn.Parameter(W.t().clone())  # tied init
        self.b_enc = nn.Parameter(torch.zeros(n_groups * group_size))

        if paper_version:
            # single scalar threshold, unconstrained parameterisation
            self.log_theta = nn.Parameter(torch.zeros(()))
        else:
            # per-block threshold learned via STE
            self.raw_theta = nn.Parameter(torch.zeros(n_groups))
            self.register_buffer('bandwidth', torch.ones(()))
            self.register_buffer('inited', torch.zeros((), dtype=torch.bool))

    
    def theta_default(self):
        return F.softplus(self.gain * self.raw_theta)

    @torch.no_grad()
    def init_theta(self, gn):
        """Cold-start: place theta so ~target_l0 blocks fire initially."""
        q = 1.0 - self.target_l0 / self.n_groups
        thr = torch.quantile(gn.flatten(), q).clamp_min(1e-3)
        self.raw_theta.copy_(torch.log(torch.expm1(thr)) / self.gain)
        self.bandwidth.copy_(gn.std() * 0.5 + 1e-6)
        self.inited.fill_(True)
        self._theta_inited = True
        self._bandwidth_f = float(self.bandwidth)

    def gate_default(self, gn):
        if self.training and not self._theta_inited:
            self.init_theta(gn)
        return BlockJumpReLU.apply(
            gn, self.theta_default(), self._bandwidth_f, self.grad_scale)

    def load_state_dict(self, state_dict, *args, **kwargs):
        # restore the python mirrors of the inited/bandwidth buffers so a resumed
        # model does not re-cold-start theta on its first training batch.
        out = super().load_state_dict(state_dict, *args, **kwargs)
        if not self.paper_version:
            self._theta_inited = bool(self.inited)
            self._bandwidth_f = float(self.bandwidth)
        return out

    
    def preact(self, x):
        return (x @ self.W_enc + self.b_enc).reshape(-1, self.n_groups, self.group_size)

    def encode(self, x):
        a = self.preact(x)
        if self.paper_version:
            theta = self.log_theta.exp()
            return block_soft_threshold(a, theta)
        else:
            return a * self.gate_default(a.norm(dim=-1)).unsqueeze(-1)

    def loss(self, x, target=None):
        target = x if target is None else target
        a = self.preact(x)

        if self.paper_version:
            theta = self.log_theta.exp()
            z = block_soft_threshold(a, theta)
            recon = (target - self.decode(z)).pow(2).mean()
            l21 = z.norm(dim=-1).sum(-1).mean()  # ℓ_{2,1} penalty
            l0 = (z.norm(dim=-1) > 1e-6).float().sum(-1).mean()  # for logging
            return recon + self.coef * l21, {'recon': recon.detach(), 'l0': l0.detach()}
        else:
            gate = self.gate_default(a.norm(dim=-1))
            z = a * gate.unsqueeze(-1)
            recon = (target - self.decode(z)).pow(2).mean()
            l0 = gate.sum(-1).mean()
            return recon + self.coef * l0, {'recon': recon.detach(), 'l0': l0.detach()}
