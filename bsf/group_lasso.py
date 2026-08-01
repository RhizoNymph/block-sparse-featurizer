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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BSF, unit_blocks



class BlockJumpReLU(torch.autograd.Function):
    """Hard block gate H(||a_g|| - theta) with a straight-through pseudo-derivative
    for theta (rectangle kernel). Passes no gradient to ||a_g|| -- the encoder
    learns through the magnitude path z = a * gate.

    ``gn`` is expected already divided by the model's running block-norm scale,
    so theta and the kernel width ``rel_bandwidth`` both live in scale-free
    units. That is what keeps the window alive: an absolute width around an
    absolute threshold has to be calibrated to one particular scale, and both
    the norms and the threshold move over a run. Measured failures in both
    directions -- theta stranded *above* the distribution (the shipped layer-32
    checkpoint: 0.011% of pairs in-window, 79% of blocks with zero threshold
    gradient) and stranded *below* it (norms drifting up 5x, every block
    saturating). See ``tests/test_threshold_schedule.py``.

    ``grad_scale`` multiplies the theta gradient. It is 1.0 by default: the
    upstream grad ``g`` already carries the ``1/N`` factor from the ``.mean()``
    loss, so the ``.sum(0)`` here is batch-normalised exactly like every other
    parameter's gradient, and DDP's cross-rank averaging over equal per-rank
    batches reproduces the single-GPU global-batch gradient with no correction.
    Exposed as a knob so the DDP-equivalence test can settle it empirically."""

    @staticmethod
    def forward(ctx, gn, theta, rel_bandwidth, grad_scale):
        if not rel_bandwidth > 0:
            raise ValueError(
                f'rel_bandwidth must be > 0, got {rel_bandwidth}: a zero-width '
                f'STE window passes no gradient and freezes theta')
        ctx.save_for_backward(gn, theta)
        ctx.rel_bw = float(rel_bandwidth)
        ctx.grad_scale = float(grad_scale)
        return (gn > theta).to(gn.dtype)

    @staticmethod
    def backward(ctx, g):
        gn, theta = ctx.saved_tensors
        # Cauchy pseudo-derivative K(u) = 1/(pi*tau*(1+u^2)), u = (gn-theta)/tau.
        #
        # Not a rectangle: a rectangle is exactly zero outside its width, and
        # theta necessarily sits far out in the tail of the block-norm
        # distribution (at L0=16 of 4096 it is the 99.6th percentile), where
        # almost nothing lands inside any fixed width -- measured 0.039% of
        # pairs, leaving most blocks with exactly zero gradient and a permanently
        # frozen threshold.
        #
        # Not a logistic either: its tail decays like e^-|u|, which underflows to
        # exactly zero around |u| ~ 200 in fp32 -- the same cliff, moved. Cauchy
        # decays polynomially, so a threshold that has been stranded far from the
        # data still feels a real pull back toward it. Integrates to 1, so the
        # gradient scale does not depend on this choice.
        tau = ctx.rel_bw
        u = (gn - theta) / tau
        K = 1.0 / (math.pi * tau * (1.0 + u * u))
        return None, -(g * K).sum(0) * ctx.grad_scale, None, None


@torch.no_grad()
def window_occupancy(gn, theta, rel_bandwidth, eps=1e-8):
    """Diagnostic: how much gradient signal theta actually receives.

    ``gn`` must already be scale-normalised, matching ``gate_default``.

    ``in_window`` is the mean Cauchy-kernel mass per (token, block) pair,
    normalised by the kernel peak (1/(pi*tau)) so it reads as a fraction. It is
    a continuous stand-in for "what share of the batch is near a threshold", and
    is not comparable to the pre-fix rectangle version.
    ``dead_blocks`` counts blocks whose total kernel mass is below ``eps``. With
    a polynomial tail this should stay 0; a nonzero count means a threshold has
    run so absurdly far from the data that even 1/u^2 underflowed.
    """
    tau = float(rel_bandwidth)
    u = (gn - theta) / tau
    K = 1.0 / (math.pi * tau * (1.0 + u * u))
    # normalised by the kernel peak so it reads as a fraction
    return {'in_window': float((K * (math.pi * tau)).mean()),
            'dead_blocks': int((K.sum(0) < eps).sum())}


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
        rel_bandwidth=0.1,
        scale_momentum=0.99,
        l0_control=0.0,
        l0_control_clip=2.0,
        l0_control_kp=2.0,
    ):
        super().__init__(d, n_groups, group_size)
        self.coef = coef
        self.target_l0 = target_l0
        self.gain = gain
        self.paper_version = paper_version
        # theta-gradient scale for the STE (see BlockJumpReLU); 1.0 is correct
        # for DDP -- kept configurable for the equivalence test.
        self.grad_scale = grad_scale
        # STE kernel width, in units of the running block-norm scale.
        if not rel_bandwidth > 0:
            raise ValueError(f'rel_bandwidth must be > 0, got {rel_bandwidth}')
        self.rel_bandwidth = float(rel_bandwidth)
        self.scale_momentum = float(scale_momentum)
        # Dual ascent on the constraint E[L0] = target_l0. With a FIXED `coef`,
        # target_l0 only places the cold-start threshold and the operating point
        # is wherever `coef` happens to balance reconstruction -- on the layer-32
        # captures coef=1e-2 drives L0 to ~8.6 against a target of 32, and the
        # only reason it took 30 epochs to get there was a frozen STE window
        # slowing the descent. With l0_control > 0 the coefficient is the
        # multiplier and target_l0 is the thing actually held.
        self.l0_control = float(l0_control)
        self.l0_control_clip = float(l0_control_clip)
        self.l0_control_kp = float(l0_control_kp)
        # python mirror of the `inited` buffer so the steady-state (compiled)
        # gate has no data-dependent branch on a GPU tensor.
        self._theta_inited = False

        W = unit_blocks(torch.randn(n_groups * group_size, d), n_groups, group_size)
        self.W_dec = nn.Parameter(W)
        self.W_enc = nn.Parameter(W.t().clone())  # tied init
        self.b_enc = nn.Parameter(torch.zeros(n_groups * group_size))

        if paper_version:
            # single scalar threshold, unconstrained parameterisation
            self.log_theta = nn.Parameter(torch.zeros(()))
        else:
            # per-block threshold learned via STE, in units of `norm_scale`
            self.raw_theta = nn.Parameter(torch.zeros(n_groups))
            # running mean block norm. The gate compares ||a_g|| / norm_scale to
            # theta, so the threshold is scale-free and cannot be left behind by
            # a drifting encoder. Kept as a buffer (not recomputed per batch) so
            # train and eval gate identically and small batches do not jitter the
            # threshold. Under DDP it is EMA'd per rank and exactly re-synced
            # every `scale_sync_every` steps by the trainer.
            self.register_buffer('norm_scale', torch.ones(()))
            self.register_buffer('inited', torch.zeros((), dtype=torch.bool))
            # log-space offset applied to `coef` by the L0 controller; 0 when off
            self.register_buffer('log_coef_adj', torch.zeros(()))   # integral
            self.register_buffer('log_coef_prop', torch.zeros(()))  # proportional
            self.register_buffer('l0_ema', torch.zeros(()))

    
    def theta_default(self):
        return F.softplus(self.gain * self.raw_theta)

    @torch.no_grad()
    def init_theta(self, gn):
        """Cold-start: set the norm scale, then place theta so ~target_l0 blocks
        fire initially. Both are in scale-free units from here on."""
        self.norm_scale.copy_(gn.mean().clamp_min(1e-6))
        q = 1.0 - self.target_l0 / self.n_groups
        thr = torch.quantile((gn / self.norm_scale).flatten(), q).clamp_min(1e-3)
        self.raw_theta.copy_(torch.log(torch.expm1(thr)) / self.gain)
        self.inited.fill_(True)
        self._theta_inited = True

    @torch.no_grad()
    def _track_scale(self, gn):
        m = self.scale_momentum
        self.norm_scale.mul_(m).add_(gn.mean().clamp_min(1e-6), alpha=1.0 - m)

    @torch.no_grad()
    def _control_l0(self, l0):
        """One dual-ascent step on E[L0] = target_l0.

        Too many active blocks -> raise the penalty; too few -> lower it.

        The error is measured in LOG space, so being 2x too dense and 2x too
        sparse push equally hard. A linear relative error does not: clipped to
        +-1 it saturates at 2x too dense but only reaches -0.5 at 2x too sparse,
        biasing the loop toward over-sparsification -- which is the direction
        that was already broken. Clipped to +-1, i.e. a factor of e per step's
        worth of signal, so the opening transient (L0 can start 8x target)
        cannot slam the multiplier into its bound.

        ``l0_control`` is a PER-STEP gain. At ~1200 steps/epoch a gain of 0.02
        can move the multiplier by e^24 in one epoch; keep it near 1e-3 so the
        multiplier moves O(1) per epoch and the objective the dictionary sees is
        approximately stationary.
        """
        m = self.scale_momentum
        if not bool(self.inited) or float(self.l0_ema) == 0.0:
            self.l0_ema.copy_(l0)
        else:
            self.l0_ema.mul_(m).add_(l0, alpha=1.0 - m)
        ratio = (self.l0_ema.clamp_min(1e-6) / max(self.target_l0, 1))
        err = ratio.log().clamp(-1.0, 1.0)
        # Integral term, tightly bounded. A wide bound is an windup trap: a pure
        # integrator driven to a +-6 log-unit rail needs ~2.5 epochs at this gain
        # just to come back off it, and L0 diverges meanwhile (measured: 10 ->
        # 17 -> 70 -> 266 over four epochs). Anti-windup: stop accumulating in
        # the direction that would push further into the rail.
        adj = self.log_coef_adj
        railed = ((adj >= self.l0_control_clip) & (err > 0)) | \
                 ((adj <= -self.l0_control_clip) & (err < 0))
        adj.add_(torch.where(railed, torch.zeros_like(err),
                             self.l0_control * err))
        adj.clamp_(-self.l0_control_clip, self.l0_control_clip)
        # Proportional term: responds to the CURRENT error immediately instead
        # of waiting for the integral to unwind. This is what damps the loop.
        self.log_coef_prop.copy_(self.l0_control_kp * err)

    def effective_coef(self):
        """The L0 multiplier actually applied this step."""
        if not self.l0_control or self.paper_version:
            return self.coef
        total = (self.log_coef_prop + self.log_coef_adj).clamp(
            -2.0 * self.l0_control_clip, 2.0 * self.l0_control_clip)
        return self.coef * total.exp()

    def gate_default(self, gn):
        if self.training and not self._theta_inited:
            self.init_theta(gn)
        if self.training:
            self._track_scale(gn)
        gn = gn / self.norm_scale.clamp_min(1e-6)
        return BlockJumpReLU.apply(
            gn, self.theta_default(), self.rel_bandwidth, self.grad_scale)

    def window_stats(self, x):
        """STE-window diagnostics for a batch (see ``window_occupancy``)."""
        return self.window_stats_from_norms(self.preact(x).norm(dim=-1))

    def window_stats_from_norms(self, gn):
        """As ``window_stats`` but from raw block norms already to hand."""
        return window_occupancy(gn / self.norm_scale.clamp_min(1e-6),
                                self.theta_default().detach(),
                                self.rel_bandwidth)

    def load_state_dict(self, state_dict, *args, **kwargs):
        # restore the python mirror of `inited` so a resumed model does not
        # re-cold-start theta on its first training batch.
        if not self.paper_version and 'bandwidth' in state_dict:
            # Pre-fix checkpoint: `bandwidth` was an absolute STE width and theta
            # was an absolute threshold. There is no norm_scale to recover, so
            # loading one into the current gate would silently change its
            # meaning -- refuse rather than produce a plausible wrong model.
            raise ValueError(
                'checkpoint predates the scale-free threshold (it carries a '
                '`bandwidth` buffer and an absolute theta). Its gate is not '
                'convertible; retrain, or load it with the pre-fix code.')
        out = super().load_state_dict(state_dict, *args, **kwargs)
        if not self.paper_version:
            self._theta_inited = bool(self.inited)
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
            if self.training and self.l0_control:
                self._control_l0(l0.detach())
            coef = self.effective_coef()
            return recon + coef * l0, {'recon': recon.detach(), 'l0': l0.detach(),
                                       'coef': torch.as_tensor(coef).detach()}
