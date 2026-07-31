"""The block-JumpReLU threshold must stay trainable for the whole run.

Background — the bug these tests pin down. ``init_theta`` used to set a single
scalar ``bandwidth = gn.std() * 0.5`` from the FIRST training batch and freeze
it (a python float, deliberately, so ``torch.compile`` sees no host sync). The
STE's rectangle kernel only passes gradient to theta where
``|‖a_g‖ - theta| <= bandwidth/2``. As the encoder trains, block norms move and
theta drifts, so an absolute batch-0 bandwidth stops covering the distribution
it is meant to threshold. Measured on the shipped layer-32 checkpoint:

    block norm  median 4.42, p99 11.6      bandwidth 0.197
    theta       median 14.94 (12.2-18.2)   window = +-0.66% of theta
    -> 0.011% of (token, block) pairs in-window
    -> 79% of blocks got EXACTLY zero theta gradient
    -> realized L0 7.9 against ``--target-l0 32``

The fix gates in scale-free units: the model tracks a running mean block norm
(``norm_scale``) and compares ``‖a_g‖ / norm_scale`` to theta, so the threshold
and the fixed-width window cannot be left behind by a drifting encoder.

Making the threshold movable then exposed a second, independent problem it had
been masking: with a FIXED ``coef`` the sparsity penalty is unopposed, so L0
slides to wherever ``coef`` balances reconstruction regardless of ``target_l0``.
The layer-32 baseline took 30 epochs to slide 261 -> 8.6; with a live threshold
the same slide took 5. ``l0_control`` closes that loop by dual ascent on the
multiplier, which is what makes ``target_l0`` mean anything.
"""
import math

import pytest
import torch

from bsf.group_lasso import BlockJumpReLU, GroupLassoBSF


D, G, GS = 32, 16, 4


def _model(**kw):
    torch.manual_seed(0)
    return GroupLassoBSF(D, G, group_size=GS, **kw)


def _norms(n=512, scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, G, generator=g) * scale


# ---------------------------------------------------------------------------
# the kernel itself
# ---------------------------------------------------------------------------
def test_kernel_is_nonzero_arbitrarily_far_from_theta():
    """The whole point of the smooth kernel: no block is ever cut off.

    theta necessarily sits far out in the tail of the block-norm distribution
    (at L0 = 16 of 4096 it is the 99.6th percentile), so a rectangle kernel
    caught 0.039% of pairs and left most blocks with exactly zero gradient.
    """
    rel = 0.1
    for delta in (0.06, 1.0, 5.0, 20.0):
        t = torch.tensor([1.0], requires_grad=True)
        BlockJumpReLU.apply(torch.tensor([[1.0 + delta]]), t, rel, 1.0).sum().backward()
        assert float(t.grad.abs()) > 0, f'kernel died at delta={delta}'


def test_kernel_integrates_to_one():
    """The pseudo-derivative must stay a density, so the gradient scale is
    unchanged by the choice of kernel shape."""
    rel = 0.25
    theta = torch.tensor([0.0]).requires_grad_(True)
    # Riemann sum of the kernel over a wide grid around theta
    gn = torch.arange(-20.0, 20.0, 0.001).reshape(-1, 1)
    BlockJumpReLU.apply(gn, theta, rel, 1.0).sum().backward()
    # Cauchy has a heavy tail, so a finite grid captures slightly less than 1
    assert float(theta.grad.abs()) * 0.001 == pytest.approx(1.0, rel=0.02)


def test_kernel_peaks_at_theta_and_decays():
    rel = 0.1
    def mass(delta):
        t = torch.tensor([1.0], requires_grad=True)
        BlockJumpReLU.apply(torch.tensor([[1.0 + delta]]), t, rel, 1.0).sum().backward()
        return float(t.grad.abs())
    assert mass(0.0) > mass(0.05) > mass(0.2) > mass(1.0)
    import math
    assert mass(0.0) == pytest.approx(1.0 / (math.pi * rel), rel=1e-6)


def test_zero_rel_bandwidth_is_rejected():
    theta = torch.tensor([1.0], requires_grad=True)
    with pytest.raises(ValueError):
        BlockJumpReLU.apply(torch.ones(1, 1), theta, 0.0, 1.0)


# ---------------------------------------------------------------------------
# the property that actually failed in production
# ---------------------------------------------------------------------------
def test_window_survives_a_10x_scale_shift():
    """Train-time drift simulation: block norms grow 10x after cold start.

    An absolute window calibrated at cold start empties and theta freezes. The
    scale-normalised gate keeps the same *fraction* of pairs in-window, because
    ``norm_scale`` tracks the drift.
    """
    m = _model(target_l0=4)
    m.train()
    m.init_theta(_norms(scale=1.0))

    def occupancy(scale, steps=400):
        gn = _norms(scale=scale, seed=1)
        for _ in range(steps):        # let the EMA converge to the new scale
            m._track_scale(gn)
        return m.window_stats_from_norms(gn)['in_window']

    small = occupancy(1.0)
    large = occupancy(10.0)
    assert small > 0, 'sanity: window is populated at the calibration scale'
    assert large == pytest.approx(small, rel=0.35), (
        f'occupancy collapsed under a 10x scale shift: {small:.4f} -> {large:.4f}')


def test_every_block_keeps_a_live_window():
    """A block whose theta drifted high must still receive gradient."""
    m = _model(target_l0=4)
    m.eval()                                    # no EMA update inside this check
    m.init_theta(_norms(scale=1.0))
    with torch.no_grad():                      # push half the blocks 5x higher
        m.raw_theta[: G // 2] += math.log(math.expm1(5.0)) / m.gain
    theta = m.theta_default().detach()
    # each block's own threshold, in raw units: theta * norm_scale
    gn = theta.unsqueeze(0) * float(m.norm_scale) * 1.001
    m.gate_default(gn).sum().backward()
    assert torch.all(m.raw_theta.grad != 0), (
        f'{int((m.raw_theta.grad == 0).sum())}/{G} blocks got no theta gradient')


def test_norm_scale_tracks_the_data():
    """The running scale must converge to the mean block norm, not batch 0's."""
    m = _model(target_l0=4)
    m.train()
    m.init_theta(_norms(scale=1.0))
    gn = _norms(scale=8.0, seed=3)
    for _ in range(800):
        m._track_scale(gn)
    assert float(m.norm_scale) == pytest.approx(float(gn.mean()), rel=0.02)


def test_eval_does_not_move_the_scale():
    """Train/eval must gate identically; eval batches must not shift theta."""
    m = _model(target_l0=4)
    m.train()
    m.init_theta(_norms(scale=1.0))
    before = float(m.norm_scale)
    m.eval()
    m.gate_default(_norms(scale=50.0, seed=4))
    assert float(m.norm_scale) == before


# ---------------------------------------------------------------------------
# the trainer-facing diagnostic
# ---------------------------------------------------------------------------
def test_window_occupancy_reports_dead_blocks():
    from bsf.group_lasso import window_occupancy

    m = _model(target_l0=4)
    m.init_theta(_norms(scale=1.0))
    theta = m.theta_default().detach()
    # every block sitting exactly at its threshold -> peak kernel mass
    live = window_occupancy(theta.unsqueeze(0).clone(), theta, m.rel_bandwidth)
    assert live['dead_blocks'] == 0
    assert live['in_window'] == pytest.approx(1.0, rel=1e-6)
    # far away -> much less mass, but NOT zero: that is the point of the
    # polynomial tail, and no block may be reported dead.
    far = window_occupancy(theta.unsqueeze(0) * 100.0, theta, m.rel_bandwidth)
    assert far['in_window'] < live['in_window'] / 1000
    assert far['in_window'] > 0
    assert far['dead_blocks'] == 0


def test_l0_target_check_flags_the_shipped_regime():
    """7.9 realized against a target of 32 must be flagged."""
    from bsf.train import l0_off_target

    assert l0_off_target(7.9, 32, tol=2.0)
    assert l0_off_target(80.0, 32, tol=2.0)
    assert not l0_off_target(30.0, 32, tol=2.0)
    assert not l0_off_target(32.0, 32, tol=2.0)
    assert not l0_off_target(5.0, None, tol=2.0), 'no target -> never off-target'


# ---------------------------------------------------------------------------
# end-to-end: theta must still be moving late in training
# ---------------------------------------------------------------------------
def test_theta_stays_trainable_through_a_short_run():
    """After 200 steps on drifting data, theta must not be frozen."""
    torch.manual_seed(0)
    m = _model(target_l0=4, coef=1e-2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x = torch.randn(256, D)

    grads = []
    for step in range(200):
        # deliberately drift the input scale so an absolute window would die
        xb = x * (1.0 + 4.0 * step / 200)
        loss, _ = m.loss(xb)
        opt.zero_grad()
        loss.backward()
        grads.append(float((m.raw_theta.grad != 0).float().mean()))
        opt.step()
        m.normalize_decoder()

    late = sum(grads[-50:]) / 50
    assert late > 0.05, (
        f'only {late:.1%} of blocks had a live theta gradient over the last 50 '
        f'steps — the threshold has gone dead')


# ---------------------------------------------------------------------------
# the L0 controller: make target_l0 the thing that is actually held
# ---------------------------------------------------------------------------
def test_controller_off_by_default():
    m = _model(target_l0=4)
    assert m.l0_control == 0.0
    assert m.effective_coef() == m.coef


def test_controller_raises_penalty_when_too_dense():
    m = _model(target_l0=4, l0_control=0.05)
    m.train()
    m.inited.fill_(True)
    for _ in range(200):
        m._control_l0(torch.tensor(40.0))       # 10x the target
    assert m.effective_coef() > m.coef * 5


def test_controller_lowers_penalty_when_too_sparse():
    m = _model(target_l0=32, l0_control=0.05)
    m.train()
    m.inited.fill_(True)
    for _ in range(200):
        m._control_l0(torch.tensor(8.6))        # the shipped run's endpoint
    assert m.effective_coef() < m.coef / 5


def test_controller_is_a_fixed_point_at_target():
    m = _model(target_l0=16, l0_control=0.05)
    m.train()
    m.inited.fill_(True)
    for _ in range(500):
        m._control_l0(torch.tensor(16.0))
    assert m.effective_coef() == pytest.approx(m.coef, rel=1e-6)


def test_controller_multiplier_is_bounded():
    m = _model(target_l0=4, l0_control=1.0, l0_control_clip=2.0)
    m.train()
    m.inited.fill_(True)
    for _ in range(10_000):
        m._control_l0(torch.tensor(4000.0))
    assert float(m.log_coef_adj) == pytest.approx(2.0)


def test_controller_pulls_l0_toward_target_end_to_end():
    """The property the whole feature exists for."""
    torch.manual_seed(0)
    target = 6
    m = _model(target_l0=target, coef=1e-2, l0_control=0.02)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x = torch.randn(256, D)
    for _ in range(1500):
        loss, info = m.loss(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        m.normalize_decoder()
    l0 = float(info['l0'])
    assert abs(l0 - target) < 0.5 * target, (
        f'controller left L0 at {l0:.1f} against target {target}')


def _final_l0(target, l0_control, steps=1500, seed=0):
    torch.manual_seed(seed)
    m = _model(target_l0=target, coef=1e-2, l0_control=l0_control)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x = torch.randn(256, D)
    info = {}
    for _ in range(steps):
        loss, info = m.loss(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        m.normalize_decoder()
    return float(info['l0'])


def test_control_lands_closer_to_target_than_a_fixed_coef():
    """The comparison that matters: same run, controller on vs off.

    A fixed `coef` settles wherever it balances reconstruction, which is only
    the requested sparsity by luck. How far off depends on the data — 1.4x on
    this toy, 3.7x on the real layer-32 captures (target 32, realized 8.6).
    """
    target = 6
    off = _final_l0(target, 0.0)
    on = _final_l0(target, 0.02)
    assert abs(on - target) < abs(off - target), (
        f'controller did not improve on the fixed coefficient: '
        f'|{on:.1f}-{target}| vs |{off:.1f}-{target}|')


def test_controller_off_is_numerically_identical():
    """Existing configs must be bit-unaffected by the controller's existence."""
    x = torch.randn(64, D)
    outs = []
    for _ in range(2):
        torch.manual_seed(0)
        m = GroupLassoBSF(D, G, group_size=GS, target_l0=4, coef=1e-2,
                          l0_control=0.0)
        m.train()
        loss, info = m.loss(x)
        outs.append((float(loss.detach()), float(info['l0'])))
    assert outs[0] == outs[1]

    torch.manual_seed(0)
    ref = GroupLassoBSF(D, G, group_size=GS, target_l0=4, coef=1e-2)
    ref.train()
    ref_loss, _ = ref.loss(x)
    assert float(ref_loss.detach()) == outs[0][0], 'default ctor must match l0_control=0'


def test_control_error_is_symmetric_in_log_space():
    """2x too dense and 2x too sparse must push equally hard.

    A linear relative error clipped to +-1 does not: it saturates at 2x dense
    (+1.0) but reaches only -0.5 at 2x sparse, biasing toward the failure mode
    this feature exists to prevent.
    """
    target = 32
    dense = _model(target_l0=target, l0_control=0.01)
    sparse = _model(target_l0=target, l0_control=0.01)
    for m in (dense, sparse):
        m.train()
        m.inited.fill_(True)
    dense._control_l0(torch.tensor(float(2 * target)))
    sparse._control_l0(torch.tensor(float(target / 2)))
    assert float(dense.log_coef_adj) == pytest.approx(-float(sparse.log_coef_adj),
                                                      rel=1e-6)


def test_control_gain_is_per_step_and_bounded_per_step():
    """One step moves log_coef_adj by at most the gain (error clips at +-1)."""
    m = _model(target_l0=32, l0_control=0.003)
    m.train()
    m.inited.fill_(True)
    m._control_l0(torch.tensor(1e6))            # absurdly dense
    assert float(m.log_coef_adj) == pytest.approx(0.003, rel=1e-6)
