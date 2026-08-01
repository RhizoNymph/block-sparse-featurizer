"""Auxiliary revival loss: give silent concepts a gradient path back.

Why this exists. Once the operating point was pinned at L0=32 (see
``docs/features/threshold_schedule.md``) a quarter of the dictionary went
permanently silent — measured over 40k held-out tokens, 4066/4096 concepts fired
in the baseline but only ~3050 in the fixed runs. A block that never wins gets no
reconstruction gradient at all: ``z = a * gate``, so ``dz/da = gate = 0``, and
nothing reaches its encoder columns. It cannot come back on its own.

The remedy is the block analogue of AuxK: track how long each block has been
silent, and add a small auxiliary term in which the top-``k_aux`` *dead* blocks
try to reconstruct the residual the live blocks failed to explain. Dead blocks
compete only against each other, so reviving them never steals capacity from
working concepts, and the residual target is detached so the auxiliary path
cannot perturb the main reconstruction.

Scaling: ``dead_after`` is in TOKENS, not steps, so the same setting means the
same thing at any batch size.
"""
import pytest
import torch

from bsf.revival import DeadBlockTracker, aux_revival_loss


D, G, K = 32, 16, 4


def _fired(rows, groups, n=8):
    f = torch.zeros(n, G, dtype=torch.bool)
    for r, g in zip(rows, groups):
        f[r, g] = True
    return f


# ---------------------------------------------------------------------------
# the tracker
# ---------------------------------------------------------------------------
def test_tracker_starts_with_nothing_dead():
    t = DeadBlockTracker(G, dead_after=100)
    assert int(t.dead_mask().sum()) == 0


def test_firing_resets_the_counter():
    t = DeadBlockTracker(G, dead_after=100)
    t.update(_fired([0], [3]))                  # only block 3 fired, 8 tokens
    assert float(t.tokens_since_fired[3]) == 0.0
    assert float(t.tokens_since_fired[0]) == 8.0


def test_counter_accumulates_in_tokens_not_steps():
    """Same token budget, different batch sizes -> same state."""
    a = DeadBlockTracker(G, dead_after=100)
    b = DeadBlockTracker(G, dead_after=100)
    for _ in range(4):
        a.update(torch.zeros(16, G, dtype=torch.bool))      # 4 x 16 = 64
    for _ in range(8):
        b.update(torch.zeros(8, G, dtype=torch.bool))       # 8 x 8  = 64
    assert torch.equal(a.tokens_since_fired, b.tokens_since_fired)


def test_block_becomes_dead_only_past_the_threshold():
    t = DeadBlockTracker(G, dead_after=100)
    for _ in range(12):                                      # 96 tokens
        t.update(torch.zeros(8, G, dtype=torch.bool))
    assert int(t.dead_mask().sum()) == 0, 'not dead at 96 of 100'
    t.update(torch.zeros(8, G, dtype=torch.bool))            # 104
    assert int(t.dead_mask().sum()) == G


def test_a_revived_block_leaves_the_dead_set():
    t = DeadBlockTracker(G, dead_after=50)
    for _ in range(10):
        t.update(torch.zeros(8, G, dtype=torch.bool))
    assert bool(t.dead_mask()[5])
    t.update(_fired([0], [5]))
    assert not bool(t.dead_mask()[5])


def test_eval_mode_does_not_move_the_tracker():
    t = DeadBlockTracker(G, dead_after=50)
    t.eval()
    t.update(torch.zeros(8, G, dtype=torch.bool))
    assert float(t.tokens_since_fired.max()) == 0.0


# ---------------------------------------------------------------------------
# the auxiliary loss
# ---------------------------------------------------------------------------
def _setup(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, G, K, generator=g)
    atoms = torch.randn(G * K, D, generator=g)
    residual = torch.randn(n, D, generator=g)
    return a, atoms, residual


def test_no_dead_blocks_means_no_aux_loss():
    a, atoms, residual = _setup()
    dead = torch.zeros(G, dtype=torch.bool)
    loss = aux_revival_loss(a, dead, residual, atoms, k_aux=4)
    assert float(loss) == 0.0
    assert not loss.requires_grad or float(loss) == 0.0


def test_aux_loss_uses_only_dead_blocks():
    """A live block must never be recruited, however large its norm."""
    a, atoms, residual = _setup()
    a[:, 0] *= 100.0                            # block 0 is huge but ALIVE
    dead = torch.zeros(G, dtype=torch.bool)
    dead[7] = True
    a.requires_grad_(True)
    aux_revival_loss(a, dead, residual, atoms, k_aux=4).backward()
    assert float(a.grad[:, 0].abs().sum()) == 0.0, 'live block received aux gradient'
    assert float(a.grad[:, 7].abs().sum()) > 0.0, 'the dead block got none'


def test_aux_loss_picks_the_largest_dead_blocks():
    a, atoms, residual = _setup()
    dead = torch.zeros(G, dtype=torch.bool)
    dead[[2, 5, 9, 11]] = True
    a[:, 5] *= 50.0
    a[:, 9] *= 20.0
    a.requires_grad_(True)
    aux_revival_loss(a, dead, residual, atoms, k_aux=2).backward()
    got = {g for g in range(G) if float(a.grad[:, g].abs().sum()) > 0}
    assert got == {5, 9}, f'expected the two largest dead blocks, got {got}'


def test_k_aux_larger_than_the_dead_set_is_clamped():
    a, atoms, residual = _setup()
    dead = torch.zeros(G, dtype=torch.bool)
    dead[[1, 2]] = True
    a.requires_grad_(True)
    aux_revival_loss(a, dead, residual, atoms, k_aux=99).backward()
    got = {g for g in range(G) if float(a.grad[:, g].abs().sum()) > 0}
    assert got == {1, 2}


def test_aux_loss_is_lower_when_dead_atoms_explain_the_residual():
    """Sanity that it is a reconstruction objective, not an arbitrary penalty."""
    torch.manual_seed(0)
    a = torch.zeros(4, G, K)
    a[:, 3, 0] = 1.0                            # block 3 uses its first atom
    atoms = torch.zeros(G * K, D)
    atoms[3 * K + 0, 0] = 1.0                   # ...which points at dim 0
    dead = torch.zeros(G, dtype=torch.bool)
    dead[3] = True
    good = torch.zeros(4, D)
    good[:, 0] = 1.0                            # residual IS that direction
    bad = torch.zeros(4, D)
    bad[:, 1] = 1.0                             # orthogonal to it
    assert float(aux_revival_loss(a, dead, good, atoms, 1)) < \
           float(aux_revival_loss(a, dead, bad, atoms, 1))


def test_residual_target_is_treated_as_a_constant():
    """The aux path must not push gradient back into the main reconstruction."""
    a, atoms, residual = _setup()
    a.requires_grad_(True)                       # the aux path's real parameter
    residual = residual.clone().requires_grad_(True)
    dead = torch.zeros(G, dtype=torch.bool)
    dead[4] = True
    aux_revival_loss(a, dead, residual, atoms, k_aux=2).backward()
    assert float(a.grad.abs().sum()) > 0.0, 'sanity: the aux path has a gradient'
    assert residual.grad is None or float(residual.grad.abs().sum()) == 0.0


# ---------------------------------------------------------------------------
# wired into the featurizers
# ---------------------------------------------------------------------------
def test_revival_off_is_numerically_identical():
    from bsf.vanilla import VanillaBSF

    x = torch.randn(32, D)
    outs = []
    for alpha in (0.0, 0.0):
        torch.manual_seed(0)
        m = VanillaBSF(D, G, K, l0=4, revival_alpha=alpha)
        m.train()
        outs.append(float(m.loss(x)[0].detach()))
    torch.manual_seed(0)
    ref = VanillaBSF(D, G, K, l0=4)              # default ctor, no revival
    ref.train()
    assert float(ref.loss(x)[0].detach()) == outs[0] == outs[1]


def test_revival_reports_the_dead_count():
    from bsf.vanilla import VanillaBSF

    torch.manual_seed(0)
    m = VanillaBSF(D, G, K, l0=2, revival_alpha=1 / 32, dead_after=8)
    m.train()
    x = torch.randn(16, D)
    for _ in range(6):
        m.loss(x)
    _, info = m.loss(x)
    assert 'dead' in info and int(info['dead']) > 0


def _silent_block(m, x, l0):
    """A block that wins TopK for no token in this batch, or None."""
    gn = m.preact(x).norm(dim=-1)
    won = set(gn.topk(l0, dim=-1).indices.flatten().tolist())
    idle = [g for g in range(m.n_groups) if g not in won]
    return idle[0] if idle else None


def test_a_silent_block_gets_no_gradient_without_revival():
    """The problem, stated as a test: silence is self-perpetuating.

    ``z = a * mask``, so a block that wins nowhere has ``dz/da = 0`` and its
    encoder columns receive nothing. This is why dead blocks never recover.
    """
    from bsf.vanilla import VanillaBSF

    torch.manual_seed(0)
    m = VanillaBSF(D, G, K, l0=2)                    # no revival
    m.train()
    x = torch.randn(8, D)                            # few tokens -> idle blocks
    idle = _silent_block(m, x, 2)
    assert idle is not None, 'sanity: this batch must leave some block idle'
    m.loss(x)[0].backward()
    cols = slice(idle * K, (idle + 1) * K)
    assert float(m.W_enc.grad[:, cols].abs().sum()) == 0.0


def test_revival_gives_dead_blocks_a_gradient():
    """...and the auxiliary term is exactly the path back."""
    from bsf.vanilla import VanillaBSF

    torch.manual_seed(0)
    m = VanillaBSF(D, G, K, l0=2, revival_alpha=1 / 32, dead_after=1)
    m.train()
    x = torch.randn(8, D)
    m.loss(x)                                        # age the tracker
    m.zero_grad()
    m.loss(x)[0].backward()

    dead = torch.nonzero(m.dead_tracker.dead_mask()).flatten().tolist()
    assert dead, 'sanity: some block should be marked dead by now'
    total = sum(float(m.W_enc.grad[:, g * K:(g + 1) * K].abs().sum()) for g in dead)
    assert total > 0.0, 'no dead block received any auxiliary gradient'
