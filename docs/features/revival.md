# Concept revival — a gradient path back for silent blocks

## Scope
The auxiliary loss that lets dead blocks re-enter a trained dictionary, its
dead-block tracker, and the wiring into `VanillaBSF`.

## Non-scope
- The operating point itself — see `docs/features/threshold_schedule.md`. Dead
  blocks are the *cost* identified there, not its cause.
- `GroupLassoBSF` / `GrassmannianBSF`. The mixin is featurizer-agnostic and would
  drop into either, but only the block-TopK path is wired and measured.

## The problem

Once the operating point was pinned at L0=32, roughly a quarter of a 4096-block
dictionary went permanently silent — 3068/4096 concepts ever fired over 40k
held-out tokens.

Silence is self-perpetuating, and the reason is one line of the forward pass.
The code is `z = a * mask`, so for a block that wins TopK for no token,
`dz/da = mask = 0`: its encoder columns receive **exactly** zero gradient, and
its decoder rows receive none either because `z` is zero there. Nothing about
the reconstruction objective can bring it back, however wrong the dictionary is
without it. `tests/test_revival.py::test_a_silent_block_gets_no_gradient_without_revival`
pins this directly.

## The mechanism

The block analogue of AuxK. Track how long each block has been silent; each
step, let the top-`k_aux` **dead** blocks try to reconstruct the residual the
live blocks failed to explain, and add `revival_alpha` times that error.

```
tokens_since_fired += batch_rows;  zeroed wherever a block fired
dead        = tokens_since_fired > dead_after
e           = target - x_hat                      (detached)
recruited   = top-k_aux of `dead` by block norm    (per token)
aux         = mean((e - decode(a * recruited))^2)
loss        = recon + revival_alpha * aux
```

Three properties that matter, each with a test:

- **Dead blocks compete only with each other.** Live blocks are masked to `-inf`
  before the top-k, so reviving one never takes capacity from a working concept
  however large its norm.
- **The residual target is detached**, so the auxiliary path steers only the
  dead blocks' own parameters and cannot perturb the main reconstruction.
- **Silence is counted in tokens, not steps**, so `dead_after` means the same
  thing at any batch size.

With `revival_alpha=0` (the default) no buffers are created and the loss is
numerically identical to before, so existing configs are unaffected.

## Results (layer 32, block-TopK, G=4096, k=4, l0=32, 40k held-out tokens)

| | no revival | + revival |
|---|---|---|
| realized L0 | 32.00 | 32.00 |
| R^2 | 0.5659 | **0.5815** |
| concepts that ever fire | 3068 (74.9%) | **4096 (100.0%)** |
| concepts with >=400 firings | 1067 | 995 |
| within-block PC shares | .362/.270/.211/.154 | .361/.257/.210/.159 |
| chordal median | 1.9989 | 1.9984 |

The whole dictionary is live, and reconstruction *improves* — the revived blocks
do useful work rather than displacing specialists. Geometry is unchanged.

**One honest caveat.** Concepts with >=400 firings went slightly *down*
(1067 -> 995). That is arithmetic, not regression: the firing budget is fixed at
`l0` per token, so spreading it over 4096 live concepts instead of 3068 gives
each fewer firings on average (312 vs 417 over 40k tokens). The >=400 threshold
therefore sits at a different point of the distribution in the two runs and is
not comparable across them. If you need many heavily-populated concepts rather
than a fully-used dictionary, a smaller `G` is the lever, not this.

**A warning about testing it.** On a saturated toy (R^2 0.999) revival makes
things *worse* — 15 live blocks against 56 — because with no residual left to
explain, a revived block becomes a generic high-norm direction that displaces
specialists through the TopK competition. On an under-parameterised toy nothing
dies and the comparison is vacuous. Neither resembles the real regime. The
end-to-end benefit is verified on real captures; the unit tests pin the
*mechanism* (dead blocks receive gradient they otherwise could not), which is
deterministic.

## Related files
- `bsf/revival.py` — `DeadBlockTracker`, `aux_revival_loss`, `RevivalMixin`.
- `bsf/vanilla.py` — `VanillaBSF(revival_alpha=, k_aux=, dead_after=)`; `loss`
  now computes `preact`/`mask` explicitly so the auxiliary term can see them.
- `bsf/cli.py` — `--revival-alpha`, `--k-aux`, `--dead-after`.
- `bsf/train.py` — `silent=` in the epoch log.
- `tests/test_revival.py`.

## Invariants / constraints
- `dead_after > 0` tokens and `revival_alpha >= 0`, both enforced in the ctor.
- `DeadBlockTracker.update` is a no-op outside training mode: an evaluation pass
  can never mark a block dead or resurrect one.
- `aux_revival_loss` returns exactly zero when nothing is dead, and recruits at
  most `min(k_aux, n_dead)` blocks.
- A live block must never receive auxiliary gradient.
- `revival_alpha=0` must leave the loss bit-identical to the pre-feature path.
