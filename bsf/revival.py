"""Auxiliary revival loss -- a gradient path back for silent concepts.

A block that never fires receives no reconstruction gradient at all: the code is
``z = a * gate``, so ``dz/da = gate = 0`` and nothing reaches its encoder
columns. Whatever killed it, it cannot come back. Measured on layer 32 at L0=32,
roughly a quarter of a 4096-block dictionary ends up in that state.

This is the block analogue of AuxK: track how long each block has been silent,
then add a small term in which the top-``k_aux`` *dead* blocks try to reconstruct
the residual the live blocks failed to explain. Two properties matter:

  * dead blocks compete only against each other, so reviving one never takes
    capacity from a working concept;
  * the residual target is detached, so the auxiliary path cannot perturb the
    main reconstruction -- it only steers the dead blocks' own parameters.

Silence is counted in TOKENS rather than steps, so ``dead_after`` means the same
thing at any batch size.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DeadBlockTracker(nn.Module):
    """Tokens observed since each block last fired.

    ``update`` is a no-op outside training mode, so an evaluation pass can never
    mark a block dead or resurrect one.
    """

    def __init__(self, n_groups: int, dead_after: float = 1_000_000):
        super().__init__()
        if dead_after <= 0:
            raise ValueError(f'dead_after must be > 0 tokens, got {dead_after}')
        self.dead_after = float(dead_after)
        self.register_buffer('tokens_since_fired', torch.zeros(n_groups))

    @torch.no_grad()
    def update(self, fired: torch.Tensor) -> None:
        """``fired``: (N, G) bool -- did each block fire for each token?"""
        if not self.training:
            return
        self.tokens_since_fired += float(fired.shape[0])
        self.tokens_since_fired.masked_fill_(fired.any(0), 0.0)

    def dead_mask(self) -> torch.Tensor:
        """(G,) bool -- blocks silent for longer than ``dead_after`` tokens."""
        return self.tokens_since_fired > self.dead_after

    def n_dead(self) -> int:
        return int(self.dead_mask().sum())


def aux_revival_loss(a: torch.Tensor, dead_mask: torch.Tensor,
                     residual: torch.Tensor, atoms: torch.Tensor,
                     k_aux: int) -> torch.Tensor:
    """Let the largest ``k_aux`` dead blocks reconstruct the residual.

    Args:
        a: (N, G, group_size) pre-activations.
        dead_mask: (G,) bool from :meth:`DeadBlockTracker.dead_mask`.
        residual: (N, d) what the live blocks failed to explain. Detached here,
            so this term never flows back into the main reconstruction.
        atoms: (G * group_size, d) decoder.
        k_aux: how many dead blocks to recruit; clamped to the number available.

    Returns a 0-dim tensor, exactly zero when nothing is dead.
    """
    n_dead = int(dead_mask.sum())
    k = min(int(k_aux), n_dead)
    if k <= 0:
        return a.new_zeros(())

    gn = a.norm(dim=-1)                                   # (N, G)
    # -inf on live blocks: they cannot be selected however large their norm.
    gn = gn.masked_fill(~dead_mask.unsqueeze(0), float('-inf'))
    idx = gn.topk(k, dim=-1).indices
    mask = torch.zeros_like(gn).scatter_(1, idx, 1.0)

    z = a * mask.unsqueeze(-1)
    e_hat = z.reshape(z.shape[0], -1) @ atoms
    return (residual.detach() - e_hat).pow(2).mean()


class RevivalMixin:
    """Adds dead-block tracking + the auxiliary term to a featurizer.

    Subclasses call :meth:`_init_revival` from ``__init__`` and
    :meth:`_revival_term` from ``loss``. With ``revival_alpha == 0`` (the
    default) no buffers are created and the loss is numerically unchanged, so
    existing configs are unaffected by this module's existence.
    """

    def _init_revival(self, revival_alpha: float = 0.0, k_aux: int | None = None,
                      dead_after: float = 1_000_000) -> None:
        if revival_alpha < 0:
            raise ValueError(f'revival_alpha must be >= 0, got {revival_alpha}')
        self.revival_alpha = float(revival_alpha)
        # OpenAI's AuxK uses roughly the main sparsity budget; default to that.
        self.k_aux = int(k_aux) if k_aux is not None else max(
            1, int(getattr(self, 'l0', None) or getattr(self, 'target_l0', 16)))
        self.dead_tracker = (DeadBlockTracker(self.n_groups, dead_after)
                             if self.revival_alpha > 0 else None)

    def _revival_term(self, a: torch.Tensor, fired: torch.Tensor,
                      target: torch.Tensor, x_hat: torch.Tensor):
        """-> (term, info). ``term`` is 0-dim; ``info`` carries the dead count."""
        if self.dead_tracker is None:
            return None, {}
        self.dead_tracker.update(fired)
        dead = self.dead_tracker.dead_mask()
        aux = aux_revival_loss(a, dead, target - x_hat, self.decoder_atoms(),
                               self.k_aux)
        return self.revival_alpha * aux, {
            'dead': torch.as_tensor(self.dead_tracker.n_dead()),
            'aux': aux.detach(),
        }


__all__ = ['DeadBlockTracker', 'aux_revival_loss', 'RevivalMixin']
