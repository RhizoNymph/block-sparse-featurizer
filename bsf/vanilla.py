"""Vanilla BSF.

A free linear encoder produces signed per-block codes; the block projection
Pi_l keeps the `l0` blocks of largest norm and zeroes the rest. Encoder and
decoder are untied; sparsity is enforced by construction (no penalty). With
`group_size=1` this is an ordinary absolute-TopK SAE.
"""
import torch
import torch.nn as nn

from .base import BSF, group_topk, unit_blocks
from .revival import RevivalMixin


class VanillaBSF(RevivalMixin, BSF):
    def __init__(self, d, n_groups, group_size=3, l0=16, revival_alpha=0.0,
                 k_aux=None, dead_after=1_000_000):
        super().__init__(d, n_groups, group_size)
        self.l0 = l0
        W = unit_blocks(torch.randn(n_groups * group_size, d), n_groups, group_size)
        self.W_dec = nn.Parameter(W)
        # tied init
        self.W_enc = nn.Parameter(W.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(n_groups * group_size))
        self._init_revival(revival_alpha, k_aux, dead_after)

    def preact(self, x):
        return (x @ self.W_enc + self.b_enc).reshape(-1, self.n_groups,
                                                     self.group_size)

    def encode(self, x):
        a = self.preact(x)
        # per-sample block TopK
        mask = group_topk(a.norm(dim=-1), self.l0)
        return a * mask.unsqueeze(-1)

    def loss(self, x, target=None):
        # target != x -> denoising
        target = x if target is None else target
        a = self.preact(x)
        mask = group_topk(a.norm(dim=-1), self.l0)
        z = a * mask.unsqueeze(-1)
        x_hat = self.decode(z)
        recon = (target - x_hat).pow(2).mean()
        info = {'recon': recon.detach()}
        aux, aux_info = self._revival_term(a, mask > 0, target, x_hat)
        info.update(aux_info)
        return recon if aux is None else recon + aux, info
