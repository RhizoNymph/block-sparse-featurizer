"""Grassmannian BSF.

Each concept is an orthonormal `group_size`-frame D_g (a point on a
Grassmannian). The constraint moves to the dictionary: encoder and decoder are
tied to the same frame, the code is z_g = gamma * x D_g^T (one learned scalar
gamma compensates the energy lost by tying), and the block projection Pi_l keeps
the `l0` frames of largest energy. A QR each forward keeps the frames
orthonormal, so the decoder needs no normalisation.
"""
import torch
import torch.nn as nn

from .base import BSF, group_topk


class GrassmannianBSF(BSF):
    def __init__(self, d, n_groups, group_size=3, l0=16):
        super().__init__(d, n_groups, group_size)
        self.l0 = l0
        B = torch.randn(n_groups, d, group_size)
        # orthonormal columns
        B, _ = torch.linalg.qr(B)
        self.B_raw = nn.Parameter(B)
        # gamma = exp(log_gamma)
        self.log_gamma = nn.Parameter(torch.zeros(()))

    # (n_groups * group_size, d), orthonormal within each concept
    def decoder_atoms(self):
        # (n_groups, d, group_size)
        B, _ = torch.linalg.qr(self.B_raw)
        return B.permute(0, 2, 1).reshape(-1, self.d)

    def encode(self, x):
        # (n_groups * group_size, d)
        atoms = self.decoder_atoms()
        # tied encoder = gamma * D
        z = torch.exp(self.log_gamma) * (x @ atoms.t())
        z = z.reshape(-1, self.n_groups, self.group_size)
        mask = group_topk(z.norm(dim=-1), self.l0)
        return z * mask.unsqueeze(-1)

    def loss(self, x, target=None):
        # target != x -> denoising
        target = x if target is None else target
        x_hat, _ = self(x)
        recon = (target - x_hat).pow(2).mean()
        return recon, {'recon': recon.detach()}
