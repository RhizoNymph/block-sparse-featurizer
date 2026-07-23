"""bsf — block-sparse featurizers for vision activations.

A *concept* is a block of `group_size` latent dims rather than a single
direction. Three featurizers instantiate this block-sparse principle, share one
decoder/forward interface (`BSF`), and train with one agnostic trainer:

    VanillaBSF      free encoder + per-sample block TopK
    GrassmannianBSF tied orthonormal frames + block TopK
    GroupLassoBSF   free encoder + thresholded block gate, theta learned by STE
"""
from .base import BSF
from .vanilla import VanillaBSF
from .grassmannian import GrassmannianBSF
from .group_lasso import GroupLassoBSF
from .train import train, fit, recon_r2
from .sources import ActivationSource, VisionSource, CapturesSource
from . import data, viz, sources, capture_format, normalize

__all__ = [
    'BSF', 'VanillaBSF', 'GrassmannianBSF', 'GroupLassoBSF',
    'train', 'fit', 'recon_r2',
    'ActivationSource', 'VisionSource', 'CapturesSource',
    'data', 'viz', 'sources', 'capture_format', 'normalize',
]
