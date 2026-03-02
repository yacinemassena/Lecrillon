# Loss module - clean exports without GraphGym dependencies
from .vix_losses import (
    get_loss,
    build_loss,
    VIXAwareLoss,
    AsymmetricVIXLoss,
    RegimeAwareLoss,
    MultiHorizonVIXLoss,
    CombinedVIXLoss,
    MaskedVIXLoss,
)

__all__ = [
    'get_loss',
    'build_loss',
    'VIXAwareLoss',
    'AsymmetricVIXLoss',
    'RegimeAwareLoss',
    'MultiHorizonVIXLoss',
    'CombinedVIXLoss',
    'MaskedVIXLoss',
]
