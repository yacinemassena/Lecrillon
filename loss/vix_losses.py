# vix_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class VIXAwareLoss(nn.Module):
    """VIX-specific loss that heavily penalizes missing volatility spikes"""
    
    def __init__(self, spike_threshold=25.0, spike_penalty=3.0):
        super().__init__()
        self.spike_threshold = spike_threshold
        self.spike_penalty = spike_penalty
        
    def forward(self, pred, true):
        base_loss = F.mse_loss(pred, true)
        spike_mask = true > self.spike_threshold
        
        if spike_mask.any():
            spike_loss = F.mse_loss(pred[spike_mask], true[spike_mask])
            total_loss = base_loss + self.spike_penalty * spike_loss
        else:
            total_loss = base_loss
            
        return total_loss


class AsymmetricVIXLoss(nn.Module):
    """Asymmetric loss - penalize under-prediction more than over-prediction"""
    
    def __init__(self, under_penalty=2.0):
        super().__init__()
        self.under_penalty = under_penalty
        
    def forward(self, pred, true):
        diff = pred - true
        loss = torch.where(diff < 0,
                          self.under_penalty * diff**2,
                          diff**2)
        return loss.mean()


class RegimeAwareLoss(nn.Module):
    """Market regime-aware loss with different penalties per regime"""
    
    def __init__(self, calm_weight=1.0, stress_weight=2.0, crisis_weight=5.0):
        super().__init__()
        self.calm_weight = calm_weight
        self.stress_weight = stress_weight
        self.crisis_weight = crisis_weight
        
    def forward(self, pred, true, regime):
        base_loss = F.mse_loss(pred, true, reduction='none')
        weights = torch.where(regime == 0, self.calm_weight,
                  torch.where(regime == 1, self.stress_weight, self.crisis_weight))
        weighted_loss = base_loss * weights
        return weighted_loss.mean()


class MultiHorizonVIXLoss(nn.Module):
    """Multi-horizon VIX prediction loss"""
    
    def __init__(self, weights=(1.0, 0.5, 0.3, 0.2)):
        super().__init__()
        self.weights = weights
        
    def forward(self, pred, true):
        """pred/true: [B, num_horizons]"""
        total_loss = 0
        for i, w in enumerate(self.weights):
            total_loss += w * F.mse_loss(pred[:, i], true[:, i])
        return total_loss


class CombinedVIXLoss(nn.Module):
    """
    Multi-horizon + Asymmetric + Spike-aware loss.
    
    Combines three key insights for VIX prediction:
    1. Short-term predictions matter more (horizon weighting)
    2. Under-predicting volatility is worse than over-predicting (asymmetric)
    3. Missing spikes is catastrophic (spike penalty)
    """
    
    def __init__(
        self,
        horizon_weights=(1.0, 0.5, 0.3, 0.2),
        under_penalty=2.0,
        spike_threshold=25.0,
        spike_penalty=2.0
    ):
        super().__init__()
        self.horizon_weights = horizon_weights
        self.under_penalty = under_penalty
        self.spike_threshold = spike_threshold
        self.spike_penalty = spike_penalty
        
    def forward(self, pred, true):
        """
        Args:
            pred: [B, num_horizons] predictions
            true: [B, num_horizons] targets
        """
        total_loss = 0
        
        for i, w in enumerate(self.horizon_weights):
            if i >= pred.shape[1]:
                break
                
            p, t = pred[:, i], true[:, i]
            
            # Asymmetric base loss
            diff = p - t
            base = torch.where(
                diff < 0,
                self.under_penalty * diff**2,
                diff**2
            ).mean()
            
            # Spike penalty
            spike_mask = t > self.spike_threshold
            if spike_mask.any():
                spike_loss = F.mse_loss(p[spike_mask], t[spike_mask])
                base = base + self.spike_penalty * spike_loss
            
            total_loss += w * base
            
        return total_loss


class MaskedVIXLoss(nn.Module):
    """
    Loss that handles missing targets (NaN values).
    Useful when some horizons may not have ground truth.
    """
    
    def __init__(self, base_loss=None):
        super().__init__()
        self.base_loss = base_loss or nn.L1Loss(reduction='none')
        
    def forward(self, pred, true):
        """
        Args:
            pred: [B, num_horizons]
            true: [B, num_horizons] - may contain NaN
        """
        # Create mask for valid (non-NaN) targets
        valid_mask = ~torch.isnan(true)
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Compute loss only on valid targets
        loss = self.base_loss(pred, torch.nan_to_num(true, nan=0.0))
        masked_loss = loss * valid_mask.float()
        
        return masked_loss.sum() / valid_mask.float().sum()


def get_loss(name='combined', **kwargs):
    """
    Factory function to get loss by name.
    
    Args:
        name: One of 'l1', 'mse', 'asymmetric', 'spike', 'multi', 'combined'
        **kwargs: Arguments passed to the loss class
        
    Returns:
        Loss module
    """
    losses = {
        'l1': nn.L1Loss,
        'mse': nn.MSELoss,
        'asymmetric': AsymmetricVIXLoss,
        'spike': VIXAwareLoss,
        'multi': MultiHorizonVIXLoss,
        'combined': CombinedVIXLoss,
        'regime': RegimeAwareLoss,
    }
    
    if name not in losses:
        raise ValueError(f"Unknown loss: {name}. Available: {list(losses.keys())}")
    
    return losses[name](**kwargs)


def build_loss(config) -> nn.Module:
    """
    Build loss function from config dataclass.
    
    Uses config.train.loss_name and associated hyperparams:
        - under_penalty: For asymmetric loss
        - spike_threshold: VIX level considered a spike
        - spike_penalty: Extra penalty for missing spikes
    
    Args:
        config: Config dataclass with train.loss_name, etc.
        
    Returns:
        Loss module
    """
    name = config.train.loss_name.lower()
    
    if name == 'asymmetric':
        return AsymmetricVIXLoss(under_penalty=config.train.under_penalty)
    elif name == 'spike' or name == 'vix_aware':
        return VIXAwareLoss(
            spike_threshold=config.train.spike_threshold,
            spike_penalty=config.train.spike_penalty
        )
    elif name == 'combined':
        return CombinedVIXLoss(
            under_penalty=config.train.under_penalty,
            spike_threshold=config.train.spike_threshold,
            spike_penalty=config.train.spike_penalty
        )
    elif name == 'l1':
        return nn.L1Loss()
    elif name == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss: {name}. Available: l1, mse, asymmetric, spike, combined")