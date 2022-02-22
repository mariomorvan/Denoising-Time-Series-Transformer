import warnings

import torch
import torch.nn as nn

from utils.stats import iqr, naniqr


class _MaskedLoss(nn.Module):
    """Base class for masked losses"""

    def __init__(self, reduction='mean', ignore_nans=True):
        super().__init__()
        self.reduction = reduction
        self.ignore_nans = ignore_nans

    def forward(self, input, target, mask=None):
        """Compute a loss between input and target for given mask.
        Note that this implementation is faster than loss(input[mask], target[mask])
        for a given loss, and is nan-proof."""
        if not (target.size() == input.size()):
            warnings.warn(
                "Using a target size ({}) that is different to the input size ({}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.".format(
                    target.size(), input.size()),
                stacklevel=2,
            )
        if mask is None:
            mask = torch.ones_like(input, dtype=bool)

        target_proxy = target
        if self.ignore_nans:
            target_proxy = target.clone()
            nans = torch.isnan(target)
            if nans.any():
                with torch.no_grad():
                    mask = mask & ~nans
                    target_proxy[nans] = 0
        full_loss = self.criterion(input, target_proxy)

        if not mask.any():
            warnings.warn(
                "Evaluation mask is False everywhere, this might lead to incorrect results.")
        full_loss[~mask] = 0

        if self.reduction == 'none':
            return full_loss
        if self.reduction == 'sum':
            return full_loss.sum()
        if self.reduction == 'mean':
            return full_loss.sum() / mask.to(full_loss.dtype).sum()


class MaskedMSELoss(_MaskedLoss):
    """Masked MSE loss"""

    def __init__(self, reduction='mean', ignore_nans=True):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans)
        self.criterion = nn.MSELoss(reduction='none')


class MaskedL1Loss(_MaskedLoss):
    """Masked L1 loss."""

    def __init__(self, reduction='mean', ignore_nans=True):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans)
        self.criterion = nn.L1Loss(reduction='none')


class MaskedHuberLoss(_MaskedLoss):
    """Masked L1 loss."""

    def __init__(self, reduction='mean', ignore_nans=True, delta=1):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans)
        self.criterion = nn.HuberLoss(reduction='none', delta=delta)


class IQRLoss(nn.Module):
    "IQR of the residuals"

    def __init__(self, reduction='nanmean', ignore_nans=True):
        super().__init__()
        self.reduction = reduction
        self.ignore_nans = ignore_nans

    def forward(self, input, target=0.):
        if isinstance(target, torch.Tensor) and not (target.size() == input.size()):
            warnings.warn(
                "Using a target size ({}) that is different to the input size ({}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.".format(
                    target.size(), input.size()),
                stacklevel=2,
            )
        if self.ignore_nans:
            return naniqr(target-input, reduction=self.reduction)
        else:
            return iqr(target-input, reduction=self.reduction)
