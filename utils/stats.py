import numpy as np
import torch
from statsmodels.stats.stattools import durbin_watson


def nanstd(input, dim=None, keepdim=False):
    mu = torch.nanmean(input, dim=dim, keepdim=True)
    return torch.sqrt(torch.nanmean((input - mu)**2, dim=dim, keepdim=keepdim))


def iqr(batch, dim=None, reduction='mean'):
    if dim is None:
        if len(batch.shape) == 1:
            dim = 0
        else:
            dim = 1
    if isinstance(batch, np.ndarray):
        out = np.quantile(batch, 0.75, axis=dim) - \
            np.quantile(batch, 0.25, axis=dim)
    elif isinstance(batch, torch.Tensor):
        out = torch.quantile(batch, 0.75, dim=dim) - \
            torch.quantile(batch, 0.25, dim=dim)
    if reduction == 'none':
        return out
    elif reduction == 'mean':
        return out.mean()
    else:
        raise NotImplementedError


def naniqr(batch, dim=None, reduction='none'):
    if dim is None:
        if len(batch.shape) == 1:
            dim = 0
        else:
            dim = 1
    if isinstance(batch, np.ndarray):
        out = np.nanquantile(batch, 0.75, axis=dim) - \
            np.nanquantile(batch, 0.25, axis=dim)
    elif isinstance(batch, torch.Tensor):
        out = torch.nanquantile(batch, 0.75, dim=dim) - \
            torch.nanquantile(batch, 0.25, dim=dim)
    if reduction == 'none':
        return out
    elif reduction == 'mean':
        return out.mean()
    elif reduction == 'nanmean':
        return torch.nanmean(out)
    else:
        raise NotImplementedError


def compute_dw(res, dim=1, replace_missing=0., reduction='none'):
    """Durbin-Watson statistics
    https://www.statsmodels.org/devel/generated/statsmodels.stats.stattools.durbin_watson.html
    """
    if isinstance(res, torch.Tensor):
        res = res.detach().cpu().numpy()
    if replace_missing is not None:
        res = res.copy()
        res[np.isnan(res)] = replace_missing
    out = durbin_watson(res, axis=dim)
    if reduction == 'mean':
        return out.mean()
    elif reduction == 'none':
        return out
    elif reduction == 'median':
        return np.median(out)


def estimate_noise(x, dim=1, window_size=10, step=5, reduce='nanmean', keepdim=True):
    noises = nanstd(x.unfold(dim, window_size, step), -1, keepdim=False)
    if reduce=='nanmedian':
        return noises.nanmedian(dim, keepdim=keepdim).values
    if reduce=='nanmean':
        return noises.nanmean(dim, keepdim=keepdim)
    if reduce=='median':
        return noises.median(dim, keepdim=keepdim).values
    if reduce=='mean':
        return noises.mean(dim, keepdim=keepdim)
    if reduce=='none':
        return noises
    raise ValueError
