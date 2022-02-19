import numpy as np
import torch


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


# def acf(x, fill_missing=None, keepdims=False):
#     if fill_missing is not None:
#         x[np.isnan(x)] = fill_missing
#     y = np.squeeze(x)
#     if len(y.shape) == 1:
#         out = np.correlate(y, y, mode = 'same')
#     elif len(y.shape) == 2:
#         out = np.array([acf(xi) for xi in y])
#     else:
#         raise ValueError
#     if keepdims:
#         return out.reshape(x.shape)
#     return out

# acf(res[0,:,0], fill_missing=0).shape

# out = acf(res, fill_missing=0, keepdims=True)#, fill_missing=0)

# plt.plot(out[:,:,0].T)
