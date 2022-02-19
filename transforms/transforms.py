import warnings

import numpy as np
import torch

from . import functional_array as F_np
from . import functional_tensor as F_t
from utils.stats import nanstd


class Compose:
    """Composes several transforms together. 
    Adapted from https://pytorch.org/vision/master/_modules/torchvision/transforms/transforms.html#Compose

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None, info=None):
        out = x
        for t in self.transforms:
            out, mask, info = t(out, mask=mask, info=info)
        return out, mask, info

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class FillNans(object):
    def __init__(self, value):
        self.value = value

    def __call__(self, x, mask=None, info=None):
        if isinstance(x, np.ndarray):
            out = F_np.fill_nans(x, self.value)
        else:
            out = F_t.fill_nans(x, self.value)
        return out, mask, info


class Mask(object):
    def __init__(self,
                 mask_ratio,
                 block_len=None,
                 block_mode='geom',
                 interval_mode='geom',
                 overlap_mode='random',
                 value=np.nan,
                 exclude_mask=True
                 ):
        # None default argument for value prevents from modiying the input's values at mask location.
        self.mask_ratio = mask_ratio
        self.block_len = block_len
        self.overlap_mode = overlap_mode
        self.block_mode = block_mode
        self.interval_mode = interval_mode
        self.value = value
        self.exclude_mask = exclude_mask

    def __call__(self, x, mask=None, info=None):
        if isinstance(x, np.ndarray):
            out = x
        else:
            raise NotImplementedError
        temp_out = out
        if self.exclude_mask and mask is not None:
            # only implemented for univariate at the moment.
            assert x.shape[-1] == 1
            temp_out = out[~mask][:, np.newaxis]

        temp_mask = F_np.create_mask_like(temp_out, self.mask_ratio, block_len=self.block_len,
                                          block_mode=self.block_mode, interval_mode=self.interval_mode,
                                          overlap_mode=self.overlap_mode)
        if self.value is not None:
            temp_out[temp_mask] = self.value

        if mask is None:
            mask = temp_mask
            out = temp_out
        elif self.exclude_mask:
            out[~mask] = temp_out.squeeze()
            mask[~mask] = temp_mask.squeeze()
        else:
            mask = mask | temp_mask
            out = temp_out

        return out, mask, info

    def __repr__(self):
        return (f"Mask(ratio={self.mask_ratio}" + f" ; overlap={self.overlap_mode}" +
                ((f" ; block_length={self.block_len } ; block_mode={self.block_mode} ;" +
                  f" interval_mode={self.interval_mode})") if self.block_len else ")"))


class AddGaussianNoise(object):
    def __init__(self, sigma=1.0, exclude_mask=False, mask_only=False):
        self.sigma = sigma
        self.exclude_mask = exclude_mask
        self.mask_only = mask_only
        assert not (exclude_mask and mask_only)

    def __call__(self, x, mask=None, info=None):
        exclude_mask = None
        if mask is not None:
            if self.exclude_mask:
                exclude_mask = mask
            elif self.mask_only:
                exclude_mask = ~mask
        if isinstance(x, np.ndarray):
            out = F_np.add_gaussian_noise(
                x, self.sigma, mask=exclude_mask)
        else:
            out = F_t.add_gaussian_noise(
                x, self.sigma, mask=exclude_mask)
        return out, mask, info


# TODO: behaviour relative to input mask
class Scaler(object):
    def __init__(self, dim, centers=None, norms=None,  eps=1e-10):
        super().__init__()
        self.dim = dim
        self.centers = centers
        self.norms = norms
        self.eps = eps

    def transform(self, x, mask=None):
        if mask is None:
            return (x - self.centers) / self.norms
        else:
            return (x - self.centers) / self.norms, mask

    def fit(self, x, mask=None):
        raise NotImplementedError

    def fit_transform(self, x, mask=None):
        self.fit(x, mask=mask)
        return self.transform(x)

    def inverse_transform(self, y):
        return (y * self.norms) + self.centers

    def __call__(self, x, mask=None, info=None):
        out = self.fit_transform(x, mask=mask)
        info['mu'] = self.centers
        info['sigma'] = self.norms
        return out, mask, info


class StandardScaler(Scaler):
    def fit(self, x, mask=None):
        xm = x
        if isinstance(x, np.ndarray):
            if mask is not None:
                xm = x.copy()
                xm[mask] = np.nan
            self.centers = np.nanmean(xm, self.dim, keepdims=True)
            self.norms = np.nanstd(xm, self.dim, keepdims=True) + self.eps
        elif isinstance(x, torch.Tensor):
            if mask is not None:
                xm = x.clone()
                xm[mask] = np.nan
            self.centers = torch.nanmean(xm, self.dim, keepdim=True)
            self.norms = nanstd(xm, self.dim, keepdim=True) + self.eps
        else:
            raise NotImplementedError


class DownSample:
    def __init__(self, factor=1):
        self.factor = factor

    def __call__(self, x, mask=None, info=None):
        return x[::self.factor], mask[::self.factor], info


class RandomCrop:
    def __init__(self, width, exclude_missing_threshold=None):
        self.width = width
        self.exclude_missing_threshold = exclude_missing_threshold
        assert exclude_missing_threshold is None or 0 <= exclude_missing_threshold <= 1

    def __call__(self, x, mask=None, info=None):
        seq_len = x.shape[0]
        if seq_len < self.width:
            self.left_crop = 0
            warnings.warn(
                'cannot crop because width smaller than sequence length')
        else:
            left_crop = np.random.randint(seq_len-self.width)
        info['left_crop'] = left_crop

        out_x = x[left_crop:left_crop+self.width]

        if self.exclude_missing_threshold is not None and np.isnan(out_x).mean() >= self.exclude_missing_threshold:
            return self.__call__(x, mask=mask, info=info)
        out_m = mask[left_crop:left_crop+self.width]

        return (out_x, out_m, info)

