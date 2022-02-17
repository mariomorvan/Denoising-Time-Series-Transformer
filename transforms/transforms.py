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

    def __call__(self, x, mask=None):
        out = x
        for t in self.transforms:
            result = t(out, mask=mask)
            if isinstance(result, tuple):
                out, mask = result
            else:
                out = result
        if mask is not None:
            return out, mask
        else:
            return out

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

    def __call__(self, x, mask=None):
        if isinstance(x, np.ndarray):
            out = F_np.fill_nans(x, self.value)
        else:
            out = F_t.fill_nans(x, self.value)
        if mask is not None:
            return out, mask
        else:
            return out


class Mask(object):
    def __init__(self,
                 mask_ratio,
                 block_len=None,
                 block_mode='geom',
                 interval_mode='geom',
                 overlap_mode='random',
                 value=np.nan):
        # None default argument for value prevents from modiying the input's values at mask location.
        self.mask_ratio = mask_ratio
        self.block_len = block_len
        self.overlap_mode = overlap_mode
        self.block_mode = block_mode
        self.interval_mode = interval_mode
        self.value = value

    def __call__(self, x, mask=None):
        if isinstance(x, np.ndarray):
            out = x
        else:
            raise NotImplementedError
        temp_mask = F_np.create_mask_like(out, self.mask_ratio, block_len=self.block_len,
                                          block_mode=self.block_mode, interval_mode=self.interval_mode,
                                          overlap_mode=self.overlap_mode)
        if self.value is not None:
            out[temp_mask] = self.value
        out_mask = temp_mask if mask is None else mask | temp_mask
        return out, out_mask

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

    def __call__(self, x, mask=None):
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
        if mask is not None:
            return out, mask
        else:
            return out


# TODO: behaviour relative to input mask
class Scaler(object):
    def __init__(self, centers=None, norms=None, dim=1):
        super().__init__()
        self.dim = dim
        self.centers = centers
        self.norms = norms

    def transform(self, x, mask=None):
        if mask is None:
            return (x - self.centers) / self.norms
        else:
            return (x - self.centers) / self.norms, mask

    def fit(self, x, mask=None):
        raise NotImplementedError

    def fit_transform(self, x, mask=None):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, y):
        return (y * self.norms) + self.centers

    def __call__(self, x, mask=None):
        out = self.fit_transform(x)
        if mask is not None:
            return out, mask
        else:
            return out


class StandardScaler(Scaler):
    def fit(self, x):
        if isinstance(x, np.ndarray):
            self.centers = np.nanmean(x, self.dim, keepdims=True)
            self.norms = np.nanstd(x, self.dim, keepdims=True)
        elif isinstance(x, torch.Tensor):
            self.centers = torch.nanmean(x, self.dim, keepdim=True)
            self.norms = nanstd(x, self.dim, keepdim=True)
        else:
            raise NotImplementedError

        if (self.norms == 0).any():
            warnings.warn('zero norms')
            self.norms[self.norms == 0] = 1


class DownSample:
    def __init__(self, factor=1):
        self.factor = factor

    def __call__(self, x, mask=None):
        if mask is not None:
            return x[::self.factor], mask[::self.factor]
        return x[::self.factor]

# Might need to be batch wise


class RandomCrop:
    def __init__(self, width):
        self.width = width
        self.left_crop = None

    def __call__(self, x, mask=None):
        seq_len = x.shape[0]
        if seq_len < self.width:
            self.left_crop = 0
            warnings.warn(
                'cannot crop because width smaller than sequence length')
        else:
            self.left_crop = np.random.randint(seq_len-self.width)
        if mask is not None:
            return (x[self.left_crop:self.left_crop+self.width],
                    mask[self.left_crop:self.left_crop+self.width])
        return x[self.left_crop:self.left_crop+self.width]
