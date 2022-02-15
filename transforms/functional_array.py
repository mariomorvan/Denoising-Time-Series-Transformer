import math
from typing import Optional

import numpy as np


def fill_nans(x: np.ndarray, value: float, return_missing: bool = False):
    out = x.copy()
    missing = np.isnan(out)
    out[missing] = value
    if return_missing:
        return out, missing
    else:
        return out


def add_gaussian_noise(x: np.ndarray, sigma: float, mask: Optional[np.ndarray] = None):
    out = x.copy()
    if mask is None:
        out = np.random.normal(x, sigma).astype(out.dtype)
    else:
        out[~mask] = np.random.normal(
            out[~mask], sigma).astype(dtype=out.dtype)
    return out


def create_mask(seq_len, mask_ratio, n_dim=1, block_len=None, block_mode='geom', interval_mode='geom', overlap_mode='random'):
    """Create an boolean array mask where masked values are True and non masked values are False."""
    if n_dim == 1:
        return create_mask_1d(seq_len, mask_ratio, block_len=block_len, block_mode=block_mode, interval_mode=interval_mode)[:, np.newaxis]
    if overlap_mode == 'random':  # also called MCAR i.e. Missing Completely At Random
        mask = np.stack([create_mask_1d(seq_len, mask_ratio, block_len=block_len, block_mode=block_mode,
                        interval_mode=interval_mode) for _ in range(n_dim)], axis=1)
    elif overlap_mode == 'blackout':
        mask = np.tile(create_mask_1d(seq_len, mask_ratio, block_len=block_len,
                       block_mode=block_mode, interval_mode=interval_mode)[:, np.newaxis], (1, n_dim))
    else:
        raise NotImplementedError
    return mask


def create_mask_like(x, mask_ratio, block_len=None, block_mode='geom', interval_mode='geom', overlap_mode='random'):
    """A wrapper to create masks with shape from input"""
    seq_len, n_dim = x.shape
    return create_mask(seq_len, mask_ratio, block_len=block_len, n_dim=n_dim, block_mode=block_mode, interval_mode=interval_mode, overlap_mode=overlap_mode)


def create_batch_mask_like(x, mask_ratio, block_len=None, block_mode='geom', interval_mode='geom', overlap_mode='random'):
    """A wrapper to create masks with shape from input"""
    batch_size, seq_len, n_dim = x.shape
    masks = [create_mask(seq_len, mask_ratio, block_len=block_len, n_dim=n_dim,
                         block_mode=block_mode, interval_mode=interval_mode, overlap_mode=overlap_mode)
             for _ in range(batch_size)]
    return np.stack(masks, axis=0)


def create_mask_1d(seq_len, mask_ratio, block_len=None, block_mode='geom', interval_mode='geom'):
    """Create a 1D boolean array mask where masked values are True and non masked values are False."""
    # geom masking inspired from https://github.com/gzerveas/mvts_transformer, just more efficient and general here
    assert int(seq_len) == seq_len
    assert 0 <= mask_ratio <= 1
    seq_len = int(seq_len)

    # No mask
    if mask_ratio == 0:
        return np.zeros(seq_len, dtype=bool)

    # full mask
    if mask_ratio == 1:
        return np.ones(seq_len, dtype=bool)

    # No defined block, masking proba independent for all elements
    if block_len is None:
        return np.random.choice([True, False], size=seq_len, replace=True, p=(mask_ratio, 1 - mask_ratio))

    # Partition in equal patches
    if block_mode == 'equal':
        n_patches = math.ceil(seq_len / block_len)
        n_masked_patches = math.ceil(n_patches * mask_ratio)
        indices = np.random.choice(np.arange(n_patches), size=n_masked_patches,
                                   replace=True)
        mask = np.zeros((n_patches, 1), dtype=bool)
        mask[indices] = True
        return mask.repeat(block_len, 1).flatten()[:seq_len]

    max_sequences = int(math.ceil(seq_len / block_len / mask_ratio)) * 5
    # Block lengths
    if block_mode == 'geom':
        p_block = 1 / block_len  # p of stopping a block
        block_lengths = np.random.geometric(p_block, max_sequences)
    elif block_mode == 'fixed':
        block_lengths = np.array([block_len]*max_sequences)
    else:
        raise NotImplementedError

    # Interval lengths
    p_interval = mask_ratio / (1 - mask_ratio) / block_len
    if interval_mode == 'geom':
        intervals = np.random.geometric(p_interval, max_sequences)
    elif interval_mode == 'fixed':
        intervals = np.array([int(1/p_interval)]*max_sequences)

    # initialisation
    mask = np.zeros(seq_len, dtype=bool)
    i = 0
    if np.random.rand() < mask_ratio:  # start with missing block
        k = np.random.randint(1, block_lengths[-1]+1)
        mask[:k] = True
        k += intervals[-1]
    else:  # start with interval
        k = np.random.randint(intervals[-1])

    # iterating through lengths
    while k < seq_len:
        mask[k:min(k+block_lengths[i], seq_len-1)] = True
        k += block_lengths[i]
        k += intervals[i]
        i += 1
    return mask
