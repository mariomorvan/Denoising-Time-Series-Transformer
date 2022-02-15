from typing import Optional

import torch


def fill_nans(x: torch.Tensor, value: float, return_missing: bool = False):
    out = x.clone()
    missing = torch.isnan(out)
    out[missing] = value
    if return_missing:
        return out, missing
    else:
        return out


def add_gaussian_noise(x: torch.Tensor, sigma: float, mask: Optional[torch.Tensor] = None):
    out = x.clone()
    if mask is None:
        out += torch.randn_like(out) * sigma
    else:
        out[~mask] += torch.randn_like(out[~mask]) * sigma
    return out
