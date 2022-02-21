import numpy as np
import torch
import matplotlib.pylab as plt
from statsmodels.graphics.tsaplots import plot_acf

plt.rcParams.update({'font.size': 13})


def inverse_standardise_batch(x, mu, sigma):
    return x * sigma + mu


def detrend(x, trend):
    return x / trend


def cast_array(x):
    out = x
    if isinstance(x, torch.Tensor):
        out = out.cpu().detach().numpy()
    elif isinstance(x, list):
        out = np.array(x)
    if isinstance(out, np.ndarray):
        out = out.squeeze()
    return out


def compute_rollout_attention(attention_maps):
    """attention rollout as introduced in https://arxiv.org/pdf/2005.00928.pdf"""
    Ar = attention_maps[0]
    for l in range(1, len(attention_maps)):
        Ar *= attention_maps[l]
    return Ar


def plot_pred_diagnostic(x, y, y_pred, mask=None, ar=None, mu=None, sigma=None, targetid=None):
    x = cast_array(x)
    y = cast_array(y)
    y_pred = cast_array(y_pred)
    mask = cast_array(mask)
    mu = cast_array(mu)
    sigma = cast_array(sigma)
    missing = np.isnan(y)
    if mask is not None:
        mask = mask & ~missing

    res = y - y_pred

    plot_ar = ar is not None
    if plot_ar:
        fig, ax = plt.subplots(4, 2, figsize=(15, 8), gridspec_kw={
            'width_ratios': [2, 1], 'height_ratios': [2, 1, 2, 2]},
            sharex='col')
        fig.delaxes(ax[2, 1])
    else:
        fig, ax = plt.subplots(3, 2, figsize=(15, 8), gridspec_kw={
            'width_ratios': [2, 1]}, sharex='col')
    fig.delaxes(ax[0, 1])
    fig.delaxes(ax[1, 1])

    # INPUT
    ax[0, 0].set_title('Input')
    ax[-1, 0].set_xlabel('time steps')
    ax[0, 0].set_ylabel('stand. flux')
    ax[0, 0].scatter(range(len(x)), y, label='input',
                     color='black', s=3, alpha=0.4)
    if not np.isclose(x, y, equal_nan=True).all():
        ax[0, 0].scatter(range(len(x)), y, label='target',
                         color='green', s=3, alpha=0.4)

    ymin, ymax = ax[0, 0].get_ylim()
    if missing.any():
        ax[0, 0].fill_between(range(len(x)), ymin, ymax,
                              where=missing, alpha=0.4, label='missing')
    if mask is not None and mask.any():
        ax[0, 0].fill_between(range(len(x)), ymin, ymax,
                              where=mask, alpha=0.4, label='random mask')

    ax[0, 0].legend(bbox_to_anchor=(1.3, 1))

    # Attention
    if ar is not None:
        m, M = np.min(ar), np.max(ar)
        alpha = (ar-m)/(M-m)/1.002 + m + 1e-5
        s = (((ar-m)/(M-m)+m)) * 50 + 1
        ax[1, 0].set_title('Rollout attention')
        ax[1, 0].set_ylabel('norm. scores')
        ax[1, 0].plot(alpha)
    else:
        alpha = 0.4
        s = 3

    # PRED
    ax[1+plot_ar, 0].set_title('Prediction')
    ax[1+plot_ar, 0].set_ylabel('stand. flux')

    ax[1+plot_ar, 0].scatter(range(len(x)), y, label='input',
                             color='black', s=s, alpha=alpha)

    if not np.isclose(x, y, equal_nan=True).all():
        ax[1+plot_ar, 0].scatter(range(len(x)), y, label='target',
                                 color='green', s=3, alpha=0.4)
    ax[1+plot_ar, 0].plot(y_pred, label='pred', color='red', lw=2, alpha=0.7)

    # RESIDUALS

    # Star-normalised
    if mu is not None and sigma is not None:
        ax[2+plot_ar, 0].set_title('Detrended LC')
        y_o = inverse_standardise_batch(y, mu, sigma)
        pred_o = inverse_standardise_batch(y_pred, mu, sigma)
        y_d = detrend(y_o, pred_o)

        ax[2+plot_ar, 0].set_ylabel('star-norm. flux')
        ax[2+plot_ar, 0].plot([0, len(res)], [1, 1],
                              linestyle='dashed', c='black', alpha=0.5)
        ax[2+plot_ar, 0].scatter(range(len(res)), y_d, color='red',
                                 alpha=0.7, s=5, label='Detrended')
    else:
        ax[2+plot_ar, 0].set_title('Residual Errors')
        ax[2+plot_ar, 0].set_ylabel('stand. flux units')
        ax[2+plot_ar, 0].plot([0, len(res)], [0, 0],
                              linestyle='dashed', c='black', alpha=0.5)
        ax[2+plot_ar, 0].scatter(range(len(res)), res, color='red',
                                 alpha=0.7, s=5, label='residuals')

    # ACF
    if (~np.isnan(res)).sum() > 1:
        first_present = (np.isnan(y)).argmin()
        last_present = len(y) - np.argmin(np.isnan(y[::-1]))
        res_mod = res[first_present:last_present]
        plot_acf(res_mod, lags=len(res_mod)//2,
                 ax=ax[2+plot_ar, 1], missing='drop')
        ax[2+plot_ar, 1].set_ylim(-0.5, 0.5)

    # title
    title = 'Prediction Diagnostic ' + \
        (f'(targetid = {targetid})' if targetid is not None else '')
    fig.suptitle(title, fontsize=15)
    return ax
