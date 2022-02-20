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


def plot_pred_diagnostic(x, y, y_pred, mask=None, mu=None, sigma=None, targetid=None):
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

    fig, ax = plt.subplots(3, 2, figsize=(15, 8), gridspec_kw={
                         'width_ratios': [2, 1]}, sharex='col')
    fig.delaxes(ax[0,1])
    fig.delaxes(ax[1,1])

    # PREDICTION
    ax[0,0].set_title('Network Prediction')
    ax[0,0].set_ylabel('stand. flux')
    ax[0, 0].scatter(range(len(x)), y, label='input',
                     color='black', s=3, alpha=0.4)
    if not np.isclose(x, y, equal_nan=True).all():
        ax[0, 0].scatter(range(len(x)), y, label='target',
                         color='green', s=3, alpha=0.4)

    ymin, ymax = ax[0, 0].get_ylim()
    if missing.any():
        ax[0, 0].fill_between(range(len(x)), ymin, ymax, where=missing, alpha=0.4, label='missing')
    if mask is not None and mask.any():
        ax[0, 0].fill_between(range(len(x)), ymin, ymax, where=mask, alpha=0.4, label='random mask')

    #ax[0,0].plot(y, label='target', color='green', alpha=0.7)
    ax[0, 0].plot(y_pred, label='pred', color='red', alpha=0.7)
    # ax[0,0].scatter(range(len(res)), pred_m, marker="s", color='red')
    ax[0, 0].legend(bbox_to_anchor=(1.3, 1))

    # RESIDUALS
    ax[1,1].set_title('Residual Errors')
    ax[1, 0].set_ylabel('stand. flux units')
    ax[1, 0].plot([0, len(res)], [0, 0],
                  linestyle='dashed', c='black', alpha=0.5)
    ax[1, 0].scatter(range(len(res)), res, color='red',
                     alpha=0.7, s=5, label='residuals')

    ymin, ymax = ax[1, 0].get_ylim()
    if missing.any():
        ax[1, 0].fill_between(range(len(x)), ymin, ymax, where=missing, alpha=0.4, label='missing')
    if mask is not None and mask.any():
        ax[1, 0].fill_between(range(len(x)), ymin, ymax, where=mask, alpha=0.4, label='random mask')
    ax[1, 0].legend(bbox_to_anchor=(1.3, 0.8))

    # Star-normalised
    if mu is not None and sigma is not None:
        ax[2,0].set_title('Detrended LC')
        y_o = inverse_standardise_batch(y, mu, sigma)
        pred_o = inverse_standardise_batch(y_pred, mu, sigma)
        y_d = detrend(y_o, pred_o)

        ax[2, 0].set_ylabel('star-norm. flux')
        ax[2, 0].plot([0, len(res)], [1, 1],
                      linestyle='dashed', c='black', alpha=0.5)
        ax[2, 0].scatter(range(len(res)), y_d, color='red',
                         alpha=0.7, s=5, label='Detrended')

        ymin, ymax = ax[2, 0].get_ylim()
        if missing.any():
            ax[2, 0].fill_between(range(len(x)), ymin, ymax, where=missing, alpha=0.4, label='missing')
        if mask is not None and mask.any():
            ax[2, 0].fill_between(range(len(x)), ymin, ymax, where=mask, alpha=0.4, label='random mask')
        #ax[2,0].legend()
    # ACF
    if (~np.isnan(res)).sum() > 1:
        first_present  = (np.isnan(y)).argmin()
        last_present = len(y) - np.argmin(np.isnan(y[::-1]))
        res_mod = res[first_present:last_present]
        plot_acf(res_mod, lags=len(res_mod)//2, ax=ax[2, 1], missing='drop')
        ax[2, 1].set_ylim(-0.5, 0.5)

    # PACF
    # plot_pacf(res, lags=100, ax=ax[0, 1], method='ywm')
    # ax[0, 1].set_ylim(-0.5, 0.5)

    # title
    title = 'Prediction Diagnostic ' + \
        (str(targetid) if targetid is not None else '')
    fig.suptitle(title, fontsize=15)
    return ax
