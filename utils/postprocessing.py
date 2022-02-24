import warnings
from datetime import datetime
from tqdm import tqdm 
import numpy as np
import torch
import matplotlib.pylab as plt
from statsmodels.graphics.tsaplots import plot_acf

from utils.stats import naniqr, compute_dw

plt.rcParams.update({'font.size': 13})


def inverse_standardise_batch(x, mu, sigma):
    return x * sigma + mu


def fold_back(x, skip=0, seq_len=None):
    # Assumes no skip at the start
    if skip == 0:
        out = x.flatten()
    else:
        out = [x[0,:skip].flatten(), x[:,skip:-skip].flatten()]
        if isinstance(x, torch.Tensor):
            out = torch.cat(out)
        elif isinstance(x, np.ndarray):
            out = np.concatenate(out)
    if seq_len is not None:
        out = out[:seq_len]
    return out


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
        fig, ax = plt.subplots(4,1, figsize=(8, 7), gridspec_kw={'height_ratios': [2, 1, 2, 1.5]},
            sharex='col')
    else:
        fig, ax = plt.subplots(3, 1, figsize=(8, 7), gridspec_kw={
            'height_ratios': [2, 2, 1]}, sharex='col')

    # INPUT
    #ax[0].set_title('Input')
    ax[-1].set_xlabel('time steps')
    ax[0].set_ylabel('stand. flux')
    ax[0].scatter(range(len(x)), y, label='$x$',
                     color='black', s=3, alpha=0.4)
    if not np.isclose(x, y, equal_nan=True).all():
        ax[0].scatter(range(len(x)), y, label='$y$',
                         color='green', s=3, alpha=0.4)
    ax[0].plot(y_pred, label='$\hat{y}$', color='red', lw=2, alpha=0.7)

    ymin, ymax = ax[0].get_ylim()
    if missing.any():
        ax[0].fill_between(range(len(x)), ymin, ymax,
                              where=missing, alpha=0.4, label='$m_{missing}$')
    if mask is not None and mask.any():
        ax[0].fill_between(range(len(x)), ymin, ymax,
                              where=mask, alpha=0.4, label='$m_{random}$')
    ax[0].legend()

    # Attention
    if ar is not None:
        m, M = np.min(ar), np.max(ar)
        alpha = (ar-m)/(M-m)/1.002 + m + 1e-5
        s = (((ar-m)/(M-m)+m)) * 50 + 1
        ax[1].set_ylabel('norm. scores')
        ax[1].plot(alpha, label='rollout attention')
        ax[1].legend()
    else:
        alpha = 0.4
        s = 3

    # RESIDUALS

    # Star-normalised
    if mu is not None and sigma is not None:
        y_o = inverse_standardise_batch(y, mu, sigma)
        pred_o = inverse_standardise_batch(y_pred, mu, sigma)
        y_d = detrend(y_o, pred_o)

        ax[1+plot_ar].set_ylabel('star-norm. flux')
        ax[1+plot_ar].plot([0, len(res)], [1, 1],
                              linestyle='dashed', c='black', alpha=0.5)
        ax[1+plot_ar].scatter(range(len(res)), y_d, color='red',
                                 alpha=0.7, s=5, label='residuals')
    else:
        ax[1+plot_ar].set_ylabel('stand. flux units')
        ax[1+plot_ar].plot([0, len(res)], [0, 0],
                              linestyle='dashed', c='black', alpha=0.5)
        ax[1+plot_ar].scatter(range(len(res)), res, color='red',
                                 alpha=0.7, s=5, label='residuals')
    ax[1+plot_ar].legend()

    # ACF
    if (~np.isnan(res)).sum() > 1:
        ax[2+plot_ar].set_ylabel('ACF')
        first_present = (np.isnan(y)).argmin()
        last_present = len(y) - np.argmin(np.isnan(y[::-1]))
        res_mod = res[first_present:last_present]
        try:
            plot_acf(res_mod, lags=(~np.isnan(res_mod)).sum()-1,
                     ax=ax[2+plot_ar], missing='drop', markersize=3)
            ax[2+plot_ar].set_ylim(-0.5, 0.5)
            ax[2+plot_ar].set_title(None)

        except ValueError as e:
            raise e
            warnings.warn('Issue in ACF plot, passing')

    # title
    title = f'TESS ID = {targetid}' if targetid is not None else ''
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.align_labels()
    return ax


def predict_full_inputs(lit_model, loader, test_dataset, skip, device=None):
    lit_model.eval()
    target_test = np.vstack([test_dataset.get_pretransformed_sample(idx).squeeze() 
                             for idx in range(len(test_dataset))])
    seq_len = target_test.shape[1]
    exec_time = []
    out = []
    lit_model.eval().to(device)
    
    for X, Y, M, I in tqdm(loader):
        # access original TS non transformed 
        idx = I['idx'][0]
        Y_intact = test_dataset.get_pretransformed_sample(idx).squeeze()
        seq_len = len(Y_intact)
        
        with torch.no_grad():
            t0 = datetime.now()
            Y_pred = lit_model(X.to(device), mask=M.to(device)).cpu()
            Y_pred_o = inverse_standardise_batch(Y_pred, I['mu'], I['sigma'])
            Y_pred_of = fold_back(Y_pred_o, skip=skip, seq_len=seq_len)
            out += [Y_pred_of]
            exec_time += [datetime.now()-t0]

            t0 = datetime.now()
    print('Mean batch exec time', np.mean(exec_time))
    final_preds = np.vstack(out)
    return final_preds

def eval_full_inputs(lit_model, loader, test_dataset, skip, device=None):
    target_test = np.vstack([test_dataset.get_pretransformed_sample(idx).squeeze() 
                             for idx in range(len(test_dataset))])
    final_preds = predict_full_inputs(lit_model, loader, test_dataset, skip, device=device)
    pred_d = target_test / final_preds
    iqr = naniqr(pred_d, dim=1, reduction='none')
    dw = compute_dw(pred_d-1, dim=1, reduction='none')
    return iqr, dw
    