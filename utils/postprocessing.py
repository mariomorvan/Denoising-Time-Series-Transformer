import numpy as np
import matplotlib.pylab as plt
from statsmodels.graphics.tsaplots import plot_acf
plt.rcParams.update({'font.size': 13})


def plot_pred_diagnostic(x, y, y_pred, mask=None, info=None):
    res = y - y_pred

    f, ax = plt.subplots(2, 2, figsize=(15, 8), gridspec_kw={
                         'width_ratios': [2, 1]})

    # PREDICTION
    ax[0, 0].scatter(range(len(x)), y, label='input',
                     color='black', s=3, alpha=0.4)
    if not np.isclose(x, y, equal_nan=True).all():
        ax[0, 0].scatter(range(len(x)), y, label='target',
                         color='green', s=3, alpha=0.4)

    if mask is not None:
        ymin, ymax = ax[0, 0].get_ylim()
        ax[0, 0].fill_between(range(len(x)), [ymin]*len(x), [ymax]
                              * len(x), where=mask, alpha=0.4, label='input mask')

    #ax[0,0].plot(y, label='target', color='green', alpha=0.7)
    ax[0, 0].plot(y_pred, label='pred', color='red', alpha=0.7)
    # ax[0,0].scatter(range(len(res)), pred_m, marker="s", color='red')
    ax[0, 0].legend()

    # RESIDUALS
    ax[1, 0].plot([0, len(res)], [0, 0],
                  linestyle='dashed', c='black', alpha=0.5)
    ax[1, 0].scatter(range(len(res)), res, color='red',
                     alpha=0.7, s=5, label='Residuals')
    if mask is not None:
        ymin, ymax = ax[1, 0].get_ylim()
        ax[1, 0].fill_between(range(len(x)), [ymin]*len(x), [ymax]
                              * len(x), where=mask, alpha=0.4, label='input mask')
    ax[1, 0].legend()

    # ACF
    if (~np.isnan(res)).sum() > 1:
        plot_acf(res, lags=100, ax=ax[1,1], missing='drop')
        ax[1,1].set_ylim(-0.5, 0.5)

    # PACF
    # plot_pacf(res, lags=100, ax=ax[0, 1], method='ywm')
    # ax[0, 1].set_ylim(-0.5, 0.5)

    # title
    title = 'Prediction Diagnostic ' + (str(info) if info is not None else '')
    f.suptitle(title, fontsize=15)
    return ax
