import numpy as np
import torch
import wotan


def predict_batch_wotan(y, cadence=1/48, method='biweight', **kwargs):
    # y in flux units
    # cadence for 30min cadence by default
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().squeeze()
    batch_size, len_seq = y.shape
    time = np.arange(len_seq) * cadence
    list_flat = []
    list_trend = []
    for i in range(batch_size):
        flattened_y, trend_y = wotan.flatten(time, y[i], method=method, return_trend=True, **kwargs)
        list_flat += [flattened_y]
        list_trend += [trend_y]
    return np.stack(list_flat), np.stack(list_trend)