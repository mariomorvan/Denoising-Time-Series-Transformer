import warnings

import numpy as np
import torch
from torch import nn

from transforms.transforms import StandardScaler


class CollatePred(object):
    # May need to pad outside to ensure the totatlity is contained in output :-)
    def __init__(self, window, step=1, standardise=None):
        self.window = window
        self.step = step
        self.scaler = StandardScaler(dim=1)

    def __call__(self, batch):
        if len(batch) > 1:
            warnings.warn('Pred Collate fn was designed to process single time series.')
        x_out_list = []
        y_out_list = []
        m_out_list = []
        info_out_list = []

        seq_len = batch[0][0].shape[0]
        d = (seq_len - self.window) / self.step + 1
        if int(d) == d:
            padding = 0
        else:
            padding = self.step
            pad = nn.ConstantPad1d((0, padding), value=np.nan)

        for i in range(len(batch)):
            x, y, m, info = batch[i]

            x_out = torch.tensor(x)
            y_out = torch.tensor(y)
            m_out = torch.tensor(m)

            if padding:
                x_out = pad(x_out.T).T
                y_out = pad(y_out.T).T
                m_out = pad(m_out.T).T

            x_out = x_out.unfold(0, size=self.window,
                                 step=self.step).transpose(1, 2)
            y_out = y_out.unfold(0, size=self.window,
                                 step=self.step).transpose(1, 2)
            m_out = m_out.unfold(0, size=self.window,
                                 step=self.step).transpose(1, 2)

            info_out = {k: torch.tensor([v]*len(x_out))
                        for k, v in info.items()}

            x_out = self.scaler.fit_transform(x_out)
            y_out = self.scaler.transform(y_out)
            info_out['left_crop'] = torch.arange(0, seq_len, self.step)
            info_out['mu'] = self.scaler.centers
            info_out['sigma'] = self.scaler.norms

            x_out_list += [x_out]
            y_out_list += [y_out]
            m_out_list += [m_out]
            info_out_list += [info_out]
        return (torch.cat(x_out_list),
                torch.cat(y_out_list),
                torch.cat(m_out_list),
                {k: torch.cat([info_out_list[i][k] for i in range(len(batch))]) for k in info_out})
