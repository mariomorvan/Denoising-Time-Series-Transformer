import math
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as opt
import pytorch_lightning as pl

from .loss import MaskedMSELoss, MaskedL1Loss, MaskedHuberLoss, IQRLoss
from utils.stats import estimate_noise


class LitImputer(pl.LightningModule):
    def __init__(self,
                 n_dim=1,
                 d_model=64,
                 nhead=8,
                 dim_feedforward=128,
                 eye=0,
                 dropout=0.1,
                 num_layers=3,
                 lr=0.001,
                 learned_pos=False,
                 norm='batch',
                 attention='full',
                 seq_len=None,
                 keep_ratio=0.,
                 random_ratio=1.,
                 token_ratio=0.,
                 uniform_bound=2.,
                 train_unit='standard',
                 train_loss='mae',
                 **kwargs
                 ):
        """Instanciate a Lit TPT imputer module

        Args:
            n_dim (int, optional): number of input dimensions. Defaults to 1.
            d_model (int, optional): Encoder latent dimension. Defaults to 128.
            nhead (int, optional): Number of heads. Defaults to 8.
            dim_feedforward (int, optional): number of feedforward units in the encoder.
                Defaults to 256.
            dropout (float, optional): Encoder dropout. Defaults to 0.1.
            num_layers (int, optional): Number of encoder layer(s). Defaults to 3.
            lr (float, optional): AdamW earning rate. Defaults to 0.001.
        """
        super().__init__()
        self.save_hyperparameters()
        self.n_dim = n_dim
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_layers = num_layers
        self.lr = lr
        self.norm = norm
        # self.zero_ratio =  zero_ratio
        self.keep_ratio = keep_ratio
        self.random_ratio = random_ratio
        self.uniform_bound = uniform_bound
        self.token_ratio = token_ratio
        self.train_unit = train_unit
        assert train_unit in ['standard', 'noise', 'flux', 'star']

        self.ie = nn.Linear(n_dim, d_model)
        self.pe = PosEmbedding(d_model, learned=learned_pos)
        self.ea = EyeAttention(eye)
        if attention == 'linear':
            self.encoder = Linformer(
                dim=d_model,
                seq_len=seq_len,
                depth=num_layers,
                heads=nhead,
                k=32,
                one_kv_head=True,
                share_kv=True
            )
        else:
            encoder_layer = TransformerEncoderLayer(d_model,
                                                    nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=0.1,
                                                    batch_first=True,
                                                    norm=norm, seq_len=seq_len,
                                                    attention=attention
                                                    )
            self.encoder = TransformerEncoder(encoder_layer, num_layers)
        self.recons_head = nn.Linear(d_model, n_dim)
        self.msk_token_emb = nn.Parameter(torch.randn(1, 1, d_model))

        if train_loss == 'mse':
            self.criterion = MaskedMSELoss()  # masked or not masked
        elif train_loss == 'mae':
            self.criterion = MaskedL1Loss()
        elif train_loss == 'huber':
            self.criterion = MaskedHuberLoss()
        else:
            raise NotImplementedError
        self.mae_loss = MaskedL1Loss()
        self.mse_loss = MaskedMSELoss()
        self.iqr_loss = IQRLoss()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitImputer")
        parser.add_argument("--n_dim", type=int)
        parser.add_argument("--d_model", type=int)
        parser.add_argument("--nhead", type=int)
        parser.add_argument("--dim_feedforward", type=int)
        parser.add_argument("--eye", type=int)
        parser.add_argument("--dropout", type=float)
        parser.add_argument("--num_layers", type=int)
        parser.add_argument("--lr", type=float)
        parser.add_argument("--learned_pos", action='store_true')
        parser.add_argument("--norm", type=str)
        parser.add_argument("--attention", type=str)
        parser.add_argument("--seq_len", type=int)
        parser.add_argument("--keep_ratio", type=float)
        parser.add_argument("--random_ratio", type=float)
        parser.add_argument("--token_ratio", type=float)
        parser.add_argument("--uniform_bound", type=float)
        parser.add_argument("--train_unit", type=str,
                            choices=['standard', 'noise', 'star'])
        parser.add_argument("--train_loss", type=str,
                            choices=['mae', 'mse', 'huber'])
        return parent_parser

    def configure_optimizers(self):
        optimiser = opt.Adam(self.parameters(), lr=self.lr)
        return optimiser

    def apply_mask(self, x, mask):
        if mask is None:
            out = x
            out[torch.isnan(out)] = 0.
            return out, torch.zeros_like(x)

        r = torch.rand_like(x)
        keep_mask = (~mask | (r <= self.keep_ratio)).to(x.dtype)
        random_mask = (mask & (self.keep_ratio < r)
                       & (r <= self.keep_ratio+self.random_ratio)).to(x.dtype)
        token_mask = (mask & ((1-self.token_ratio) < r)).to(x.dtype)
        xm, xM = -self.uniform_bound, self.uniform_bound
        out = x * keep_mask + (torch.rand_like(x)*(xM-xm)+xm) * random_mask
        out[torch.isnan(out)] = 0.
        return out, token_mask

    def forward(self, x, mask=None):
        out, token_mask = self.apply_mask(x, mask)
        out = self.ie(out)
        if self.token_ratio:
            out = self.msk_token_emb * token_mask + (1-token_mask) * out

        out = out + self.pe(out)

        attention_mask = self.ea(x)
        out = self.encoder(out, mask=attention_mask)
        out = self.recons_head(out)
        return out

    def get_attention_maps(self, x, mask=None):
        out, token_mask = self.apply_mask(x, mask)
        out = self.ie(out)
        if self.token_ratio:
            out = self.msk_token_emb * token_mask + (1-token_mask) * out

        out = out + self.pe(out)

        attention_mask = self.ea(x)
        out = self.encoder.get_attention_maps(out, mask=attention_mask)
        return out

    def training_step(self, batch, batch_index):
        x, y, m, info = batch
        pred = self.forward(x, m)

        if self.train_unit == 'standard':
            loss = self.criterion(pred, y, m)
        elif self.train_unit == 'noise':
            noise = estimate_noise(y)
            loss = self.criterion(pred/noise, y/noise, m)
        elif self.train_unit == 'star':
            y_o = inverse_standardise_batch(y, info['mu'], info['sigma'])
            pred_o = inverse_standardise_batch(pred, info['mu'], info['sigma'])
            y_d = detrend(y_o, pred_o)
            loss = self.criterion(y_d, torch.ones_like(y_d), m)
        if torch.isnan(loss):
            print('Pred has nans?', torch.isnan(pred).sum().item())
            print('Y has nans?', torch.isnan(
                y).sum().item(), f' shape({y.shape})')
            print('M has fully masked items?',
                  ((m.int()-1).sum((1, 2)) == 0).sum().item())
            print('mu has nans?', torch.isnan(info['mu']).sum().item())
            raise ValueError('Nan Loss found during training')
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss)

    def validation_step(self, batch, batch_index, dataloader_idx=None):
        variable_noise = 0.5
        x, y, m, info = batch
        pred = self.forward(x, m)

        noise = estimate_noise(y)
        variable = (noise <= variable_noise).squeeze()
        n_variables = variable.sum()
        pred_noise = pred / noise
        y_noise = y / noise

        # star normalised unit space
        y_o = inverse_standardise_batch(y, info['mu'], info['sigma'])
        pred_o = inverse_standardise_batch(pred, info['mu'], info['sigma'])
        y_d = detrend(y_o, pred_o)

        out = dict()
        if dataloader_idx is None or dataloader_idx == 0:  # Imputing
            # Imputation
            rmse = torch.sqrt(self.mse_loss(pred, y, m))
            rmse_noise = torch.sqrt(self.mse_loss(pred_noise, y_noise, m))
            rmse_star = torch.sqrt(self.mse_loss(torch.ones_like(y_d), y_d, m))
            mae = self.mae_loss(pred, y, m)
            mae_noise = self.mae_loss(pred_noise, y_noise, m)
            mae_star = self.mae_loss(torch.ones_like(y_d), y_d, m)

            out.update({'val_mrmse': rmse, 'val_mmae': mae,
                        'val_mrmse_noise': rmse_noise, 'val_mmae_noise': mae_noise,
                        'val_mrmse_star': rmse_star, 'val_mmae_star': mae_star
                        })

        if dataloader_idx is None or dataloader_idx == 1:
            # Bias
            rmse = torch.sqrt(self.mse_loss(pred, y))
            rmse_noise = torch.sqrt(self.mse_loss(pred_noise, y_noise))
            rmse_star = torch.sqrt(self.mse_loss(torch.ones_like(y_d), y_d))
            mae = self.mae_loss(pred, y)
            mae_noise = self.mae_loss(pred_noise, y_noise)
            mae_star = self.mae_loss(torch.ones_like(y_d), y_d)

            out.update({'val_rmse': rmse, 'val_mae': mae,
                        'val_rmse_noise': rmse_noise, 'val_mae_noise': mae_noise,
                        'val_rmse_star': rmse_star, 'val_mae_star': mae_star
                        })

            # Denoising
            iqr = self.iqr_loss(pred, y)
            iqr_variable = torch.tensor(np.nan, device=pred.device)
            if n_variables:
                iqr_variable = self.iqr_loss((pred-y)[variable])
            iqr_noise = self.iqr_loss(pred_noise, y_noise)
            iqr_variable_noise = torch.tensor(np.nan, device=pred.device)
            if n_variables:
                iqr_variable_noise = self.iqr_loss(
                    (pred_noise-y_noise)[variable])
            iqr_star = self.iqr_loss(y_d)
            iqr_variable_star = torch.tensor(np.nan, device=pred.device)
            if n_variables:
                iqr_variable_star = self.iqr_loss(y_d[variable])

            out.update({'val_IQR': iqr, 'val_IQR_var': iqr_variable,
                        'val_IQR_noise': iqr_noise, 'val_IQR_var_noise': iqr_variable_noise,
                        'val_IQR_star': iqr_star, 'val_IQR_var_star': iqr_variable_star,
                        })
        return out

    def validation_epoch_end(self, outputs):
        if len(outputs) > 1:
            for dataloader_idx in range(len(outputs)):
                for name in outputs[dataloader_idx][0].keys():
                    score = torch.stack([x[name]
                                        for x in outputs[dataloader_idx]]).mean()
                    self.log(name, score, prog_bar=True)
        else:
            for name in outputs[0].keys():
                score = torch.stack([x[name]
                                    for x in outputs]).mean()
                self.log(name, score, prog_bar=True)

    def test_step(self, batch, batch_index, dataloader_idx=None):
        d_out = self.validation_step(batch, batch_index, dataloader_idx)
        return {k.replace('val', 'test'): v for k, v in d_out.items()}

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)


class PosEmbedding(nn.Module):
    def __init__(self, d_model, learned=False, max_len=5000, dtype=torch.float32):
        super(PosEmbedding, self).__init__()
        if learned:
            self.pe = LearnedPosEmbedding(
                d_model, max_len=max_len, dtype=dtype)
        else:
            self.pe = FixedPosEmbedding(d_model, max_len=max_len, dtype=dtype)

    def forward(self, x):
        return self.pe(x)


class FixedPosEmbedding(nn.Module):
    def __init__(self, d_model, max_len=10000, dtype=torch.float32):
        super(FixedPosEmbedding, self).__init__()
        # Compute the positional embeddings once in log space.
        pe = torch.zeros(max_len, d_model, dtype=dtype)
        pe.requires_grad = False

        position = torch.arange(0, max_len, dtype=dtype).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2, dtype=dtype)
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class LearnedPosEmbedding(nn.Module):
    def __init__(self, d_model, max_len=10000, dtype=torch.float32):
        super(LearnedPosEmbedding, self).__init__()
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model, dtype=dtype))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required). (B, L, D)
        Shape:
            output: tensor of shape (B, L, D)
        """
        return self.pe[:, :x.size(1), :]


class BatchNorm(nn.BatchNorm1d):
    """Overrides nn.BatchNorm1d to define shape structure identical
    to LayerNorm, i.e. (N, L, C) and not (N, C, L)"""

    def forward(self, input):
        return super().forward(input.transpose(1, 2)).transpose(1, 2)


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    r"""Overrides nn.TransformerEncoderLayer class with
    - BatchNorm option as suggested by Zerveas et al https://arxiv.org/abs/2010.02803
    - PrboSparse attention from Zhou et al 
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 norm='layer', attention='full', seq_len=None,
                 device=None, dtype=None) -> None:
        # this combination of shapes hasn't been dealt with yet
        assert batch_first or norm == 'layer'
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation,
                                                      layer_norm_eps, batch_first, norm_first, device, dtype)

        if attention == 'full':
            pass
        else:
            raise NotImplementedError
        if norm == 'layer':
            pass
        elif norm == 'batch':
            self.norm1 = BatchNorm(
                d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = BatchNorm(
                d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            raise NotImplementedError


class TransformerEncoder(nn.TransformerEncoder):
    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, x, x, attn_mask=mask)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps


def inverse_standardise_batch(x, mu, sigma):
    return x * sigma + mu


def detrend(x, trend):
    return x / trend
