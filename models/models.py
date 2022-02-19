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


class ProbAttention(nn.Module):
    """from Zhang 2020"""

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert(L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
            np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1./math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    """from Zhang 2020"""

    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask, key_padding_mask=None, need_weights=None):

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class LinearAttentionHead(nn.Module):
    """
    Linear attention, as proposed by the linformer paper
    """

    def __init__(self, dim, dropout, E_proj, F_proj, causal_mask, full_attention=False):
        super(LinearAttentionHead, self).__init__()
        self.E = E_proj
        self.F = F_proj
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.P_bar = None
        self.full_attention = full_attention
        self.causal_mask = causal_mask
        self.is_proj_tensor = isinstance(E_proj, torch.Tensor)

    def forward(self, Q, K, V, **kwargs):
        """
        Assume Q, K, V have same dtype
        E, F are `nn.Linear` modules
        """
        input_mask = kwargs["input_mask"] if "input_mask" in kwargs else None
        embeddings_mask = kwargs["embeddings_mask"] if "embeddings_mask" in kwargs else None

        # Instead of classic masking, we have to do this, because the classic mask is of size nxn
        if input_mask is not None:
            # This is for k, v
            mask = input_mask[:, :, None]
            K = K.masked_fill_(~mask, 0.0)
            V = V.masked_fill_(~mask, 0.0)
            del mask

        if embeddings_mask is not None:
            mask = embeddings_mask[:, :, None]
            Q = Q.masked_fill_(~mask, 0.0)
            del mask

        K = K.transpose(1, 2)
        if not self.full_attention:
            if self.is_proj_tensor:
                self.E = self.E.to(K.device)
                K = torch.matmul(K, self.E)
            else:
                K = self.E(K)
        Q = torch.matmul(Q, K)

        P_bar = Q / \
            torch.sqrt(torch.tensor(self.dim).type(Q.type())).to(Q.device)
        if self.causal_mask is not None:
            self.causal_mask = self.causal_mask.to(Q.device)
            P_bar = P_bar.masked_fill_(~self.causal_mask, float('-inf'))
        P_bar = P_bar.softmax(dim=-1)

        # Only save this when visualizing
        if "visualize" in kwargs and kwargs["visualize"] == True:
            self.P_bar = P_bar

        P_bar = self.dropout(P_bar)

        if not self.full_attention:
            V = V.transpose(1, 2)
            if self.is_proj_tensor:
                self.F = self.F.to(V.device)
                V = torch.matmul(V, self.F)
            else:
                V = self.F(V)
            V = V.transpose(1, 2)
        out_tensor = torch.matmul(P_bar, V)

        return out_tensor

    # import torch
# from linformer import LinformerSelfAttention


def default(val, default_val):
    return val if val is not None else default_val


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class LinformerSelfAttention(nn.Module):
    def __init__(self, dim, seq_len, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False, dropout=0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x,
                src_mask=None, src_key_padding_mask=None,
                context=None, **kwargs):
        if src_mask is not None:
            warnings.warn('SRC MASK not used in Linformer')
        if src_key_padding_mask is not None:
            warnings.warn('src_key_padding_mask not used in Linformer')
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        kv_len = n if context is None else context.shape[1]
        assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x)

        def proj_seq_len(args): return torch.einsum('bnd,nk->bkd', *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (
            self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        def merge_key_values(t): return t.reshape(
            b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., activation=None, glu=False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x


class SequentialSequence(nn.Module):
    def __init__(self, layers, args_route={}, layer_dropout=0.):
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values(
        )), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route
        self.layer_dropout = layer_dropout

    def forward(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        if self.training and self.layer_dropout > 0:
            layers_and_args = layer_drop(layers_and_args, self.layer_dropout)

        for (f, g), (f_args, g_args) in layers_and_args:
            x = x + f(x, **f_args)
            x = x + g(x, **g_args)
        return x


def route_args(router, args, depth):
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            new_f_args, new_g_args = map(lambda route: (
                {key: val} if route else {}), routes)
            routed_args[depth] = ({**f_args, **new_f_args},
                                  {**g_args, **new_g_args})
    return routed_args


class Linformer(nn.Module):
    def __init__(self, dim, seq_len, depth, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False, reversible=False, dropout=0.):
        super().__init__()
        layers = nn.ModuleList([])
        for _ in range(depth):
            attn = LinformerSelfAttention(dim, seq_len, k=k, heads=heads, dim_head=dim_head,
                                          one_kv_head=one_kv_head, share_kv=share_kv, dropout=dropout)
            ff = FeedForward(dim, dropout=dropout)

            layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, ff)
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence
        self.net = execute_type(layers)

    def forward(self, x,
                mask=None
                ):
        return self.net(x)

    import torch.nn.functional as F


def eye_large(n, width=1, dtype=torch.float32, device=None):
    out = np.eye(n)
    for k in range(1, width):
        out += np.eye(n, k=k)
        out += np.eye(n, k=-k)
    return torch.tensor(out, dtype=dtype, device=device)


class EyeAttention(nn.Module):
    def __init__(self, width=1, max_len=500):
        super(EyeAttention, self).__init__()
        # "don't attend now"
        if width == 1:
            mask = torch.eye(max_len, dtype=bool)
        elif width == 0:
            mask = None
        else:
            mask = eye_large(max_len, width=width, dtype=bool)
        self.register_buffer('mask', mask)

    def forward(self, x):
        if self.mask is None:
            return None
        return self.mask[:x.shape[1], :x.shape[1]]


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
    """adapted from Zhang 2020"""

    def __init__(self, d_model, max_len=5000, dtype=torch.float32):
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
    """adapted from Zerveas 2020"""

    def __init__(self, d_model, max_len=5000, dtype=torch.float32):
        super(LearnedPosEmbedding, self).__init__()
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        # requires_grad automatically set to True
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model, dtype=dtype))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required). (B, L, D)
        Shape:
            output: [batch_size, sequence length, embed dim] (B, L, D)
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
        elif attention == "prob":
            self.self_attn = AttentionLayer(ProbAttention(False,
                                                          factor=5,
                                                          attention_dropout=dropout,
                                                          output_attention=False),
                                            d_model, nhead, mix=False)
#         elif attention=='linear':  # doesn't work like that. Raises the quesion for probsparse!
#             assert seq_len is not None
#             self.self_attn = LinformerSelfAttention(
#                                                 dim = d_model,
#                                                 seq_len = seq_len,
#                                                 heads = nhead,
#                                                 k = 256,
#                                                 one_kv_head = True,
#                                                 share_kv = True
#                                             )
        if norm == 'layer':
            pass
        elif norm == 'batch':
            self.norm1 = BatchNorm(
                d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = BatchNorm(
                d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            raise NotImplementedError


class LitImputer(pl.LightningModule):
    def __init__(self, n_dim=1, d_model=128, nhead=8, dim_feedforward=256, eye=0,
                 dropout=0.1, num_layers=3, lr=0.001,
                 learned_pos=False, norm='batch', attention='full', seq_len=None,
                 zero_ratio=None, keep_ratio=None, random_ratio=None, token_ratio=None,
                 train_unit='standard', train_loss='mse'
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
        if self.n_dim == 1:
            self.zero_ratio = 0. if zero_ratio is None else zero_ratio
            self.keep_ratio = 0. if keep_ratio is None else keep_ratio
            self.random_ratio = 0.1 if random_ratio is None else random_ratio
            self.token_ratio = 0.9 if token_ratio is None else token_ratio
        elif self.n_dim > 1:
            self.zero_ratio = 0.1 if zero_ratio is None else zero_ratio
            self.keep_ratio = 0. if keep_ratio is None else keep_ratio
            self.random_ratio = 0.9 if random_ratio is None else random_ratio
            self.token_ratio = 0. if token_ratio is None else token_ratio
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
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.recons_head = nn.Linear(d_model, n_dim)
        self.msk_token_emb = nn.Parameter(torch.randn(1, 1, d_model))

        if train_loss == 'mse':
            self.criterion = MaskedMSELoss()  # masked or not masked
        elif train_loss == 'mae':
            self.criterion = MaskedL1Loss() 
        elif train_loss == 'huber':
            self.criterion = MaskedHuberLoss() 
        self.iqr_loss = IQRLoss()

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
        # keep_mask = (~missing | (artif_mask & (r <= self.keep_ratio))).to(x.dtype)
        random_mask = (mask & (self.keep_ratio < r)
                       & (r <= self.keep_ratio+self.random_ratio)).to(x.dtype)
        # zero_mask = (mask & (self.keep_ratio + self.random_ratio < r)   # Shouldn't need it by construction
        #              & (r <= (1-self.token_ratio))).to(x.dtype)
        token_mask = (mask & ((1-self.token_ratio) < r)).to(x.dtype)
        # xm, xM = x.min(1, keepdim=True).values, x.max(1, keepdim=True).values   # problem due to Nans
        xm, xM = -2, 2
        out = x * keep_mask + (torch.rand_like(x)*(xM-xm)+xm) * random_mask
        out[torch.isnan(out)] = 0.
        return out, token_mask

    def forward(self, x, mask=None):
        out, token_mask = self.apply_mask(x, mask)
        out = self.ie(out)
        #print(self.msk_token_emb.shape, token_mask.shape, out.shape)
        if self.token_ratio:
            out = self.msk_token_emb * token_mask + (1-token_mask) * out

        out = out + self.pe(out)

        attention_mask = self.ea(x)
        out = self.encoder(out, mask=attention_mask)
        out = self.recons_head(out)
        return out

    def training_step(self, batch, batch_index):
        x, y, m, info = batch
        pred = self.forward(x, m)
        assert not torch.isnan(pred).any()

        if self.train_unit == 'standard':
            loss = self.criterion(pred, y, m)
        elif self.train_unit == 'noise':
            noise = estimate_noise(y)
            # noise[torch.isnan(noise)] = 1.
            # noise[noise == 0] = 1.
            # nans = torch.isnan(y)
            # m = m & ~nans
            loss = self.criterion(pred/noise, y/noise, m)
        elif self.train_unit == 'star':
            # nans = torch.isnan(y)
            # m = m & ~nans
            # y[nans] = 0.
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

    def validation_step(self, batch, batch_index):
        variable_noise = 0.5
        x, y, m, info = batch
        pred = self.forward(x, m)
        noise = estimate_noise(y)
        variable = (noise <= variable_noise).squeeze()
        n_variables = variable.sum()

        # standard unit space
        mse = self.criterion(pred, y, m)
        iqr = self.iqr_loss(pred, y)
        iqr_variable = np.nan
        if variable.sum():
            iqr_variable = self.iqr_loss((pred-y)[variable])

        # white noise unit space
        pred_noise = pred / noise
        y_noise = y / noise
        mse_noise = self.criterion(pred_noise, y_noise, m)
        iqr_noise = self.iqr_loss(pred_noise, y_noise)
        iqr_variable_noise = np.nan
        if n_variables:
            iqr_variable_noise = self.iqr_loss((pred_noise-y_noise)[variable])

        # star normalised unit space
        y_o = inverse_standardise_batch(y, info['mu'], info['sigma'])
        pred_o = inverse_standardise_batch(pred, info['mu'], info['sigma'])
        y_d = detrend(y_o, pred_o)
        iqr_star = self.iqr_loss(y_d)
        mse_star = self.criterion( torch.ones_like(y_d), y_d, m)
        iqr_variable_star = np.nan
        if n_variables:
            iqr_variable_star = self.iqr_loss(y_d[variable])

        return {'val_rmse': torch.sqrt(mse), 'val_IQR': iqr, 'val_IQR_var':iqr_variable,
                'val_rmse_noise': torch.sqrt(mse_noise), 'val_IQR_noise': iqr_noise, 'val_IQR_var_noise': iqr_variable_noise,
                'val_rmse_star': torch.sqrt(mse_star), 'val_IQR_star': iqr_star, 'val_IQR_var_star': iqr_variable_star,
                #"n_variables": n_variables,
                }

    def validation_epoch_end(self, outputs):
        for name in outputs[0].keys():
            score = torch.stack([x[name] for x in outputs]).mean()
            self.log(name, score, prog_bar=True)

    def test_step(self, batch, batch_index):
        d_out = self.validation_step(batch, batch_index)
        return {k.replace('val', 'test') : v for k,v in d_out.items()}

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)


def inverse_standardise_batch(x, mu, sigma):
    return x * sigma + mu


def detrend(x, trend):
    return x / trend
