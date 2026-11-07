"""Attention and Transformer layers."""
from __future__ import annotations
import numpy as np
import math
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


def _softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class MultiheadAttention(Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout=0.0,
                 bias=True, batch_first=False, kdim=None, vdim=None):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout
        self.batch_first = batch_first
        self.kdim = kdim or embed_dim
        self.vdim = vdim or embed_dim
        k = 1.0 / math.sqrt(embed_dim)
        self.q_proj_weight = Tensor(np.random.uniform(-k, k, (embed_dim, embed_dim)).astype(np.float32), requires_grad=True)
        self.k_proj_weight = Tensor(np.random.uniform(-k, k, (embed_dim, self.kdim)).astype(np.float32), requires_grad=True)
        self.v_proj_weight = Tensor(np.random.uniform(-k, k, (embed_dim, self.vdim)).astype(np.float32), requires_grad=True)
        self.out_proj_weight = Tensor(np.random.uniform(-k, k, (embed_dim, embed_dim)).astype(np.float32), requires_grad=True)
        for name in ('q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'out_proj_weight'):
            self._parameters[name] = getattr(self, name)
        self.in_proj_bias = None
        self.out_proj_bias = None
        if bias:
            self.in_proj_bias = Tensor(np.zeros(3 * embed_dim, dtype=np.float32), requires_grad=True)
            self.out_proj_bias = Tensor(np.zeros(embed_dim, dtype=np.float32), requires_grad=True)
            self._parameters['in_proj_bias'] = self.in_proj_bias
            self._parameters['out_proj_bias'] = self.out_proj_bias

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                attn_mask=None, key_padding_mask=None):
        q, k, v = query._data, key._data, value._data
        if self.batch_first:
            q = q.transpose(1, 0, 2)
            k = k.transpose(1, 0, 2)
            v = v.transpose(1, 0, 2)
        T, N, E = q.shape
        S = k.shape[0]
        qb = kb = vb = None
        if self.in_proj_bias is not None:
            qb = self.in_proj_bias._data[:E]
            kb = self.in_proj_bias._data[E:2*E]
            vb = self.in_proj_bias._data[2*E:]
        Q = q.reshape(T*N, E) @ self.q_proj_weight._data.T
        if qb is not None: Q += qb
        K = k.reshape(S*N, E) @ self.k_proj_weight._data.T
        if kb is not None: K += kb
        V = v.reshape(S*N, E) @ self.v_proj_weight._data.T
        if vb is not None: V += vb
        Q = Q.reshape(T, N, self.num_heads, self.head_dim).transpose(1, 2, 0, 3)  # (N,H,T,D)
        K = K.reshape(S, N, self.num_heads, self.head_dim).transpose(1, 2, 0, 3)  # (N,H,S,D)
        V = V.reshape(S, N, self.num_heads, self.head_dim).transpose(1, 2, 0, 3)  # (N,H,S,D)
        scale = math.sqrt(self.head_dim)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / scale  # (N,H,T,S)
        if attn_mask is not None:
            scores = scores + attn_mask
        if key_padding_mask is not None:
            scores = scores + key_padding_mask[:, None, None, :] * -1e9
        attn_weights = _softmax(scores, axis=-1)
        if self.dropout_p > 0 and self.training:
            mask = (np.random.rand(*attn_weights.shape) > self.dropout_p).astype(np.float32)
            attn_weights = attn_weights * mask / (1 - self.dropout_p)
        out = np.matmul(attn_weights, V)  # (N,H,T,D)
        out = out.transpose(2, 0, 1, 3).reshape(T, N, E)  # (T,N,E)
        out = out.reshape(T*N, E) @ self.out_proj_weight._data.T
        if self.out_proj_bias is not None:
            out += self.out_proj_bias._data
        out = out.reshape(T, N, E)
        if self.batch_first:
            out = out.transpose(1, 0, 2)
        avg_attn = attn_weights.mean(axis=1)
        return Tensor(out.astype(np.float32), device=query.device), Tensor(avg_attn.astype(np.float32))

    def __repr__(self):
        return f'MultiheadAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads})'


class TransformerEncoderLayer(Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward=2048,
                 dropout=0.1, activation='relu', batch_first=False, norm_first=False):
        super().__init__()
        from vectorphile.nn.layers import Linear, LayerNorm, Dropout
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = activation
        self.norm_first = norm_first
        for name, mod in [('self_attn', self.self_attn), ('linear1', self.linear1),
                           ('linear2', self.linear2), ('norm1', self.norm1), ('norm2', self.norm2)]:
            self._parameters.update({f'{name}_{k}': v for k, v in mod._parameters.items()})

    def _act(self, x):
        if self.activation == 'relu':
            return Tensor(np.maximum(0, x._data))
        elif self.activation == 'gelu':
            return Tensor(x._data * 0.5 * (1.0 + np.tanh(math.sqrt(2/math.pi) * (x._data + 0.044715 * x._data**3))))
        return x

    def forward(self, src: Tensor, src_mask=None, src_key_padding_mask=None) -> Tensor:
        if self.norm_first:
            src2 = self.norm1(src)
            attn_out, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
            src = Tensor(src._data + self.dropout1(attn_out)._data)
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self._act(self.linear1(src2))))
            src = Tensor(src._data + self.dropout2(src2)._data)
        else:
            attn_out, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
            src = self.norm1(Tensor(src._data + self.dropout1(attn_out)._data))
            src2 = self.linear2(self.dropout(self._act(self.linear1(src))))
            src = self.norm2(Tensor(src._data + self.dropout2(src2)._data))
        return src


class TransformerEncoder(Module):
    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int, norm=None):
        super().__init__()
        self.layers = [encoder_layer]
        for i in range(1, num_layers):
            import copy
            lyr = copy.deepcopy(encoder_layer)
            self.layers.append(lyr)
            self._parameters.update({f'layer_{i}_{k}': v for k, v in lyr._parameters.items()})
        self.norm = norm

    def forward(self, src: Tensor, mask=None, src_key_padding_mask=None) -> Tensor:
        out = src
        for layer in self.layers:
            out = layer(out, mask, src_key_padding_mask)
        if self.norm is not None:
            out = self.norm(out)
        return out


class TransformerDecoderLayer(Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward=2048,
                 dropout=0.1, activation='relu', batch_first=False):
        super().__init__()
        from vectorphile.nn.layers import Linear, LayerNorm, Dropout
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.activation = activation

    def _act(self, x):
        if self.activation == 'gelu':
            return Tensor(x._data * 0.5 * (1.0 + np.tanh(math.sqrt(2/math.pi) * (x._data + 0.044715 * x._data**3))))
        return Tensor(np.maximum(0, x._data))

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        attn1, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(Tensor(tgt._data + self.dropout1(attn1)._data))
        attn2, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = self.norm2(Tensor(tgt._data + self.dropout2(attn2)._data))
        ff = self.linear2(self.dropout(self._act(self.linear1(tgt))))
        tgt = self.norm3(Tensor(tgt._data + self.dropout3(ff)._data))
        return tgt


class TransformerDecoder(Module):
    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int, norm=None):
        super().__init__()
        self.layers = [decoder_layer]
        import copy
        for i in range(1, num_layers):
            lyr = copy.deepcopy(decoder_layer)
            self.layers.append(lyr)
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        out = tgt
        for layer in self.layers:
            out = layer(out, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        if self.norm is not None:
            out = self.norm(out)
        return out


class Transformer(Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation='relu', batch_first=False):
        super().__init__()
        enc_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first)
        dec_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first)
        self.encoder = TransformerEncoder(enc_layer, num_encoder_layers)
        self.decoder = TransformerDecoder(dec_layer, num_decoder_layers)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        memory = self.encoder(src, src_mask, src_key_padding_mask)
        out = self.decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return out


class PositionalEncoding(Module):
    def __init__(self, d_model: int, dropout=0.1, max_len=5000, batch_first=False):
        super().__init__()
        from vectorphile.nn.layers import Dropout
        self.dropout = Dropout(dropout)
        self.batch_first = batch_first
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pos = np.arange(max_len)[:, None]
        div = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[:d_model//2])
        self._pe = pe  # (max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        data = x._data
        if self.batch_first:
            T = data.shape[1]
            data = data + self._pe[:T][None, :, :]
        else:
            T = data.shape[0]
            data = data + self._pe[:T][:, None, :]
        return self.dropout(Tensor(data.astype(np.float32), device=x.device))
