"""Complete Transformer architectures — GPT, BERT, T5, LLaMA, Mistral."""
from __future__ import annotations
import numpy as np
from typing import Optional, Dict, List, Tuple
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor
from vectorphile.nn.architectures.attention import (
    GroupedQueryAttention, RotaryEmbedding,
    scaled_dot_product_attention, causal_mask, _softmax,
)


# ── Common building blocks ────────────────────────────────────────────────────

class RMSNorm(Module):
    """Root Mean Square Layer Normalization — Zhang & Sennrich 2019. Used in LLaMA/T5."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        w = Tensor(np.ones(dim, dtype=np.float32))
        w.requires_grad = True
        self._parameters['weight'] = w
        self.weight = w

    def forward(self, x: Tensor) -> Tensor:
        d = x._data
        rms = np.sqrt((d ** 2).mean(axis=-1, keepdims=True) + self.eps)
        return Tensor((d / rms) * self.weight._data)


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = Tensor(np.ones(dim, dtype=np.float32))
        self.bias = Tensor(np.zeros(dim, dtype=np.float32))
        self.weight.requires_grad = True
        self.bias.requires_grad = True
        self._parameters['weight'] = self.weight
        self._parameters['bias'] = self.bias

    def forward(self, x: Tensor) -> Tensor:
        d = x._data
        mean = d.mean(axis=-1, keepdims=True)
        var = d.var(axis=-1, keepdims=True)
        return Tensor(((d - mean) / np.sqrt(var + self.eps)) * self.weight._data + self.bias._data)


def _linear(x: np.ndarray, w: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    out = x @ w
    if b is not None:
        out = out + b
    return out


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        scale = np.sqrt(2.0 / in_features)
        w = Tensor(np.random.randn(in_features, out_features).astype(np.float32) * scale)
        w.requires_grad = True
        self._parameters['weight'] = w
        self.weight = w
        self.bias_param = None
        if bias:
            b = Tensor(np.zeros(out_features, dtype=np.float32))
            b.requires_grad = True
            self._parameters['bias'] = b
            self.bias_param = b

    def forward(self, x: Tensor) -> Tensor:
        b = self.bias_param._data if self.bias_param is not None else None
        return Tensor(_linear(x._data, self.weight._data, b))


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        w = Tensor(np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02)
        w.requires_grad = True
        self._parameters['weight'] = w
        self.weight = w
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, indices: Tensor) -> Tensor:
        idx = indices._data.astype(int)
        return Tensor(self.weight._data[idx])


class Dropout(Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = (np.random.random(x._data.shape) > self.p).astype(np.float32) / (1 - self.p)
        return Tensor(x._data * mask)


# ── FFN variants ──────────────────────────────────────────────────────────────

class GELUMLP(Module):
    """Standard FFN with GELU — used in BERT, GPT-2."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.fc1 = Linear(embed_dim, ffn_dim, bias=bias)
        self.fc2 = Linear(ffn_dim, embed_dim, bias=bias)
        self.drop = Dropout(dropout)
        self._modules['fc1'] = self.fc1
        self._modules['fc2'] = self.fc2

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc1(x)
        # GELU approximation
        d = h._data
        gelu = 0.5 * d * (1 + np.tanh(np.sqrt(2 / np.pi) * (d + 0.044715 * d ** 3)))
        return self.fc2(self.drop(Tensor(gelu)))


class SwiGLU(Module):
    """SwiGLU FFN — Noam Shazeer 2020. Used in LLaMA, PaLM, Gemma.

    FFN(x) = (Swish(W1 x) ⊙ W3 x) W2
    dim_expansion typically 8/3 * embed_dim rounded to nearest 256.
    """

    def __init__(self, embed_dim: int, ffn_dim: Optional[int] = None, bias: bool = False):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = int(8 / 3 * embed_dim)
            ffn_dim = ((ffn_dim + 255) // 256) * 256   # round up to 256
        self.w1 = Linear(embed_dim, ffn_dim, bias=bias)
        self.w2 = Linear(ffn_dim, embed_dim, bias=bias)
        self.w3 = Linear(embed_dim, ffn_dim, bias=bias)
        self._modules.update({'w1': self.w1, 'w2': self.w2, 'w3': self.w3})

    def forward(self, x: Tensor) -> Tensor:
        gate = self.w1(x)._data
        # SiLU (Swish-1): x * sigmoid(x)
        silu = gate * (1 / (1 + np.exp(-gate)))
        v = self.w3(x)._data
        return self.w2(Tensor(silu * v))


class GeGLU(Module):
    """GeGLU FFN variant — Noam Shazeer 2020. Used in some T5 variants."""

    def __init__(self, embed_dim: int, ffn_dim: Optional[int] = None, bias: bool = True):
        super().__init__()
        ffn_dim = ffn_dim or embed_dim * 4
        self.w = Linear(embed_dim, ffn_dim * 2, bias=bias)
        self.w2 = Linear(ffn_dim, embed_dim, bias=bias)
        self._modules.update({'w': self.w, 'w2': self.w2})

    def forward(self, x: Tensor) -> Tensor:
        projected = self.w(x)._data
        half = projected.shape[-1] // 2
        x1, x2 = projected[..., :half], projected[..., half:]
        gelu = 0.5 * x1 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x1 + 0.044715 * x1 ** 3)))
        return self.w2(Tensor(gelu * x2))


# ── GPT-2 ─────────────────────────────────────────────────────────────────────

class GPTBlock(Module):
    """Single GPT-2 decoder block: pre-norm → causal MHA → pre-norm → MLP."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)
        self.drop = Dropout(dropout)
        self.ffn = GELUMLP(embed_dim, ffn_dim, dropout=dropout, bias=bias)
        # QKV in one matrix for efficiency
        self.c_attn = Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.c_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self._modules.update({'ln1': self.ln1, 'ln2': self.ln2,
                               'ffn': self.ffn, 'c_attn': self.c_attn, 'c_proj': self.c_proj})

    def forward(self, x: Tensor, kv_cache: Optional[dict] = None) -> Tensor:
        # Pre-norm + causal self-attention
        normed = self.ln1(x)
        qkv = self.c_attn(normed)._data
        B, T, _ = qkv.shape
        q, k, v = np.split(qkv, 3, axis=-1)

        def _split(t):
            return t.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = _split(q), _split(k), _split(v)

        if kv_cache is not None:
            if 'k' in kv_cache:
                k = np.concatenate([kv_cache['k'], k], axis=2)
                v = np.concatenate([kv_cache['v'], v], axis=2)
            kv_cache['k'] = k
            kv_cache['v'] = v

        T_kv = k.shape[2]
        mask = causal_mask(T_kv)
        # Only last T rows needed when using KV cache
        if T_kv > T:
            mask = mask[-T:, :]
        attn_out, _ = scaled_dot_product_attention(q, k, v, mask=mask[None, None, :, :])
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        x = Tensor(x._data + self.drop(self.c_proj(Tensor(attn_out)))._data)
        x = Tensor(x._data + self.drop(self.ffn(self.ln2(x)))._data)
        return x


class GPT2(Module):
    """GPT-2 — Radford et al. 2019.

    Decoder-only Transformer with learned positional embeddings.
    Configurations: small (117M), medium (345M), large (762M), xl (1.5B).

    Args:
        vocab_size: vocabulary size
        embed_dim: model dimension (d_model)
        num_heads: attention heads
        num_layers: transformer blocks
        max_seq_len: maximum sequence length
        ffn_dim: feed-forward hidden dimension (default 4 * embed_dim)
        dropout: dropout probability
        bias: use bias in linear layers
    """

    CONFIGS = {
        'small':  dict(embed_dim=768,  num_heads=12, num_layers=12),
        'medium': dict(embed_dim=1024, num_heads=16, num_layers=24),
        'large':  dict(embed_dim=1280, num_heads=20, num_layers=36),
        'xl':     dict(embed_dim=1600, num_heads=25, num_layers=48),
    }

    def __init__(self, vocab_size: int = 50257, embed_dim: int = 768,
                 num_heads: int = 12, num_layers: int = 12,
                 max_seq_len: int = 1024, ffn_dim: Optional[int] = None,
                 dropout: float = 0.1, bias: bool = True):
        super().__init__()
        ffn_dim = ffn_dim or embed_dim * 4
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_emb = Embedding(vocab_size, embed_dim)
        self.pos_emb = Embedding(max_seq_len, embed_dim)
        self.drop = Dropout(dropout)
        self.blocks = [GPTBlock(embed_dim, num_heads, ffn_dim, dropout, bias) for _ in range(num_layers)]
        self.ln_f = LayerNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)

        self._modules['token_emb'] = self.token_emb
        self._modules['pos_emb'] = self.pos_emb
        for i, b in enumerate(self.blocks):
            self._modules[f'block_{i}'] = b
        self._modules['ln_f'] = self.ln_f
        self._modules['lm_head'] = self.lm_head

    @classmethod
    def from_config(cls, name: str, **kwargs) -> 'GPT2':
        return cls(**{**cls.CONFIGS[name], **kwargs})

    def forward(self, input_ids: Tensor, kv_caches: Optional[List[dict]] = None) -> Tensor:
        B, T = input_ids._data.shape
        pos = Tensor(np.arange(T, dtype=np.float32).reshape(1, T))
        x = Tensor(self.token_emb(input_ids)._data + self.pos_emb(pos)._data)
        x = self.drop(x)
        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)
        for block, cache in zip(self.blocks, kv_caches):
            x = block(x, cache)
        x = self.ln_f(x)
        return self.lm_head(x)   # (B, T, vocab_size)

    def generate(self, input_ids: Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None) -> np.ndarray:
        """Auto-regressive generation with temperature, top-k, and nucleus (top-p) sampling."""
        ids = input_ids._data.copy()
        kv_caches = [{} for _ in self.blocks]
        self.eval()
        for _ in range(max_new_tokens):
            inp = Tensor(ids[:, -self.max_seq_len:])
            logits = self.forward(inp, kv_caches)._data[:, -1, :]  # (B, vocab)
            logits = logits / max(temperature, 1e-8)
            if top_k is not None:
                v = np.sort(logits, axis=-1)[:, -top_k:][:, :1]
                logits = np.where(logits < v, -1e9, logits)
            if top_p is not None:
                sorted_logits = np.sort(logits, axis=-1)[:, ::-1]
                cumprobs = np.cumsum(_softmax(sorted_logits), axis=-1)
                cutoff = sorted_logits[0, np.searchsorted(cumprobs[0], top_p)]
                logits = np.where(logits < cutoff, -1e9, logits)
            probs = _softmax(logits)
            next_id = np.array([[np.random.choice(probs.shape[-1], p=probs[b]) for b in range(len(probs))]])
            ids = np.concatenate([ids, next_id.T], axis=1)
        return ids


# ── BERT ──────────────────────────────────────────────────────────────────────

class BERTEmbeddings(Module):
    """Token + Position + Segment embeddings for BERT."""

    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int = 512,
                 type_vocab_size: int = 2, dropout: float = 0.1):
        super().__init__()
        self.word_emb = Embedding(vocab_size, embed_dim)
        self.pos_emb = Embedding(max_seq_len, embed_dim)
        self.seg_emb = Embedding(type_vocab_size, embed_dim)
        self.ln = LayerNorm(embed_dim)
        self.drop = Dropout(dropout)
        for k, v in [('word_emb', self.word_emb), ('pos_emb', self.pos_emb),
                     ('seg_emb', self.seg_emb), ('ln', self.ln)]:
            self._modules[k] = v

    def forward(self, input_ids: Tensor, segment_ids: Optional[Tensor] = None) -> Tensor:
        B, T = input_ids._data.shape
        pos = Tensor(np.arange(T, dtype=np.float32).reshape(1, T))
        emb = self.word_emb(input_ids)._data + self.pos_emb(pos)._data
        if segment_ids is not None:
            emb = emb + self.seg_emb(segment_ids)._data
        return self.drop(self.ln(Tensor(emb)))


class BERTSelfAttention(Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = Linear(embed_dim, embed_dim)
        self.key = Linear(embed_dim, embed_dim)
        self.value = Linear(embed_dim, embed_dim)
        self.attn_drop = Dropout(dropout)
        for k, v in [('query', self.query), ('key', self.key), ('value', self.value)]:
            self._modules[k] = v

    def forward(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        B, T, _ = x._data.shape
        def _split(t):
            return t._data.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k, v = _split(self.query(x)), _split(self.key(x)), _split(self.value(x))
        out, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        return Tensor(out.transpose(0, 2, 1, 3).reshape(B, T, -1))


class BERTLayer(Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = BERTSelfAttention(embed_dim, num_heads, dropout)
        self.attn_out = Linear(embed_dim, embed_dim)
        self.ln1 = LayerNorm(embed_dim)
        self.ffn = GELUMLP(embed_dim, ffn_dim, dropout=dropout)
        self.ln2 = LayerNorm(embed_dim)
        self.drop = Dropout(dropout)
        for k, v in [('attn', self.attn), ('attn_out', self.attn_out),
                     ('ln1', self.ln1), ('ffn', self.ffn), ('ln2', self.ln2)]:
            self._modules[k] = v

    def forward(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        attn_out = self.drop(self.attn_out(self.attn(x, mask)))
        x = self.ln1(Tensor(x._data + attn_out._data))
        ffn_out = self.drop(self.ffn(x))
        return self.ln2(Tensor(x._data + ffn_out._data))


class BERTEncoder(Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int,
                 ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layers = [BERTLayer(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        for i, l in enumerate(self.layers):
            self._modules[f'layer_{i}'] = l

    def forward(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class BERT(Module):
    """BERT — Devlin et al. 2018.

    Bidirectional encoder. Configurations: base (110M), large (340M).

    Args:
        vocab_size: vocabulary size (default 30522 — WordPiece)
        embed_dim: d_model
        num_heads: attention heads
        num_layers: transformer layers
        ffn_dim: feed-forward hidden size
        max_seq_len: maximum position embeddings
        dropout: dropout probability
    """

    CONFIGS = {
        'base':  dict(embed_dim=768,  num_heads=12, num_layers=12, ffn_dim=3072),
        'large': dict(embed_dim=1024, num_heads=16, num_layers=24, ffn_dim=4096),
    }

    def __init__(self, vocab_size: int = 30522, embed_dim: int = 768,
                 num_heads: int = 12, num_layers: int = 12,
                 ffn_dim: int = 3072, max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.embeddings = BERTEmbeddings(vocab_size, embed_dim, max_seq_len, dropout=dropout)
        self.encoder = BERTEncoder(embed_dim, num_heads, num_layers, ffn_dim, dropout)
        self.pooler = Linear(embed_dim, embed_dim)  # [CLS] pooling
        self._modules.update({'embeddings': self.embeddings,
                               'encoder': self.encoder, 'pooler': self.pooler})

    @classmethod
    def from_config(cls, name: str, **kwargs) -> 'BERT':
        return cls(**{**cls.CONFIGS[name], **kwargs})

    def forward(self, input_ids: Tensor, segment_ids: Optional[Tensor] = None,
                attention_mask: Optional[np.ndarray] = None) -> Tuple[Tensor, Tensor]:
        x = self.embeddings(input_ids, segment_ids)
        # Convert padding mask (1=attend, 0=pad) to additive mask
        if attention_mask is not None:
            attn_mask = (1 - attention_mask[:, None, None, :]).astype(bool)
        else:
            attn_mask = None
        sequence_output = self.encoder(x, attn_mask)
        # [CLS] token → pooled
        cls = sequence_output._data[:, 0, :]
        pooled = np.tanh(_linear(cls, self.pooler.weight._data,
                                  self.pooler.bias_param._data if self.pooler.bias_param else None))
        return sequence_output, Tensor(pooled)


class BERTForMLM(Module):
    """BERT with Masked Language Modelling head."""

    def __init__(self, **bert_kwargs):
        super().__init__()
        self.bert = BERT(**bert_kwargs)
        embed_dim = bert_kwargs.get('embed_dim', 768)
        vocab_size = bert_kwargs.get('vocab_size', 30522)
        self.dense = Linear(embed_dim, embed_dim)
        self.ln = LayerNorm(embed_dim)
        self.decoder = Linear(embed_dim, vocab_size, bias=False)
        for k, v in [('bert', self.bert), ('dense', self.dense),
                     ('ln', self.ln), ('decoder', self.decoder)]:
            self._modules[k] = v

    def forward(self, input_ids: Tensor, **kwargs) -> Tensor:
        seq_out, _ = self.bert(input_ids, **kwargs)
        d = seq_out._data
        # GELU activation
        h = _linear(d, self.dense.weight._data, self.dense.bias_param._data if self.dense.bias_param else None)
        gelu = 0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h ** 3)))
        h = self.ln(Tensor(gelu))
        return self.decoder(h)   # (B, T, vocab_size)


# ── T5 ────────────────────────────────────────────────────────────────────────

class T5RelativePositionBias(Module):
    """T5-style relative position bias — shared across all layers."""

    def __init__(self, num_heads: int, num_buckets: int = 32, max_distance: int = 128,
                 bidirectional: bool = True):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        w = Tensor(np.random.randn(num_buckets, num_heads).astype(np.float32) * 0.02)
        w.requires_grad = True
        self._parameters['emb'] = w
        self.emb = w

    def _bucket(self, relative_position: np.ndarray) -> np.ndarray:
        n = -relative_position
        num_buckets = self.num_buckets
        ret = np.zeros_like(n)
        if self.bidirectional:
            num_buckets //= 2
            ret += (n < 0).astype(int) * num_buckets
            n = np.abs(n)
        else:
            n = np.maximum(n, 0)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            np.log(n.astype(float) / max_exact + 1e-8) /
            np.log(self.max_distance / max_exact) * (num_buckets - max_exact)
        ).astype(int)
        val_if_large = np.minimum(val_if_large, num_buckets - 1)
        return np.where(is_small, n, val_if_large) + ret

    def forward(self, seq_q: int, seq_k: int) -> np.ndarray:
        ctx = np.arange(seq_q)[:, None]
        mem = np.arange(seq_k)[None, :]
        rel = mem - ctx   # (seq_q, seq_k)
        buckets = self._bucket(rel)
        bias = self.emb._data[buckets]   # (seq_q, seq_k, num_heads)
        return bias.transpose(2, 0, 1)[None, :, :, :]   # (1, heads, seq_q, seq_k)


class T5Block(Module):
    """T5 encoder/decoder block with relative position bias and optional cross-attention."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 is_decoder: bool = False, dropout: float = 0.1):
        super().__init__()
        self.is_decoder = is_decoder
        head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embed_dim = embed_dim

        self.ln1 = RMSNorm(embed_dim)
        self.sa_q = Linear(embed_dim, embed_dim, bias=False)
        self.sa_k = Linear(embed_dim, embed_dim, bias=False)
        self.sa_v = Linear(embed_dim, embed_dim, bias=False)
        self.sa_o = Linear(embed_dim, embed_dim, bias=False)

        if is_decoder:
            self.ln_cross = RMSNorm(embed_dim)
            self.ca_q = Linear(embed_dim, embed_dim, bias=False)
            self.ca_k = Linear(embed_dim, embed_dim, bias=False)
            self.ca_v = Linear(embed_dim, embed_dim, bias=False)
            self.ca_o = Linear(embed_dim, embed_dim, bias=False)
            self._modules.update({'ln_cross': self.ln_cross, 'ca_q': self.ca_q,
                                   'ca_k': self.ca_k, 'ca_v': self.ca_v, 'ca_o': self.ca_o})

        self.ln2 = RMSNorm(embed_dim)
        self.ffn = GeGLU(embed_dim, ffn_dim, bias=False)
        self.drop = Dropout(dropout)
        for k, v in [('ln1', self.ln1), ('sa_q', self.sa_q), ('sa_k', self.sa_k),
                     ('sa_v', self.sa_v), ('sa_o', self.sa_o), ('ln2', self.ln2), ('ffn', self.ffn)]:
            self._modules[k] = v

    def _attn(self, q_lin, k_lin, v_lin, o_lin, q_inp, kv_inp,
               mask=None, pos_bias=None):
        B, Tq, _ = q_inp._data.shape
        Tk = kv_inp._data.shape[1]
        q = q_lin(q_inp)._data.reshape(B, Tq, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k_lin(kv_inp)._data.reshape(B, Tk, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v_lin(kv_inp)._data.reshape(B, Tk, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = q @ k.swapaxes(-2, -1) * scale
        if pos_bias is not None:
            scores = scores + pos_bias
        if mask is not None:
            scores = np.where(mask, -1e9, scores)
        weights = _softmax(scores)
        out = weights @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, Tq, -1)
        return o_lin(Tensor(out))

    def forward(self, x: Tensor, encoder_out: Optional[Tensor] = None,
                self_mask=None, cross_mask=None, pos_bias=None) -> Tensor:
        # Self-attention
        sa = self._attn(self.sa_q, self.sa_k, self.sa_v, self.sa_o,
                        self.ln1(x), self.ln1(x), mask=self_mask, pos_bias=pos_bias)
        x = Tensor(x._data + self.drop(sa)._data)
        # Cross-attention (decoder only)
        if self.is_decoder and encoder_out is not None:
            ca = self._attn(self.ca_q, self.ca_k, self.ca_v, self.ca_o,
                            self.ln_cross(x), encoder_out, mask=cross_mask)
            x = Tensor(x._data + self.drop(ca)._data)
        # FFN
        ffn_out = self.drop(self.ffn(self.ln2(x)))
        return Tensor(x._data + ffn_out._data)


class T5(Module):
    """T5 (Text-To-Text Transfer Transformer) — Raffel et al. 2019.

    Encoder-decoder with relative position bias (not absolute), RMSNorm,
    GeGLU FFN (v1.1), no biases in linear layers.

    Configurations: small (60M), base (220M), large (770M), xl (3B), xxl (11B).
    """

    CONFIGS = {
        'small': dict(embed_dim=512,  num_heads=8,  num_layers=6,  ffn_dim=1024),
        'base':  dict(embed_dim=768,  num_heads=12, num_layers=12, ffn_dim=2048),
        'large': dict(embed_dim=1024, num_heads=16, num_layers=24, ffn_dim=2816),
        'xl':    dict(embed_dim=2048, num_heads=32, num_layers=24, ffn_dim=5120),
    }

    def __init__(self, vocab_size: int = 32128, embed_dim: int = 768,
                 num_heads: int = 12, num_layers: int = 12,
                 ffn_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.shared_emb = Embedding(vocab_size, embed_dim)
        self.enc_blocks = [T5Block(embed_dim, num_heads, ffn_dim, is_decoder=False, dropout=dropout)
                           for _ in range(num_layers)]
        self.dec_blocks = [T5Block(embed_dim, num_heads, ffn_dim, is_decoder=True, dropout=dropout)
                           for _ in range(num_layers)]
        self.enc_ln = RMSNorm(embed_dim)
        self.dec_ln = RMSNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)
        self.pos_bias_enc = T5RelativePositionBias(num_heads, bidirectional=True)
        self.pos_bias_dec = T5RelativePositionBias(num_heads, bidirectional=False)
        self.drop = Dropout(dropout)
        self._modules['shared_emb'] = self.shared_emb
        for i, b in enumerate(self.enc_blocks):
            self._modules[f'enc_{i}'] = b
        for i, b in enumerate(self.dec_blocks):
            self._modules[f'dec_{i}'] = b
        self._modules.update({'enc_ln': self.enc_ln, 'dec_ln': self.dec_ln,
                               'lm_head': self.lm_head,
                               'pos_bias_enc': self.pos_bias_enc,
                               'pos_bias_dec': self.pos_bias_dec})

    @classmethod
    def from_config(cls, name: str, **kwargs) -> 'T5':
        return cls(**{**cls.CONFIGS[name], **kwargs})

    def encode(self, input_ids: Tensor) -> Tensor:
        x = self.drop(self.shared_emb(input_ids))
        T = x._data.shape[1]
        pos_bias = self.pos_bias_enc(T, T)
        for block in self.enc_blocks:
            x = block(x, pos_bias=pos_bias)
        return self.enc_ln(x)

    def decode(self, decoder_ids: Tensor, encoder_out: Tensor) -> Tensor:
        x = self.drop(self.shared_emb(decoder_ids))
        Tq = x._data.shape[1]
        Tk = encoder_out._data.shape[1]
        pos_bias = self.pos_bias_dec(Tq, Tq)
        from vectorphile.nn.architectures.attention import causal_mask as _cmask
        dec_mask = _cmask(Tq)[None, None, :, :]
        for block in self.dec_blocks:
            x = block(x, encoder_out=encoder_out, self_mask=dec_mask, pos_bias=pos_bias)
        return self.dec_ln(x)

    def forward(self, input_ids: Tensor, decoder_ids: Tensor) -> Tensor:
        enc_out = self.encode(input_ids)
        dec_out = self.decode(decoder_ids, enc_out)
        return self.lm_head(dec_out)   # (B, T_dec, vocab_size)


# ── LLaMA ────────────────────────────────────────────────────────────────────

class LLaMABlock(Module):
    """LLaMA / LLaMA-2 / LLaMA-3 decoder block.

    Uses: RMSNorm, RoPE, SwiGLU, GQA (LLaMA-2+), no bias anywhere.
    """

    def __init__(self, embed_dim: int, num_q_heads: int, num_kv_heads: int,
                 ffn_dim: Optional[int] = None, dropout: float = 0.0,
                 rope: Optional[RotaryEmbedding] = None):
        super().__init__()
        self.ln1 = RMSNorm(embed_dim)
        self.ln2 = RMSNorm(embed_dim)
        self.attn = GroupedQueryAttention(embed_dim, num_q_heads, num_kv_heads,
                                           dropout=dropout, bias=False, rope=rope)
        self.ffn = SwiGLU(embed_dim, ffn_dim, bias=False)
        for k, v in [('ln1', self.ln1), ('ln2', self.ln2),
                     ('attn', self.attn), ('ffn', self.ffn)]:
            self._modules[k] = v

    def forward(self, x: Tensor, mask: Optional[np.ndarray] = None,
                kv_cache: Optional[dict] = None) -> Tensor:
        # Pre-norm + self-attention + residual
        attn_out = self.attn(self.ln1(x), mask=mask, kv_cache=kv_cache)
        x = Tensor(x._data + attn_out._data)
        # Pre-norm + FFN + residual
        ffn_out = self.ffn(self.ln2(x))
        return Tensor(x._data + ffn_out._data)


class LLaMA(Module):
    """LLaMA family (LLaMA 1/2/3) — Touvron et al. 2023.

    Decoder-only Transformer with:
    - RoPE positional embeddings (no learned pos embeddings)
    - SwiGLU FFN activation
    - RMSNorm (not LayerNorm)
    - Grouped Query Attention (GQA) — LLaMA 2 and later
    - No biases in linear layers

    Configurations: 7B, 13B, 70B parameter families.

    Args:
        vocab_size: vocabulary size (default 32000 for LLaMA 1/2, 128256 for LLaMA 3)
        embed_dim: d_model
        num_q_heads: query attention heads
        num_kv_heads: KV heads (use num_q_heads for MHA, 8 for LLaMA-2 70B GQA)
        num_layers: transformer blocks
        ffn_dim: SwiGLU hidden dim (auto-computed if None)
        max_seq_len: maximum context length
        rope_base: RoPE frequency base (10000 original, 500000 LLaMA-3)
        dropout: dropout (0.0 in production)
    """

    CONFIGS = {
        '7b':  dict(embed_dim=4096, num_q_heads=32, num_kv_heads=32, num_layers=32),
        '13b': dict(embed_dim=5120, num_q_heads=40, num_kv_heads=40, num_layers=40),
        '70b': dict(embed_dim=8192, num_q_heads=64, num_kv_heads=8,  num_layers=80),
    }

    def __init__(self, vocab_size: int = 32000, embed_dim: int = 4096,
                 num_q_heads: int = 32, num_kv_heads: int = 32,
                 num_layers: int = 32, ffn_dim: Optional[int] = None,
                 max_seq_len: int = 4096, rope_base: int = 10000,
                 dropout: float = 0.0):
        super().__init__()
        head_dim = embed_dim // num_q_heads
        self.rope = RotaryEmbedding(head_dim, max_seq_len=max_seq_len, base=rope_base)
        self.token_emb = Embedding(vocab_size, embed_dim)
        self.blocks = [LLaMABlock(embed_dim, num_q_heads, num_kv_heads,
                                   ffn_dim, dropout, self.rope)
                       for _ in range(num_layers)]
        self.norm = RMSNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)
        self._modules['token_emb'] = self.token_emb
        for i, b in enumerate(self.blocks):
            self._modules[f'block_{i}'] = b
        self._modules.update({'norm': self.norm, 'lm_head': self.lm_head})
        self.max_seq_len = max_seq_len

    @classmethod
    def from_config(cls, name: str, **kwargs) -> 'LLaMA':
        return cls(**{**cls.CONFIGS[name], **kwargs})

    def forward(self, input_ids: Tensor,
                kv_caches: Optional[List[dict]] = None) -> Tensor:
        x = self.token_emb(input_ids)
        T = x._data.shape[1]
        mask = causal_mask(T)[None, None, :, :]  # (1, 1, T, T)
        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)
        for block, cache in zip(self.blocks, kv_caches):
            x = block(x, mask=mask, kv_cache=cache)
        x = self.norm(x)
        return self.lm_head(x)   # (B, T, vocab_size)

    def generate(self, input_ids: Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: Optional[int] = 50) -> np.ndarray:
        ids = input_ids._data.copy()
        kv_caches = [{} for _ in self.blocks]
        self.eval()
        for _ in range(max_new_tokens):
            inp = Tensor(ids[:, -self.max_seq_len:])
            logits = self.forward(inp, kv_caches)._data[:, -1, :]
            logits = logits / max(temperature, 1e-8)
            if top_k is not None:
                v = np.sort(logits, axis=-1)[:, -top_k:][:, :1]
                logits = np.where(logits < v, -1e9, logits)
            probs = _softmax(logits)
            next_id = np.array([[np.random.choice(probs.shape[-1], p=probs[b]) for b in range(len(probs))]])
            ids = np.concatenate([ids, next_id.T], axis=1)
        return ids


# ── Mistral ───────────────────────────────────────────────────────────────────

class MistralBlock(Module):
    """Mistral 7B block: RMSNorm + SlidingWindowAttention (GQA) + SwiGLU."""

    def __init__(self, embed_dim: int, num_q_heads: int, num_kv_heads: int,
                 ffn_dim: Optional[int], window_size: int = 4096,
                 dropout: float = 0.0, rope: Optional[RotaryEmbedding] = None):
        super().__init__()
        from vectorphile.nn.architectures.attention import SlidingWindowAttention
        self.ln1 = RMSNorm(embed_dim)
        self.ln2 = RMSNorm(embed_dim)
        # Use GQA with sliding window
        self.attn = GroupedQueryAttention(embed_dim, num_q_heads, num_kv_heads,
                                           dropout=dropout, bias=False, rope=rope)
        self.window_size = window_size
        self.ffn = SwiGLU(embed_dim, ffn_dim, bias=False)
        for k, v in [('ln1', self.ln1), ('ln2', self.ln2),
                     ('attn', self.attn), ('ffn', self.ffn)]:
            self._modules[k] = v

    def forward(self, x: Tensor, kv_cache: Optional[dict] = None) -> Tensor:
        from vectorphile.nn.architectures.attention import sliding_window_mask, causal_mask as _cm
        T = x._data.shape[1]
        mask = (_cm(T) | sliding_window_mask(T, self.window_size))[None, None, :, :]
        attn_out = self.attn(self.ln1(x), mask=mask, kv_cache=kv_cache)
        x = Tensor(x._data + attn_out._data)
        ffn_out = self.ffn(self.ln2(x))
        return Tensor(x._data + ffn_out._data)


class Mistral(Module):
    """Mistral 7B — Jiang et al. 2023.

    Key innovations over LLaMA:
    - Sliding Window Attention (window_size=4096 by default)
    - Grouped Query Attention (GQA, num_kv_heads=8)
    - Longer effective context via rolling KV cache
    """

    def __init__(self, vocab_size: int = 32000, embed_dim: int = 4096,
                 num_q_heads: int = 32, num_kv_heads: int = 8,
                 num_layers: int = 32, ffn_dim: int = 14336,
                 window_size: int = 4096, max_seq_len: int = 32768,
                 rope_base: int = 10000, dropout: float = 0.0):
        super().__init__()
        head_dim = embed_dim // num_q_heads
        self.rope = RotaryEmbedding(head_dim, max_seq_len=max_seq_len, base=rope_base)
        self.token_emb = Embedding(vocab_size, embed_dim)
        self.blocks = [MistralBlock(embed_dim, num_q_heads, num_kv_heads,
                                     ffn_dim, window_size, dropout, self.rope)
                       for _ in range(num_layers)]
        self.norm = RMSNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)
        self._modules['token_emb'] = self.token_emb
        for i, b in enumerate(self.blocks):
            self._modules[f'block_{i}'] = b
        self._modules.update({'norm': self.norm, 'lm_head': self.lm_head})
        self.max_seq_len = max_seq_len

    def forward(self, input_ids: Tensor,
                kv_caches: Optional[List[dict]] = None) -> Tensor:
        x = self.token_emb(input_ids)
        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)
        for block, cache in zip(self.blocks, kv_caches):
            x = block(x, kv_cache=cache)
        return self.lm_head(self.norm(x))
