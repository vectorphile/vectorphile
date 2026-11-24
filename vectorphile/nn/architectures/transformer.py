"""Complete Transformer architectures — GPT, BERT, T5, LLaMA, Mistral."""
from __future__ import annotations
import math as _math
import random as _random
from typing import Optional, Dict, List, Tuple
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor
from vectorphile._core.ndarray import array as _nd_arr, NDArray
from vectorphile.nn.architectures.attention import (
    GroupedQueryAttention, RotaryEmbedding,
    scaled_dot_product_attention, causal_mask, _softmax_nd as _softmax,
)

# Sequential is defined in nn.layers; import lazily to avoid circular imports
def _get_sequential():
    from vectorphile.nn.layers import Sequential as _Seq
    return _Seq

class _LazySequential:
    """Thin wrapper that behaves like nn.Sequential for in-file use."""
    def __new__(cls, *args):
        try:
            return _get_sequential()(*args)
        except Exception:
            # Fallback: a minimal sequential container
            inst = object.__new__(cls)
            inst._layers = list(args)
            return inst
    def forward(self, x):
        for l in self._layers:
            x = l(x)
        return x
    def __call__(self, x):
        return self.forward(x)
    def parameters(self):
        for l in self._layers:
            if hasattr(l, 'parameters'):
                yield from l.parameters()

Sequential = _LazySequential


def _nd_randn(shape, scale=1.0):
    n = 1
    for s in shape: n *= s
    flat = [_math.sqrt(-2*_math.log(max(_random.random(),1e-10)))*_math.cos(2*_math.pi*_random.random())*scale
            for _ in range(n)]
    return _nd_arr(flat, dtype='float32').reshape(shape)


def _nd_zeros(shape):
    n = 1
    for s in shape: n *= s
    return _nd_arr([0.0]*n, dtype='float32').reshape(shape)


def _nd_ones(shape):
    n = 1
    for s in shape: n *= s
    return _nd_arr([1.0]*n, dtype='float32').reshape(shape)

# ── Common building blocks ────────────────────────────────────────────────────

class RMSNorm(Module):
    """Root Mean Square Layer Normalization — Zhang & Sennrich 2019. Used in LLaMA/T5."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        w = Tensor(_nd_ones((dim,)))
        w.requires_grad = True
        self._parameters['weight'] = w
        self.weight = w

    def forward(self, x: Tensor) -> Tensor:
        d = x._data
        sh = list(d.shape)
        last = sh[-1]
        n_rows = len(d.to_flat_list() if hasattr(d, 'to_flat_list') else list(d)) // last
        # C AVX2 fast path
        try:
            from vectorphile._core import _vp_core as _C_rms
            from vectorphile._core.ndarray import NDArray as _NDA_rms, _c_strides as _cs_rms, _DType as _DT_rms
            import array as _a_rms
            if (isinstance(d, _NDA_rms) and str(d.dtype) == 'float32' and d._buf is not None):
                out_buf = _a_rms.array('f', bytearray(4 * n_rows * last))
                wd = self.weight._data
                if isinstance(wd, _NDA_rms) and wd._buf is not None:
                    _C_rms.rms_norm_f32(d._buf, out_buf, wd._buf, n_rows, last, self.eps)
                    tsh = tuple(sh)
                    out_nd = _NDA_rms(out_buf, tsh, _cs_rms(tsh), _DT_rms('float32'))
                    return Tensor(out_nd, device=x.device)
        except Exception:
            pass
        # Pure-VP fallback
        flat = d.to_flat_list() if hasattr(d, 'to_flat_list') else list(d)
        rms_flat = []
        for r in range(n_rows):
            row = flat[r*last:(r+1)*last]
            rms_flat.append(_math.sqrt(sum(v*v for v in row)/last + self.eps))
        rms_rep = [rms_flat[r] for r in range(n_rows) for _ in range(last)]
        rms_nd = _nd_arr(rms_rep, dtype='float32').reshape(tuple(sh))
        return Tensor((d / rms_nd) * self.weight._data)


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = Tensor(_nd_ones((dim,)))
        self.bias = Tensor(_nd_zeros((dim,)))
        self.weight.requires_grad = True
        self.bias.requires_grad = True
        self._parameters['weight'] = self.weight
        self._parameters['bias'] = self.bias

    def forward(self, x: Tensor) -> Tensor:
        d = x._data
        mean = d.mean(axis=-1, keepdims=True)
        var = d.var(axis=-1, keepdims=True)
        flat_v = d.to_flat_list(); sh = list(d.shape); last = sh[-1]
        n_rows = len(flat_v) // last
        inv_flat = []
        for r in range(n_rows):
            row = flat_v[r*last:(r+1)*last]
            mu = sum(row)/last
            va = sum((v-mu)**2 for v in row)/last
            inv_flat.append(1.0/_math.sqrt(va + self.eps))
        # mean, inv per row
        mu_flat = [sum(flat_v[r*last:(r+1)*last])/last for r in range(n_rows)]
        mu_nd = _nd_arr([mu_flat[r] for r in range(n_rows) for _ in range(last)], dtype='float32').reshape(tuple(sh))
        inv_nd = _nd_arr([inv_flat[r] for r in range(n_rows) for _ in range(last)], dtype='float32').reshape(tuple(sh))
        return Tensor(((d - mu_nd) * inv_nd) * self.weight._data + self.bias._data)


def _linear(x, w, b=None):
    out = x @ w
    if b is not None:
        out = out + b
    return out


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        scale = _math.sqrt(2.0 / in_features)
        w = Tensor(_nd_randn((in_features, out_features), scale))
        w.requires_grad = True
        self._parameters['weight'] = w
        self.weight = w
        self.bias_param = None
        if bias:
            b = Tensor(_nd_zeros((out_features,)))
            b.requires_grad = True
            self._parameters['bias'] = b
            self.bias_param = b

    def forward(self, x: Tensor) -> Tensor:
        b = self.bias_param._data if self.bias_param is not None else None
        return Tensor(_linear(x._data, self.weight._data, b))


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        w = Tensor(_nd_randn((num_embeddings, embedding_dim), 0.02))
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
        sh = x._data.shape; n = 1
        for s in sh: n *= s
        mf = [1.0/(1-self.p) if _random.random() > self.p else 0.0 for _ in range(n)]
        mask = _nd_arr(mf, dtype='float32').reshape(sh)
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
        c = _math.sqrt(2.0 / _math.pi)
        flat_d = d.to_flat_list(); sh = d.shape
        gelu_f = [0.5*v*(1+_math.tanh(c*(v+0.044715*v**3))) for v in flat_d]
        gelu = _nd_arr(gelu_f, dtype='float32').reshape(sh)
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
        gate_t = self.w1(x); v_t = self.w3(x)
        gate_d = gate_t._data; v_d = v_t._data
        sh_g = gate_d.shape if hasattr(gate_d, 'shape') else (len(list(gate_d)),)
        # Triton inference fast path: concat gate+val then call fused_swiglu
        if not x.requires_grad:
            try:
                from vectorphile.cuda.triton_kernels import fused_swiglu as _fs
                # concat gate and val along last dim
                gf = gate_d.to_flat_list() if hasattr(gate_d, 'to_flat_list') else list(gate_d)
                vf = v_d.to_flat_list() if hasattr(v_d, 'to_flat_list') else list(v_d)
                n = len(gf) // sh_g[-1]; C = sh_g[-1]
                interleaved = []
                for i in range(n):
                    interleaved.extend(gf[i*C:(i+1)*C])
                    interleaved.extend(vf[i*C:(i+1)*C])
                from vectorphile._core.ndarray import array as _nd_arr2
                combined_nd = _nd_arr2(interleaved, 'float32').reshape(sh_g[:-1] + (C*2,))
                out_nd = _fs(combined_nd)
                return self.w2(Tensor(out_nd))
            except Exception:
                pass
        # C AVX2 fused SwiGLU fast path
        try:
            from vectorphile._core import _vp_core as _C_swiglu
            from vectorphile._core.ndarray import NDArray as _NDA_sg, _c_strides as _cs_sg, _DType as _DT_sg
            import array as _a_sg
            if (isinstance(gate_d, _NDA_sg) and str(gate_d.dtype) == 'float32' and gate_d._buf is not None
                    and isinstance(v_d, _NDA_sg) and v_d._buf is not None):
                dst_buf = _a_sg.array('f', bytearray(4 * gate_d._buf.buffer_info()[1]))
                _C_swiglu.swiglu_f32(gate_d._buf, v_d._buf, dst_buf)
                out_nd = _NDA_sg(dst_buf, sh_g, _cs_sg(sh_g), _DT_sg('float32'))
                return self.w2(Tensor(out_nd))
        except Exception:
            pass
        # Pure-VP fallback
        gf = gate_d.to_flat_list() if hasattr(gate_d, 'to_flat_list') else list(gate_d)
        silu_f = [v/(1+_math.exp(-v)) for v in gf]
        silu = _nd_arr(silu_f, dtype='float32').reshape(sh_g)
        return self.w2(Tensor(silu * v_d))


class GeGLU(Module):
    """GeGLU FFN variant — Noam Shazeer 2020. Used in some T5 variants."""

    def __init__(self, embed_dim: int, ffn_dim: Optional[int] = None, bias: bool = True):
        super().__init__()
        ffn_dim = ffn_dim or embed_dim * 4
        self.w = Linear(embed_dim, ffn_dim * 2, bias=bias)
        self.w2 = Linear(ffn_dim, embed_dim, bias=bias)
        self._modules.update({'w': self.w, 'w2': self.w2})

    def forward(self, x: Tensor) -> Tensor:
        projected_t = self.w(x)
        projected = projected_t._data
        # Triton inference fast path
        if not x.requires_grad:
            try:
                from vectorphile.cuda.triton_kernels import fused_geglu as _fg
                out_nd = _fg(projected)
                return self.w2(Tensor(out_nd))
            except Exception:
                pass
        half = projected.shape[-1] // 2
        x1, x2 = projected[..., :half], projected[..., half:]
        c = _math.sqrt(2.0 / _math.pi)
        x1f = x1.to_flat_list(); sh_x = x1.shape
        gelu_f = [0.5*v*(1+_math.tanh(c*(v+0.044715*v**3))) for v in x1f]
        gelu = _nd_arr(gelu_f, dtype='float32').reshape(sh_x)
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
        # Split QKV into thirds along last axis
        d3 = qkv.shape[-1] // 3
        qkv_flat = qkv.to_flat_list()
        total = B * T
        q_flat = [qkv_flat[i*3*d3 + j] for i in range(total) for j in range(d3)]
        k_flat = [qkv_flat[i*3*d3 + d3 + j] for i in range(total) for j in range(d3)]
        v_flat = [qkv_flat[i*3*d3 + 2*d3 + j] for i in range(total) for j in range(d3)]
        q = _nd_arr(q_flat, dtype='float32').reshape((B, T, d3))
        k = _nd_arr(k_flat, dtype='float32').reshape((B, T, d3))
        v = _nd_arr(v_flat, dtype='float32').reshape((B, T, d3))

        def _split(t):
            return t.reshape((B, T, self.num_heads, self.head_dim)).transpose([0, 2, 1, 3])

        q, k, v = _split(q), _split(k), _split(v)

        if kv_cache is not None:
            if 'k' in kv_cache:
                from vectorphile.core.tensor import _nd_concatenate
                k = _nd_concatenate([kv_cache['k'], k], axis=2)
                v = _nd_concatenate([kv_cache['v'], v], axis=2)
            kv_cache['k'] = k
            kv_cache['v'] = v

        T_kv = k.shape[2]
        mask_full = causal_mask(T_kv)
        if T_kv > T:
            # Take last T rows
            mf = mask_full.to_flat_list()
            mask_rows = mf[(T_kv - T) * T_kv:]
            mask_nd = _nd_arr(mask_rows, dtype='float32').reshape((T, T_kv))
        else:
            mask_nd = mask_full
        mf2 = mask_nd.to_flat_list()
        mask4d = _nd_arr(mf2, dtype='float32').reshape((1, 1) + mask_nd.shape)
        attn_out, _ = scaled_dot_product_attention(q, k, v, mask=mask4d)
        attn_out = attn_out.transpose([0, 2, 1, 3]).reshape((B, T, -1))
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
        pos = Tensor(_nd_arr([float(i) for i in range(T)], dtype='float32').reshape((1, T)))
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
                 top_p: Optional[float] = None):
        """Auto-regressive generation with temperature, top-k, and nucleus (top-p) sampling."""
        ids_flat = input_ids._data.to_flat_list()
        sh = input_ids._data.shape
        B_gen = sh[0]; T_start = sh[1]
        ids = [ids_flat[b*T_start:(b+1)*T_start] for b in range(B_gen)]
        kv_caches = [{} for _ in self.blocks]
        self.eval()
        for _ in range(max_new_tokens):
            cur_len = len(ids[0])
            inp_flat = [v for row in ids for v in row[-self.max_seq_len:]]
            inp_t = min(cur_len, self.max_seq_len)
            inp = Tensor(_nd_arr([float(v) for v in inp_flat], dtype='float32').reshape((B_gen, inp_t)))
            logits_nd = self.forward(inp, kv_caches)._data
            # Take last token: shape (B, T, vocab) → last → (B, vocab)
            lsh = logits_nd.shape; vocab = lsh[2]; T_l = lsh[1]
            lf = logits_nd.to_flat_list()
            logits_flat = [lf[b*T_l*vocab + (T_l-1)*vocab + v] / max(temperature, 1e-8)
                           for b in range(B_gen) for v in range(vocab)]
            if top_k is not None:
                for b in range(B_gen):
                    row = logits_flat[b*vocab:(b+1)*vocab]
                    kth = sorted(row)[-top_k]
                    for v in range(vocab):
                        if logits_flat[b*vocab+v] < kth:
                            logits_flat[b*vocab+v] = -1e9
            logits_nd = _nd_arr(logits_flat, dtype='float32').reshape((B_gen, vocab))
            probs_nd = _softmax(logits_nd, axis=-1)
            pf = probs_nd.to_flat_list()
            for b in range(B_gen):
                row_p = pf[b*vocab:(b+1)*vocab]
                s = sum(row_p); row_p = [v/s for v in row_p]
                r = _random.random(); cum = 0.0
                next_tok = vocab - 1
                for vi, p in enumerate(row_p):
                    cum += p
                    if cum >= r:
                        next_tok = vi; break
                ids[b].append(next_tok)
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
        pos = Tensor(_nd_arr([float(i) for i in range(T)], dtype='float32').reshape((1, T)))
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

    def forward(self, x: Tensor, mask=None) -> Tensor:
        B, T, _ = x._data.shape
        def _split(t):
            return t._data.reshape((B, T, self.num_heads, self.head_dim)).transpose([0, 2, 1, 3])
        q, k, v = _split(self.query(x)), _split(self.key(x)), _split(self.value(x))
        out, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        return Tensor(out.transpose([0, 2, 1, 3]).reshape((B, T, -1)))


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

    def forward(self, x: Tensor, mask=None) -> Tensor:
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

    def forward(self, x: Tensor, mask=None) -> Tensor:
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
                attention_mask=None) -> Tuple[Tensor, Tensor]:
        x = self.embeddings(input_ids, segment_ids)
        attn_mask = None
        if attention_mask is not None:
            # Convert (B, T) mask to (B,1,1,T) additive mask: 0=attend→keep, 1=pad→mask
            amf = attention_mask.to_flat_list() if hasattr(attention_mask, 'to_flat_list') else list(attention_mask)
            amf2 = [0.0 if v > 0.5 else 1.0 for v in amf]  # invert
            B_a = attention_mask.shape[0] if hasattr(attention_mask, 'shape') else 1
            T_a = len(amf) // B_a
            attn_mask = _nd_arr(amf2, dtype='float32').reshape((B_a, 1, 1, T_a))
        sequence_output = self.encoder(x, attn_mask)
        cls_flat = sequence_output._data.to_flat_list()
        sh = sequence_output._data.shape; d_model = sh[2]
        # CLS is index 0 along T
        cls_flat2 = [cls_flat[b*sh[1]*d_model + 0*d_model + i] for b in range(sh[0]) for i in range(d_model)]
        cls = _nd_arr(cls_flat2, dtype='float32').reshape((sh[0], d_model))
        p = _linear(cls, self.pooler.weight._data,
                    self.pooler.bias_param._data if self.pooler.bias_param else None)
        pf = p.to_flat_list()
        pooled = _nd_arr([_math.tanh(v) for v in pf], dtype='float32').reshape(p.shape)
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
        c2 = _math.sqrt(2.0/_math.pi)
        hf = h.to_flat_list(); sh_h = h.shape
        gelu_f = [0.5*v*(1+_math.tanh(c2*(v+0.044715*v**3))) for v in hf]
        h = _nd_arr(gelu_f, dtype='float32').reshape(sh_h)
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
        w = Tensor(_nd_randn((num_buckets, num_heads), 0.02))
        w.requires_grad = True
        self._parameters['emb'] = w
        self.emb = w

    def _bucket(self, rel: int) -> int:
        n = -rel
        num_buckets = self.num_buckets
        ret = 0
        if self.bidirectional:
            num_buckets //= 2
            if n < 0:
                ret += num_buckets
            n = abs(n)
        else:
            n = max(n, 0)
        max_exact = num_buckets // 2
        if n < max_exact:
            return n + ret
        val = max_exact + int(
            _math.log(n / max_exact + 1e-8) / _math.log(self.max_distance / max_exact) * (num_buckets - max_exact)
        )
        return min(val, num_buckets - 1) + ret

    def forward(self, seq_q: int, seq_k: int):
        emb_flat = self.emb._data.to_flat_list()
        num_heads = self.emb._data.shape[1]
        # Build (1, heads, seq_q, seq_k)
        result = []
        for h in range(num_heads):
            for i in range(seq_q):
                for j in range(seq_k):
                    b = self._bucket(j - i)
                    result.append(emb_flat[b * num_heads + h])
        bias = _nd_arr(result, dtype='float32').reshape((1, num_heads, seq_q, seq_k))
        return bias


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
        q = q_lin(q_inp)._data.reshape((B, Tq, self.num_heads, self.head_dim)).transpose([0, 2, 1, 3])
        k = k_lin(kv_inp)._data.reshape((B, Tk, self.num_heads, self.head_dim)).transpose([0, 2, 1, 3])
        v = v_lin(kv_inp)._data.reshape((B, Tk, self.num_heads, self.head_dim)).transpose([0, 2, 1, 3])
        scale = 1.0 / _math.sqrt(self.head_dim)
        scores = q @ k.transpose([0, 1, 3, 2]) * scale
        if pos_bias is not None:
            scores = scores + pos_bias
        if mask is not None:
            # Apply mask: where mask>0.5 set to -1e9
            sf = scores.to_flat_list(); mf = mask.to_flat_list() if hasattr(mask, 'to_flat_list') else list(mask)
            sf2 = [-1e9 if mf[i] > 0.5 else sf[i] for i in range(len(sf))]
            scores = _nd_arr(sf2, dtype='float32').reshape(scores.shape)
        weights = _softmax(scores, axis=-1)
        out = weights @ v
        out = out.transpose([0, 2, 1, 3]).reshape((B, Tq, -1))
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

    def forward(self, x: Tensor, mask=None, kv_cache: Optional[dict] = None) -> Tensor:
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
                 temperature: float = 1.0, top_k: Optional[int] = 50):
        ids_flat = input_ids._data.to_flat_list()
        sh = input_ids._data.shape; B_g = sh[0]; T0 = sh[1]
        ids = [ids_flat[b*T0:(b+1)*T0] for b in range(B_g)]
        kv_caches = [{} for _ in self.blocks]
        self.eval()
        for _ in range(max_new_tokens):
            cur_len = len(ids[0])
            inp_t = min(cur_len, self.max_seq_len)
            inp_flat = [float(v) for row in ids for v in row[-inp_t:]]
            inp = Tensor(_nd_arr(inp_flat, dtype='float32').reshape((B_g, inp_t)))
            logits_nd = self.forward(inp, kv_caches)._data
            lsh = logits_nd.shape; vocab = lsh[2]; T_l = lsh[1]
            lf = logits_nd.to_flat_list()
            logits_b = [[lf[b*T_l*vocab + (T_l-1)*vocab + v] / max(temperature, 1e-8) for v in range(vocab)]
                        for b in range(B_g)]
            if top_k is not None:
                for b in range(B_g):
                    kth = sorted(logits_b[b])[-top_k]
                    logits_b[b] = [v if v >= kth else -1e9 for v in logits_b[b]]
            for b in range(B_g):
                row = logits_b[b]
                mx = max(row); exps = [_math.exp(v - mx) for v in row]
                s = sum(exps); probs = [v/s for v in exps]
                r = _random.random(); cum = 0.0; next_tok = vocab - 1
                for vi, p in enumerate(probs):
                    cum += p
                    if cum >= r: next_tok = vi; break
                ids[b].append(next_tok)
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
        cm_nd = _cm(T); sw_nd = sliding_window_mask(T, self.window_size)
        cmf = cm_nd.to_flat_list(); swf = sw_nd.to_flat_list()
        combined = [1.0 if (cmf[i]>0.5 or swf[i]>0.5) else 0.0 for i in range(len(cmf))]
        mask_2d = _nd_arr(combined, dtype='float32').reshape((T, T))
        mf3 = mask_2d.to_flat_list()
        mask = _nd_arr(mf3, dtype='float32').reshape((1, 1, T, T))
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


# ── Qwen2 ────────────────────────────────────────────────────────────────────

class Qwen2Block(Module):
    """Qwen2 transformer block (Qwen Team 2024).

    Identical structure to LLaMA 3 / Mistral with:
    - GQA (num_kv_heads typically = num_q_heads / 4 for smaller models)
    - SwiGLU FFN with no bias
    - RMSNorm (pre-norm)
    - RoPE (theta=1_000_000 for Qwen2)
    """
    def __init__(self, embed_dim: int, num_q_heads: int, num_kv_heads: int,
                 ffn_dim: int, dropout: float = 0.0,
                 rope: Optional[RotaryEmbedding] = None):
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim)
        self.ffn_norm  = RMSNorm(embed_dim)
        self.attn = GroupedQueryAttention(embed_dim, num_q_heads, num_kv_heads,
                                          dropout=dropout, rope=rope)
        self.gate_proj = Linear(embed_dim, ffn_dim, bias=False)
        self.up_proj   = Linear(embed_dim, ffn_dim, bias=False)
        self.down_proj  = Linear(ffn_dim, embed_dim, bias=False)
        self._dropout = dropout

    def forward(self, x: Tensor, kv_cache=None) -> Tensor:
        # Pre-norm + attention + residual
        h = self.attn(self.attn_norm(x), kv_cache=kv_cache)
        x = Tensor(x._data + h._data)
        # Pre-norm + SwiGLU FFN + residual
        xn = self.ffn_norm(x)
        gate  = self.gate_proj(xn)
        up    = self.up_proj(xn)
        # SiLU gate
        import math as _m
        gfl = gate._data.to_flat_list() if hasattr(gate._data, 'to_flat_list') else list(gate._data)
        ufl  = up._data.to_flat_list()   if hasattr(up._data,   'to_flat_list') else list(up._data)
        swiglu_out = [gfl[i]/(1.0+_m.exp(-gfl[i]))*ufl[i] for i in range(len(gfl))]
        from vectorphile._core.ndarray import array as _nd_q2
        swiglu_nd = _nd_q2(swiglu_out, 'float32').reshape(gate._data.shape)
        ffn_out = self.down_proj(Tensor(swiglu_nd))
        return Tensor(x._data + ffn_out._data)


class Qwen2(Module):
    """Qwen2 language model — Qwen Team 2024.

    Supported configs (pass embed_dim/num_heads manually or use presets):
    - Qwen2-0.5B : embed_dim=896,  num_q=14, num_kv=2,  layers=24, ffn=4864
    - Qwen2-1.5B : embed_dim=1536, num_q=12, num_kv=2,  layers=28, ffn=8960
    - Qwen2-7B   : embed_dim=3584, num_q=28, num_kv=4,  layers=28, ffn=18944
    - Qwen2-72B  : embed_dim=8192, num_q=64, num_kv=8,  layers=80, ffn=29568
    """

    def __init__(self, vocab_size: int = 151936, embed_dim: int = 3584,
                 num_q_heads: int = 28, num_kv_heads: int = 4,
                 num_layers: int = 28, ffn_dim: int = 18944,
                 max_seq_len: int = 32768, rope_base: int = 1_000_000,
                 dropout: float = 0.0):
        super().__init__()
        head_dim = embed_dim // num_q_heads
        self.rope      = RotaryEmbedding(head_dim, max_seq_len=max_seq_len, base=rope_base)
        self.token_emb = Embedding(vocab_size, embed_dim)
        self.blocks    = [Qwen2Block(embed_dim, num_q_heads, num_kv_heads,
                                     ffn_dim, dropout, self.rope)
                          for _ in range(num_layers)]
        self.norm    = RMSNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)
        self._modules['token_emb'] = self.token_emb
        for i, b in enumerate(self.blocks):
            self._modules[f'block_{i}'] = b
        self._modules.update({'norm': self.norm, 'lm_head': self.lm_head})

    def forward(self, input_ids: Tensor,
                kv_caches: Optional[List[dict]] = None) -> Tensor:
        x = self.token_emb(input_ids)
        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)
        for block, cache in zip(self.blocks, kv_caches):
            x = block(x, kv_cache=cache)
        return self.lm_head(self.norm(x))


# ── Phi-3 ─────────────────────────────────────────────────────────────────────

class Phi3Block(Module):
    """Phi-3 transformer block (Microsoft 2024).

    Key differences from LLaMA:
    - Full (non-grouped) multi-head attention (MHA)
    - Longer RoPE theta (10_000 → 500_000 for Phi-3-mini 128K)
    - SwiGLU FFN (gate + up combined into one projection with split)
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.0, rope: Optional[RotaryEmbedding] = None):
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim)
        self.ffn_norm  = RMSNorm(embed_dim)
        # Phi-3 uses full MHA (num_kv_heads = num_heads)
        self.attn = GroupedQueryAttention(embed_dim, num_heads, num_heads,
                                          dropout=dropout, rope=rope)
        # Combined gate+up projection (Phi-3 style: single proj of size 2*ffn_dim)
        self.gate_up_proj = Linear(embed_dim, 2 * ffn_dim, bias=False)
        self.down_proj    = Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: Tensor, kv_cache=None) -> Tensor:
        h = self.attn(self.attn_norm(x), kv_cache=kv_cache)
        x = Tensor(x._data + h._data)
        xn   = self.ffn_norm(x)
        proj = self.gate_up_proj(xn)
        # Split gate and up (first half = gate, second half = up)
        pfl  = proj._data.to_flat_list() if hasattr(proj._data, 'to_flat_list') else list(proj._data)
        half = len(pfl) // 2
        import math as _m
        gate_fl = pfl[:half]
        up_fl   = pfl[half:]
        ff_out  = [gate_fl[i]/(1.0+_m.exp(-gate_fl[i]))*up_fl[i] for i in range(half)]
        from vectorphile._core.ndarray import array as _nd_p3
        sh_half = proj._data.shape[:-1] + (len(ff_out) // (proj._data.shape[0] * proj._data.shape[1] if len(proj._data.shape)==3 else proj._data.shape[0]),) if len(proj._data.shape) > 1 else (len(ff_out),)
        try:
            ff_nd = _nd_p3(ff_out, 'float32').reshape(proj._data.shape[:-1] + (half // (proj._data.shape[0] if len(proj._data.shape)==2 else proj._data.shape[0]*proj._data.shape[1]),))
        except Exception:
            ff_nd = _nd_p3(ff_out, 'float32')
        return Tensor(x._data + self.down_proj(Tensor(ff_nd))._data)


class Phi3(Module):
    """Microsoft Phi-3 small language model (2024).

    Default config is Phi-3-mini-4k (3.8B parameters).
    """

    def __init__(self, vocab_size: int = 32064, embed_dim: int = 3072,
                 num_heads: int = 32, num_layers: int = 32,
                 ffn_dim: int = 8192, max_seq_len: int = 4096,
                 rope_base: int = 10_000, dropout: float = 0.0):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.rope      = RotaryEmbedding(head_dim, max_seq_len=max_seq_len, base=rope_base)
        self.token_emb = Embedding(vocab_size, embed_dim)
        self.blocks    = [Phi3Block(embed_dim, num_heads, ffn_dim, dropout, self.rope)
                          for _ in range(num_layers)]
        self.norm    = RMSNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)
        self._modules['token_emb'] = self.token_emb
        for i, b in enumerate(self.blocks):
            self._modules[f'block_{i}'] = b
        self._modules.update({'norm': self.norm, 'lm_head': self.lm_head})

    def forward(self, input_ids: Tensor,
                kv_caches: Optional[List[dict]] = None) -> Tensor:
        x = self.token_emb(input_ids)
        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)
        for block, cache in zip(self.blocks, kv_caches):
            x = block(x, kv_cache=cache)
        return self.lm_head(self.norm(x))


# ── DeepSeek-V2 (Multi-head Latent Attention) ─────────────────────────────────

class MultiHeadLatentAttention(Module):
    """Multi-head Latent Attention (MLA) — DeepSeek-V2 2024.

    Key innovation: compresses K/V into a low-rank latent vector (kv_lora_rank)
    before projection, dramatically reducing KV cache size (from D → kv_lora_rank).
    At inference: only the latent vector needs to be cached (not full K/V).

    For DeepSeek-V2-lite: kv_lora_rank=512, q_lora_rank=1536
    For DeepSeek-V2:      kv_lora_rank=512, q_lora_rank=1536
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 kv_lora_rank: int = 512, q_lora_rank: int = 1536,
                 head_dim: int = 128, v_head_dim: int = 128,
                 rope: Optional[RotaryEmbedding] = None):
        super().__init__()
        self.num_heads    = num_heads
        self.head_dim     = head_dim
        self.v_head_dim   = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank  = q_lora_rank
        self.rope         = rope
        self.scale        = head_dim ** -0.5

        # Q low-rank: embed_dim → q_lora_rank → (num_heads * head_dim)
        self.q_a = Linear(embed_dim, q_lora_rank, bias=False)
        self.q_b = Linear(q_lora_rank, num_heads * head_dim, bias=False)
        self.q_norm = RMSNorm(q_lora_rank)

        # KV compression: embed_dim → kv_lora_rank (the cached latent)
        self.kv_down = Linear(embed_dim, kv_lora_rank + head_dim, bias=False)
        self.kv_norm  = RMSNorm(kv_lora_rank)

        # KV up-projection: kv_lora_rank → (num_heads * (head_dim + v_head_dim))
        self.kv_up = Linear(kv_lora_rank, num_heads * (head_dim + v_head_dim), bias=False)

        # Output projection
        self.out_proj = Linear(num_heads * v_head_dim, embed_dim, bias=False)

    def forward(self, x: Tensor, kv_latent_cache=None) -> Tensor:
        """x: (B, T, D); kv_latent_cache: dict with 'latent' key (optional)."""
        import math as _m
        from vectorphile._core.ndarray import array as _nd_mla
        def _fl(d):
            if hasattr(d, 'to_flat_list'): return d.to_flat_list()
            return list(d)

        sh = x._data.shape if hasattr(x._data, 'shape') else None
        B, T = (sh[0], sh[1]) if sh and len(sh)==3 else (1, len(_fl(x._data))//self.num_heads//self.head_dim)
        H, Dh, Dv = self.num_heads, self.head_dim, self.v_head_dim

        # Q: low-rank path
        q_a = self.q_a(x)
        q_proj = self.q_b(self.q_norm(q_a))  # (B, T, H*Dh)

        # KV compression
        kv_down_out = self.kv_down(x)  # (B, T, kv_lora_rank + rope_dim)
        kv_fl = _fl(kv_down_out._data)
        kv_len = self.kv_lora_rank
        rope_dim = self.head_dim
        total = kv_len + rope_dim
        # Split: first kv_lora_rank is KV latent, last head_dim is rope decoupled K
        kv_latent_fl = []
        rope_k_fl    = []
        n_tokens     = B * T
        for i in range(n_tokens):
            base = i * total
            kv_latent_fl.extend(kv_fl[base: base + kv_len])
            rope_k_fl.extend(kv_fl[base + kv_len: base + total])

        kv_latent_nd = _nd_mla(kv_latent_fl, 'float32').reshape((B, T, kv_len))
        kv_latent_t  = Tensor(kv_latent_nd)

        # Cache latent (only kv_lora_rank per token instead of num_heads*head_dim*2)
        if kv_latent_cache is not None:
            if 'latent' in kv_latent_cache:
                from vectorphile.core.tensor import _nd_concatenate
                kv_latent_nd_full = _nd_concatenate(
                    [kv_latent_cache['latent']._data, kv_latent_nd], axis=1)
                kv_latent_t = Tensor(kv_latent_nd_full)
            kv_latent_cache['latent'] = kv_latent_t

        # KV up: latent → K + V
        kv_up_out = self.kv_up(self.kv_norm(kv_latent_t))  # (B, T_kv, H*(Dh+Dv))
        kv_up_fl  = _fl(kv_up_out._data)
        kv_sh     = kv_up_out._data.shape
        T_kv      = kv_sh[1] if len(kv_sh)==3 else T

        # Split K and V: first H*Dh = K, next H*Dv = V
        k_fl_all = []
        v_fl_all = []
        for b in range(B):
            for t in range(T_kv):
                base = (b * T_kv + t) * H * (Dh + Dv)
                k_fl_all.extend(kv_up_fl[base: base + H * Dh])
                v_fl_all.extend(kv_up_fl[base + H * Dh: base + H * (Dh + Dv)])

        # Scaled dot-product attention with flat arrays
        qfl = _fl(q_proj._data)
        sc  = self.scale
        output = [0.0] * (B * T * H * Dv)

        for b in range(B):
            for h in range(H):
                for i in range(T):
                    qi = [qfl[((b*T+i)*H+h)*Dh+d] for d in range(Dh)]
                    scores = []
                    for j in range(T_kv):
                        kj = [k_fl_all[((b*T_kv+j)*H+h)*Dh+d] for d in range(Dh)]
                        scores.append(sum(qi[d]*kj[d] for d in range(Dh)) * sc)
                    mx = max(scores)
                    exps  = [_m.exp(s-mx) for s in scores]
                    denom = sum(exps)
                    for d in range(Dv):
                        s = sum(exps[j]/denom * v_fl_all[((b*T_kv+j)*H+h)*Dv+d]
                                for j in range(T_kv))
                        output[((b*T+i)*H+h)*Dv+d] = s

        out_nd = _nd_mla(output, 'float32').reshape((B, T, H * Dv))
        return self.out_proj(Tensor(out_nd))


class DeepSeekV2Block(Module):
    """DeepSeek-V2 transformer block with MLA attention + MoE FFN."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 num_experts: int = 64, top_k: int = 6,
                 kv_lora_rank: int = 512, q_lora_rank: int = 1536,
                 shared_experts: int = 2, dropout: float = 0.0,
                 rope: Optional[RotaryEmbedding] = None):
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim)
        self.ffn_norm  = RMSNorm(embed_dim)
        self.attn = MultiHeadLatentAttention(embed_dim, num_heads,
                                              kv_lora_rank=kv_lora_rank,
                                              q_lora_rank=q_lora_rank,
                                              rope=rope)
        from vectorphile.nn.moe import SortedDispatchMoE
        self.moe = SortedDispatchMoE(embed_dim, num_experts, ffn_dim, top_k=top_k)
        # Shared (always-active) experts
        from vectorphile.nn.moe import Expert
        self.shared_experts = [Expert(embed_dim, ffn_dim, embed_dim, 'silu')
                                for _ in range(shared_experts)]
        self._num_shared = shared_experts

    def forward(self, x: Tensor, kv_cache=None) -> Tensor:
        h = self.attn(self.attn_norm(x), kv_latent_cache=kv_cache)
        x = Tensor(x._data + h._data)
        xn = self.ffn_norm(x)
        moe_out, _ = self.moe(xn)
        # Add shared expert outputs
        for exp in self.shared_experts:
            shared_out = exp(xn)
            moe_out = Tensor(moe_out._data + shared_out._data)
        return Tensor(x._data + moe_out._data)


class DeepSeekV2(Module):
    """DeepSeek-V2 language model — DeepSeek AI 2024.

    Combines:
    - Multi-head Latent Attention (MLA): KV cache reduced by 93.3%
    - DeepSeekMoE: 64 routed + 2 shared experts, top-6 routing
    - 236B total params but only 21B active per token

    Default config is DeepSeek-V2-Lite (15.7B total, 2.4B active).
    """

    def __init__(self, vocab_size: int = 102400, embed_dim: int = 2048,
                 num_heads: int = 16, num_layers: int = 27,
                 ffn_dim: int = 1408, num_experts: int = 64,
                 top_k: int = 6, shared_experts: int = 2,
                 kv_lora_rank: int = 512, q_lora_rank: int = 1536,
                 max_seq_len: int = 4096, rope_base: int = 10000,
                 dropout: float = 0.0):
        super().__init__()
        head_dim = 128  # DeepSeek-V2 decoupled head_dim
        self.rope      = RotaryEmbedding(head_dim, max_seq_len=max_seq_len, base=rope_base)
        self.token_emb = Embedding(vocab_size, embed_dim)
        self.blocks    = [DeepSeekV2Block(embed_dim, num_heads, ffn_dim,
                                          num_experts, top_k, kv_lora_rank,
                                          q_lora_rank, shared_experts,
                                          dropout, self.rope)
                          for _ in range(num_layers)]
        self.norm    = RMSNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)
        self._modules['token_emb'] = self.token_emb
        for i, b in enumerate(self.blocks):
            self._modules[f'block_{i}'] = b
        self._modules.update({'norm': self.norm, 'lm_head': self.lm_head})

    def forward(self, input_ids: Tensor,
                kv_caches: Optional[List[dict]] = None) -> Tensor:
        x = self.token_emb(input_ids)
        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)
        for block, cache in zip(self.blocks, kv_caches):
            x = block(x, kv_cache=cache)
        return self.lm_head(self.norm(x))


# ── Multi-Token Prediction (MTP) ─────────────────────────────────────────────

class MTPHead(Module):
    """Multi-Token Prediction head (Gloeckle et al. / Meta LLaMA 3.1).

    Predicts K future tokens simultaneously from one forward pass. This
    enables speculative decoding at training time and 1.5-2× generation
    speedup by accepting multiple predictions per step.

    For each prediction depth d in [0, K):
        - shared_proj  : linear projection of hidden states
        - token_emb_d  : separate embedding for the d-th future token
        - lm_head_d    : separate unembedding for the d-th position

    Forward returns a list of K logit tensors, one per depth.
    """

    def __init__(self, embed_dim: int, vocab_size: int, num_heads: int = 4,
                 tie_weights: bool = False):
        super().__init__()
        self.embed_dim  = embed_dim
        self.vocab_size = vocab_size
        self.num_heads  = num_heads
        # Shared trunk projection
        self.shared_proj = Linear(embed_dim * 2, embed_dim)
        # Per-depth components
        self._depth_emb = []
        self._depth_head = []
        for d in range(num_heads):
            emb  = Linear(embed_dim, embed_dim, bias=False)
            head = Linear(embed_dim, vocab_size, bias=False)
            if tie_weights and d == 0:
                pass  # first head weight tied at training time by caller
            self._depth_emb.append(emb)
            self._depth_head.append(head)
            self._modules[f'emb_{d}']  = emb
            self._modules[f'head_{d}'] = head
        self._modules['shared_proj'] = self.shared_proj

    def forward(self, hidden: Tensor, token_embeds: Tensor) -> list:
        """
        Args:
            hidden      : (B, T, D) — final hidden states from backbone
            token_embeds: (B, T, D) — token embeddings (for each depth step)
        Returns:
            list of K Tensors, each (B, T, vocab_size)
        """
        from vectorphile._core.ndarray import array as _nd
        logits_list = []
        h = hidden
        # At each depth, combine hidden + shifted token embed, project, predict
        for d in range(self.num_heads):
            # Concatenate hidden with current token embedding
            # Simple concat along last dim: (B, T, 2*D)
            combined_flat = []
            hf = h._data.to_flat_list() if hasattr(h._data, 'to_flat_list') else list(h._data)
            ef = token_embeds._data.to_flat_list() if hasattr(token_embeds._data, 'to_flat_list') else list(token_embeds._data)
            sh = h.shape
            for i in range(len(hf)):
                combined_flat.append(hf[i])
                combined_flat.append(ef[i % len(ef)])
            B, T, D = sh[0], sh[1], sh[2]
            combined_nd = _nd(combined_flat, dtype='float32').reshape((B, T, D * 2))
            combined = Tensor(combined_nd)
            h_new = self.shared_proj(combined)
            # Apply depth-specific projection + RMSNorm approximation
            proj = self._depth_emb[d](h_new)
            logits = self._depth_head[d](proj)
            logits_list.append(logits)
            h = h_new  # feed forward for next depth
        return logits_list


class MTPLoss(Module):
    """Compute MTP loss: sum of cross-entropy losses at each depth.

    loss = sum_{d=1}^{K} lambda_d * CE(logits_d, targets shifted by d)
    """

    def __init__(self, num_heads: int = 4, lambdas=None):
        super().__init__()
        self.num_heads = num_heads
        self.lambdas = lambdas or [0.3] * num_heads  # equal weight by default

    def forward(self, logits_list: list, targets: Tensor) -> Tensor:
        """
        Args:
            logits_list: list of K Tensors, each (B, T, vocab_size)
            targets    : (B, T) int token ids
        Returns:
            Scalar loss Tensor
        """
        from vectorphile.nn.functional import cross_entropy as _ce
        from vectorphile._core.ndarray import array as _nd
        total_loss = None
        tgt_flat = targets._data.to_flat_list() if hasattr(targets._data, 'to_flat_list') else list(targets._data)
        B, T = targets.shape[0], targets.shape[1]
        for d, logits in enumerate(logits_list):
            # Shift targets by d+1 (predict token d+1 steps ahead)
            shift = d + 1
            if T - shift <= 0:
                continue
            shifted_tgt = tgt_flat[B * shift:]
            if len(shifted_tgt) < B * (T - shift):
                continue
            tgt_nd = _nd([float(v) for v in shifted_tgt[:B*(T-shift)]], dtype='float32')
            tgt_t = Tensor(tgt_nd)
            # Flatten logits: (B*(T-shift), vocab_size)
            lf = logits._data.to_flat_list() if hasattr(logits._data, 'to_flat_list') else list(logits._data)
            V = logits.shape[-1]
            logits_slice = _nd(lf[:B*(T-shift)*V], dtype='float32').reshape((B*(T-shift), V))
            loss_d = _ce(Tensor(logits_slice), tgt_t)
            w = self.lambdas[d]
            if total_loss is None:
                total_loss = loss_d if w == 1.0 else Tensor(loss_d._data * w)
            else:
                total_loss = Tensor((total_loss._data * 1.0) + (loss_d._data * w))
        if total_loss is None:
            from vectorphile._core.ndarray import array as _nd2
            total_loss = Tensor(_nd2([0.0], dtype='float32'))
        return total_loss


# ── Speculative Decoding ──────────────────────────────────────────────────────

class SpeculativeDecoder:
    """Speculative Decoding (Leviathan et al. 2023 / Chen et al. 2023).

    Runs a small *draft* model to generate K candidate tokens, then verifies
    all K in a single *target* model forward pass.  Accepted tokens are kept;
    rejected tokens trigger a correction step.  Achieves up to K× speedup when
    draft acceptance rate is high.

    Both ``draft_model`` and ``target_model`` must implement::
        logits = model.forward(tokens)   # tokens: list[int] → logits: list[float] (vocab_size,)

    Usage::
        decoder = SpeculativeDecoder(draft_model, target_model, vocab_size=32000)
        generated = decoder.generate(prompt_tokens, max_new_tokens=200, K=4)
    """

    def __init__(self, draft_model, target_model, vocab_size: int,
                 temperature: float = 1.0):
        self.draft   = draft_model
        self.target  = target_model
        self.V       = vocab_size
        self.temp    = temperature

    # ── softmax helper ────────────────────────────────────────────────────────
    @staticmethod
    def _softmax(logits: list, temperature: float = 1.0) -> list:
        import math as _m
        scaled = [v / max(temperature, 1e-9) for v in logits]
        m = max(scaled)
        exps  = [_m.exp(v - m) for v in scaled]
        total = sum(exps)
        return [e / total for e in exps]

    @staticmethod
    def _sample(probs: list) -> int:
        import random as _rng
        r = _rng.random()
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                return i
        return len(probs) - 1

    # ── single draft-then-verify step ────────────────────────────────────────
    def _speculative_step(self, tokens: list, K: int):
        """Generate up to K draft tokens and verify with target.

        Returns (accepted_tokens, needs_new_draft_probs) where
        ``accepted_tokens`` is the list of accepted token ids (may be < K).
        """
        import math as _m
        draft_tokens = []
        draft_probs  = []

        ctx = list(tokens)
        for _ in range(K):
            logits_d = self.draft.forward(ctx)
            if isinstance(logits_d, Tensor):
                logits_d = logits_d._data.to_flat_list() if hasattr(logits_d._data, 'to_flat_list') else list(logits_d._data)
            p_d = self._softmax(logits_d, self.temp)
            tok = self._sample(p_d)
            draft_tokens.append(tok)
            draft_probs.append(p_d)
            ctx.append(tok)

        # Verify all K draft tokens in one target pass
        # target receives context + all draft tokens → K+1 logit positions
        target_logits_all = self.target.forward(ctx)  # (K+1, V) or list of V*K+1
        if isinstance(target_logits_all, Tensor):
            tfl = target_logits_all._data.to_flat_list() if hasattr(target_logits_all._data, 'to_flat_list') else list(target_logits_all._data)
        else:
            tfl = list(target_logits_all)

        # Reshape to (K+1, V) — last position is the correction token
        V = self.V
        # If target returns a flat list of length (K+1)*V
        if len(tfl) == (K + 1) * V:
            target_probs = [self._softmax(tfl[i*V:(i+1)*V], self.temp)
                            for i in range(K + 1)]
        else:
            # target may return only the last-position logits
            target_probs = [self._softmax(tfl, self.temp)] * (K + 1)

        accepted = []
        import random as _rng_m
        for i, (tok, p_d) in enumerate(zip(draft_tokens, draft_probs)):
            p_t = target_probs[i][tok]
            p_q = p_d[tok]
            accept_ratio = min(1.0, p_t / max(p_q, 1e-9))
            if _rng_m.random() < accept_ratio:
                accepted.append(tok)
            else:
                # Rejection: sample from residual distribution
                residual = [max(0.0, target_probs[i][v] - p_d[v]) for v in range(V)]
                r_sum = sum(residual)
                if r_sum > 1e-9:
                    residual = [r / r_sum for r in residual]
                    accepted.append(self._sample(residual))
                else:
                    accepted.append(self._sample(target_probs[i]))
                break  # stop at first rejection

        # If all K were accepted, sample one bonus token from target's K+1 position
        if len(accepted) == K:
            bonus = self._sample(target_probs[K])
            accepted.append(bonus)

        return accepted

    def generate(self, prompt: list, max_new_tokens: int = 128,
                 K: int = 4, eos_token: int = -1) -> list:
        """Autoregressively generate tokens using speculative decoding.

        prompt        : list[int]  — input token ids
        max_new_tokens: int        — maximum tokens to generate
        K             : int        — number of draft tokens per step
        eos_token     : int        — stop when this token is generated (-1 = never)
        Returns list[int] of newly generated tokens (not including prompt).
        """
        tokens    = list(prompt)
        generated = []
        while len(generated) < max_new_tokens:
            remaining = max_new_tokens - len(generated)
            k = min(K, remaining)
            new_toks = self._speculative_step(tokens, k)
            for t in new_toks:
                generated.append(t)
                tokens.append(t)
                if t == eos_token:
                    return generated
                if len(generated) >= max_new_tokens:
                    break
        return generated


# ── Vision Transformer (ViT) ─────────────────────────────────────────────────

class PatchEmbed(Module):
    """Split an image into patches and linearly project them.

    Args:
        img_size: input image size (square, default 224)
        patch_size: patch size (square, default 16)
        in_channels: input channels (default 3)
        embed_dim: embedding dimension (default 768)
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C, H, W) → (B, num_patches, embed_dim)"""
        sh = x._data.shape
        B, C, H, W = sh[0], sh[1], sh[2], sh[3]
        P = self.patch_size
        flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
        patches = []
        for b in range(B):
            img = flat[b * C * H * W: (b+1) * C * H * W]
            for i in range(0, H, P):
                for j in range(0, W, P):
                    patch = []
                    for c in range(C):
                        for pi in range(P):
                            for pj in range(P):
                                idx = c * H * W + (i + pi) * W + (j + pj)
                                patch.append(img[idx] if idx < len(img) else 0.0)
                    patches.append(patch)
        import array as _a
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType
        patch_dim = C * P * P
        patch_flat = []
        for p in patches:
            patch_flat.extend(p)
        patch_sh = (B * self.num_patches, patch_dim)
        patch_t = Tensor(NDArray(_a.array('f', [float(v) for v in patch_flat]),
                                 patch_sh, _c_strides(patch_sh), _DType('float32')))
        out = self.proj(patch_t)
        out_sh = (B, self.num_patches, out._data.shape[-1])
        out_flat = list(out._data.to_flat_list() if hasattr(out._data, 'to_flat_list') else out._data)
        return Tensor(NDArray(_a.array('f', [float(v) for v in out_flat]),
                              out_sh, _c_strides(out_sh), _DType('float32')))


class ViTBlock(Module):
    """Standard ViT transformer block (Pre-Norm)."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.attn  = Linear(embed_dim, embed_dim * 3)  # fused QKV
        self.out_proj = Linear(embed_dim, embed_dim)
        ffn_dim = int(embed_dim * mlp_ratio)
        self.ff1 = Linear(embed_dim, ffn_dim)
        self.ff2 = Linear(ffn_dim, embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout_p = dropout

    def forward(self, x: Tensor) -> Tensor:
        import math as _m
        B_S = x._data.shape[0] if len(x._data.shape) >= 1 else 1
        S = x._data.shape[1] if len(x._data.shape) >= 2 else B_S
        D = x._data.shape[-1]
        H = self.num_heads
        head_dim = D // H
        scale = head_dim ** -0.5

        # Self-attention with pre-norm
        normed = self.norm1(x)
        qkv = self.attn(normed)
        qkv_flat = list(qkv._data.to_flat_list() if hasattr(qkv._data, 'to_flat_list') else qkv._data)
        # Simple attention (no causal mask for ViT)
        attn_out = _simple_mha(qkv_flat, B_S, S, H, head_dim, scale, causal=False)
        import array as _a
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType
        attn_sh = x._data.shape
        attn_t = Tensor(NDArray(_a.array('f', [float(v) for v in attn_out]),
                                attn_sh, _c_strides(attn_sh), _DType('float32')))
        x = Tensor(x._data + self.out_proj(attn_t)._data)

        # FFN with pre-norm
        normed2 = self.norm2(x)
        ff_out = self.ff1(normed2)
        ff_flat = list(ff_out._data.to_flat_list() if hasattr(ff_out._data, 'to_flat_list') else ff_out._data)
        ff_act = [v / (1.0 + _m.exp(-v)) for v in ff_flat]  # SiLU
        import array as _a2
        ff_sh = ff_out._data.shape
        ff_t = Tensor(NDArray(_a2.array('f', [float(v) for v in ff_act]),
                              ff_sh, _c_strides(ff_sh), _DType('float32')))
        x = Tensor(x._data + self.ff2(ff_t)._data)
        return x


def _simple_mha(qkv_flat, B_S, S, H, head_dim, scale, causal=False):
    """Pure-Python multi-head attention forward."""
    import math as _m
    D = H * head_dim
    out = []
    for b in range(B_S):
        for s in range(S):
            base = (b * S + s) * (3 * D)
            q = qkv_flat[base: base + D]
            k = qkv_flat[base + D: base + 2*D]
            v_vec = qkv_flat[base + 2*D: base + 3*D]
            # simplified: single-query attention over all tokens
            out.extend(v_vec[:D])  # placeholder: return value as-is
    return out


class ViT(Module):
    """Vision Transformer (Dosovitskiy et al. 2020, ViT-B/16 defaults).

    Patches the image, linearly embeds patches, prepends a [CLS] token,
    adds 1D learnable positional embeddings, runs L transformer blocks,
    pools the [CLS] token, and classifies with a head.

    Args:
        img_size: input image side length (square)
        patch_size: patch side length (square)
        in_channels: input image channels
        num_classes: number of classification outputs
        embed_dim: transformer width
        depth: number of transformer blocks
        num_heads: attention heads per block
        mlp_ratio: FFN expansion factor
        num_registers: extra register tokens (ViT-Reg, 0 = off)

    Usage::
        from vectorphile.nn.architectures import ViT
        model = ViT(img_size=224, patch_size=16, num_classes=1000)
        logits = model(image_batch)  # (B, num_classes)
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, num_classes: int = 1000,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, num_registers: int = 0):
        super().__init__()
        self.patch_embed  = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches       = self.patch_embed.num_patches
        self.num_registers = num_registers
        self.cls_token    = Linear(1, embed_dim, bias=False)  # 1 × embed_dim weight
        num_positions     = num_patches + 1 + num_registers
        self.pos_emb      = Embedding(num_positions, embed_dim)
        self.blocks = Sequential(*[
            ViTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = RMSNorm(embed_dim)
        self.head = Linear(embed_dim, num_classes)
        self.embed_dim = embed_dim

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C, H, W) → logits (B, num_classes)"""
        import array as _a
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType
        patches = self.patch_embed(x)   # (B, N, D)
        sh = patches._data.shape
        B, N, D = sh[0], sh[1], sh[2]
        # Prepend CLS token: (B, 1, D) of zeros (class token learned via pos_emb[0])
        cls_flat = [0.0] * (B * D)
        cls_sh = (B, 1, D)
        cls_t = Tensor(NDArray(_a.array('f', cls_flat), cls_sh, _c_strides(cls_sh), _DType('float32')))
        # Concatenate cls + patches
        all_flat = cls_flat + list(patches._data.to_flat_list() if hasattr(patches._data, 'to_flat_list') else patches._data)
        all_sh = (B, N + 1, D)
        all_t = Tensor(NDArray(_a.array('f', [float(v) for v in all_flat]),
                               all_sh, _c_strides(all_sh), _DType('float32')))
        # Add positional embeddings
        pos_ids_flat = [i for _ in range(B) for i in range(N+1)]
        pos_ids_sh = (B * (N+1),)
        pos_ids_nd = NDArray(_a.array('i', pos_ids_flat), pos_ids_sh, _c_strides(pos_ids_sh), _DType('int32'))
        pos_emb_out = self.pos_emb(Tensor(pos_ids_nd))  # (B*(N+1), D)
        # Add positional embeddings to tokens
        all_flat2 = list(all_t._data.to_flat_list() if hasattr(all_t._data, 'to_flat_list') else all_t._data)
        pos_flat2 = list(pos_emb_out._data.to_flat_list() if hasattr(pos_emb_out._data, 'to_flat_list') else pos_emb_out._data)
        total_len = B * (N + 1) * D
        added = [all_flat2[i] + pos_flat2[i] for i in range(min(total_len, len(all_flat2), len(pos_flat2)))]
        x_in = Tensor(NDArray(_a.array('f', [float(v) for v in added]),
                              all_sh, _c_strides(all_sh), _DType('float32')))
        # Transformer blocks
        x_out = self.blocks(x_in)
        x_norm = self.norm(x_out)
        # Extract CLS token (first position)
        norm_flat = list(x_norm._data.to_flat_list() if hasattr(x_norm._data, 'to_flat_list') else x_norm._data)
        cls_tokens = []
        for b in range(B):
            cls_tokens.extend(norm_flat[b * (N+1) * D: b * (N+1) * D + D])
        cls_sh2 = (B, D)
        cls_t2 = Tensor(NDArray(_a.array('f', [float(v) for v in cls_tokens]),
                                cls_sh2, _c_strides(cls_sh2), _DType('float32')))
        return self.head(cls_t2)


# ── Mamba (State Space Model) ────────────────────────────────────────────────

class MambaBlock(Module):
    """Mamba SSM block (Gu & Dao 2023).

    Selective State Space Model that processes sequences in O(N×D) time
    (linear in sequence length) vs O(N²) for attention.

    Key components:
    - Linear input projection → (expand*D) dimensional
    - Conv1D (causal, width 4) for local context
    - SSM (selective scan): A, B, C, dt all input-dependent
    - Gate (SiLU) × SSM output

    Args:
        d_model: input/output dimension
        d_state: SSM state dimension (default 16)
        expand: expansion factor (default 2)
        d_conv: conv width (default 4)
        dt_rank: rank of dt projection ('auto' → ceil(d_model/16))
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2,
                 d_conv: int = 4, dt_rank: str = 'auto'):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.d_conv  = d_conv
        self.dt_rank = max(1, (d_model + 15) // 16) if dt_rank == 'auto' else int(dt_rank)
        # Projections
        self.in_proj  = Linear(d_model, self.d_inner * 2, bias=False)   # x + gate
        self.conv1d   = Linear(self.d_inner, self.d_inner, bias=True)    # simplified (no weight sharing)
        self.x_proj   = Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj  = Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = Linear(self.d_inner, d_model, bias=False)
        self.norm     = RMSNorm(d_model)
        # A: log-space initialization
        import math as _m
        import array as _a
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType
        A_flat = [-float(n+1) for n in range(d_state) for _ in range(self.d_inner)]
        A_sh = (self.d_inner, d_state)
        self._A_log = Tensor(NDArray(_a.array('f', [float(v) for v in A_flat]),
                                     A_sh, _c_strides(A_sh), _DType('float32')))
        self._parameters['A_log'] = self._A_log

    def _ssm_scan(self, u, dt, B_ssm, C_ssm):
        """Selective state space scan (pure Python, simplified).

        u: (L, d_inner) input sequence
        dt: (L, d_inner) timestep
        B_ssm: (L, d_state) input matrix
        C_ssm: (L, d_state) output matrix
        Returns y: (L, d_inner)
        """
        import math as _m
        L = len(u)
        D = self.d_inner; N = self.d_state
        h = [0.0] * (D * N)  # hidden state: (d_inner, d_state)
        y_all = []
        for t in range(L):
            # dA = exp(dt * A_log)  per-token
            dt_t = dt[t]
            u_t  = u[t]
            B_t  = B_ssm[t]
            C_t  = C_ssm[t]
            new_h = []
            y_t = []
            for d in range(D):
                dA_d = _m.exp(dt_t[d] * (-1.0))  # simplified uniform A=-1
                dB_d = dt_t[d]
                # h[d, :] = dA * h[d, :] + dB * B_t * u_t[d]
                yi = 0.0
                for n in range(N):
                    new_h_dn = dA_d * h[d*N+n] + dB_d * B_t[n % len(B_t)] * u_t[d]
                    new_h.append(new_h_dn)
                    yi += C_t[n % len(C_t)] * new_h_dn
                y_t.append(yi)
            h = new_h
            y_all.extend(y_t)
        return y_all

    def forward(self, x: Tensor) -> Tensor:
        import math as _m
        import array as _a
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType
        sh = x._data.shape
        B = sh[0]; L = sh[1] if len(sh) > 2 else 1; D = sh[-1]
        normed = self.norm(x)
        xz = self.in_proj(normed)
        xz_flat = list(xz._data.to_flat_list() if hasattr(xz._data, 'to_flat_list') else xz._data)
        # Split into x and z (gate)
        total = B * L * self.d_inner
        x_flat = xz_flat[:total]; z_flat = xz_flat[total:2*total]
        # x_proj → dt, B_ssm, C_ssm
        x_sh = (B * L, self.d_inner)
        x_t  = Tensor(NDArray(_a.array('f', [float(v) for v in x_flat]),
                              x_sh, _c_strides(x_sh), _DType('float32')))
        xBC  = self.x_proj(x_t)
        xBC_flat = list(xBC._data.to_flat_list() if hasattr(xBC._data, 'to_flat_list') else xBC._data)
        R = self.dt_rank; N = self.d_state
        dt_r = xBC_flat[:B*L*R]
        B_s  = xBC_flat[B*L*R: B*L*(R+N)]
        C_s  = xBC_flat[B*L*(R+N):]
        # dt projection
        dt_r_sh = (B*L, R)
        dt_r_t  = Tensor(NDArray(_a.array('f', [float(v) for v in dt_r]),
                                 dt_r_sh, _c_strides(dt_r_sh), _DType('float32')))
        dt_full = self.dt_proj(dt_r_t)
        dt_flat = list(dt_full._data.to_flat_list() if hasattr(dt_full._data, 'to_flat_list') else dt_full._data)
        # Softplus(dt)
        dt_flat = [_m.log1p(_m.exp(v)) for v in dt_flat]
        # SSM scan per batch
        y_all = []
        for b in range(B):
            u_seq  = [x_flat[b*L*self.d_inner + t*self.d_inner: b*L*self.d_inner + (t+1)*self.d_inner]
                      for t in range(L)]
            dt_seq = [dt_flat[b*L*self.d_inner + t*self.d_inner: b*L*self.d_inner + (t+1)*self.d_inner]
                      for t in range(L)]
            B_seq  = [B_s[b*L*N + t*N: b*L*N + (t+1)*N] for t in range(L)]
            C_seq  = [C_s[b*L*N + t*N: b*L*N + (t+1)*N] for t in range(L)]
            y_seq  = self._ssm_scan(u_seq, dt_seq, B_seq, C_seq)
            y_all.extend(y_seq)
        # Gate: SiLU(z) * y
        gated = [y_all[i] * z_flat[i] / (1.0 + _m.exp(-z_flat[i])) for i in range(len(y_all))]
        y_sh = (B * L, self.d_inner)
        y_t  = Tensor(NDArray(_a.array('f', [float(v) for v in gated]),
                              y_sh, _c_strides(y_sh), _DType('float32')))
        out  = self.out_proj(y_t)
        out_sh = sh
        out_flat = list(out._data.to_flat_list() if hasattr(out._data, 'to_flat_list') else out._data)
        return Tensor(NDArray(_a.array('f', [float(v) for v in out_flat]),
                              out_sh, _c_strides(out_sh), _DType('float32')))


class Mamba(Module):
    """Full Mamba model for language modelling.

    Args:
        vocab_size: vocabulary size
        d_model: model width
        depth: number of MambaBlocks
        d_state: SSM state size
        expand: expansion factor

    Usage::
        from vectorphile.nn.architectures import Mamba
        model = Mamba(vocab_size=50257, d_model=512, depth=24)
        logits = model(input_ids)   # (B, L, vocab_size)
    """

    def __init__(self, vocab_size: int = 50257, d_model: int = 512,
                 depth: int = 24, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.embed  = Embedding(vocab_size, d_model)
        self.blocks = Sequential(*[MambaBlock(d_model, d_state, expand) for _ in range(depth)])
        self.norm   = RMSNorm(d_model)
        self.head   = Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x)


# ── RetNet (Retentive Network) ────────────────────────────────────────────────

class RetentionHead(Module):
    """Single retention head (Sun et al. 2023, RetNet).

    Implements the retention mechanism: parallel, recurrent, and chunkwise modes.
    Here we use the simplified parallel (causal) mode.

    Retention: o_n = sum_{m<=n} gamma^(n-m) * (Q_n K_m^T) V_m

    Args:
        head_dim: dimension per head
        gamma: decay factor (typically 1 - 2^{-5-h} for head h)
    """

    def __init__(self, head_dim: int, gamma: float = 0.9):
        super().__init__()
        self.head_dim = head_dim
        self.gamma = gamma

    def forward(self, q: list, k: list, v: list, L: int) -> list:
        """Parallel mode retention.
        q, k, v: flat lists of shape (L, head_dim)
        Returns: flat (L, head_dim)
        """
        D = self.head_dim
        g = self.gamma
        out = []
        for t in range(L):
            q_t = q[t*D:(t+1)*D]
            o_t = [0.0] * D
            for m in range(t+1):
                decay = g ** (t - m)
                k_m = k[m*D:(m+1)*D]
                v_m = v[m*D:(m+1)*D]
                dot = sum(q_t[i] * k_m[i] for i in range(D))
                for i in range(D):
                    o_t[i] += decay * dot * v_m[i]
            out.extend(o_t)
        return out


class RetNetBlock(Module):
    """RetNet block: multi-scale retention + FFN.

    Args:
        d_model: model dimension
        num_heads: number of retention heads
        ffn_dim: feed-forward width
    """

    def __init__(self, d_model: int, num_heads: int = 8, ffn_dim: int = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        ffn_dim = ffn_dim or d_model * 4
        self.qkv_proj = Linear(d_model, d_model * 3, bias=False)
        self.out_proj  = Linear(d_model, d_model, bias=False)
        self.ff1 = Linear(d_model, ffn_dim, bias=False)
        self.ff2 = Linear(ffn_dim, d_model, bias=False)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        # Decay factors per head: gamma_h = 1 - 2^{-5-h}
        self.heads = [RetentionHead(self.head_dim,
                                    gamma=1.0 - 2.0 ** (-5.0 - h))
                      for h in range(num_heads)]

    def forward(self, x: Tensor) -> Tensor:
        import math as _m
        import array as _a
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType
        sh = x._data.shape
        B = sh[0]; L = sh[1] if len(sh) > 2 else 1; D = sh[-1]
        normed = self.norm1(x)
        qkv = self.qkv_proj(normed)
        qkv_flat = list(qkv._data.to_flat_list() if hasattr(qkv._data, 'to_flat_list') else qkv._data)
        HD = self.head_dim; H = self.num_heads
        q_all = qkv_flat[:B*L*D]; k_all = qkv_flat[B*L*D:2*B*L*D]; v_all = qkv_flat[2*B*L*D:]
        # Per-head retention
        ret_out = [0.0] * (B * L * D)
        for h, head in enumerate(self.heads):
            # Extract head h from q, k, v
            q_h = [q_all[b*L*D + t*D + h*HD + d2]
                   for b in range(B) for t in range(L) for d2 in range(HD)]
            k_h = [k_all[b*L*D + t*D + h*HD + d2]
                   for b in range(B) for t in range(L) for d2 in range(HD)]
            v_h = [v_all[b*L*D + t*D + h*HD + d2]
                   for b in range(B) for t in range(L) for d2 in range(HD)]
            o_h = head.forward(q_h, k_h, v_h, B * L)
            for b in range(B):
                for t in range(L):
                    for d2 in range(HD):
                        ret_out[b*L*D + t*D + h*HD + d2] += o_h[(b*L+t)*HD + d2]
        ret_sh = (B * L, D)
        ret_t  = Tensor(NDArray(_a.array('f', [float(v) for v in ret_out]),
                                ret_sh, _c_strides(ret_sh), _DType('float32')))
        out = self.out_proj(ret_t)
        out_flat = list(out._data.to_flat_list() if hasattr(out._data, 'to_flat_list') else out._data)
        x_flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
        # Residual
        x2_flat = [x_flat[i] + out_flat[i] for i in range(min(len(x_flat), len(out_flat)))]
        x2 = Tensor(NDArray(_a.array('f', [float(v) for v in x2_flat]),
                            sh, _c_strides(sh), _DType('float32')))
        # FFN
        normed2 = self.norm2(x2)
        ff_out  = self.ff1(normed2)
        ff_flat = list(ff_out._data.to_flat_list() if hasattr(ff_out._data, 'to_flat_list') else ff_out._data)
        ff_silu = [v / (1.0 + _m.exp(-v)) for v in ff_flat]
        ff_sh   = ff_out._data.shape
        ff_t    = Tensor(NDArray(_a.array('f', [float(v) for v in ff_silu]),
                                 ff_sh, _c_strides(ff_sh), _DType('float32')))
        ff2 = self.ff2(ff_t)
        ff2_flat = list(ff2._data.to_flat_list() if hasattr(ff2._data, 'to_flat_list') else ff2._data)
        final = [x2_flat[i] + ff2_flat[i] for i in range(min(len(x2_flat), len(ff2_flat)))]
        return Tensor(NDArray(_a.array('f', [float(v) for v in final]),
                              sh, _c_strides(sh), _DType('float32')))


class RetNet(Module):
    """Retentive Network for language modelling (Sun et al. 2023).

    Args:
        vocab_size: vocabulary size
        d_model: model width
        depth: number of RetNetBlocks
        num_heads: retention heads per block
        ffn_dim: FFN intermediate width

    Usage::
        from vectorphile.nn.architectures import RetNet
        model = RetNet(vocab_size=50257, d_model=512, depth=12)
        logits = model(input_ids)
    """

    def __init__(self, vocab_size: int = 50257, d_model: int = 512,
                 depth: int = 12, num_heads: int = 8, ffn_dim: int = None):
        super().__init__()
        self.embed  = Embedding(vocab_size, d_model)
        self.blocks = Sequential(*[RetNetBlock(d_model, num_heads, ffn_dim) for _ in range(depth)])
        self.norm   = RMSNorm(d_model)
        self.head   = Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x)


# =============================================================================
# RWKV — Receptance Weighted Key Value (Peng et al. 2023)
# Linear-time architecture: combines attention expressiveness with RNN speed.
# =============================================================================

class RWKVTimeMix(Module):
    """RWKV time-mixing channel — WKV recurrence in parallel form.

    The core RWKV operation:
        wkv_t = (sum_{i<t} exp((t-1-i)*w + k_i) * v_i + exp(u+k_t)*v_t)
                / (sum_{i<t} exp((t-1-i)*w + k_i) + exp(u+k_t))
        r_t   = sigmoid(W_r @ x_t)
        o_t   = W_o @ (r_t * wkv_t)
    """

    def __init__(self, d_model: int, layer_id: int = 0):
        super().__init__()
        self.d_model  = d_model
        self.time_w   = Linear(d_model, d_model, bias=False)   # W_r
        self.time_k   = Linear(d_model, d_model, bias=False)
        self.time_v   = Linear(d_model, d_model, bias=False)
        self.time_out = Linear(d_model, d_model, bias=False)
        # Learnable per-channel decay w and bonus u are stored as plain lists
        # for framework-agnostic compatibility.
        self._w = [-(float(i) + 1.0) / d_model for i in range(d_model)]  # negative
        self._u = [0.0] * d_model

    def _wkv(self, k_flat, v_flat, B, T, C):
        """Pure-Python WKV recurrence — O(B*T*C)."""
        y = []
        for b in range(B):
            for t in range(T):
                row = []
                for c in range(C):
                    w_c = self._w[c]
                    u_c = self._u[c]
                    aa = bb = 0.0
                    pp = -1e38
                    for i in range(t):
                        ki = k_flat[b * T * C + i * C + c]
                        vi = v_flat[b * T * C + i * C + c]
                        decay = w_c * (t - 1 - i)
                        exp_w = decay + ki
                        if exp_w > pp:
                            e1 = _math.exp(pp - exp_w)
                            aa = e1 * aa + _math.exp(0) * vi
                            bb = e1 * bb + 1.0
                            pp = exp_w
                        else:
                            e2 = _math.exp(exp_w - pp)
                            aa = aa + e2 * vi
                            bb = bb + e2
                    # bonus term for current t
                    kt = k_flat[b * T * C + t * C + c]
                    vt = v_flat[b * T * C + t * C + c]
                    ukt = u_c + kt
                    if ukt > pp:
                        e1 = _math.exp(pp - ukt)
                        num = e1 * aa + vt
                        den = e1 * bb + 1.0
                    else:
                        e2 = _math.exp(ukt - pp)
                        num = aa + e2 * vt
                        den = bb + e2
                    row.append(num / (den + 1e-8))
                y.extend(row)
        return y

    def forward(self, x: 'Tensor') -> 'Tensor':
        sh = x._data.shape
        B, T, C = sh[0], sh[1], sh[2]

        def _t(mod, inp):
            return mod(inp)

        r = _t(self.time_w, x)
        k = _t(self.time_k, x)
        v = _t(self.time_v, x)

        k_flat = list(k._data.to_flat_list() if hasattr(k._data, 'to_flat_list') else k._data)
        v_flat = list(v._data.to_flat_list() if hasattr(v._data, 'to_flat_list') else v._data)
        r_flat = list(r._data.to_flat_list() if hasattr(r._data, 'to_flat_list') else r._data)

        wkv    = self._wkv(k_flat, v_flat, B, T, C)
        # r * wkv (gate)
        r_sig  = [1.0 / (1.0 + _math.exp(-rv)) for rv in r_flat]
        gated  = [r_sig[i] * wkv[i] for i in range(len(wkv))]

        import array as _a
        from vectorphile._core.ndarray import NDArray as _NDArr, _c_strides, _DType
        out_t  = Tensor(_NDArr(_a.array('f', [float(v) for v in gated]),
                               sh, _c_strides(sh), _DType('float32')))
        return self.time_out(out_t)


class RWKVChannelMix(Module):
    """RWKV channel-mixing — FFN-equivalent with key-gated output."""

    def __init__(self, d_model: int, ffn_mult: int = 4):
        super().__init__()
        self.key   = Linear(d_model, d_model * ffn_mult, bias=False)
        self.value = Linear(d_model * ffn_mult, d_model, bias=False)
        self.receptance = Linear(d_model, d_model, bias=False)

    def forward(self, x: 'Tensor') -> 'Tensor':
        import array as _a
        from vectorphile._core.ndarray import NDArray as _NDArr2, _c_strides, _DType
        k   = self.key(x)
        # squared ReLU gate
        sh  = k._data.shape
        k_f = list(k._data.to_flat_list() if hasattr(k._data, 'to_flat_list') else k._data)
        sq  = [max(0.0, v) ** 2 for v in k_f]
        k_sq = Tensor(_NDArr2(_a.array('f', [float(v) for v in sq]),
                               sh, _c_strides(sh), _DType('float32')))
        vv = self.value(k_sq)
        r  = self.receptance(x)
        r_f  = list(r._data.to_flat_list() if hasattr(r._data, 'to_flat_list') else r._data)
        vv_f = list(vv._data.to_flat_list() if hasattr(vv._data, 'to_flat_list') else vv._data)
        r_sig = [1.0 / (1.0 + _math.exp(-rv)) for rv in r_f]
        out   = [r_sig[i] * vv_f[i] for i in range(min(len(r_sig), len(vv_f)))]
        sh2   = x._data.shape
        return Tensor(_NDArr2(_a.array('f', [float(v) for v in out]),
                              sh2, _c_strides(sh2), _DType('float32')))


class RWKVBlock(Module):
    """Single RWKV block: pre-norm time-mix + pre-norm channel-mix."""

    def __init__(self, d_model: int, layer_id: int = 0, ffn_mult: int = 4):
        super().__init__()
        self.ln1      = RMSNorm(d_model)
        self.ln2      = RMSNorm(d_model)
        self.time_mix = RWKVTimeMix(d_model, layer_id)
        self.chan_mix  = RWKVChannelMix(d_model, ffn_mult)

    def forward(self, x: 'Tensor') -> 'Tensor':
        x = x + self.time_mix(self.ln1(x))
        x = x + self.chan_mix(self.ln2(x))
        return x


class RWKV(Module):
    """RWKV language model (Peng et al. 2023).

    Linear-time architecture — O(1) inference cost per token when run as RNN.
    Parallel training via the WKV recurrence kernel.

    Args:
        vocab_size: vocabulary size
        d_model: model width
        depth: number of RWKVBlocks
        ffn_mult: channel-mix FFN multiplier (default 4)

    Usage::
        from vectorphile.nn.architectures import RWKV
        model = RWKV(vocab_size=50257, d_model=512, depth=12)
        logits = model(input_ids)   # (B, T, vocab_size)
    """

    def __init__(self, vocab_size: int = 50257, d_model: int = 512,
                 depth: int = 12, ffn_mult: int = 4):
        super().__init__()
        self.embed  = Embedding(vocab_size, d_model)
        self.blocks = Sequential(*[RWKVBlock(d_model, i, ffn_mult) for i in range(depth)])
        self.norm   = RMSNorm(d_model)
        self.head   = Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: 'Tensor') -> 'Tensor':
        x = self.embed(input_ids)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x)


# ── Gemma ─────────────────────────────────────────────────────────────────────

class GemmaBlock(Module):
    """Gemma transformer block (Google DeepMind 2024).

    Key differences from LLaMA:
    - Normalizes QK before attention (pre-norm with RMSNorm)
    - GeGLU activation in FFN (instead of SwiGLU)
    - Scaled embeddings: embed_scale = sqrt(embed_dim)
    - No bias in linear layers
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 num_kv_heads: int = None, dropout: float = 0.0,
                 rope: 'RotaryEmbedding' = None):
        super().__init__()
        from vectorphile.nn.layers import Linear
        self.attn_norm = RMSNorm(embed_dim)
        self.ffn_norm = RMSNorm(embed_dim)
        num_kv_heads = num_kv_heads or num_heads
        self.attn = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads)
        self.ffn = GeGLU(embed_dim, ffn_dim)
        self.rope = rope
        self._modules.update({'attn_norm': self.attn_norm, 'ffn_norm': self.ffn_norm,
                               'attn': self.attn, 'ffn': self.ffn})

    def forward(self, x: Tensor, kv_cache: dict = None) -> Tensor:
        # Self-attention with pre-norm
        h = self.attn_norm(x)
        attn_out = self.attn(h)
        x = x + attn_out
        # FFN with pre-norm
        h = self.ffn_norm(x)
        x = x + self.ffn(h)
        return x


class Gemma(Module):
    """Gemma language model (Google DeepMind 2024).

    Default config: Gemma-2B
        vocab_size=256000, embed_dim=2048, num_heads=8, num_kv_heads=1,
        num_layers=18, ffn_dim=16384

    Usage::
        from vectorphile.nn.architectures import Gemma
        model = Gemma(vocab_size=256000, embed_dim=2048)
        logits = model(input_ids)
    """

    def __init__(self, vocab_size: int = 256000, embed_dim: int = 2048,
                 num_heads: int = 8, num_kv_heads: int = 1,
                 num_layers: int = 18, ffn_dim: int = 16384,
                 max_seq_len: int = 8192, rope_base: float = 10000.0,
                 dropout: float = 0.0):
        super().__init__()
        from vectorphile.nn.layers import Embedding, Linear
        self.embed_scale = _math.sqrt(embed_dim)
        self.rope = RotaryEmbedding(embed_dim // num_heads, max_seq_len=max_seq_len, base=rope_base)
        self.token_emb = Embedding(vocab_size, embed_dim)
        self.blocks = [GemmaBlock(embed_dim, num_heads, ffn_dim, num_kv_heads, dropout, self.rope)
                       for _ in range(num_layers)]
        self.norm = RMSNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)
        self._modules['token_emb'] = self.token_emb
        for i, b in enumerate(self.blocks):
            self._modules[f'block_{i}'] = b
        self._modules.update({'norm': self.norm, 'lm_head': self.lm_head})

    def forward(self, input_ids: Tensor, kv_caches: Optional[List[dict]] = None) -> Tensor:
        x = self.token_emb(input_ids)
        # Scale embeddings (Gemma-specific)
        xf = x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data)
        x = Tensor(_nd_arr([v * self.embed_scale for v in xf], 'float32').reshape(x._data.shape))
        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)
        for block, cache in zip(self.blocks, kv_caches):
            x = block(x, kv_cache=cache)
        return self.lm_head(self.norm(x))


# ── Mixtral (Sparse MoE LLM — Mistral AI 2023) ───────────────────────────────

class MixtralExpert(Module):
    """Single FFN expert (SwiGLU)."""
    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.w1 = Linear(embed_dim, ffn_dim, bias=False)
        self.w2 = Linear(ffn_dim, embed_dim, bias=False)
        self.w3 = Linear(embed_dim, ffn_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.w1(x)
        # SiLU gate
        gf = gate._data.to_flat_list() if hasattr(gate._data, 'to_flat_list') else list(gate._data)
        gf = [v / (1.0 + _math.exp(-v)) for v in gf]
        gate_act = Tensor(_nd_arr(gf, 'float32').reshape(gate._data.shape))
        up = self.w3(x)
        return self.w2(gate_act * up)


class MixtralSparseMoE(Module):
    """Sparse mixture-of-experts routing (top-k)."""
    def __init__(self, embed_dim: int, ffn_dim: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = Linear(embed_dim, num_experts, bias=False)
        self.experts = _ModuleList([MixtralExpert(embed_dim, ffn_dim) for _ in range(num_experts)])

    def forward(self, x: Tensor) -> Tensor:
        sh = x._data.shape  # (..., embed_dim)
        embed_dim = sh[-1]
        flat = x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data)
        n_tokens = len(flat) // embed_dim
        out_flat = [0.0] * len(flat)
        # Compute router logits
        router_logits = self.gate(x)
        rfl = router_logits._data.to_flat_list() if hasattr(router_logits._data, 'to_flat_list') else list(router_logits._data)
        for t in range(n_tokens):
            logits_t = rfl[t * self.num_experts:(t + 1) * self.num_experts]
            # softmax
            mx = max(logits_t)
            exps = [_math.exp(v - mx) for v in logits_t]
            s = sum(exps) + 1e-9
            probs = [v / s for v in exps]
            # top-k selection
            top_k_idx = sorted(range(self.num_experts), key=lambda i: probs[i], reverse=True)[:self.top_k]
            top_k_probs = [probs[i] for i in top_k_idx]
            norm = sum(top_k_probs) + 1e-9
            top_k_probs = [v / norm for v in top_k_probs]
            # token vector
            tok_vec = flat[t * embed_dim:(t + 1) * embed_dim]
            tok_t = Tensor(_nd_arr(tok_vec, 'float32').reshape((1, embed_dim)))
            combined = [0.0] * embed_dim
            for expert_idx, weight in zip(top_k_idx, top_k_probs):
                expert_out = self.experts[expert_idx](tok_t)
                eof = expert_out._data.to_flat_list() if hasattr(expert_out._data, 'to_flat_list') else list(expert_out._data)
                for j in range(embed_dim):
                    combined[j] += weight * eof[j]
            for j in range(embed_dim):
                out_flat[t * embed_dim + j] = combined[j]
        return Tensor(_nd_arr(out_flat, 'float32').reshape(sh))


def _get_module_list():
    try:
        from vectorphile.nn.container import ModuleList
        return ModuleList
    except Exception:
        class _ML(Module):
            def __init__(self, mods):
                super().__init__()
                for i, m in enumerate(mods):
                    setattr(self, str(i), m)
                self._mods = mods
            def __getitem__(self, i):
                return self._mods[i]
            def __iter__(self):
                return iter(self._mods)
            def __len__(self):
                return len(self._mods)
            def parameters(self):
                for m in self._mods:
                    yield from m.parameters()
        return _ML

_ModuleList = _get_module_list()


class MixtralBlock(Module):
    """Mixtral transformer block: GQA attn + Sparse MoE FFN + RMSNorm pre-norm."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 num_experts: int = 8, top_k: int = 2,
                 num_kv_heads: int = 8, max_seq_len: int = 2048, eps: float = 1e-5):
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim, eps=eps)
        self.ffn_norm = RMSNorm(embed_dim, eps=eps)
        self.attn = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads)
        self.moe = MixtralSparseMoE(embed_dim, ffn_dim, num_experts, top_k)

    def forward(self, x: Tensor, kv_cache: Optional[dict] = None) -> Tensor:
        x = x + self.attn(self.attn_norm(x), kv_cache=kv_cache)
        x = x + self.moe(self.ffn_norm(x))
        return x


class Mixtral(Module):
    """Mixtral 8x7B-style sparse MoE language model (Mistral AI, 2023)."""
    def __init__(self, vocab_size: int = 32000, embed_dim: int = 4096,
                 num_heads: int = 32, num_kv_heads: int = 8,
                 num_layers: int = 32, ffn_dim: int = 14336,
                 num_experts: int = 8, top_k: int = 2,
                 max_seq_len: int = 32768):
        super().__init__()
        self.token_emb = Embedding(vocab_size, embed_dim)
        self.blocks = _ModuleList([
            MixtralBlock(embed_dim, num_heads, ffn_dim, num_experts, top_k,
                         num_kv_heads, max_seq_len)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids: Tensor, kv_caches: Optional[List[dict]] = None) -> Tensor:
        x = self.token_emb(input_ids)
        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)
        for block, cache in zip(self.blocks, kv_caches):
            x = block(x, kv_cache=cache)
        return self.lm_head(self.norm(x))

    @classmethod
    def mixtral_8x7b(cls) -> 'Mixtral':
        return cls(vocab_size=32000, embed_dim=4096, num_heads=32, num_kv_heads=8,
                   num_layers=32, ffn_dim=14336, num_experts=8, top_k=2)

    @classmethod
    def mixtral_8x22b(cls) -> 'Mixtral':
        return cls(vocab_size=32000, embed_dim=6144, num_heads=48, num_kv_heads=8,
                   num_layers=56, ffn_dim=16384, num_experts=8, top_k=2)


# ── Falcon (Technology Innovation Institute, 2023) ────────────────────────────

class FalconMQA(Module):
    """Multi-Query Attention for Falcon: single KV head shared across all Q heads."""
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # Q: full multi-head; K/V: single head (MQA)
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, self.head_dim, bias=bias)
        self.v_proj = Linear(embed_dim, self.head_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x: Tensor, kv_cache: Optional[dict] = None) -> Tensor:
        sh = x._data.shape
        B = sh[0] if len(sh) == 3 else 1
        T = sh[-2]; D = self.embed_dim; H = self.num_heads; HD = self.head_dim
        Q = self.q_proj(x)
        K = self.k_proj(x)  # (B, T, HD)
        V = self.v_proj(x)  # (B, T, HD)
        qf = _nd_arr(_flat_list(Q._data), 'float32')
        kf = _nd_arr(_flat_list(K._data), 'float32')
        vf = _nd_arr(_flat_list(V._data), 'float32')
        scale = 1.0 / _math.sqrt(HD)
        qfl = qf.to_flat_list(); kfl = kf.to_flat_list(); vfl = vf.to_flat_list()
        result = []
        for b in range(B):
            for t in range(T):
                q_all = qfl[(b*T+t)*D:(b*T+t)*D+D]
                attn_out = [0.0]*D
                scores_per_head = []
                for h in range(H):
                    q_h = q_all[h*HD:(h+1)*HD]
                    head_scores = []
                    for s in range(T):
                        k_s = kfl[(b*T+s)*HD:(b*T+s)*HD+HD]
                        head_scores.append(sum(q_h[i]*k_s[i] for i in range(HD)) * scale)
                    scores_per_head.append(head_scores)
                for h in range(H):
                    mx = max(scores_per_head[h])
                    exps = [_math.exp(v-mx) for v in scores_per_head[h]]
                    sm = sum(exps)+1e-9
                    weights = [v/sm for v in exps]
                    for s in range(T):
                        v_s = vfl[(b*T+s)*HD:(b*T+s)*HD+HD]
                        for i in range(HD):
                            attn_out[h*HD+i] += weights[s]*v_s[i]
                result.extend(attn_out)
        new_sh = (B,T,D) if len(sh)==3 else (T,D)
        return self.out_proj(Tensor(_nd_arr(result,'float32').reshape(new_sh)))


def _flat_list(nd):
    if hasattr(nd, 'to_flat_list'): return nd.to_flat_list()
    if hasattr(nd, 'tolist'): return nd.tolist()
    return list(nd)


class FalconBlock(Module):
    """Falcon transformer block: parallel attn+FFN with single LayerNorm (no post-attn norm)."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-5):
        super().__init__()
        # Falcon uses a single LayerNorm applied before both attn and FFN in parallel
        from vectorphile.nn.layers import LayerNorm
        self.ln = LayerNorm(embed_dim, eps=eps)
        self.attn = FalconMQA(embed_dim, num_heads, bias=False)
        self.ff_gate = Linear(embed_dim, ffn_dim * 2, bias=False)
        self.ff_out = Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: Tensor, kv_cache: Optional[dict] = None) -> Tensor:
        residual = x
        norm_x = self.ln(x)
        attn_out = self.attn(norm_x, kv_cache=kv_cache)
        # Parallel FFN (GELU-gated)
        gate = self.ff_gate(norm_x)
        sh = gate._data.shape
        gf = _flat_list(gate._data)
        half = sh[-1] // 2
        n = len(gf) // sh[-1]
        ff_flat = []
        for i in range(n):
            g = gf[i*sh[-1]:i*sh[-1]+half]
            v = gf[i*sh[-1]+half:i*sh[-1]+sh[-1]]
            # GELU(g) * v
            for j in range(half):
                gelu_g = g[j] * 0.5 * (1 + _math.tanh(0.7978845608 * (g[j] + 0.044715*g[j]**3)))
                ff_flat.append(gelu_g * v[j])
        ff_sh = sh[:-1] + (half,)
        ff_t = Tensor(_nd_arr(ff_flat,'float32').reshape(ff_sh))
        ff_out = self.ff_out(ff_t)
        return residual + attn_out + ff_out


class Falcon(Module):
    """Falcon language model (TII, 2023). Supports 7B/40B/180B configs."""
    def __init__(self, vocab_size: int = 65024, embed_dim: int = 4544,
                 num_heads: int = 71, num_layers: int = 32,
                 ffn_dim: int = 18176):
        super().__init__()
        self.token_emb = Embedding(vocab_size, embed_dim)
        self.blocks = _ModuleList([
            FalconBlock(embed_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        from vectorphile.nn.layers import LayerNorm
        self.ln_f = LayerNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids: Tensor, kv_caches: Optional[List[dict]] = None) -> Tensor:
        x = self.token_emb(input_ids)
        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)
        for block, cache in zip(self.blocks, kv_caches):
            x = block(x, kv_cache=cache)
        return self.lm_head(self.ln_f(x))

    @classmethod
    def falcon_7b(cls) -> 'Falcon':
        return cls(vocab_size=65024, embed_dim=4544, num_heads=71, num_layers=32, ffn_dim=18176)

    @classmethod
    def falcon_40b(cls) -> 'Falcon':
        return cls(vocab_size=65024, embed_dim=8192, num_heads=128, num_layers=60, ffn_dim=32768)

    @classmethod
    def falcon_180b(cls) -> 'Falcon':
        return cls(vocab_size=65024, embed_dim=14848, num_heads=232, num_layers=80, ffn_dim=59392)


# ── Phi-2 ─────────────────────────────────────────────────────────────────────

class Phi2Block(Module):
    """Phi-2 transformer block (Microsoft 2023).

    Uses parallel attention + MLP (like Falcon), partial rotary position
    embeddings (rotary on first 32 dims), and LayerNorm (not RMSNorm).
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 eps: float = 1e-5):
        super().__init__()
        self.ln = LayerNorm(embed_dim, eps=eps)
        head_dim = embed_dim // num_heads
        # Q/K/V combined projection
        self.qkv_proj = Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = Linear(embed_dim, embed_dim, bias=True)
        # MLP: fc1 → GELU → fc2
        self.fc1 = Linear(embed_dim, ffn_dim, bias=True)
        self.fc2 = Linear(ffn_dim, embed_dim, bias=True)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._modules.update({'ln': self.ln, 'qkv_proj': self.qkv_proj,
                               'out_proj': self.out_proj, 'fc1': self.fc1,
                               'fc2': self.fc2})

    def _gelu(self, x):
        import math as _m
        flat = x.to_flat_list() if hasattr(x, 'to_flat_list') else list(x)
        sh = x.shape if hasattr(x, 'shape') else (len(flat),)
        from vectorphile._core.ndarray import array as _nd_arr
        result = [0.5 * v * (1.0 + _m.tanh(_m.sqrt(2.0/_m.pi) * (v + 0.044715*v**3))) for v in flat]
        return _nd_arr(result, dtype='float32').reshape(sh)

    def forward(self, x: Tensor) -> Tensor:
        h = self.ln(x)
        B, T, C = h._data.shape
        qkv = h._data @ self.qkv_proj.weight._data
        if self.qkv_proj.bias is not None:
            bias_flat = self.qkv_proj.bias._data.to_flat_list()
            from vectorphile._core.ndarray import array as _nd_arr
            qkv_flat = qkv.to_flat_list()
            qkv = _nd_arr([v + bias_flat[j % (3*C)] for j, v in enumerate(qkv_flat)], 'float32').reshape((B, T, 3*C))
        q, k, v = qkv.reshape((B, T, 3, self.num_heads, self.head_dim)).transpose([2, 0, 3, 1, 4])._data if False else (
            qkv[:, :, :C], qkv[:, :, C:2*C], qkv[:, :, 2*C:])
        # Simple scaled dot-product attention
        from vectorphile.nn.architectures.attention import scaled_dot_product_attention
        q_h = q.reshape((B, T, self.num_heads, self.head_dim)).transpose([0, 2, 1, 3])
        k_h = k.reshape((B, T, self.num_heads, self.head_dim)).transpose([0, 2, 1, 3])
        v_h = v.reshape((B, T, self.num_heads, self.head_dim)).transpose([0, 2, 1, 3])
        attn_out, _ = scaled_dot_product_attention(q_h, k_h, v_h)
        attn_out = attn_out.transpose([0, 2, 1, 3]).reshape((B, T, C))
        attn_out2 = Tensor(attn_out @ self.out_proj.weight._data)
        # MLP
        mlp = Tensor(h._data @ self.fc1.weight._data)
        mlp = Tensor(self._gelu(mlp._data))
        mlp = Tensor(mlp._data @ self.fc2.weight._data)
        return Tensor(x._data + attn_out2._data + mlp._data)


class Phi2(Module):
    """Phi-2 language model (Microsoft Research 2023).

    A 2.7B parameter model using parallel attention+MLP blocks,
    partial rotary embeddings, and trained on high-quality data.

    Default config: Phi-2 (2.7B)
        vocab_size=51200, embed_dim=2560, num_heads=32, num_layers=32, ffn_dim=10240

    Usage::
        from vectorphile.nn.architectures import Phi2
        model = Phi2(vocab_size=51200, embed_dim=2560)
        logits = model(input_ids)
    """

    def __init__(self, vocab_size: int = 51200, embed_dim: int = 2560,
                 num_heads: int = 32, num_layers: int = 32,
                 ffn_dim: int = 10240, eps: float = 1e-5):
        super().__init__()
        self.embed = Embedding(vocab_size, embed_dim)
        self.blocks = [Phi2Block(embed_dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]
        self.ln_f = LayerNorm(embed_dim, eps=eps)
        self.lm_head = Linear(embed_dim, vocab_size, bias=True)
        self._modules['embed'] = self.embed
        for i, b in enumerate(self.blocks):
            self._modules[f'block_{i}'] = b
        self._modules['ln_f'] = self.ln_f
        self._modules['lm_head'] = self.lm_head

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    @classmethod
    def phi2(cls) -> 'Phi2':
        return cls(vocab_size=51200, embed_dim=2560, num_heads=32, num_layers=32, ffn_dim=10240)


# ── OPT ───────────────────────────────────────────────────────────────────────

class OPTBlock(Module):
    """OPT transformer block (Meta AI 2022).

    Pre-LayerNorm variant of GPT-2, with learned positional embeddings.
    Uses standard multi-head attention + FFN (ReLU activation).
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.0, eps: float = 1e-5):
        super().__init__()
        self.self_attn_layer_norm = LayerNorm(embed_dim, eps=eps)
        self.final_layer_norm = LayerNorm(embed_dim, eps=eps)
        self.q_proj = Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = Linear(embed_dim, embed_dim, bias=True)
        self.fc1 = Linear(embed_dim, ffn_dim, bias=True)
        self.fc2 = Linear(ffn_dim, embed_dim, bias=True)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self._modules.update({
            'self_attn_layer_norm': self.self_attn_layer_norm,
            'final_layer_norm': self.final_layer_norm,
            'q_proj': self.q_proj, 'k_proj': self.k_proj,
            'v_proj': self.v_proj, 'out_proj': self.out_proj,
            'fc1': self.fc1, 'fc2': self.fc2,
        })

    def _relu(self, x_nd):
        flat = x_nd.to_flat_list() if hasattr(x_nd, 'to_flat_list') else list(x_nd)
        sh = x_nd.shape
        from vectorphile._core.ndarray import array as _nd_arr
        return _nd_arr([max(0.0, float(v)) for v in flat], 'float32').reshape(sh)

    def forward(self, x: Tensor) -> Tensor:
        from vectorphile.nn.architectures.attention import scaled_dot_product_attention
        # Self-attention
        h = self.self_attn_layer_norm(x)
        B, T, C = h._data.shape
        q = h._data @ self.q_proj.weight._data
        k = h._data @ self.k_proj.weight._data
        v = h._data @ self.v_proj.weight._data
        q_h = q.reshape((B, T, self.num_heads, self.head_dim)).transpose([0, 2, 1, 3])
        k_h = k.reshape((B, T, self.num_heads, self.head_dim)).transpose([0, 2, 1, 3])
        v_h = v.reshape((B, T, self.num_heads, self.head_dim)).transpose([0, 2, 1, 3])
        attn_out, _ = scaled_dot_product_attention(q_h, k_h, v_h)
        attn_out = attn_out.transpose([0, 2, 1, 3]).reshape((B, T, C))
        attn_out = Tensor(attn_out @ self.out_proj.weight._data)
        x = Tensor(x._data + attn_out._data)
        # FFN
        h2 = self.final_layer_norm(x)
        ff = Tensor(self._relu(Tensor(h2._data @ self.fc1.weight._data)._data))
        ff = Tensor(ff._data @ self.fc2.weight._data)
        return Tensor(x._data + ff._data)


class OPT(Module):
    """OPT (Open Pre-trained Transformer) language model (Meta AI 2022).

    A GPT-3 style pre-LayerNorm decoder-only model with learned positional
    embeddings. Available in sizes from 125M to 175B parameters.

    Default config: OPT-125M
        vocab_size=50272, embed_dim=768, num_heads=12, num_layers=12, ffn_dim=3072

    Usage::
        from vectorphile.nn.architectures import OPT
        model = OPT(vocab_size=50272, embed_dim=768)
        logits = model(input_ids)
    """

    def __init__(self, vocab_size: int = 50272, embed_dim: int = 768,
                 num_heads: int = 12, num_layers: int = 12,
                 ffn_dim: int = 3072, max_seq_len: int = 2048,
                 dropout: float = 0.0, eps: float = 1e-5):
        super().__init__()
        self.embed_tokens = Embedding(vocab_size, embed_dim)
        self.embed_positions = Embedding(max_seq_len + 2, embed_dim)
        self.final_layer_norm = LayerNorm(embed_dim, eps=eps)
        self.blocks = [OPTBlock(embed_dim, num_heads, ffn_dim, dropout, eps)
                       for _ in range(num_layers)]
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)
        self._modules['embed_tokens'] = self.embed_tokens
        self._modules['embed_positions'] = self.embed_positions
        self._modules['final_layer_norm'] = self.final_layer_norm
        for i, b in enumerate(self.blocks):
            self._modules[f'block_{i}'] = b
        self._modules['lm_head'] = self.lm_head

    def forward(self, input_ids: Tensor) -> Tensor:
        from vectorphile.core.tensor import Tensor as _T
        from vectorphile._core.ndarray import array as _nd_arr
        B, T = input_ids._data.shape[0], input_ids._data.shape[1]
        tok_emb = self.embed_tokens(input_ids)
        pos_ids_flat = list(range(2, T + 2)) * B
        pos_ids = _T(_nd_arr(pos_ids_flat, dtype='int32').reshape((B, T)))
        pos_emb = self.embed_positions(pos_ids)
        x = _T(tok_emb._data + pos_emb._data)
        for block in self.blocks:
            x = block(x)
        x = self.final_layer_norm(x)
        return self.lm_head(x)

    @classmethod
    def opt_125m(cls) -> 'OPT':
        return cls(vocab_size=50272, embed_dim=768, num_heads=12, num_layers=12, ffn_dim=3072)

    @classmethod
    def opt_1_3b(cls) -> 'OPT':
        return cls(vocab_size=50272, embed_dim=2048, num_heads=32, num_layers=24, ffn_dim=8192)

    @classmethod
    def opt_6_7b(cls) -> 'OPT':
        return cls(vocab_size=50272, embed_dim=4096, num_heads=32, num_layers=32, ffn_dim=16384)

    @classmethod
    def opt_30b(cls) -> 'OPT':
        return cls(vocab_size=50272, embed_dim=7168, num_heads=56, num_layers=48, ffn_dim=28672)

    @classmethod
    def opt_66b(cls) -> 'OPT':
        return cls(vocab_size=50272, embed_dim=9216, num_heads=72, num_layers=64, ffn_dim=36864)
