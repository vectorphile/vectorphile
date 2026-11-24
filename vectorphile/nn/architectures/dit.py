"""DiT — Diffusion Transformer (Peebles & Xie 2022).

The architecture behind Stable Diffusion 3, FLUX, and PixArt.

Key ideas
---------
- Treats the noisy latent as a sequence of patches (like ViT)
- Conditioning (timestep t, class c) injected via AdaLN-Zero:
    scale, shift = MLP(embed(t) + embed(c))
    x = scale * LayerNorm(x) + shift
- Zero-init: gate weights init to 0 so each block starts as identity
- Output: predict noise ε or (ε, Σ) for learned variance

Variants implemented
--------------------
  DiT-S/2  : depth=12, hidden=384, heads=6,  patch=2  (33M params)
  DiT-B/4  : depth=12, hidden=768, heads=12, patch=4  (130M params)
  DiT-L/4  : depth=24, hidden=1024,heads=16, patch=4  (458M params)
  DiT-XL/2 : depth=28, hidden=1152,heads=16, patch=2  (675M params)
"""
from __future__ import annotations
import math as _math
from typing import Optional, Tuple
from vectorphile._core.ndarray import (
    NDArray, zeros as _nd_zeros, ones as _nd_ones, array as _nd_array,
)
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor
from vectorphile._vp_helpers import *
from vectorphile.nn.architectures.attention import (
    GroupedQueryAttention, scaled_dot_product_attention,
)


def _hash(x):
    x = x ^ (x >> 16)
    x = (x * 0x45d9f3b) & 0xFFFFFFFF
    x = x ^ (x >> 16)
    return x

class _Lin(Module):
    def __init__(self, i, o, bias=True, zero_init=False):
        super().__init__()
        s = _math.sqrt(1.0 / i)
        self.w = Tensor(_nd_array([[(_hash(i*o+j+100)%2000)/1000.0-1 for j in range(i)] for _ in range(o)], dtype='float32') * (0.0 if zero_init else s), requires_grad=True)
        self._parameters['w'] = self.w
        if bias:
            self.b = Tensor(_nd_zeros(o, dtype='float32'), requires_grad=True)
            self._parameters['b'] = self.b
        else:
            self.b = None
    def forward(self, x: Tensor) -> Tensor:
        out = x._data @ self.w._data.transpose()
        if self.b is not None:
            out = out + self.b._data
        return Tensor(out.astype('float32'), device=x.device)

class _LN(Module):
    def __init__(self, d, eps=1e-6, affine=False):
        super().__init__()
        self.eps = eps
        if affine:
            self.w = Tensor(_nd_ones(d, dtype='float32'), requires_grad=True)
            self.b = Tensor(_nd_zeros(d, dtype='float32'), requires_grad=True)
            self._parameters['w'] = self.w; self._parameters['b'] = self.b
        else:
            self.w = None; self.b = None
    def forward(self, x: Tensor) -> Tensor:
        d = x._data
        m = d.mean(axis=-1, keepdims=True)
        v = d.var(axis=-1, keepdims=True)
        out = (d - m) / (v + self.eps)**0.5
        if self.w is not None:
            out = out * self.w._data + self.b._data
        return Tensor(out.astype('float32'), device=x.device)


# ── Sinusoidal timestep embedding ─────────────────────────────────────────────

class TimestepEmbedding(Module):
    """Sinusoidal timestep embedding → 2-layer MLP → hidden_size.

    t: (B,) integer or float timesteps
    output: (B, hidden_size)
    """
    def __init__(self, freq_dim: int = 256, hidden_size: int = 512):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp1 = _Lin(freq_dim, hidden_size)
        self.mlp2 = _Lin(hidden_size, hidden_size)
        self._modules.update({'mlp1': self.mlp1, 'mlp2': self.mlp2})

    def forward(self, t: Tensor) -> Tensor:
        d = t._data  # (B,)
        B = d.shape[0] if d.ndim > 0 else 1
        half = self.freq_dim // 2
        import array as _a
        from vectorphile._core.ndarray import NDArray as _ND
        # sinusoidal frequencies
        freqs = [_math.exp(-_math.log(10000.0) * i / (half - 1)) for i in range(half)]
        emb_buf = _a.array('f', [0.0] * (B * self.freq_dim))
        for b in range(B):
            tv = float(d[b]) if d.ndim > 0 else float(d)
            for i, f in enumerate(freqs):
                emb_buf[b * self.freq_dim + i]        = _math.sin(tv * f)
                emb_buf[b * self.freq_dim + half + i] = _math.cos(tv * f)
        emb = _ND(emb_buf, (B, self.freq_dim), (self.freq_dim, 1), 'float32')
        x = self.mlp1(Tensor(emb))
        x = Tensor(x._data / (1.0 + _math_exp(-x._data)))  # SiLU
        return self.mlp2(x)  # (B, hidden_size)


class LabelEmbedding(Module):
    """Class label embedding for conditional DiT."""
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.embedding_table = Tensor(
            _nd_array([[(_hash(i*hidden_size+j)%2000)/1000.0-1
                        for j in range(hidden_size)] for i in range(num_classes + 1)],
                      dtype='float32') * _math.sqrt(1.0/hidden_size),
            requires_grad=True)
        self._parameters['embedding_table'] = self.embedding_table

    def forward(self, labels: Tensor, train: bool = False) -> Tensor:
        ids = labels._data
        B = ids.shape[0]
        d = self.embedding_table._data.shape[-1]
        import array as _a
        from vectorphile._core.ndarray import NDArray as _ND
        buf = _a.array('f', [0.0] * (B * d))
        for b in range(B):
            c = int(ids[b])
            # classifier-free guidance: randomly replace with null class (last row)
            if train and _math.random() < self.dropout_prob:
                c = self.embedding_table._data.shape[0] - 1
            for j in range(d):
                buf[b * d + j] = float(self.embedding_table._data[c, j])
        return Tensor(_ND(buf, (B, d), (d, 1), 'float32'))


# ── AdaLN-Zero ────────────────────────────────────────────────────────────────

class AdaLNZero(Module):
    """Adaptive LayerNorm-Zero block.

    Computes 6 modulation params from conditioning vector c:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    shift/scale applied to pre-norm; gate (init=0) multiplied after block.
    """
    def __init__(self, hidden_size: int, cond_size: int):
        super().__init__()
        self.silu = lambda x: x / (1.0 + _math_exp(-x))
        self.linear = _Lin(cond_size, 6 * hidden_size, zero_init=True)
        self._modules['linear'] = self.linear

    def forward(self, c: Tensor) -> Tensor:
        """c: (B, cond_size) → (B, 6, hidden_size)."""
        h = self.silu(c._data)
        out = h @ self.linear.w._data.transpose()
        if self.linear.b is not None:
            out = out + self.linear.b._data
        B = out.shape[0]
        d = out.shape[-1] // 6
        return Tensor(out.reshape(B, 6, d).astype('float32'), device=c.device)


# ── DiT Block ─────────────────────────────────────────────────────────────────

class DiTBlock(Module):
    """Single DiT block with AdaLN-Zero conditioning.

    Flow:
        c_params = AdaLNZero(c)      # 6 params: s1,sh1,g1, s2,sh2,g2
        x = x + g1 * Attn(s1 * LN(x) + sh1)
        x = x + g2 * MLP(s2 * LN(x) + sh2)
    """
    def __init__(self, hidden_size: int, num_heads: int, cond_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln1 = _LN(hidden_size)
        self.ln2 = _LN(hidden_size)
        self.attn = GroupedQueryAttention(hidden_size, num_heads, num_heads, bias=True)
        ffn = int(hidden_size * mlp_ratio)
        self.mlp1 = _Lin(hidden_size, ffn)
        self.mlp2 = _Lin(ffn, hidden_size, zero_init=True)
        self.adaLN = AdaLNZero(hidden_size, cond_size)
        for k, v in [('ln1', self.ln1), ('ln2', self.ln2), ('attn', self.attn),
                     ('mlp1', self.mlp1), ('mlp2', self.mlp2), ('adaLN', self.adaLN)]:
            self._modules[k] = v

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        # c: (B, cond_size) → 6 modulation params each (B, hidden_size)
        params = self.adaLN(c)._data  # (B, 6, D)
        B, _, D = params.shape
        s1 = params[:, 0, :][:, None, :]; sh1 = params[:, 1, :][:, None, :]
        g1 = params[:, 2, :][:, None, :]
        s2 = params[:, 3, :][:, None, :]; sh2 = params[:, 4, :][:, None, :]
        g2 = params[:, 5, :][:, None, :]

        # Attention sub-block
        normed1 = Tensor((1 + s1) * self.ln1(x)._data + sh1)
        a = self.attn(normed1)
        x = Tensor(x._data + g1 * a._data)

        # MLP sub-block
        normed2 = Tensor((1 + s2) * self.ln2(x)._data + sh2)
        h = self.mlp1(normed2)._data
        h = h / (1.0 + _math_exp(-h))  # SiLU
        mlp_out = self.mlp2(Tensor(h))
        x = Tensor(x._data + g2 * mlp_out._data)
        return x


class FinalLayer(Module):
    """Final DiT layer: AdaLN-Zero + linear → patch output."""
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, cond_size: int):
        super().__init__()
        self.ln = _LN(hidden_size)
        self.adaLN = AdaLNZero(hidden_size, cond_size)
        out_dim = patch_size * patch_size * out_channels
        self.linear = _Lin(hidden_size, out_dim, zero_init=True)
        for k, v in [('ln', self.ln), ('adaLN', self.adaLN), ('linear', self.linear)]:
            self._modules[k] = v
    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        params = self.adaLN(c)._data
        s1 = params[:, 0, :][:, None, :]; sh1 = params[:, 1, :][:, None, :]
        normed = Tensor((1 + s1) * self.ln(x)._data + sh1)
        return self.linear(normed)


# ── DiT ───────────────────────────────────────────────────────────────────────

class DiT(Module):
    """Diffusion Transformer (Peebles & Xie 2022, Stability AI / Meta).

    Takes a noisy latent image + timestep + class label and predicts noise.
    Can also predict (noise, log_variance) for learned variance (learn_sigma=True).

    Args:
        input_size   : spatial size of the latent (default 32 → 256px with VAE-8×)
        patch_size   : patch size for patchification (2 or 4)
        in_channels  : latent channels (SD: 4)
        hidden_size  : transformer hidden dimension
        depth        : number of DiT blocks
        num_heads    : attention heads
        mlp_ratio    : FFN expansion ratio (default 4.0)
        num_classes  : conditioning classes (1000 for ImageNet)
        learn_sigma  : predict both noise and log variance (doubles out_channels)

    Configurations::
        DiT.from_config('XL/2')   # 675M params, patch=2, d=1152, 28 layers
        DiT.from_config('L/4')    # 458M params, patch=4, d=1024, 24 layers
        DiT.from_config('B/4')    # 130M params, patch=4, d=768,  12 layers
        DiT.from_config('S/2')    # 33M  params, patch=2, d=384,  12 layers
    """

    CONFIGS = {
        'XL/2': dict(patch_size=2, hidden_size=1152, depth=28, num_heads=16),
        'XL/4': dict(patch_size=4, hidden_size=1152, depth=28, num_heads=16),
        'L/4':  dict(patch_size=4, hidden_size=1024, depth=24, num_heads=16),
        'L/2':  dict(patch_size=2, hidden_size=1024, depth=24, num_heads=16),
        'B/4':  dict(patch_size=4, hidden_size=768,  depth=12, num_heads=12),
        'B/2':  dict(patch_size=2, hidden_size=768,  depth=12, num_heads=12),
        'S/4':  dict(patch_size=4, hidden_size=384,  depth=12, num_heads=6),
        'S/2':  dict(patch_size=2, hidden_size=384,  depth=12, num_heads=6),
    }

    def __init__(self, input_size=32, patch_size=2, in_channels=4,
                 hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0,
                 num_classes=1000, learn_sigma=True):
        super().__init__()
        self.patch_size   = patch_size
        self.in_channels  = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_patches  = (input_size // patch_size) ** 2
        cond_size         = hidden_size

        # Patch embedding
        patch_dim = in_channels * patch_size * patch_size
        self.x_embedder  = _Lin(patch_dim, hidden_size, bias=True)
        self.t_embedder  = TimestepEmbedding(freq_dim=256, hidden_size=hidden_size)
        self.y_embedder  = LabelEmbedding(num_classes, hidden_size)

        # Fixed sinusoidal position embedding
        import array as _a
        from vectorphile._core.ndarray import NDArray as _ND
        pos_buf = _a.array('f', [0.0] * (self.num_patches * hidden_size))
        half = hidden_size // 2
        for i in range(self.num_patches):
            for j in range(half):
                freq = _math.exp(-_math.log(10000.0) * j / max(half - 1, 1))
                pos_buf[i * hidden_size + j]        = _math.sin(i * freq)
                pos_buf[i * hidden_size + half + j] = _math.cos(i * freq)
        self.pos_embed = _ND(pos_buf, (1, self.num_patches, hidden_size),
                             (self.num_patches * hidden_size, hidden_size, 1), 'float32')

        self.blocks = [DiTBlock(hidden_size, num_heads, cond_size, mlp_ratio)
                       for _ in range(depth)]
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, cond_size)

        self._modules['x_embedder']   = self.x_embedder
        self._modules['t_embedder']   = self.t_embedder
        self._modules['y_embedder']   = self.y_embedder
        self._modules['final_layer']  = self.final_layer
        for i, b in enumerate(self.blocks):
            self._modules[f'block_{i}'] = b

    @classmethod
    def from_config(cls, name: str, **kwargs) -> 'DiT':
        cfg = cls.CONFIGS[name.upper().replace(' ', '/')]
        return cls(**{**cfg, **kwargs})

    def _patchify(self, x: NDArray) -> NDArray:
        """(B, C, H, W) → (B, num_patches, C*p*p)."""
        B, C, H, W = x.shape
        p = self.patch_size
        H_p, W_p = H // p, W // p
        patch_dim = C * p * p
        import array as _a
        from vectorphile._core.ndarray import NDArray as _ND
        buf = _a.array('f', [0.0] * (B * H_p * W_p * patch_dim))
        for b in range(B):
            for hi in range(H_p):
                for wi in range(W_p):
                    idx = hi * W_p + wi
                    offset = (b * H_p * W_p + idx) * patch_dim
                    for c in range(C):
                        for ph in range(p):
                            for pw in range(p):
                                buf[offset + c*p*p + ph*p + pw] = \
                                    float(x[b, c, hi*p+ph, wi*p+pw])
        return _ND(buf, (B, H_p*W_p, patch_dim), (H_p*W_p*patch_dim, patch_dim, 1), 'float32')

    def _unpatchify(self, x: NDArray, H: int, W: int) -> NDArray:
        """(B, num_patches, out_channels*p*p) → (B, out_channels, H, W)."""
        B, N, Cpp2 = x.shape
        p = self.patch_size
        C = Cpp2 // (p * p)
        H_p, W_p = H // p, W // p
        import array as _a
        from vectorphile._core.ndarray import NDArray as _ND
        buf = _a.array('f', [0.0] * (B * C * H * W))
        for b in range(B):
            for hi in range(H_p):
                for wi in range(W_p):
                    idx = hi * W_p + wi
                    for c in range(C):
                        for ph in range(p):
                            for pw in range(p):
                                val = float(x[b, idx, c*p*p + ph*p + pw])
                                buf[((b * C + c) * H + hi*p+ph) * W + wi*p+pw] = val
        return _ND(buf, (B, C, H, W), (C*H*W, H*W, W, 1), 'float32')

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        """
        Args:
            x : (B, C, H, W)  noisy latent
            t : (B,)           diffusion timestep
            y : (B,)           class label (int)
        Returns:
            out : (B, out_channels, H, W) predicted noise (or noise + log_var)
        """
        d_in = x._data
        B, C, H, W = d_in.shape

        # 1. Patchify + embed
        patches = self._patchify(d_in)       # (B, N, patch_dim)
        xe = self.x_embedder(Tensor(patches)) # (B, N, hidden)
        xe = Tensor(xe._data + self.pos_embed)

        # 2. Conditioning
        te = self.t_embedder(t)   # (B, hidden)
        ye = self.y_embedder(y)   # (B, hidden)
        c  = Tensor(te._data + ye._data)   # (B, hidden)

        # 3. Transformer blocks
        for block in self.blocks:
            xe = block(xe, c)

        # 4. Final layer → (B, N, out_channels*p*p)
        out_patches = self.final_layer(xe, c)
        # 5. Unpatchify
        out_nd = self._unpatchify(out_patches._data, H, W)
        return Tensor(out_nd, device=x.device)
