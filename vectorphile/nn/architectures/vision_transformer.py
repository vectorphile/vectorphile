"""Vision Transformers — ViT (Dosovitskiy et al. 2020) and DeiT-style distillation."""
from __future__ import annotations
import math as _math
import random as _random
from typing import Optional, Tuple
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor
from vectorphile._core.ndarray import array as _nd_arr, NDArray


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


def _softmax(x: NDArray, axis: int = -1) -> NDArray:
    flat = x.to_flat_list(); sh = list(x.shape)
    ax = axis % len(sh); ax_size = sh[ax]
    n_outer = 1
    for s in sh[:ax]: n_outer *= s
    n_inner = 1
    for s in sh[ax+1:]: n_inner *= s
    result = flat[:]
    for outer in range(n_outer):
        for inner in range(n_inner):
            row = [flat[outer*ax_size*n_inner + k*n_inner + inner] for k in range(ax_size)]
            mx = max(row); exps = [_math.exp(v-mx) for v in row]; s = sum(exps)+1e-9
            for k in range(ax_size):
                result[outer*ax_size*n_inner + k*n_inner + inner] = exps[k]/s
    return _nd_arr(result, dtype='float32').reshape(tuple(sh))


def _layer_norm(x: NDArray, w, b, eps: float = 1e-6) -> NDArray:
    flat = x.to_flat_list(); sh = list(x.shape); last = sh[-1]
    n_rows = len(flat)//last; result = []
    wf = w.to_flat_list() if hasattr(w,'to_flat_list') else list(w)
    bf = b.to_flat_list() if hasattr(b,'to_flat_list') else list(b)
    for r in range(n_rows):
        row = flat[r*last:(r+1)*last]
        mu = sum(row)/last; va = sum((v-mu)**2 for v in row)/last
        inv = 1.0/_math.sqrt(va+eps)
        result.extend([(v-mu)*inv*wf[i]+bf[i] for i,v in enumerate(row)])
    return _nd_arr(result, dtype='float32').reshape(tuple(sh))


def _gelu(x: NDArray) -> NDArray:
    c = _math.sqrt(2.0/_math.pi)
    flat = x.to_flat_list()
    return _nd_arr([0.5*v*(1+_math.tanh(c*(v+0.044715*v**3))) for v in flat], dtype='float32').reshape(x.shape)


# ── Patch embedding ────────────────────────────────────────────────────────────

class PatchEmbedding(Module):
    """Splits image into non-overlapping patches and linearly projects each.

    Args:
        image_size: input spatial size (H = W assumed)
        patch_size: patch spatial size
        in_channels: image channels
        embed_dim: projected embedding dimension
    """

    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        # Linear projection of flattened patches
        self.proj_w = Tensor(_nd_randn((self.patch_dim, embed_dim), _math.sqrt(2.0 / self.patch_dim)))
        self.proj_b = Tensor(_nd_zeros((embed_dim,)))
        self.proj_w.requires_grad = True
        self.proj_b.requires_grad = True
        self._parameters['proj_w'] = self.proj_w
        self._parameters['proj_b'] = self.proj_b

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            patches: (B, num_patches, embed_dim)
        """
        B, C, H, W = x._data.shape
        p = self.patch_size
        # Reshape into patches: (B, C, H/p, p, W/p, p) → (B, num_patches, C*p*p)
        h_patches = H // p
        w_patches = W // p
        d = x._data.reshape(B, C, h_patches, p, w_patches, p)
        d = d.transpose([0, 2, 4, 1, 3, 5])           # (B, h, w, C, p, p)
        d = d.reshape(B, h_patches * w_patches, -1)  # (B, num_patches, C*p*p)
        return Tensor(d @ self.proj_w._data + self.proj_b._data)


# ── Multi-head self-attention ──────────────────────────────────────────────────

class ViTAttention(Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale = 1.0 / _math.sqrt(self.head_dim)
        self.dropout = dropout
        s = 0.02
        for name in ['w_q', 'w_k', 'w_v', 'w_o']:
            p = Tensor(_nd_randn((embed_dim, embed_dim), s))
            p.requires_grad = True
            setattr(self, name, p)
            self._parameters[name] = p
        for name in ['b_q', 'b_k', 'b_v', 'b_o']:
            p = Tensor(_nd_zeros((embed_dim,)))
            p.requires_grad = True
            setattr(self, name, p)
            self._parameters[name] = p

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x._data.shape
        H, D = self.num_heads, self.head_dim
        def proj(d, w, b): return (d @ w + b).reshape((B, T, H, D)).transpose([0, 2, 1, 3])
        q = proj(x._data, self.w_q._data, self.b_q._data)
        k = proj(x._data, self.w_k._data, self.b_k._data)
        v = proj(x._data, self.w_v._data, self.b_v._data)
        scores = q @ k.swapaxes(-2, -1) * self.scale
        attn = _softmax(scores)
        if self.dropout > 0 and self.training:
            mf = [1.0 if _random.random() < (1 - self.dropout) else 0.0 for _ in range(1)]
            # apply dropout mask per element
            af = attn.to_flat_list()
            af = [v * (1.0 if _random.random() < (1 - self.dropout) else 0.0) / (1 - self.dropout) for v in af]
            from vectorphile._core.ndarray import array as _nd_arr2
            attn = _nd_arr2(af, dtype='float32').reshape(attn.shape)
        out = (attn @ v).transpose([0, 2, 1, 3]).reshape((B, T, C))
        return Tensor(out @ self.w_o._data + self.b_o._data)


# ── MLP block ─────────────────────────────────────────────────────────────────

class ViTMLP(Module):
    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        s = _math.sqrt(2.0 / embed_dim)
        self.fc1_w = Tensor(_nd_randn((embed_dim, mlp_dim), s))
        self.fc1_b = Tensor(_nd_zeros((mlp_dim,)))
        self.fc2_w = Tensor(_nd_randn((mlp_dim, embed_dim), s))
        self.fc2_b = Tensor(_nd_zeros((embed_dim,)))
        for n, p in [('fc1_w',self.fc1_w),('fc1_b',self.fc1_b),
                     ('fc2_w',self.fc2_w),('fc2_b',self.fc2_b)]:
            p.requires_grad = True
            self._parameters[n] = p

    def forward(self, x: Tensor) -> Tensor:
        d = _gelu(x._data @ self.fc1_w._data + self.fc1_b._data)
        if self.dropout > 0 and self.training:
            df = d.to_flat_list()
            df = [v * (1.0 if _random.random() < (1 - self.dropout) else 0.0) / (1 - self.dropout) for v in df]
            d = _nd_arr(df, dtype='float32').reshape(d.shape)
        return Tensor(d @ self.fc2_w._data + self.fc2_b._data)


# ── Transformer encoder block ──────────────────────────────────────────────────

class ViTBlock(Module):
    """ViT transformer block: pre-norm with LN → MHSA and LN → MLP."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.attn = ViTAttention(embed_dim, num_heads, dropout)
        self.mlp  = ViTMLP(embed_dim, mlp_dim, dropout)
        self._modules.update({'attn': self.attn, 'mlp': self.mlp})
        for suffix in ['1', '2']:
            w = Tensor(_nd_ones((embed_dim,)))
            b = Tensor(_nd_zeros((embed_dim,)))
            w.requires_grad = True
            b.requires_grad = True
            setattr(self, f'ln{suffix}_w', w)
            setattr(self, f'ln{suffix}_b', b)
            self._parameters[f'ln{suffix}_w'] = w
            self._parameters[f'ln{suffix}_b'] = b

    def forward(self, x: Tensor) -> Tensor:
        d = x._data
        # Pre-norm + attention + residual
        d = d + self.attn(Tensor(_layer_norm(d, self.ln1_w._data, self.ln1_b._data)))._data
        # Pre-norm + MLP + residual
        d = d + self.mlp(Tensor(_layer_norm(d, self.ln2_w._data, self.ln2_b._data)))._data
        return Tensor(d)


# ── ViT ───────────────────────────────────────────────────────────────────────

class ViT(Module):
    """Vision Transformer (ViT) — Dosovitskiy et al. 2020.

    "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale"

    Args:
        image_size: input image spatial size (H = W)
        patch_size: patch spatial size (e.g. 16 or 32)
        in_channels: input channels
        num_classes: classification head output size (None = return embeddings)
        embed_dim: transformer embedding dimension
        num_layers: number of transformer blocks
        num_heads: attention heads
        mlp_dim: FFN hidden dim (typically 4× embed_dim)
        dropout: attention and FFN dropout
        representation_size: optional MLP head hidden size (ViT-H style)

    Configurations:
        ViT-S/16: embed=384, layers=12, heads=6
        ViT-B/16: embed=768, layers=12, heads=12
        ViT-L/16: embed=1024, layers=24, heads=16
        ViT-H/14: embed=1280, layers=32, heads=16
    """

    CONFIGS = {
        'tiny':   dict(embed_dim=192,  num_layers=12, num_heads=3,  mlp_dim=768),
        'small':  dict(embed_dim=384,  num_layers=12, num_heads=6,  mlp_dim=1536),
        'base':   dict(embed_dim=768,  num_layers=12, num_heads=12, mlp_dim=3072),
        'large':  dict(embed_dim=1024, num_layers=24, num_heads=16, mlp_dim=4096),
        'huge':   dict(embed_dim=1280, num_layers=32, num_heads=16, mlp_dim=5120),
    }

    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, num_classes: int = 1000,
                 embed_dim: int = 768, num_layers: int = 12,
                 num_heads: int = 12, mlp_dim: int = 3072,
                 dropout: float = 0.1, representation_size: Optional[int] = None):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self._modules['patch_embed'] = self.patch_embed
        num_patches = self.patch_embed.num_patches

        # [CLS] token
        self.cls_token = Tensor(_nd_zeros((1, 1, embed_dim)))
        self.cls_token.requires_grad = True
        self._parameters['cls_token'] = self.cls_token

        # Learned position embeddings: (1, num_patches+1, embed_dim)
        self.pos_embed = Tensor(_nd_randn((1, num_patches + 1, embed_dim), 0.02))
        self.pos_embed.requires_grad = True
        self._parameters['pos_embed'] = self.pos_embed

        self.dropout_rate = dropout

        # Transformer blocks
        self.blocks = []
        for i in range(num_layers):
            block = ViTBlock(embed_dim, num_heads, mlp_dim, dropout)
            self.blocks.append(block)
            self._modules[f'block_{i}'] = block

        # Final layer norm
        self.ln_w = Tensor(_nd_ones((embed_dim,)))
        self.ln_b = Tensor(_nd_zeros((embed_dim,)))
        self.ln_w.requires_grad = True
        self.ln_b.requires_grad = True
        self._parameters['ln_w'] = self.ln_w
        self._parameters['ln_b'] = self.ln_b

        # Classification head
        if representation_size is not None:
            self.pre_logits_w = Tensor(_nd_randn((embed_dim, representation_size), 0.02))
            self.pre_logits_b = Tensor(_nd_zeros((representation_size,)))
            self.pre_logits_w.requires_grad = True
            self.pre_logits_b.requires_grad = True
            self._parameters['pre_logits_w'] = self.pre_logits_w
            self._parameters['pre_logits_b'] = self.pre_logits_b
            head_in = representation_size
        else:
            head_in = embed_dim

        if num_classes is not None:
            self.head_w = Tensor(_nd_zeros((head_in, num_classes)))
            self.head_b = Tensor(_nd_zeros((num_classes,)))
            self.head_w.requires_grad = True
            self.head_b.requires_grad = True
            self._parameters['head_w'] = self.head_w
            self._parameters['head_b'] = self.head_b

    @classmethod
    def from_config(cls, name: str, patch_size: int = 16, **kwargs) -> 'ViT':
        return cls(patch_size=patch_size, **{**cls.CONFIGS[name], **kwargs})

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, num_classes) logits or (B, embed_dim) class token embeddings
        """
        B = x._data.shape[0]
        patches = self.patch_embed(x)._data   # (B, num_patches, embed_dim)

        # Prepend [CLS] token — broadcast cls_token (1,1,E) → (B,1,E) via flat list repeat
        E = self.cls_token._data.shape[-1]
        cls_flat = self.cls_token._data.to_flat_list()  # length E
        cls_rep = cls_flat * B
        from vectorphile._core.ndarray import array as _nd_arr_vit
        cls_nd = _nd_arr_vit(cls_rep, dtype='float32').reshape((B, 1, E))
        from vectorphile.core.tensor import _nd_concatenate as _vit_cat
        d = _vit_cat([cls_nd, patches], axis=1)   # (B, 1+num_patches, E)

        # Add positional embeddings
        d = d + self.pos_embed._data

        # Dropout
        if self.dropout_rate > 0 and self.training:
            df = d.to_flat_list()
            df = [v * (1.0 if _random.random() < (1 - self.dropout_rate) else 0.0) / (1 - self.dropout_rate) for v in df]
            d = _nd_arr_vit(df, dtype='float32').reshape(d.shape)

        # Transformer blocks
        for block in self.blocks:
            d = block(Tensor(d))._data

        # Final norm
        d = _layer_norm(d, self.ln_w._data, self.ln_b._data)

        # CLS token output — d shape (B, T, E); extract position 0 along axis 1
        d_flat = d.to_flat_list()
        T_seq = d.shape[1]
        cls_out_flat = [d_flat[b * T_seq * E + 0 * E + e] for b in range(B) for e in range(E)]
        cls_out = _nd_arr_vit(cls_out_flat, dtype='float32').reshape((B, E))

        if hasattr(self, 'pre_logits_w'):
            raw = cls_out @ self.pre_logits_w._data + self.pre_logits_b._data
            rf = raw.to_flat_list()
            rf = [_math.tanh(v) for v in rf]
            cls_out = _nd_arr_vit(rf, dtype='float32').reshape(raw.shape)

        if hasattr(self, 'head_w'):
            return Tensor(cls_out @ self.head_w._data + self.head_b._data)
        return Tensor(cls_out)


# ── DeiT ──────────────────────────────────────────────────────────────────────

class DeiT(Module):
    """Data-efficient Image Transformer (DeiT) — Touvron et al. 2021.

    Adds a distillation token alongside the [CLS] token for knowledge
    distillation from a CNN teacher. At inference, averages [CLS] and [dist].

    Inherits ViT architecture; adds a second special token and head.
    """

    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, num_classes: int = 1000,
                 embed_dim: int = 384, num_layers: int = 12,
                 num_heads: int = 6, mlp_dim: int = 1536,
                 dropout: float = 0.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self._modules['patch_embed'] = self.patch_embed
        num_patches = self.patch_embed.num_patches

        self.cls_token  = Tensor(_nd_zeros((1, 1, embed_dim)))
        self.dist_token = Tensor(_nd_zeros((1, 1, embed_dim)))
        self.cls_token.requires_grad = True
        self.dist_token.requires_grad = True
        self._parameters['cls_token']  = self.cls_token
        self._parameters['dist_token'] = self.dist_token

        # +2 for cls and dist tokens
        self.pos_embed = Tensor(_nd_randn((1, num_patches + 2, embed_dim), 0.02))
        self.pos_embed.requires_grad = True
        self._parameters['pos_embed'] = self.pos_embed

        self.dropout_rate = dropout
        self.blocks = []
        for i in range(num_layers):
            block = ViTBlock(embed_dim, num_heads, mlp_dim, dropout)
            self.blocks.append(block)
            self._modules[f'block_{i}'] = block

        self.ln_w = Tensor(_nd_ones((embed_dim,)))
        self.ln_b = Tensor(_nd_zeros((embed_dim,)))
        self.ln_w.requires_grad = True
        self.ln_b.requires_grad = True
        self._parameters['ln_w'] = self.ln_w
        self._parameters['ln_b'] = self.ln_b

        # Dual heads: classification head and distillation head
        for name in ['cls_head', 'dist_head']:
            w = Tensor(_nd_zeros((embed_dim, num_classes)))
            b = Tensor(_nd_zeros((num_classes,)))
            w.requires_grad = True
            b.requires_grad = True
            setattr(self, f'{name}_w', w)
            setattr(self, f'{name}_b', b)
            self._parameters[f'{name}_w'] = w
            self._parameters[f'{name}_b'] = b

    def forward(self, x: Tensor) -> Tensor:
        """
        Returns:
            Training: (cls_logits, dist_logits) each (B, num_classes)
            Inference: average of both heads (B, num_classes)
        """
        B = x._data.shape[0]
        patches = self.patch_embed(x)._data
        E = self.cls_token._data.shape[-1]
        from vectorphile._core.ndarray import array as _nd_arr_deit
        from vectorphile.core.tensor import _nd_concatenate as _deit_cat
        cls_rep  = self.cls_token._data.to_flat_list() * B
        dist_rep = self.dist_token._data.to_flat_list() * B
        cls_nd  = _nd_arr_deit(cls_rep,  dtype='float32').reshape((B, 1, E))
        dist_nd = _nd_arr_deit(dist_rep, dtype='float32').reshape((B, 1, E))
        d = _deit_cat([cls_nd, dist_nd, patches], axis=1) + self.pos_embed._data

        for block in self.blocks:
            d = block(Tensor(d))._data

        d = _layer_norm(d, self.ln_w._data, self.ln_b._data)
        # Extract cls (pos 0) and dist (pos 1) tokens
        T_seq = d.shape[1]
        d_flat = d.to_flat_list()
        cls_out_flat  = [d_flat[b * T_seq * E + 0 * E + e] for b in range(B) for e in range(E)]
        dist_out_flat = [d_flat[b * T_seq * E + 1 * E + e] for b in range(B) for e in range(E)]
        cls_out  = _nd_arr_deit(cls_out_flat,  dtype='float32').reshape((B, E))
        dist_out = _nd_arr_deit(dist_out_flat, dtype='float32').reshape((B, E))

        cls_logits  = cls_out  @ self.cls_head_w._data  + self.cls_head_b._data
        dist_logits = dist_out @ self.dist_head_w._data + self.dist_head_b._data

        if self.training:
            return Tensor(cls_logits), Tensor(dist_logits)
        # Inference: average both heads
        return Tensor((cls_logits + dist_logits) / 2)


# =============================================================================
# Session 11 - ViT-Reg (Register Tokens) and DeiT-III
# =============================================================================
from vectorphile.nn.layers import LayerNorm, Linear  # noqa: E402

class ViTReg(Module):
    """Vision Transformer with Register Tokens (ViT-Reg, Darcet et al. 2023).

    Register tokens are extra learnable tokens appended to the patch sequence.
    They act as "discard slots" for redundant global information, improving
    attention map quality and downstream task performance.

    Args:
        img_size: input image side length
        patch_size: patch side length
        in_channels: number of input channels
        num_classes: number of output classes
        embed_dim: transformer embedding dim
        depth: number of ViT blocks
        num_heads: attention heads
        num_registers: number of register tokens (Darcet et al. use 4-16)

    Usage::
        from vectorphile.nn.architectures import ViTReg
        model = ViTReg(img_size=224, patch_size=16, num_classes=1000, num_registers=4)
        logits = model(x)   # x: (B, 3, 224, 224)
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, num_classes: int = 1000,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 num_registers: int = 4):
        super().__init__()
        import math as _m
        import random as _r
        import array as _a2
        from vectorphile._core.ndarray import array as _nd2, NDArray as _ND2, _c_strides, _DType
        from vectorphile.core.tensor import Tensor as _T

        self.num_patches   = (img_size // patch_size) ** 2
        self.embed_dim     = embed_dim
        self.num_registers = num_registers

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        seq_len = 1 + self.num_patches + num_registers  # CLS + patches + registers

        # CLS token
        cls_v = [_r.gauss(0, 0.02) for _ in range(embed_dim)]
        self.cls_token = _T(_ND2(_a2.array('f', cls_v), (1, embed_dim),
                                 _c_strides((1, embed_dim)), _DType('float32')), requires_grad=True)
        self._parameters['cls_token'] = self.cls_token

        # Register tokens
        reg_v = [_r.gauss(0, 0.02) for _ in range(num_registers * embed_dim)]
        self.reg_tokens = _T(_ND2(_a2.array('f', reg_v), (num_registers, embed_dim),
                                  _c_strides((num_registers, embed_dim)), _DType('float32')), requires_grad=True)
        self._parameters['reg_tokens'] = self.reg_tokens

        # Positional embedding (CLS + patches only — no pos embed for registers)
        pos_v = [_r.gauss(0, 0.02) for _ in range(seq_len * embed_dim)]
        self.pos_embed = _T(_ND2(_a2.array('f', pos_v), (seq_len, embed_dim),
                                 _c_strides((seq_len, embed_dim)), _DType('float32')), requires_grad=True)
        self._parameters['pos_embed'] = self.pos_embed

        mlp_dim = embed_dim * 4
        self.blocks = [ViTBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)]
        for i, b in enumerate(self.blocks):
            self._modules[f'block_{i}'] = b

        self.norm = LayerNorm(embed_dim)
        self.head = Linear(embed_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        import array as _a2
        from vectorphile._core.ndarray import NDArray as _ND2, _c_strides, _DType, array as _nd2
        from vectorphile.core.tensor import Tensor as _T

        patches = self.patch_embed(x)
        sh = patches._data.shape; B, N, E = sh

        # Expand CLS and register tokens to batch size
        cls_f = list(self.cls_token._data.to_flat_list() if hasattr(self.cls_token._data, 'to_flat_list')
                     else self.cls_token._data)
        reg_f = list(self.reg_tokens._data.to_flat_list() if hasattr(self.reg_tokens._data, 'to_flat_list')
                     else self.reg_tokens._data)
        pat_f = list(patches._data.to_flat_list() if hasattr(patches._data, 'to_flat_list') else patches._data)
        pos_f = list(self.pos_embed._data.to_flat_list() if hasattr(self.pos_embed._data, 'to_flat_list')
                     else self.pos_embed._data)

        R = self.num_registers
        seq_full = 1 + N + R  # CLS + patches + registers
        seq_data = []
        for b in range(B):
            seq_data.extend(cls_f)                       # CLS
            seq_data.extend(pat_f[b*N*E:(b+1)*N*E])     # patches
            seq_data.extend(reg_f)                       # registers
        # Add positional embedding (applied to CLS + patches + registers)
        for i in range(B * seq_full * E):
            pos_i = i % (seq_full * E)
            seq_data[i] += pos_f[pos_i] if pos_i < len(pos_f) else 0.0

        sh_full = (B, seq_full, E)
        t = _T(_ND2(_a2.array('f', [float(v) for v in seq_data]),
                    sh_full, _c_strides(sh_full), _DType('float32')))

        for block in self.blocks:
            t = block(t)
        t = self.norm(t)

        # Extract CLS output (position 0)
        t_flat = list(t._data.to_flat_list() if hasattr(t._data, 'to_flat_list') else t._data)
        cls_out = [t_flat[b * seq_full * E + e] for b in range(B) for e in range(E)]
        cls_t = _T(_ND2(_a2.array('f', cls_out), (B, E), _c_strides((B, E)), _DType('float32')))
        return self.head(cls_t)


class DeiT3(Module):
    """DeiT-III (Touvron et al. 2022) - improved DeiT with 3-Augmentation training.

    DeiT-III uses a simple supervised training recipe (no distillation token)
    with strong data augmentation. The architecture is identical to ViT but
    trained with 3-Augmentation (simple, global crop, local crop).

    For inference this is indistinguishable from ViT; the DeiT-III improvements
    are in the training procedure. This class tracks training mode to apply
    the correct loss in the training loop.

    Args:
        img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads:
            same as ViT

    Usage::
        from vectorphile.nn.architectures import DeiT3
        model = DeiT3(img_size=224, patch_size=16, num_classes=1000)
        logits = model(x)
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, num_classes: int = 1000,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12):
        super().__init__()
        self._vit = ViT(image_size=img_size, patch_size=patch_size,
                        in_channels=in_channels, num_classes=num_classes,
                        embed_dim=embed_dim, num_layers=depth, num_heads=num_heads)
        self._modules['vit'] = self._vit
        self._parameters.update(self._vit._parameters)

    def forward(self, x: Tensor) -> Tensor:
        return self._vit(x)

    def three_aug_loss(self, x_simple: Tensor, x_global: Tensor, x_local: Tensor,
                       labels: Tensor) -> Tensor:
        """Compute DeiT-III 3-Augmentation loss.

        Args:
            x_simple: original image crop logits
            x_global: global crop logits
            x_local: local crop logits
            labels: ground truth class indices

        Returns:
            averaged cross-entropy loss over three crops
        """
        from vectorphile.nn.losses import CrossEntropyLoss
        ce = CrossEntropyLoss()
        l1 = ce(self(x_simple), labels)
        l2 = ce(self(x_global), labels)
        l3 = ce(self(x_local),  labels)
        # Average losses — equal weight
        l1f = list(l1._data.to_flat_list() if hasattr(l1._data, 'to_flat_list') else l1._data)
        l2f = list(l2._data.to_flat_list() if hasattr(l2._data, 'to_flat_list') else l2._data)
        l3f = list(l3._data.to_flat_list() if hasattr(l3._data, 'to_flat_list') else l3._data)
        avg = [(l1f[i] + l2f[i] + l3f[i]) / 3.0 for i in range(len(l1f))]
        import array as _a2
        from vectorphile._core.ndarray import NDArray as _ND2, _c_strides, _DType
        return Tensor(_ND2(_a2.array('f', avg), (len(avg),), _c_strides((len(avg),)), _DType('float32')))
