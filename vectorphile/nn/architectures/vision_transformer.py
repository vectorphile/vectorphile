"""Vision Transformers — ViT (Dosovitskiy et al. 2020) and DeiT-style distillation."""
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-9)


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(-1, keepdims=True)
    var  = x.var(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


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
        self.proj_w = Tensor(np.random.randn(self.patch_dim, embed_dim).astype(np.float32)
                             * np.sqrt(2.0 / self.patch_dim))
        self.proj_b = Tensor(np.zeros(embed_dim, dtype=np.float32))
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
        d = d.transpose(0, 2, 4, 1, 3, 5)           # (B, h, w, C, p, p)
        d = d.reshape(B, h_patches * w_patches, -1)  # (B, num_patches, C*p*p)
        return Tensor(d @ self.proj_w._data + self.proj_b._data)


# ── Multi-head self-attention ──────────────────────────────────────────────────

class ViTAttention(Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.dropout = dropout
        s = 0.02
        for name in ['w_q', 'w_k', 'w_v', 'w_o']:
            p = Tensor(np.random.randn(embed_dim, embed_dim).astype(np.float32) * s)
            p.requires_grad = True
            setattr(self, name, p)
            self._parameters[name] = p
        for name in ['b_q', 'b_k', 'b_v', 'b_o']:
            p = Tensor(np.zeros(embed_dim, dtype=np.float32))
            p.requires_grad = True
            setattr(self, name, p)
            self._parameters[name] = p

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x._data.shape
        H, D = self.num_heads, self.head_dim
        def proj(d, w, b): return (d @ w + b).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        q = proj(x._data, self.w_q._data, self.b_q._data)
        k = proj(x._data, self.w_k._data, self.b_k._data)
        v = proj(x._data, self.w_v._data, self.b_v._data)
        scores = q @ k.swapaxes(-2, -1) * self.scale
        attn = _softmax(scores)
        if self.dropout > 0 and self.training:
            mask = np.random.binomial(1, 1 - self.dropout, attn.shape).astype(np.float32)
            attn = attn * mask / (1 - self.dropout)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return Tensor(out @ self.w_o._data + self.b_o._data)


# ── MLP block ─────────────────────────────────────────────────────────────────

class ViTMLP(Module):
    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        s = np.sqrt(2.0 / embed_dim)
        self.fc1_w = Tensor(np.random.randn(embed_dim, mlp_dim).astype(np.float32) * s)
        self.fc1_b = Tensor(np.zeros(mlp_dim, dtype=np.float32))
        self.fc2_w = Tensor(np.random.randn(mlp_dim, embed_dim).astype(np.float32) * s)
        self.fc2_b = Tensor(np.zeros(embed_dim, dtype=np.float32))
        for n, p in [('fc1_w',self.fc1_w),('fc1_b',self.fc1_b),
                     ('fc2_w',self.fc2_w),('fc2_b',self.fc2_b)]:
            p.requires_grad = True
            self._parameters[n] = p

    def forward(self, x: Tensor) -> Tensor:
        d = _gelu(x._data @ self.fc1_w._data + self.fc1_b._data)
        if self.dropout > 0 and self.training:
            mask = np.random.binomial(1, 1 - self.dropout, d.shape).astype(np.float32)
            d = d * mask / (1 - self.dropout)
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
            w = Tensor(np.ones(embed_dim, dtype=np.float32))
            b = Tensor(np.zeros(embed_dim, dtype=np.float32))
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
        self.cls_token = Tensor(np.zeros((1, 1, embed_dim), dtype=np.float32))
        self.cls_token.requires_grad = True
        self._parameters['cls_token'] = self.cls_token

        # Learned position embeddings: (1, num_patches+1, embed_dim)
        self.pos_embed = Tensor(
            np.random.randn(1, num_patches + 1, embed_dim).astype(np.float32) * 0.02
        )
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
        self.ln_w = Tensor(np.ones(embed_dim, dtype=np.float32))
        self.ln_b = Tensor(np.zeros(embed_dim, dtype=np.float32))
        self.ln_w.requires_grad = True
        self.ln_b.requires_grad = True
        self._parameters['ln_w'] = self.ln_w
        self._parameters['ln_b'] = self.ln_b

        # Classification head
        if representation_size is not None:
            self.pre_logits_w = Tensor(np.random.randn(embed_dim, representation_size).astype(np.float32) * 0.02)
            self.pre_logits_b = Tensor(np.zeros(representation_size, dtype=np.float32))
            self.pre_logits_w.requires_grad = True
            self.pre_logits_b.requires_grad = True
            self._parameters['pre_logits_w'] = self.pre_logits_w
            self._parameters['pre_logits_b'] = self.pre_logits_b
            head_in = representation_size
        else:
            head_in = embed_dim

        if num_classes is not None:
            self.head_w = Tensor(np.zeros((head_in, num_classes), dtype=np.float32))
            self.head_b = Tensor(np.zeros(num_classes, dtype=np.float32))
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

        # Prepend [CLS] token
        cls = np.broadcast_to(self.cls_token._data, (B, 1, self.cls_token._data.shape[-1]))
        d = np.concatenate([cls, patches], axis=1)   # (B, 1+num_patches, embed_dim)

        # Add positional embeddings
        d = d + self.pos_embed._data

        # Dropout
        if self.dropout_rate > 0 and self.training:
            mask = np.random.binomial(1, 1 - self.dropout_rate, d.shape).astype(np.float32)
            d = d * mask / (1 - self.dropout_rate)

        # Transformer blocks
        for block in self.blocks:
            d = block(Tensor(d))._data

        # Final norm
        d = _layer_norm(d, self.ln_w._data, self.ln_b._data)

        # CLS token output
        cls_out = d[:, 0]   # (B, embed_dim)

        if hasattr(self, 'pre_logits_w'):
            cls_out = np.tanh(cls_out @ self.pre_logits_w._data + self.pre_logits_b._data)

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

        self.cls_token  = Tensor(np.zeros((1, 1, embed_dim), dtype=np.float32))
        self.dist_token = Tensor(np.zeros((1, 1, embed_dim), dtype=np.float32))
        self.cls_token.requires_grad = True
        self.dist_token.requires_grad = True
        self._parameters['cls_token']  = self.cls_token
        self._parameters['dist_token'] = self.dist_token

        # +2 for cls and dist tokens
        self.pos_embed = Tensor(
            np.random.randn(1, num_patches + 2, embed_dim).astype(np.float32) * 0.02
        )
        self.pos_embed.requires_grad = True
        self._parameters['pos_embed'] = self.pos_embed

        self.dropout_rate = dropout
        self.blocks = []
        for i in range(num_layers):
            block = ViTBlock(embed_dim, num_heads, mlp_dim, dropout)
            self.blocks.append(block)
            self._modules[f'block_{i}'] = block

        self.ln_w = Tensor(np.ones(embed_dim, dtype=np.float32))
        self.ln_b = Tensor(np.zeros(embed_dim, dtype=np.float32))
        self.ln_w.requires_grad = True
        self.ln_b.requires_grad = True
        self._parameters['ln_w'] = self.ln_w
        self._parameters['ln_b'] = self.ln_b

        # Dual heads: classification head and distillation head
        for name in ['cls_head', 'dist_head']:
            w = Tensor(np.zeros((embed_dim, num_classes), dtype=np.float32))
            b = Tensor(np.zeros(num_classes, dtype=np.float32))
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
        cls  = np.broadcast_to(self.cls_token._data,  (B, 1, self.cls_token._data.shape[-1]))
        dist = np.broadcast_to(self.dist_token._data, (B, 1, self.dist_token._data.shape[-1]))
        d = np.concatenate([cls, dist, patches], axis=1) + self.pos_embed._data

        for block in self.blocks:
            d = block(Tensor(d))._data

        d = _layer_norm(d, self.ln_w._data, self.ln_b._data)
        cls_out  = d[:, 0]
        dist_out = d[:, 1]

        cls_logits  = cls_out  @ self.cls_head_w._data  + self.cls_head_b._data
        dist_logits = dist_out @ self.dist_head_w._data + self.dist_head_b._data

        if self.training:
            return Tensor(cls_logits), Tensor(dist_logits)
        # Inference: average both heads
        return Tensor((cls_logits + dist_logits) / 2)
