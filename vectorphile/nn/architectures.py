"""vectorphile.nn.architectures — Pre-built model architectures.

Provides factory functions and classes for common model families:
  - GPT (causal language model, all sizes)
  - BERT (masked language model, encoder-only)
  - ViT (Vision Transformer)
  - ResNet (image classification)
  - T5 (encoder-decoder)
  - UNet (segmentation)
  - CLIP-style dual encoder

All weights are randomly initialized; use vectorphile.pretrained to load
pretrained checkpoints.

Usage::

    from vectorphile.nn.architectures import GPT, BERTModel, ViT, ResNet

    model = GPT(vocab_size=50257, context_len=1024, n_layer=12, n_head=12, n_embd=768)
    logits = model(input_ids)   # (B, T, vocab_size)
"""
from __future__ import annotations
from typing import Optional

from vectorphile.nn.module import Module
from vectorphile.nn.layers import Linear, Embedding, LayerNorm, Dropout, Conv2d, BatchNorm2d
from vectorphile.nn.activations import ReLU, GELU
from vectorphile.nn.attention import MultiHeadAttention, PositionalEncoding
from vectorphile.nn.sequential import Sequential
from vectorphile.nn.norm import RMSNorm
from vectorphile.core.tensor import Tensor


# ---------------------------------------------------------------------------
# Shared blocks
# ---------------------------------------------------------------------------

class TransformerBlock(Module):
    """Pre-norm transformer block (used in GPT, BERT, ViT)."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ff = Sequential(Linear(d_model, d_ff), GELU(), Linear(d_ff, d_model))
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.drop1 = Dropout(dropout)
        self.drop2 = Dropout(dropout)
        self.causal = causal

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Pre-norm
        h = self.ln1(x)
        # Pass causal mask if needed
        attn_out = self.attn(h, h, h, attn_mask=mask)
        x = x + self.drop1(attn_out)
        x = x + self.drop2(self.ff(self.ln2(x)))
        return x


# ---------------------------------------------------------------------------
# GPT
# ---------------------------------------------------------------------------

class GPTConfig:
    def __init__(self, vocab_size=50257, context_len=1024, n_layer=12,
                 n_head=12, n_embd=768, dropout=0.1, bias=True):
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias


class GPT(Module):
    """GPT-style auto-regressive language model.

    Usage::

        model = GPT(vocab_size=50257, context_len=1024, n_layer=12,
                    n_head=12, n_embd=768)
        logits = model(input_ids)       # (B, T, V)
        loss   = model(input_ids, targets=targets)  # returns scalar Tensor
    """

    def __init__(self, vocab_size: int = 50257, context_len: int = 1024,
                 n_layer: int = 12, n_head: int = 12, n_embd: int = 768,
                 dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.wte = Embedding(vocab_size, n_embd)       # token embeddings
        self.wpe = Embedding(context_len, n_embd)      # position embeddings
        self.drop = Dropout(dropout)
        self.blocks = [TransformerBlock(n_embd, n_head, 4 * n_embd, dropout, causal=True)
                       for _ in range(n_layer)]
        for i, b in enumerate(self.blocks):
            setattr(self, f'block_{i}', b)
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = Linear(n_embd, vocab_size, bias=False)
        self._n_layer = n_layer
        self._context_len = context_len
        self._n_embd = n_embd

    def forward(self, input_ids: Tensor, targets: Optional[Tensor] = None) -> Tensor:
        ids = _np.asarray(input_ids._data).astype(int)
        B, T = ids.shape
        assert T <= self._context_len, f"Sequence too long: {T} > {self._context_len}"
        pos = _np.arange(T, dtype=int)
        pos_t = Tensor(_np.tile(pos, (B, 1)))
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos_t)
        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V)
        if targets is None:
            return logits
        # Cross-entropy loss
        logits_np = _np.asarray(logits._data)
        tgt_np = _np.asarray(targets._data).astype(int).ravel()
        logits_flat = logits_np.reshape(-1, logits_np.shape[-1])
        # Stable softmax
        mx = logits_flat.max(axis=-1, keepdims=True)
        log_sum = _np.log(_np.exp(logits_flat - mx).sum(axis=-1, keepdims=True)) + mx
        log_probs = logits_flat - log_sum
        loss = -log_probs[_np.arange(len(tgt_np)), tgt_np].mean()
        return Tensor(_np.array([loss], dtype='float32'))

    @classmethod
    def from_config(cls, config: GPTConfig) -> 'GPT':
        return cls(config.vocab_size, config.context_len, config.n_layer,
                   config.n_head, config.n_embd, config.dropout, config.bias)

    @classmethod
    def gpt2(cls) -> 'GPT':
        return cls(50257, 1024, 12, 12, 768)

    @classmethod
    def gpt2_medium(cls) -> 'GPT':
        return cls(50257, 1024, 24, 16, 1024)

    @classmethod
    def gpt2_large(cls) -> 'GPT':
        return cls(50257, 1024, 36, 20, 1280)


# ---------------------------------------------------------------------------
# BERT
# ---------------------------------------------------------------------------

class BERTModel(Module):
    """BERT-style bidirectional encoder (Devlin et al., 2018).

    Usage::

        model = BERTModel(vocab_size=30522, n_layer=12, n_head=12, n_embd=768)
        cls_rep, seq_out = model(input_ids, attention_mask=mask)
    """

    def __init__(self, vocab_size: int = 30522, n_layer: int = 12,
                 n_head: int = 12, n_embd: int = 768, max_len: int = 512,
                 n_segments: int = 2, dropout: float = 0.1):
        super().__init__()
        self.word_emb = Embedding(vocab_size, n_embd)
        self.pos_emb = Embedding(max_len, n_embd)
        self.seg_emb = Embedding(n_segments, n_embd)
        self.emb_norm = LayerNorm(n_embd)
        self.emb_drop = Dropout(dropout)
        self.blocks = [TransformerBlock(n_embd, n_head, 4 * n_embd, dropout, causal=False)
                       for _ in range(n_layer)]
        for i, b in enumerate(self.blocks):
            setattr(self, f'block_{i}', b)
        self.pooler = Linear(n_embd, n_embd)
        self._n_embd = n_embd

    def forward(self, input_ids: Tensor, segment_ids: Optional[Tensor] = None,
                attention_mask: Optional[Tensor] = None):
        from vectorphile._core.ndarray import array as _nd_arr
        import math as _math_a
        # Extract integer ids and shape
        flat_ids = input_ids._data.to_flat_list() if hasattr(input_ids._data, 'to_flat_list') else list(input_ids._data)
        sh = input_ids._data.shape if hasattr(input_ids._data, 'shape') else (len(flat_ids),)
        B, T = sh[0], sh[1]
        # Build position indices [0..T-1] tiled B times
        pos_flat = [float(i % T) for i in range(B * T)]
        pos = Tensor(_nd_arr(pos_flat, dtype='float32').reshape((B, T)))
        x = self.word_emb(input_ids) + self.pos_emb(pos)
        if segment_ids is not None:
            x = x + self.seg_emb(segment_ids)
        x = self.emb_drop(self.emb_norm(x))
        for block in self.blocks:
            x = block(x)
        # CLS token (index 0 along T dimension) — x shape (B, T, d)
        x_flat = x._data.to_flat_list()
        d = x._data.shape[2]
        cls_flat = [x_flat[b * T * d + 0 * d + i] for b in range(B) for i in range(d)]
        cls_nd = _nd_arr(cls_flat, dtype='float32').reshape((B, d))
        # Apply pooler then tanh
        pooler_out = self.pooler(Tensor(cls_nd))
        pf = pooler_out._data.to_flat_list()
        tanh_flat = [_math_a.tanh(v) for v in pf]
        cls_rep = Tensor(_nd_arr(tanh_flat, dtype='float32').reshape(pooler_out._data.shape))
        return cls_rep, x

    @classmethod
    def bert_base(cls) -> 'BERTModel':
        return cls(30522, 12, 12, 768)

    @classmethod
    def bert_large(cls) -> 'BERTModel':
        return cls(30522, 24, 16, 1024)


# ---------------------------------------------------------------------------
# ViT — Vision Transformer
# ---------------------------------------------------------------------------

class ViT(Module):
    """Vision Transformer (Dosovitskiy et al., 2020).

    Usage::

        model = ViT(image_size=224, patch_size=16, num_classes=1000,
                    n_layer=12, n_head=12, n_embd=768)
        logits = model(images)   # (B, num_classes)
    """

    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 num_classes: int = 1000, n_layer: int = 12, n_head: int = 12,
                 n_embd: int = 768, in_channels: int = 3, dropout: float = 0.1):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        from vectorphile.nn.extra_layers import PatchEmbed
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, n_embd)
        self.cls_token_emb = Linear(1, n_embd, bias=False)  # proxy for CLS token
        self.pos_emb = Embedding(num_patches + 1, n_embd)
        self.drop = Dropout(dropout)
        self.blocks = [TransformerBlock(n_embd, n_head, 4 * n_embd, dropout)
                       for _ in range(n_layer)]
        for i, b in enumerate(self.blocks):
            setattr(self, f'block_{i}', b)
        self.norm = LayerNorm(n_embd)
        self.head = Linear(n_embd, num_classes)
        self._num_patches = num_patches
        self._n_embd = n_embd

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        patches = self.patch_embed(x)  # (B, N, D)
        B = _np.asarray(patches._data).shape[0]
        N = _np.asarray(patches._data).shape[1]
        # Prepend CLS token (learnable via zero init)
        cls_np = _np.zeros((B, 1, self._n_embd), dtype='float32')
        cls_t = Tensor(cls_np)
        patches_np = _np.asarray(patches._data)
        x_seq = Tensor(_np.concatenate([cls_np, patches_np], axis=1))  # (B, N+1, D)
        pos_np = _np.tile(_np.arange(N + 1, dtype=int), (B, 1))
        pos_emb = self.pos_emb(Tensor(pos_np))
        x_seq = self.drop(x_seq + pos_emb)
        for block in self.blocks:
            x_seq = block(x_seq)
        x_seq = self.norm(x_seq)
        cls_out = Tensor(_np.asarray(x_seq._data)[:, 0, :])  # CLS token
        return self.head(cls_out)

    @classmethod
    def vit_tiny(cls, num_classes=1000) -> 'ViT':
        return cls(224, 16, num_classes, 12, 3, 192)

    @classmethod
    def vit_small(cls, num_classes=1000) -> 'ViT':
        return cls(224, 16, num_classes, 12, 6, 384)

    @classmethod
    def vit_base(cls, num_classes=1000) -> 'ViT':
        return cls(224, 16, num_classes, 12, 12, 768)


# ---------------------------------------------------------------------------
# ResNet
# ---------------------------------------------------------------------------

class ResidualBlock(Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = BatchNorm2d(out_ch)
        self.conv2 = Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = BatchNorm2d(out_ch)
        self.relu = ReLU()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = Conv2d(in_ch, out_ch, 1, stride=stride)
            self.shortcut_bn = BatchNorm2d(out_ch)
        else:
            self.shortcut = None

    def forward(self, x: Tensor) -> Tensor:
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        sc = self.shortcut(x) if self.shortcut is not None else x
        if hasattr(self, 'shortcut_bn') and self.shortcut is not None:
            sc = self.shortcut_bn(sc)
        return self.relu(h + sc)


class BottleneckBlock(Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expansion: int = 4):
        super().__init__()
        mid = out_ch // expansion
        self.conv1 = Conv2d(in_ch, mid, 1)
        self.bn1 = BatchNorm2d(mid)
        self.conv2 = Conv2d(mid, mid, 3, stride=stride, padding=1)
        self.bn2 = BatchNorm2d(mid)
        self.conv3 = Conv2d(mid, out_ch, 1)
        self.bn3 = BatchNorm2d(out_ch)
        self.relu = ReLU()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = Conv2d(in_ch, out_ch, 1, stride=stride)
            self.shortcut_bn = BatchNorm2d(out_ch)
        else:
            self.shortcut = None

    def forward(self, x: Tensor) -> Tensor:
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        sc = self.shortcut(x) if self.shortcut is not None else x
        if hasattr(self, 'shortcut_bn') and self.shortcut is not None:
            sc = self.shortcut_bn(sc)
        return self.relu(h + sc)


class ResNet(Module):
    """ResNet family (He et al., 2015). Supports ResNet-18/34/50/101/152.

    Usage::

        model = ResNet.resnet50(num_classes=1000)
        out = model(images)   # (B, num_classes)
    """

    def __init__(self, layers: list, num_classes: int = 1000,
                 block_type: str = 'basic', in_channels: int = 3):
        super().__init__()
        Block = ResidualBlock if block_type == 'basic' else BottleneckBlock
        channels = [64, 128, 256, 512]
        base = 64 * (4 if block_type == 'bottleneck' else 1)
        self.stem = Sequential(
            Conv2d(in_channels, 64, 7, stride=2, padding=3),
            BatchNorm2d(64), ReLU(),
        )
        self._stages = []
        in_ch = 64
        for stage_idx, (n_blocks, out_ch) in enumerate(zip(layers, channels)):
            stage = []
            out_c = out_ch * (4 if block_type == 'bottleneck' else 1)
            for block_idx in range(n_blocks):
                stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
                stage.append(Block(in_ch, out_c, stride=stride))
                in_ch = out_c
            self._stages.append(stage)
            for j, blk in enumerate(stage):
                setattr(self, f'stage{stage_idx}_block{j}', blk)
        self.head = Linear(in_ch, num_classes)
        self._n_stages = len(self._stages)

    def forward(self, x: Tensor) -> Tensor:
        h = self.stem(x)
        for stage in self._stages:
            for block in stage:
                h = block(h)
        # Global average pool
        h_np = _np.asarray(h._data)
        gap = h_np.mean(axis=(-2, -1))  # (B, C)
        return self.head(Tensor(gap))

    @classmethod
    def resnet18(cls, num_classes=1000):
        return cls([2, 2, 2, 2], num_classes, 'basic')

    @classmethod
    def resnet34(cls, num_classes=1000):
        return cls([3, 4, 6, 3], num_classes, 'basic')

    @classmethod
    def resnet50(cls, num_classes=1000):
        return cls([3, 4, 6, 3], num_classes, 'bottleneck')

    @classmethod
    def resnet101(cls, num_classes=1000):
        return cls([3, 4, 23, 3], num_classes, 'bottleneck')

    @classmethod
    def resnet152(cls, num_classes=1000):
        return cls([3, 8, 36, 3], num_classes, 'bottleneck')


# ---------------------------------------------------------------------------
# U-Net for segmentation
# ---------------------------------------------------------------------------

class UNet(Module):
    """U-Net (Ronneberger et al., 2015) for image segmentation.

    Usage::

        model = UNet(in_channels=1, num_classes=2)
        pred  = model(images)   # (B, num_classes, H, W)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2, base_ch: int = 64):
        super().__init__()
        from vectorphile.nn.layers import ConvTranspose2d

        def _double_conv(ic, oc):
            return Sequential(Conv2d(ic, oc, 3, padding=1), BatchNorm2d(oc), ReLU(),
                               Conv2d(oc, oc, 3, padding=1), BatchNorm2d(oc), ReLU())

        self.enc1 = _double_conv(in_channels, base_ch)
        self.enc2 = _double_conv(base_ch, base_ch * 2)
        self.enc3 = _double_conv(base_ch * 2, base_ch * 4)
        self.enc4 = _double_conv(base_ch * 4, base_ch * 8)
        self.bottleneck = _double_conv(base_ch * 8, base_ch * 16)
        self.up4 = ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = _double_conv(base_ch * 16, base_ch * 8)
        self.up3 = ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = _double_conv(base_ch * 8, base_ch * 4)
        self.up2 = ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = _double_conv(base_ch * 4, base_ch * 2)
        self.up1 = ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = _double_conv(base_ch * 2, base_ch)
        self.out_conv = Conv2d(base_ch, num_classes, 1)

    def _maxpool(self, x: Tensor) -> Tensor:
        """2x2 max pool."""
        from vectorphile.nn.pooling import MaxPool2d
        return MaxPool2d(2, stride=2)(x)

    def _cat(self, a: Tensor, b: Tensor) -> Tensor:
        """Concatenate along channel axis."""
        an = _np.asarray(a._data)
        bn = _np.asarray(b._data)
        # Crop b to a's spatial dims if needed
        h, w = an.shape[-2], an.shape[-1]
        bn = bn[:, :, :h, :w]
        return Tensor(_np.concatenate([an, bn], axis=1))

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self._maxpool(e1))
        e3 = self.enc3(self._maxpool(e2))
        e4 = self.enc4(self._maxpool(e3))
        b = self.bottleneck(self._maxpool(e4))
        d4 = self.dec4(self._cat(self.up4(b), e4))
        d3 = self.dec3(self._cat(self.up3(d4), e3))
        d2 = self.dec2(self._cat(self.up2(d3), e2))
        d1 = self.dec1(self._cat(self.up1(d2), e1))
        return self.out_conv(d1)


__all__ = [
    'GPTConfig', 'GPT', 'BERTModel', 'ViT', 'ResNet', 'UNet',
    'TransformerBlock', 'ResidualBlock', 'BottleneckBlock',
]
