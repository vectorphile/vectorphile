"""U-Net architectures — vanilla U-Net, ResU-Net, U-Net with attention gates."""
from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


def _conv2d(x, w, b=None, stride=1, padding=1):
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1
    if padding > 0:
        x = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
    out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
    for i in range(H_out):
        for j in range(W_out):
            out[:, :, i, j] = np.einsum(
                'nchw,ochw->no', x[:, :, i*stride:i*stride+kH, j*stride:j*stride+kW], w)
    if b is not None:
        out += b[None, :, None, None]
    return out


def _relu(x): return np.maximum(x, 0)
def _sigmoid(x): return 1 / (1 + np.exp(-x))


def _maxpool2d(x, k=2, s=2):
    N, C, H, W = x.shape
    H_o = (H - k) // s + 1
    W_o = (W - k) // s + 1
    out = np.zeros((N, C, H_o, W_o), dtype=np.float32)
    for i in range(H_o):
        for j in range(W_o):
            out[:, :, i, j] = x[:, :, i*s:i*s+k, j*s:j*s+k].max((2, 3))
    return out


def _upsample_bilinear(x, scale=2):
    """Bilinear upsampling by integer scale factor."""
    N, C, H, W = x.shape
    H2, W2 = H * scale, W * scale
    out = np.zeros((N, C, H2, W2), dtype=np.float32)
    for i in range(H2):
        for j in range(W2):
            src_i = min(i // scale, H - 1)
            src_j = min(j // scale, W - 1)
            out[:, :, i, j] = x[:, :, src_i, src_j]
    return out


def _batch_norm2d(x, w, b, rm, rv, training, mom=0.1, eps=1e-5):
    if training:
        mean = x.mean((0, 2, 3), keepdims=True)
        var  = x.var((0, 2, 3), keepdims=True)
        rm[:] = (1 - mom) * rm + mom * x.mean((0, 2, 3))
        rv[:] = (1 - mom) * rv + mom * x.var((0, 2, 3))
    else:
        mean = rm[None, :, None, None]
        var  = rv[None, :, None, None]
    return (x - mean) / np.sqrt(var + eps) * w[None, :, None, None] + b[None, :, None, None]


# ── Double conv block ──────────────────────────────────────────────────────────

class DoubleConv(Module):
    """Two consecutive Conv → BN → ReLU blocks (standard U-Net encoder unit)."""

    def __init__(self, in_c: int, out_c: int, mid_c: Optional[int] = None):
        super().__init__()
        mid_c = mid_c or out_c
        channels = [(in_c, mid_c), (mid_c, out_c)]
        self.convs_w = []
        self.bn_w, self.bn_b = [], []
        self.rm, self.rv = [], []
        for i, (ic, oc) in enumerate(channels):
            w = Tensor(np.random.randn(oc, ic, 3, 3).astype(np.float32) * np.sqrt(2 / (ic * 9)))
            w.requires_grad = True; self._parameters[f'conv_w{i}'] = w; self.convs_w.append(w)
            bw = Tensor(np.ones(oc, dtype=np.float32)); bw.requires_grad = True
            bb = Tensor(np.zeros(oc, dtype=np.float32)); bb.requires_grad = True
            self._parameters[f'bn_w{i}'] = bw; self._parameters[f'bn_b{i}'] = bb
            self.bn_w.append(bw); self.bn_b.append(bb)
            self.rm.append(np.zeros(oc, dtype=np.float32))
            self.rv.append(np.ones(oc, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        d = x._data
        for i, cw in enumerate(self.convs_w):
            d = _conv2d(d, cw._data, None)
            d = _batch_norm2d(d, self.bn_w[i]._data, self.bn_b[i]._data,
                              self.rm[i], self.rv[i], self.training)
            d = _relu(d)
        return Tensor(d)


# ── U-Net ─────────────────────────────────────────────────────────────────────

class UNet(Module):
    """U-Net — Ronneberger, Fischer & Brox 2015.

    Encoder–decoder with symmetric skip connections.
    Default configuration matches the original medical image segmentation paper.

    Args:
        in_channels: input channels
        out_channels: output channels (e.g. number of segmentation classes)
        base_channels: initial feature map count (doubles per level)
        depth: number of encoder/decoder levels
        bilinear: use bilinear upsampling (True) or transposed conv (False)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 2,
                 base_channels: int = 64, depth: int = 4, bilinear: bool = True):
        super().__init__()
        self.depth = depth
        self.bilinear = bilinear
        # Encoder
        self.encoder_blocks = []
        in_c = in_channels
        channel_list = []
        c = base_channels
        for i in range(depth):
            blk = DoubleConv(in_c, c)
            self.encoder_blocks.append(blk)
            self._modules[f'enc_{i}'] = blk
            channel_list.append(c)
            in_c = c
            c = min(c * 2, 1024)
        # Bottleneck
        self.bottleneck = DoubleConv(in_c, c)
        self._modules['bottleneck'] = self.bottleneck
        in_c = c
        # Decoder
        self.decoder_blocks = []
        self.up_convs_w = []
        for i, skip_c in reversed(list(enumerate(channel_list))):
            # 1×1 conv to halve channels after concat
            uc_w = Tensor(np.random.randn(in_c, in_c // 2, 1, 1).astype(np.float32) * 0.02)
            uc_w.requires_grad = True; self._parameters[f'up_conv_w{i}'] = uc_w
            self.up_convs_w.append(uc_w)
            blk = DoubleConv(in_c // 2 + skip_c, skip_c)
            self.decoder_blocks.append(blk)
            self._modules[f'dec_{i}'] = blk
            in_c = skip_c
        # Output
        self.out_w = Tensor(np.random.randn(out_channels, in_c, 1, 1).astype(np.float32) * 0.02)
        self.out_b = Tensor(np.zeros(out_channels, dtype=np.float32))
        self.out_w.requires_grad = True; self.out_b.requires_grad = True
        self._parameters['out_w'] = self.out_w; self._parameters['out_b'] = self.out_b

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, H, W)
        Returns:
            (B, out_channels, H, W) logits (use softmax/sigmoid for predictions)
        """
        d = x._data
        skips = []
        for blk in self.encoder_blocks:
            d = blk(Tensor(d))._data
            skips.append(d)
            d = _maxpool2d(d)
        d = self.bottleneck(Tensor(d))._data
        for uc_w, blk, skip in zip(self.up_convs_w, self.decoder_blocks, reversed(skips)):
            d = _upsample_bilinear(d)
            d = _relu(_conv2d(d, uc_w._data, None, stride=1, padding=0))
            # Crop skip connection to match spatial size
            h, w = min(d.shape[2], skip.shape[2]), min(d.shape[3], skip.shape[3])
            d_crop    = d[:, :, :h, :w]
            skip_crop = skip[:, :, :h, :w]
            d = np.concatenate([d_crop, skip_crop], axis=1)
            d = blk(Tensor(d))._data
        return Tensor(_conv2d(d, self.out_w._data, self.out_b._data, padding=0))


# ── ResU-Net ───────────────────────────────────────────────────────────────────

class ResConvBlock(Module):
    """Residual double-conv block for ResU-Net."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.double_conv = DoubleConv(in_c, out_c)
        self._modules['dc'] = self.double_conv
        self.skip_w = None
        if in_c != out_c:
            self.skip_w = Tensor(np.random.randn(out_c, in_c, 1, 1).astype(np.float32) * 0.02)
            self.skip_w.requires_grad = True; self._parameters['skip_w'] = self.skip_w

    def forward(self, x: Tensor) -> Tensor:
        out = self.double_conv(x)._data
        skip = x._data
        if self.skip_w is not None:
            skip = _conv2d(skip, self.skip_w._data, None, stride=1, padding=0)
        return Tensor(_relu(out + skip))


class ResUNet(Module):
    """ResU-Net — residual blocks in both encoder and decoder paths.

    Improves gradient flow for deeper networks. Useful for 3D medical imaging
    tasks when adapted to 3D convolutions.

    Args:
        in_channels: input channels
        out_channels: segmentation output channels
        base_channels: initial feature maps
        depth: encoder depth
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 2,
                 base_channels: int = 64, depth: int = 4):
        super().__init__()
        self.depth = depth
        self.encoder_blocks = []
        in_c = in_channels
        channel_list = []
        c = base_channels
        for i in range(depth):
            blk = ResConvBlock(in_c, c)
            self.encoder_blocks.append(blk)
            self._modules[f'res_enc_{i}'] = blk
            channel_list.append(c)
            in_c = c
            c = min(c * 2, 1024)
        self.bottleneck = ResConvBlock(in_c, c)
        self._modules['bottleneck'] = self.bottleneck
        in_c = c
        self.decoder_blocks = []
        self.up_convs_w = []
        for i, skip_c in reversed(list(enumerate(channel_list))):
            uc_w = Tensor(np.random.randn(in_c, in_c // 2, 1, 1).astype(np.float32) * 0.02)
            uc_w.requires_grad = True; self._parameters[f'up_w{i}'] = uc_w
            self.up_convs_w.append(uc_w)
            blk = ResConvBlock(in_c // 2 + skip_c, skip_c)
            self.decoder_blocks.append(blk)
            self._modules[f'res_dec_{i}'] = blk
            in_c = skip_c
        self.out_w = Tensor(np.random.randn(out_channels, in_c, 1, 1).astype(np.float32) * 0.02)
        self.out_b = Tensor(np.zeros(out_channels, dtype=np.float32))
        self.out_w.requires_grad = True; self.out_b.requires_grad = True
        self._parameters['out_w'] = self.out_w; self._parameters['out_b'] = self.out_b

    def forward(self, x: Tensor) -> Tensor:
        d = x._data
        skips = []
        for blk in self.encoder_blocks:
            d = blk(Tensor(d))._data
            skips.append(d)
            d = _maxpool2d(d)
        d = self.bottleneck(Tensor(d))._data
        for uc_w, blk, skip in zip(self.up_convs_w, self.decoder_blocks, reversed(skips)):
            d = _upsample_bilinear(d)
            d = _relu(_conv2d(d, uc_w._data, None, stride=1, padding=0))
            h, w = min(d.shape[2], skip.shape[2]), min(d.shape[3], skip.shape[3])
            d = np.concatenate([d[:, :, :h, :w], skip[:, :, :h, :w]], axis=1)
            d = blk(Tensor(d))._data
        return Tensor(_conv2d(d, self.out_w._data, self.out_b._data, padding=0))


# ── Attention U-Net ───────────────────────────────────────────────────────────

class AttentionGate(Module):
    """Attention gate — Schlemper et al. 2019.

    Learns to focus on relevant regions of the skip connection features
    given the upsampled decoder features as a gate signal.

    g: gating signal (decoder), x: skip connection (encoder).
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        # Theta_x: skip → int
        self.theta_w = Tensor(np.random.randn(F_int, F_l, 1, 1).astype(np.float32) * 0.02)
        self.theta_w.requires_grad = True; self._parameters['theta_w'] = self.theta_w
        # Phi_g: gate → int
        self.phi_w = Tensor(np.random.randn(F_int, F_g, 1, 1).astype(np.float32) * 0.02)
        self.phi_w.requires_grad = True; self._parameters['phi_w'] = self.phi_w
        # Psi: int → 1
        self.psi_w = Tensor(np.random.randn(1, F_int, 1, 1).astype(np.float32) * 0.02)
        self.psi_b = Tensor(np.zeros(1, dtype=np.float32))
        self.psi_w.requires_grad = True; self.psi_b.requires_grad = True
        self._parameters['psi_w'] = self.psi_w; self._parameters['psi_b'] = self.psi_b

    def forward(self, g: Tensor, x: Tensor) -> Tensor:
        """
        Args:
            g: (B, F_g, H, W) gating signal
            x: (B, F_l, H', W') skip connection (H', W' >= H, W)
        Returns:
            attended x: (B, F_l, H', W')
        """
        x1 = _conv2d(x._data, self.theta_w._data, None, stride=1, padding=0)
        g1 = _conv2d(g._data, self.phi_w._data, None, stride=1, padding=0)
        # Match spatial size of x1 and g1
        h = min(x1.shape[2], g1.shape[2])
        w = min(x1.shape[3], g1.shape[3])
        x1 = x1[:, :, :h, :w]
        g1 = g1[:, :, :h, :w]
        psi = _relu(x1 + g1)
        psi = _conv2d(psi, self.psi_w._data, self.psi_b._data, padding=0)
        alpha = _sigmoid(psi)   # attention map (B, 1, H, W)
        # Upsample alpha to match x spatial size
        if alpha.shape[2] != x._data.shape[2] or alpha.shape[3] != x._data.shape[3]:
            scale_h = x._data.shape[2] // alpha.shape[2]
            scale_w = x._data.shape[3] // alpha.shape[3]
            if scale_h > 1:
                alpha = alpha.repeat(scale_h, axis=2).repeat(scale_w, axis=3)
            alpha = alpha[:, :, :x._data.shape[2], :x._data.shape[3]]
        return Tensor(x._data * alpha)


class AttentionUNet(Module):
    """Attention U-Net — adds attention gates on skip connections.

    Useful for segmentation tasks where target structures are small or
    localized (e.g. organs in CT/MRI scans).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 2,
                 base_channels: int = 64, depth: int = 4):
        super().__init__()
        self.depth = depth
        self.encoder_blocks = []
        in_c = in_channels
        channel_list = []
        c = base_channels
        for i in range(depth):
            blk = DoubleConv(in_c, c)
            self.encoder_blocks.append(blk)
            self._modules[f'enc_{i}'] = blk
            channel_list.append(c)
            in_c = c; c = min(c * 2, 1024)
        self.bottleneck = DoubleConv(in_c, c)
        self._modules['bottleneck'] = self.bottleneck
        in_c = c
        self.decoder_blocks = []
        self.up_convs_w = []
        self.attn_gates = []
        for i, skip_c in reversed(list(enumerate(channel_list))):
            uc_w = Tensor(np.random.randn(in_c, in_c // 2, 1, 1).astype(np.float32) * 0.02)
            uc_w.requires_grad = True; self._parameters[f'up_w{i}'] = uc_w
            self.up_convs_w.append(uc_w)
            gate = AttentionGate(in_c // 2, skip_c, skip_c // 2)
            self.attn_gates.append(gate)
            self._modules[f'attn_{i}'] = gate
            blk = DoubleConv(in_c // 2 + skip_c, skip_c)
            self.decoder_blocks.append(blk)
            self._modules[f'dec_{i}'] = blk
            in_c = skip_c
        self.out_w = Tensor(np.random.randn(out_channels, in_c, 1, 1).astype(np.float32) * 0.02)
        self.out_b = Tensor(np.zeros(out_channels, dtype=np.float32))
        self.out_w.requires_grad = True; self.out_b.requires_grad = True
        self._parameters['out_w'] = self.out_w; self._parameters['out_b'] = self.out_b

    def forward(self, x: Tensor) -> Tensor:
        d = x._data
        skips = []
        for blk in self.encoder_blocks:
            d = blk(Tensor(d))._data
            skips.append(d)
            d = _maxpool2d(d)
        d = self.bottleneck(Tensor(d))._data
        for uc_w, attn, blk, skip in zip(
                self.up_convs_w, self.attn_gates, self.decoder_blocks, reversed(skips)):
            d = _upsample_bilinear(d)
            g = Tensor(_relu(_conv2d(d, uc_w._data, None, stride=1, padding=0)))
            # Apply attention gate
            skip_att = attn(g, Tensor(skip))._data
            h, w = min(g._data.shape[2], skip_att.shape[2]), min(g._data.shape[3], skip_att.shape[3])
            d = np.concatenate([g._data[:, :, :h, :w], skip_att[:, :, :h, :w]], axis=1)
            d = blk(Tensor(d))._data
        return Tensor(_conv2d(d, self.out_w._data, self.out_b._data, padding=0))
