"""U-Net architectures — vanilla U-Net, ResU-Net, U-Net with attention gates."""
from __future__ import annotations
import math as _math
import random as _random
from typing import List, Optional, Tuple
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


def _conv2d(x: NDArray, w: NDArray, b=None, stride=1, padding=1) -> NDArray:
    """Pure-NDArray conv2d (NCHW)."""
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    H_out = (H + 2*padding - kH) // stride + 1
    W_out = (W + 2*padding - kW) // stride + 1
    xf = x.to_flat_list(); wf = w.to_flat_list()
    if padding > 0:
        from vectorphile.core.tensor import _nd_pad
        x = _nd_pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
        H_p = H + 2*padding; W_p = W + 2*padding
        xf = x.to_flat_list()
    else:
        H_p, W_p = H, W
    out = [0.0] * (N * C_out * H_out * W_out)
    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    s = 0.0
                    for ci in range(C_in):
                        for ki in range(kH):
                            for kj in range(kW):
                                s += xf[n*C_in*H_p*W_p + ci*H_p*W_p + (i*stride+ki)*W_p + j*stride+kj] * \
                                     wf[co*C_in*kH*kW + ci*kH*kW + ki*kW + kj]
                    if b is not None:
                        bf = b.to_flat_list() if hasattr(b, 'to_flat_list') else list(b)
                        s += bf[co]
                    out[n*C_out*H_out*W_out + co*H_out*W_out + i*W_out + j] = s
    return _nd_arr(out, dtype='float32').reshape((N, C_out, H_out, W_out))


def _relu(x: NDArray) -> NDArray:
    flat = x.to_flat_list()
    return _nd_arr([max(0.0, v) for v in flat], dtype='float32').reshape(x.shape)

def _sigmoid(x: NDArray) -> NDArray:
    flat = x.to_flat_list()
    return _nd_arr([1/(1+_math.exp(-v)) for v in flat], dtype='float32').reshape(x.shape)


def _maxpool2d(x: NDArray, k=2, s=2) -> NDArray:
    N, C, H, W = x.shape
    H_o = (H - k) // s + 1; W_o = (W - k) // s + 1
    xf = x.to_flat_list(); out = []
    for n in range(N):
        for c in range(C):
            for i in range(H_o):
                for j in range(W_o):
                    mx = -1e30
                    for ki in range(k):
                        for kj in range(k):
                            mx = max(mx, xf[n*C*H*W + c*H*W + (i*s+ki)*W + j*s+kj])
                    out.append(mx)
    return _nd_arr(out, dtype='float32').reshape((N, C, H_o, W_o))


def _upsample_bilinear(x: NDArray, scale=2) -> NDArray:
    N, C, H, W = x.shape
    H2, W2 = H*scale, W*scale
    xf = x.to_flat_list(); out = []
    for n in range(N):
        for c in range(C):
            for i in range(H2):
                for j in range(W2):
                    si = min(i//scale, H-1); sj = min(j//scale, W-1)
                    out.append(xf[n*C*H*W + c*H*W + si*W + sj])
    return _nd_arr(out, dtype='float32').reshape((N, C, H2, W2))


def _batch_norm2d(x: NDArray, w, b, rm, rv, training, mom=0.1, eps=1e-5) -> NDArray:
    N, C, H, W = x.shape
    xf = x.to_flat_list()
    rmf = rm.to_flat_list() if hasattr(rm, 'to_flat_list') else list(rm)
    rvf = rv.to_flat_list() if hasattr(rv, 'to_flat_list') else list(rv)
    cnt = N * H * W
    if training:
        mean_c = [0.0]*C; var_c = [0.0]*C
        for n in range(N):
            for c in range(C):
                for hw in range(H*W):
                    mean_c[c] += xf[n*C*H*W + c*H*W + hw]
        mean_c = [v/cnt for v in mean_c]
        for n in range(N):
            for c in range(C):
                for hw in range(H*W):
                    var_c[c] += (xf[n*C*H*W + c*H*W + hw] - mean_c[c])**2
        var_c = [v/cnt for v in var_c]
        rmf = [(1-mom)*rmf[c]+mom*mean_c[c] for c in range(C)]
        rvf = [(1-mom)*rvf[c]+mom*var_c[c] for c in range(C)]
        try:
            import array as _arr
            rm._buf[:] = _arr.array('f', [float(v) for v in rmf])
            rv._buf[:] = _arr.array('f', [float(v) for v in rvf])
        except Exception: pass
    else:
        mean_c = rmf; var_c = rvf
    wf = w.to_flat_list() if hasattr(w, 'to_flat_list') else list(w)
    bf = b.to_flat_list() if hasattr(b, 'to_flat_list') else list(b)
    out = []
    for n in range(N):
        for c in range(C):
            inv = 1.0/_math.sqrt(var_c[c]+eps)
            for hw in range(H*W):
                out.append((xf[n*C*H*W + c*H*W + hw] - mean_c[c])*inv*wf[c] + bf[c])
    return _nd_arr(out, dtype='float32').reshape((N, C, H, W))


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
            w = Tensor(_nd_randn((oc, ic, 3, 3), _math.sqrt(2/(ic*9))))
            w.requires_grad = True; self._parameters[f'conv_w{i}'] = w; self.convs_w.append(w)
            bw = Tensor(_nd_ones((oc,))); bw.requires_grad = True
            bb = Tensor(_nd_zeros((oc,))); bb.requires_grad = True
            self._parameters[f'bn_w{i}'] = bw; self._parameters[f'bn_b{i}'] = bb
            self.bn_w.append(bw); self.bn_b.append(bb)
            self.rm.append(_nd_zeros((oc,)))
            self.rv.append(_nd_ones((oc,)))

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
            uc_w = Tensor(_nd_randn((in_c, in_c // 2, 1, 1), 0.02))
            uc_w.requires_grad = True; self._parameters[f'up_conv_w{i}'] = uc_w
            self.up_convs_w.append(uc_w)
            blk = DoubleConv(in_c // 2 + skip_c, skip_c)
            self.decoder_blocks.append(blk)
            self._modules[f'dec_{i}'] = blk
            in_c = skip_c
        # Output
        self.out_w = Tensor(_nd_randn((out_channels, in_c, 1, 1), 0.02))
        self.out_b = Tensor(_nd_zeros((out_channels,)))
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
            from vectorphile.core.tensor import _nd_concatenate
            d = _nd_concatenate([d_crop, skip_crop], axis=1)
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
            self.skip_w = Tensor(_nd_randn((out_c, in_c, 1, 1), 0.02))
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
            uc_w = Tensor(_nd_randn((in_c, in_c // 2, 1, 1), 0.02))
            uc_w.requires_grad = True; self._parameters[f'up_w{i}'] = uc_w
            self.up_convs_w.append(uc_w)
            blk = ResConvBlock(in_c // 2 + skip_c, skip_c)
            self.decoder_blocks.append(blk)
            self._modules[f'res_dec_{i}'] = blk
            in_c = skip_c
        self.out_w = Tensor(_nd_randn((out_channels, in_c, 1, 1), 0.02))
        self.out_b = Tensor(_nd_zeros((out_channels,)))
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
            from vectorphile.core.tensor import _nd_concatenate
            d = _nd_concatenate([d[:, :, :h, :w], skip[:, :, :h, :w]], axis=1)
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
        self.theta_w = Tensor(_nd_randn((F_int, F_l, 1, 1), 0.02))
        self.theta_w.requires_grad = True; self._parameters['theta_w'] = self.theta_w
        # Phi_g: gate → int
        self.phi_w = Tensor(_nd_randn((F_int, F_g, 1, 1), 0.02))
        self.phi_w.requires_grad = True; self._parameters['phi_w'] = self.phi_w
        # Psi: int → 1
        self.psi_w = Tensor(_nd_randn((1, F_int, 1, 1), 0.02))
        self.psi_b = Tensor(_nd_zeros((1,)))
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
                from vectorphile.core.tensor import _nd_tile
                alpha = _nd_tile(alpha, (1, 1, scale_h, scale_w))
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
            uc_w = Tensor(_nd_randn((in_c, in_c // 2, 1, 1), 0.02))
            uc_w.requires_grad = True; self._parameters[f'up_w{i}'] = uc_w
            self.up_convs_w.append(uc_w)
            gate = AttentionGate(in_c // 2, skip_c, skip_c // 2)
            self.attn_gates.append(gate)
            self._modules[f'attn_{i}'] = gate
            blk = DoubleConv(in_c // 2 + skip_c, skip_c)
            self.decoder_blocks.append(blk)
            self._modules[f'dec_{i}'] = blk
            in_c = skip_c
        self.out_w = Tensor(_nd_randn((out_channels, in_c, 1, 1), 0.02))
        self.out_b = Tensor(_nd_zeros((out_channels,)))
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
            from vectorphile.core.tensor import _nd_concatenate
            d = _nd_concatenate([g._data[:, :, :h, :w], skip_att[:, :, :h, :w]], axis=1)
            d = blk(Tensor(d))._data
        return Tensor(_conv2d(d, self.out_w._data, self.out_b._data, padding=0))
