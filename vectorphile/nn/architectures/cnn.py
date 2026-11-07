"""CNN architectures — LeNet, VGG, ResNet, DenseNet, MobileNetV2, EfficientNet."""
from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple, Callable
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


# ── Primitives ────────────────────────────────────────────────────────────────

def _conv2d(x: np.ndarray, w: np.ndarray, b: Optional[np.ndarray],
            stride: int, padding: int) -> np.ndarray:
    """Pure-numpy conv2d (NCHW)."""
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1
    if padding > 0:
        x = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
    out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
    for i in range(H_out):
        for j in range(W_out):
            patch = x[:, :, i*stride:i*stride+kH, j*stride:j*stride+kW]  # (N, C_in, kH, kW)
            out[:, :, i, j] = np.einsum('nchw,ochw->no', patch, w)
    if b is not None:
        out += b[None, :, None, None]
    return out


def _batch_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                running_mean: np.ndarray, running_var: np.ndarray,
                training: bool, momentum: float = 0.1, eps: float = 1e-5) -> np.ndarray:
    if training:
        axes = (0, 2, 3) if x.ndim == 4 else (0,)
        mean = x.mean(axis=axes, keepdims=True)
        var = x.var(axis=axes, keepdims=True)
        m = x.mean(axis=axes)
        v = x.var(axis=axes)
        running_mean[:] = (1 - momentum) * running_mean + momentum * m
        running_var[:]  = (1 - momentum) * running_var  + momentum * v
    else:
        shape = (-1,) + (1,) * (x.ndim - 1)
        mean = running_mean.reshape(shape)
        var  = running_var.reshape(shape)
    x_norm = (x - mean) / np.sqrt(var + eps)
    if x.ndim == 4:
        return x_norm * weight[None, :, None, None] + bias[None, :, None, None]
    return x_norm * weight + bias


def _relu(x: np.ndarray) -> np.ndarray: return np.maximum(x, 0)
def _silu(x: np.ndarray) -> np.ndarray: return x * (1 / (1 + np.exp(-x)))

def _maxpool2d(x: np.ndarray, kernel_size: int, stride: int) -> np.ndarray:
    N, C, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    out = np.zeros((N, C, H_out, W_out), dtype=np.float32)
    for i in range(H_out):
        for j in range(W_out):
            out[:, :, i, j] = x[:, :, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size].max(axis=(2,3))
    return out

def _avgpool2d_global(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=(2, 3))


class ConvBNReLU(Module):
    """Conv2d → BatchNorm → ReLU (or SiLU). Used across many CNN families."""

    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, groups: int = 1,
                 bias: bool = False, activation: str = 'relu'):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.activation = activation
        scale = np.sqrt(2.0 / (in_c * kernel_size * kernel_size))
        self.weight = Tensor(np.random.randn(out_c, in_c // groups, kernel_size, kernel_size).astype(np.float32) * scale)
        self.weight.requires_grad = True
        self._parameters['weight'] = self.weight
        self.bn_w = Tensor(np.ones(out_c, dtype=np.float32))
        self.bn_b = Tensor(np.zeros(out_c, dtype=np.float32))
        self.bn_w.requires_grad = True
        self.bn_b.requires_grad = True
        self._parameters['bn_weight'] = self.bn_w
        self._parameters['bn_bias'] = self.bn_b
        self._running_mean = np.zeros(out_c, dtype=np.float32)
        self._running_var  = np.ones(out_c, dtype=np.float32)

    def forward(self, x: Tensor) -> Tensor:
        out = _conv2d(x._data, self.weight._data, None, self.stride, self.padding)
        out = _batch_norm(out, self.bn_w._data, self.bn_b._data,
                          self._running_mean, self._running_var, self.training)
        if self.activation == 'relu':
            out = _relu(out)
        elif self.activation == 'silu':
            out = _silu(out)
        return Tensor(out)


# ── LeNet-5 ───────────────────────────────────────────────────────────────────

class LeNet5(Module):
    """LeNet-5 — LeCun et al. 1998. Classic CNN for 32x32 inputs."""

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        # C1: 1 → 6, 5x5
        self.c1_w = Tensor(np.random.randn(6, in_channels, 5, 5).astype(np.float32) * 0.1)
        self.c1_b = Tensor(np.zeros(6, dtype=np.float32))
        # C3: 6 → 16, 5x5
        self.c3_w = Tensor(np.random.randn(16, 6, 5, 5).astype(np.float32) * 0.1)
        self.c3_b = Tensor(np.zeros(16, dtype=np.float32))
        # FC
        self.f5_w = Tensor(np.random.randn(400, 120).astype(np.float32) * 0.1)
        self.f5_b = Tensor(np.zeros(120, dtype=np.float32))
        self.f6_w = Tensor(np.random.randn(120, 84).astype(np.float32) * 0.1)
        self.f6_b = Tensor(np.zeros(84, dtype=np.float32))
        self.out_w = Tensor(np.random.randn(84, num_classes).astype(np.float32) * 0.1)
        self.out_b = Tensor(np.zeros(num_classes, dtype=np.float32))
        for n, p in [('c1_w',self.c1_w),('c1_b',self.c1_b),('c3_w',self.c3_w),
                     ('c3_b',self.c3_b),('f5_w',self.f5_w),('f5_b',self.f5_b),
                     ('f6_w',self.f6_w),('f6_b',self.f6_b),('out_w',self.out_w),('out_b',self.out_b)]:
            p.requires_grad = True
            self._parameters[n] = p

    def forward(self, x: Tensor) -> Tensor:
        d = x._data
        d = _relu(_conv2d(d, self.c1_w._data, self.c1_b._data, 1, 0))
        d = _maxpool2d(d, 2, 2)
        d = _relu(_conv2d(d, self.c3_w._data, self.c3_b._data, 1, 0))
        d = _maxpool2d(d, 2, 2)
        d = d.reshape(d.shape[0], -1)
        d = _relu(d @ self.f5_w._data + self.f5_b._data)
        d = _relu(d @ self.f6_w._data + self.f6_b._data)
        return Tensor(d @ self.out_w._data + self.out_b._data)


# ── VGG ───────────────────────────────────────────────────────────────────────

class VGGBlock(Module):
    def __init__(self, in_c: int, out_c: int, num_convs: int):
        super().__init__()
        self.convs = []
        for i in range(num_convs):
            c = ConvBNReLU(in_c if i == 0 else out_c, out_c)
            self.convs.append(c)
            self._modules[f'conv_{i}'] = c

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.convs:
            x = conv(x)
        return Tensor(_maxpool2d(x._data, 2, 2))


class VGG(Module):
    """VGG — Simonyan & Zisserman 2014.

    Configurations: vgg11, vgg13, vgg16, vgg19.
    """

    CFGS: Dict = {
        'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    def __init__(self, name: str = 'vgg16', num_classes: int = 1000, in_channels: int = 3):
        super().__init__()
        cfg = self.CFGS[name]
        self.features = []
        in_c = in_channels
        for i, v in enumerate(cfg):
            if v == 'M':
                continue
            # Group convs until next M
            pass
        # Build features using ConvBNReLU layers
        self.feature_layers = []
        in_c = in_channels
        for i, v in enumerate(cfg):
            if v == 'M':
                # Pooling handled by next layer, skip
                continue
            layer = ConvBNReLU(in_c, int(v))
            self.feature_layers.append(('pool' if v == 'M' else f'conv_{i}', v, layer))
            in_c = int(v)

        self._conv_layers = []
        in_c = in_channels
        idx = 0
        for v in cfg:
            if v != 'M':
                c = ConvBNReLU(in_c, int(v))
                self._conv_layers.append((v == 'M', c))
                self._modules[f'feature_{idx}'] = c
                in_c = int(v)
                idx += 1
            else:
                self._conv_layers.append((True, None))

        # Classifier
        self.fc1_w = Tensor(np.random.randn(512 * 7 * 7, 4096).astype(np.float32) * 0.01)
        self.fc1_b = Tensor(np.zeros(4096, dtype=np.float32))
        self.fc2_w = Tensor(np.random.randn(4096, 4096).astype(np.float32) * 0.01)
        self.fc2_b = Tensor(np.zeros(4096, dtype=np.float32))
        self.fc3_w = Tensor(np.random.randn(4096, num_classes).astype(np.float32) * 0.01)
        self.fc3_b = Tensor(np.zeros(num_classes, dtype=np.float32))
        for n, p in [('fc1_w',self.fc1_w),('fc1_b',self.fc1_b),
                     ('fc2_w',self.fc2_w),('fc2_b',self.fc2_b),
                     ('fc3_w',self.fc3_w),('fc3_b',self.fc3_b)]:
            p.requires_grad = True
            self._parameters[n] = p
        self._pooling = [(v == 'M') for v in cfg]
        self._cfg = cfg

    def forward(self, x: Tensor) -> Tensor:
        d = x._data
        conv_idx = 0
        for v in self._cfg:
            if v == 'M':
                d = _maxpool2d(d, 2, 2)
            else:
                is_pool, layer = self._conv_layers[conv_idx]
                d = layer(Tensor(d))._data
                conv_idx += 1
        d = _avgpool2d_global(d.reshape(d.shape[0], d.shape[1], -1, 1) if d.ndim == 3 else d)
        d = d.reshape(d.shape[0], -1)
        d = _relu(d @ self.fc1_w._data + self.fc1_b._data)
        d = _relu(d @ self.fc2_w._data + self.fc2_b._data)
        return Tensor(d @ self.fc3_w._data + self.fc3_b._data)


# ── ResNet ────────────────────────────────────────────────────────────────────

class BasicBlock(Module):
    """ResNet BasicBlock (2 × 3×3 conv). Used in ResNet-18, ResNet-34."""
    expansion = 1

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBNReLU(in_c, out_c, 3, stride, 1)
        self.conv2 = ConvBNReLU(out_c, out_c, 3, 1, 1, activation='none')
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = ConvBNReLU(in_c, out_c, 1, stride, 0, activation='none')
            self._modules['downsample'] = self.downsample
        self._modules.update({'conv1': self.conv1, 'conv2': self.conv2})

    def forward(self, x: Tensor) -> Tensor:
        identity = x._data
        out = self.conv1(x)._data
        out = _batch_norm(
            _conv2d(out, self.conv2.weight._data, None, self.conv2.stride, self.conv2.padding),
            self.conv2.bn_w._data, self.conv2.bn_b._data,
            self.conv2._running_mean, self.conv2._running_var, self.training
        )
        if self.downsample is not None:
            identity = self.downsample(x)._data
        return Tensor(_relu(out + identity))


class Bottleneck(Module):
    """ResNet Bottleneck (1×1 → 3×3 → 1×1). Used in ResNet-50/101/152."""
    expansion = 4

    def __init__(self, in_c: int, mid_c: int, stride: int = 1):
        super().__init__()
        out_c = mid_c * 4
        self.conv1 = ConvBNReLU(in_c, mid_c, 1, 1, 0)
        self.conv2 = ConvBNReLU(mid_c, mid_c, 3, stride, 1)
        self.conv3 = ConvBNReLU(mid_c, out_c, 1, 1, 0, activation='none')
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = ConvBNReLU(in_c, out_c, 1, stride, 0, activation='none')
            self._modules['downsample'] = self.downsample
        self._modules.update({'conv1': self.conv1, 'conv2': self.conv2, 'conv3': self.conv3})

    def forward(self, x: Tensor) -> Tensor:
        identity = x._data
        out = self.conv3(self.conv2(self.conv1(x)))._data
        if self.downsample is not None:
            identity = self.downsample(x)._data
        return Tensor(_relu(out + identity))


class ResNet(Module):
    """ResNet — He et al. 2015.

    Supported depths: 18, 34, 50, 101, 152.

    Args:
        depth: one of 18, 34, 50, 101, 152
        num_classes: output classes (default 1000 for ImageNet)
        in_channels: input channels
    """

    CONFIGS = {
        18:  (BasicBlock,    [2, 2, 2, 2],  64),
        34:  (BasicBlock,    [3, 4, 6, 3],  64),
        50:  (Bottleneck,    [3, 4, 6, 3],  64),
        101: (Bottleneck,    [3, 4, 23, 3], 64),
        152: (Bottleneck,    [3, 8, 36, 3], 64),
    }

    def __init__(self, depth: int = 50, num_classes: int = 1000, in_channels: int = 3):
        super().__init__()
        block_cls, layers, base = self.CONFIGS[depth]
        self.in_c = 64

        self.stem_conv = ConvBNReLU(in_channels, 64, 7, 2, 3)
        self.layer1 = self._make_layer(block_cls, base,     layers[0], stride=1)
        self.layer2 = self._make_layer(block_cls, base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block_cls, base * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block_cls, base * 8, layers[3], stride=2)
        fc_in = (base * 8) * block_cls.expansion
        self.fc_w = Tensor(np.random.randn(fc_in, num_classes).astype(np.float32) * 0.01)
        self.fc_b = Tensor(np.zeros(num_classes, dtype=np.float32))
        self.fc_w.requires_grad = True
        self.fc_b.requires_grad = True
        self._parameters.update({'fc_w': self.fc_w, 'fc_b': self.fc_b})
        self._modules['stem'] = self.stem_conv

    def _make_layer(self, block_cls, out_c: int, num_blocks: int, stride: int) -> List:
        layers = []
        in_c = self.in_c
        expected_out = out_c * block_cls.expansion
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            if block_cls == BasicBlock:
                b = block_cls(in_c, out_c, s)
            else:
                b = block_cls(in_c, out_c, s)
            layers.append(b)
            in_c = expected_out
            self._modules[f'resblock_{id(b)}'] = b
        self.in_c = expected_out
        return layers

    def forward(self, x: Tensor) -> Tensor:
        d = self.stem_conv(x)._data
        d = _maxpool2d(d, 3, 2)
        for block in self.layer1 + self.layer2 + self.layer3 + self.layer4:
            d = block(Tensor(d))._data
        d = _avgpool2d_global(d)
        d = d.reshape(d.shape[0], -1)
        return Tensor(d @ self.fc_w._data + self.fc_b._data)


# ── DenseNet ──────────────────────────────────────────────────────────────────

class DenseLayer(Module):
    """Single DenseNet layer: BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)."""

    def __init__(self, in_c: int, growth_rate: int, bn_size: int = 4):
        super().__init__()
        inter = bn_size * growth_rate
        self.conv1 = ConvBNReLU(in_c, inter, 1, 1, 0)
        self.conv2 = ConvBNReLU(inter, growth_rate, 3, 1, 1)
        self._modules.update({'conv1': self.conv1, 'conv2': self.conv2})

    def forward(self, x: Tensor) -> Tensor:
        new_feat = self.conv2(self.conv1(x))._data
        return Tensor(np.concatenate([x._data, new_feat], axis=1))


class DenseBlock(Module):
    def __init__(self, num_layers: int, in_c: int, growth_rate: int):
        super().__init__()
        self.layers = []
        c = in_c
        for i in range(num_layers):
            l = DenseLayer(c, growth_rate)
            self.layers.append(l)
            self._modules[f'dense_{i}'] = l
            c += growth_rate
        self.out_channels = c

    def forward(self, x: Tensor) -> Tensor:
        for l in self.layers:
            x = l(x)
        return x


class Transition(Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = ConvBNReLU(in_c, out_c, 1, 1, 0)
        self._modules['conv'] = self.conv

    def forward(self, x: Tensor) -> Tensor:
        d = self.conv(x)._data
        N, C, H, W = d.shape
        return Tensor(d.reshape(N, C, H//2, 2, W//2, 2).mean(axis=(3, 5)))


class DenseNet(Module):
    """DenseNet — Huang et al. 2016. Dense connections within blocks.

    Configurations: densenet121, densenet169, densenet201, densenet264.
    """

    CONFIGS = {
        121: [6, 12, 24, 16],
        169: [6, 12, 32, 32],
        201: [6, 12, 48, 32],
        264: [6, 12, 64, 48],
    }

    def __init__(self, depth: int = 121, num_classes: int = 1000,
                 growth_rate: int = 32, init_features: int = 64, in_channels: int = 3):
        super().__init__()
        block_config = self.CONFIGS[depth]
        self.stem = ConvBNReLU(in_channels, init_features, 7, 2, 3)
        self._modules['stem'] = self.stem
        num_features = init_features
        self.dense_blocks = []
        self.transitions = []
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, growth_rate)
            self.dense_blocks.append(block)
            self._modules[f'dense_block_{i}'] = block
            num_features = block.out_channels
            if i != len(block_config) - 1:
                trans = Transition(num_features, num_features // 2)
                self.transitions.append(trans)
                self._modules[f'transition_{i}'] = trans
                num_features = num_features // 2
        self.fc_w = Tensor(np.random.randn(num_features, num_classes).astype(np.float32) * 0.01)
        self.fc_b = Tensor(np.zeros(num_classes, dtype=np.float32))
        self.fc_w.requires_grad = True
        self.fc_b.requires_grad = True
        self._parameters.update({'fc_w': self.fc_w, 'fc_b': self.fc_b})

    def forward(self, x: Tensor) -> Tensor:
        d = Tensor(_maxpool2d(self.stem(x)._data, 3, 2))
        for i, block in enumerate(self.dense_blocks):
            d = block(d)
            if i < len(self.transitions):
                d = self.transitions[i](d)
        d = _avgpool2d_global(d._data)
        d = d.reshape(d.shape[0], -1)
        return Tensor(d @ self.fc_w._data + self.fc_b._data)


# ── MobileNetV2 ───────────────────────────────────────────────────────────────

class InvertedResidual(Module):
    """MobileNetV2 inverted residual block with linear bottleneck."""

    def __init__(self, in_c: int, out_c: int, stride: int, expand_ratio: int):
        super().__init__()
        hidden = in_c * expand_ratio
        self.use_res = stride == 1 and in_c == out_c
        self.layers = []
        if expand_ratio != 1:
            self.layers.append(ConvBNReLU(in_c, hidden, 1, 1, 0, activation='silu'))
        # Depthwise conv (groups=hidden — approximate with separate conv here)
        self.layers.append(ConvBNReLU(hidden, hidden, 3, stride, 1, activation='silu'))
        self.layers.append(ConvBNReLU(hidden, out_c, 1, 1, 0, activation='none'))
        for i, l in enumerate(self.layers):
            self._modules[f'layer_{i}'] = l

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for l in self.layers:
            out = l(out)
        if self.use_res:
            return Tensor(x._data + out._data)
        return out


class MobileNetV2(Module):
    """MobileNetV2 — Sandler et al. 2018.

    Lightweight CNN for mobile/edge deployment via inverted residual blocks
    and linear bottlenecks.
    """

    # (expand_ratio, out_c, num_blocks, stride)
    _SETTINGS = [
        (1, 16,  1, 1),
        (6, 24,  2, 2),
        (6, 32,  3, 2),
        (6, 64,  4, 2),
        (6, 96,  3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0, in_channels: int = 3):
        super().__init__()
        input_channel = int(32 * width_mult)
        self.stem = ConvBNReLU(in_channels, input_channel, 3, 2, 1, activation='silu')
        self._modules['stem'] = self.stem
        self.blocks = []
        for t, c, n, s in self._SETTINGS:
            out_c = int(c * width_mult)
            for i in range(n):
                b = InvertedResidual(input_channel, out_c, s if i == 0 else 1, t)
                self.blocks.append(b)
                self._modules[f'block_{len(self.blocks)}'] = b
                input_channel = out_c
        last_c = int(1280 * max(1.0, width_mult))
        self.last_conv = ConvBNReLU(input_channel, last_c, 1, 1, 0, activation='silu')
        self._modules['last_conv'] = self.last_conv
        self.fc_w = Tensor(np.random.randn(last_c, num_classes).astype(np.float32) * 0.01)
        self.fc_b = Tensor(np.zeros(num_classes, dtype=np.float32))
        self.fc_w.requires_grad = True
        self.fc_b.requires_grad = True
        self._parameters.update({'fc_w': self.fc_w, 'fc_b': self.fc_b})

    def forward(self, x: Tensor) -> Tensor:
        d = self.stem(x)
        for block in self.blocks:
            d = block(d)
        d = self.last_conv(d)
        d = Tensor(_avgpool2d_global(d._data))
        d = Tensor(d._data.reshape(d._data.shape[0], -1))
        return Tensor(d._data @ self.fc_w._data + self.fc_b._data)


# ── EfficientNet ──────────────────────────────────────────────────────────────

class SEBlock(Module):
    """Squeeze-and-Excitation block used in EfficientNet."""

    def __init__(self, in_c: int, se_ratio: float = 0.25):
        super().__init__()
        se_c = max(1, int(in_c * se_ratio))
        self.fc1_w = Tensor(np.random.randn(in_c, se_c).astype(np.float32) * 0.01)
        self.fc2_w = Tensor(np.random.randn(se_c, in_c).astype(np.float32) * 0.01)
        for n, p in [('fc1_w', self.fc1_w), ('fc2_w', self.fc2_w)]:
            p.requires_grad = True
            self._parameters[n] = p

    def forward(self, x: Tensor) -> Tensor:
        d = x._data
        se = d.mean(axis=(2, 3))                    # (N, C)
        se = _silu(se @ self.fc1_w._data)
        se = 1 / (1 + np.exp(-(se @ self.fc2_w._data)))  # sigmoid
        return Tensor(d * se[:, :, None, None])


class MBConv(Module):
    """EfficientNet MBConv block: expand → depthwise → SE → project."""

    def __init__(self, in_c: int, out_c: int, stride: int, expand_ratio: int,
                 se_ratio: float = 0.25):
        super().__init__()
        expanded = in_c * expand_ratio
        self.use_res = stride == 1 and in_c == out_c
        self.expand = ConvBNReLU(in_c, expanded, 1, 1, 0, activation='silu') if expand_ratio > 1 else None
        self.depthwise = ConvBNReLU(expanded, expanded, 3, stride, 1, activation='silu')
        self.se = SEBlock(expanded, se_ratio)
        self.project = ConvBNReLU(expanded, out_c, 1, 1, 0, activation='none')
        for k, v in [('depthwise', self.depthwise), ('se', self.se), ('project', self.project)]:
            self._modules[k] = v
        if self.expand:
            self._modules['expand'] = self.expand

    def forward(self, x: Tensor) -> Tensor:
        out = x
        if self.expand:
            out = self.expand(out)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_res:
            return Tensor(x._data + out._data)
        return out


class EfficientNet(Module):
    """EfficientNet — Tan & Le 2019.

    Compound scaling of width, depth, and resolution.
    B0–B7 configurations via width_mult and depth_mult.

    Args:
        width_mult: network width multiplier
        depth_mult: network depth multiplier
        num_classes: output classes
    """

    # (expand_ratio, channels, num_layers, stride, kernel_size)
    _BASE = [
        (1, 16,  1, 1, 3),
        (6, 24,  2, 2, 3),
        (6, 40,  2, 2, 5),
        (6, 80,  3, 2, 3),
        (6, 112, 3, 1, 5),
        (6, 192, 4, 2, 5),
        (6, 320, 1, 1, 3),
    ]

    CONFIGS = {
        'b0': dict(width_mult=1.0,  depth_mult=1.0,  dropout=0.2),
        'b1': dict(width_mult=1.0,  depth_mult=1.1,  dropout=0.2),
        'b2': dict(width_mult=1.1,  depth_mult=1.2,  dropout=0.3),
        'b3': dict(width_mult=1.2,  depth_mult=1.4,  dropout=0.3),
        'b4': dict(width_mult=1.4,  depth_mult=1.8,  dropout=0.4),
        'b5': dict(width_mult=1.6,  depth_mult=2.2,  dropout=0.4),
        'b6': dict(width_mult=1.8,  depth_mult=2.6,  dropout=0.5),
        'b7': dict(width_mult=2.0,  depth_mult=3.1,  dropout=0.5),
    }

    def __init__(self, width_mult: float = 1.0, depth_mult: float = 1.0,
                 num_classes: int = 1000, dropout: float = 0.2, in_channels: int = 3):
        super().__init__()
        stem_c = int(32 * width_mult)
        self.stem = ConvBNReLU(in_channels, stem_c, 3, 2, 1, activation='silu')
        self._modules['stem'] = self.stem
        self.blocks = []
        in_c = stem_c
        for t, c, n, s, k in self._BASE:
            out_c = int(c * width_mult)
            num_layers = max(1, int(np.ceil(n * depth_mult)))
            for i in range(num_layers):
                b = MBConv(in_c, out_c, s if i == 0 else 1, t)
                self.blocks.append(b)
                self._modules[f'block_{len(self.blocks)}'] = b
                in_c = out_c
        last_c = int(1280 * width_mult)
        self.last_conv = ConvBNReLU(in_c, last_c, 1, 1, 0, activation='silu')
        self._modules['last_conv'] = self.last_conv
        self.fc_w = Tensor(np.random.randn(last_c, num_classes).astype(np.float32) * 0.01)
        self.fc_b = Tensor(np.zeros(num_classes, dtype=np.float32))
        self.fc_w.requires_grad = True
        self.fc_b.requires_grad = True
        self._parameters.update({'fc_w': self.fc_w, 'fc_b': self.fc_b})

    @classmethod
    def from_config(cls, name: str, **kwargs) -> 'EfficientNet':
        return cls(**{**cls.CONFIGS[name], **kwargs})

    def forward(self, x: Tensor) -> Tensor:
        d = self.stem(x)
        for block in self.blocks:
            d = block(d)
        d = self.last_conv(d)
        d = Tensor(_avgpool2d_global(d._data))
        d = Tensor(d._data.reshape(d._data.shape[0], -1))
        return Tensor(d._data @ self.fc_w._data + self.fc_b._data)
