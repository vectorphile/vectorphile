"""Quantization-Aware Training (QAT) for vectorphile.

Simulates quantization in the forward pass using fake-quantize ops with
straight-through estimators (STE) for gradient flow. After QAT training,
convert_qat() replaces fake-quantize layers with real integer weights.

Usage::

    from vectorphile.nn.qat import prepare_qat, convert_qat

    model = MyModel()
    model = prepare_qat(model, weight_bits=8, act_bits=8)
    # ... train model ...
    model = convert_qat(model)   # now uses INT8 weights
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


# ── Observer ──────────────────────────────────────────────────────────────────

class MovingAverageMinMaxObserver(Module):
    """Track min/max of activations via EMA for QAT calibration.

    Args:
        momentum: EMA momentum (higher = faster adaptation)
    """

    def __init__(self, momentum: float = 0.01):
        super().__init__()
        self.momentum = momentum
        self._min: float = float('inf')
        self._max: float = float('-inf')
        self._initialized = False

    def forward(self, x: Tensor) -> Tensor:
        """Update running statistics and return x unchanged."""
        x_min = float(x._data.min())
        x_max = float(x._data.max())
        if not self._initialized:
            self._min = x_min
            self._max = x_max
            self._initialized = True
        else:
            self._min = (1 - self.momentum) * self._min + self.momentum * x_min
            self._max = (1 - self.momentum) * self._max + self.momentum * x_max
        return x

    def get_scale_zp(self, bits: int = 8) -> Tuple[float, int]:
        """Compute scale and zero-point from observed min/max.

        Returns:
            (scale, zero_point) for affine quantization
        """
        q_min = -(2 ** (bits - 1))
        q_max = 2 ** (bits - 1) - 1
        r_min = self._min if self._initialized else -1.0
        r_max = self._max if self._initialized else 1.0
        scale = (r_max - r_min) / (q_max - q_min + 1e-9)
        zero_point = int(np.round(q_min - r_min / (scale + 1e-9)))
        zero_point = int(np.clip(zero_point, q_min, q_max))
        return scale, zero_point


# ── FakeQuantize ──────────────────────────────────────────────────────────────

class FakeQuantize(Module):
    """Simulate quantization in forward pass, straight-through for backward.

    STE: gradient passes through the quantization as if it were the identity.
    This allows gradient-based training while simulating quantization noise.

    Args:
        bits: quantization bit-width (8, 4, etc.)
        symmetric: symmetric (zero_point=0) or asymmetric quantization
        per_channel: per-output-channel scale/zp (for weights) vs per-tensor
        channel_axis: which axis is the output-channel axis (for per_channel)
    """

    def __init__(self, bits: int = 8, symmetric: bool = True,
                 per_channel: bool = False, channel_axis: int = 0):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.channel_axis = channel_axis
        self.observer = MovingAverageMinMaxObserver()
        self._modules['observer'] = self.observer
        self.enabled = True

    def forward(self, x: Tensor) -> Tensor:
        """Fake-quantize x: round to nearest integer level, then back to float."""
        if not self.enabled:
            return x

        self.observer(x)
        d = x._data

        q_min = -(2 ** (self.bits - 1)) if self.symmetric else 0
        q_max = 2 ** (self.bits - 1) - 1

        if self.per_channel:
            shape = [-1 if i == self.channel_axis else 1 for i in range(d.ndim)]
            axis = tuple(i for i in range(d.ndim) if i != self.channel_axis)
            r_min = d.min(axis=axis, keepdims=True).reshape(shape)
            r_max = d.max(axis=axis, keepdims=True).reshape(shape)
        else:
            r_min = d.min()
            r_max = d.max()

        if self.symmetric:
            abs_max = np.maximum(np.abs(r_min), np.abs(r_max))
            scale = abs_max / (q_max + 1e-9)
            zp = 0
        else:
            scale = (r_max - r_min) / (q_max - q_min + 1e-9)
            zp = q_min - r_min / (scale + 1e-9)

        # Quantize
        q = np.clip(np.round(d / (scale + 1e-9) + zp), q_min, q_max)
        # Dequantize (fake — still float32)
        x_fq = (q - zp) * scale

        # STE: in backward, treat as identity (grad passes through)
        # We store x_fq as output but the grad will pass to x unchanged
        out = Tensor(x_fq.astype(np.float32))
        # STE: copy grad_fn so backward gets identity gradient
        if x.requires_grad:
            out.requires_grad = True
            inp_ref = x
            def bwd(g):
                # Straight-through: pass grad through quantization unchanged
                return [g]
            out.grad_fn = ([x], bwd)
            out._is_leaf = False
        return out


# ── QAT-aware layers ──────────────────────────────────────────────────────────

class QATLinear(Module):
    """Linear layer with fake-quantized weights and activations.

    Args:
        in_features: input size
        out_features: output size
        weight_bits: weight quantization bits
        act_bits: activation quantization bits
        bias: use bias term
    """

    def __init__(self, in_features: int, out_features: int,
                 weight_bits: int = 8, act_bits: int = 8, bias: bool = True):
        super().__init__()
        s = np.sqrt(2.0 / in_features)
        self.weight = Tensor(np.random.randn(in_features, out_features).astype(np.float32) * s)
        self.weight.requires_grad = True
        self._parameters['weight'] = self.weight
        self._bias = None
        if bias:
            self._bias = Tensor(np.zeros(out_features, dtype=np.float32))
            self._bias.requires_grad = True
            self._parameters['bias'] = self._bias

        self.weight_fq = FakeQuantize(bits=weight_bits, symmetric=True, per_channel=True)
        self.act_fq    = FakeQuantize(bits=act_bits, symmetric=False)
        self._modules['weight_fq'] = self.weight_fq
        self._modules['act_fq']    = self.act_fq

    def forward(self, x: Tensor) -> Tensor:
        # Fake-quantize weights
        w_fq = self.weight_fq(self.weight)
        # Linear
        out = Tensor(x._data @ w_fq._data)
        if self._bias is not None:
            out = Tensor(out._data + self._bias._data)
        # Fake-quantize activations
        return self.act_fq(out)


class QATConv2d(Module):
    """Conv2d with fake-quantized weights and activations.

    Args:
        in_channels, out_channels: channel counts
        kernel_size: convolution kernel size
        stride, padding: convolution params
        weight_bits: weight quantization bits
        act_bits: activation quantization bits
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1,
                 weight_bits: int = 8, act_bits: int = 8, bias: bool = True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        s = np.sqrt(2.0 / (in_channels * kernel_size ** 2))
        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * s)
        self.weight.requires_grad = True
        self._parameters['weight'] = self.weight
        self._bias = None
        if bias:
            self._bias = Tensor(np.zeros(out_channels, dtype=np.float32))
            self._bias.requires_grad = True
            self._parameters['bias'] = self._bias

        self.weight_fq = FakeQuantize(bits=weight_bits, symmetric=True, per_channel=True)
        self.act_fq    = FakeQuantize(bits=act_bits, symmetric=False)
        self._modules['weight_fq'] = self.weight_fq
        self._modules['act_fq']    = self.act_fq

    def _conv2d(self, x, w):
        from vectorphile.backend import _numpy_conv2d
        b = self._bias._data if self._bias is not None else None
        return _numpy_conv2d(x, w, b, self.stride, self.padding)

    def forward(self, x: Tensor) -> Tensor:
        w_fq = self.weight_fq(self.weight)
        out = Tensor(self._conv2d(x._data, w_fq._data))
        return self.act_fq(out)


# ── Model preparation and conversion ─────────────────────────────────────────

def prepare_qat(model: Module, weight_bits: int = 8, act_bits: int = 8) -> Module:
    """Replace Linear and Conv2d layers in model with QAT-aware versions.

    Preserves existing weights. Returns the same model object (mutated).

    Args:
        model: Module to prepare
        weight_bits: quantization bits for weights
        act_bits: quantization bits for activations
    Returns:
        model with QAT layers inserted
    """
    try:
        from vectorphile.nn.layers import Linear, Conv2d
    except ImportError:
        return model

    for name, module in list(model._modules.items()):
        if isinstance(module, Linear):
            in_f  = module.weight._data.shape[0]
            out_f = module.weight._data.shape[1]
            qat_layer = QATLinear(in_f, out_f, weight_bits, act_bits,
                                   bias=hasattr(module, 'bias') and module.bias is not None)
            qat_layer.weight._data = module.weight._data.copy()
            if qat_layer._bias is not None and hasattr(module, 'bias') and module.bias is not None:
                qat_layer._bias._data = module.bias._data.copy()
            setattr(model, name, qat_layer)
            model._modules[name] = qat_layer
        elif isinstance(module, Conv2d):
            # Replace Conv2d with QATConv2d
            w = module.weight._data
            oc, ic, kh, kw = w.shape
            qat_layer = QATConv2d(ic, oc, kh,
                                   stride=getattr(module, 'stride', 1),
                                   padding=getattr(module, 'padding', 0),
                                   weight_bits=weight_bits, act_bits=act_bits)
            qat_layer.weight._data = w.copy()
            setattr(model, name, qat_layer)
            model._modules[name] = qat_layer
        else:
            # Recurse into submodules
            prepare_qat(module, weight_bits, act_bits)
    return model


def convert_qat(model: Module) -> Module:
    """Convert QAT fake-quantize layers to real INT8 weights.

    After QAT training is complete, this freezes quantized weights into
    integer representation (stored as float32 for compatibility).

    Args:
        model: QAT-prepared Module after training
    Returns:
        model with quantized weights (fake-quant disabled)
    """
    for name, module in model._modules.items():
        if isinstance(module, (QATLinear, QATConv2d)):
            # Apply fake quantization to get the final quantized values
            with_fq = module.weight_fq(module.weight)
            module.weight._data = with_fq._data.copy()
            # Disable fake quantization (use quantized weights as-is)
            module.weight_fq.enabled = False
            module.act_fq.enabled    = False
        else:
            convert_qat(module)
    return model
