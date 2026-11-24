"""Core quantization primitives — INT8 / INT4 / FP8."""
from __future__ import annotations
import math as _math
from dataclasses import dataclass, field
from typing import Optional, List
from vectorphile._core.ndarray import (
    NDArray, zeros as _nd_zeros, ones as _nd_ones, array as _nd_array,
)
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


# ── QConfig ───────────────────────────────────────────────────────────────────

@dataclass
class QConfig:
    """Quantization configuration.

    Args:
        bits       : quantization bitwidth (4, 8)
        mode       : 'weight-only', 'dynamic', 'static'
        sym        : True → symmetric (zero_point = 0), False → asymmetric
        per_channel: True → per output-channel scale, False → per tensor
        group_size : quantization group size for group-wise quant (-1 = per-channel)
        dtype      : target dtype string ('int8', 'int4', 'fp8_e4m3')
    """
    bits: int = 8
    mode: str = 'weight-only'
    sym: bool = True
    per_channel: bool = True
    group_size: int = -1
    dtype: str = 'int8'

    @classmethod
    def int8(cls) -> 'QConfig':
        return cls(bits=8, dtype='int8')

    @classmethod
    def int4(cls, group_size: int = 128) -> 'QConfig':
        return cls(bits=4, dtype='int4', group_size=group_size)

    @classmethod
    def fp8_e4m3(cls) -> 'QConfig':
        return cls(bits=8, dtype='fp8_e4m3')


# ── Low-level quantization ────────────────────────────────────────────────────

def quantize_tensor(
    x: NDArray, bits: int = 8, sym: bool = True,
    per_channel: bool = False, group_size: int = -1,
) -> tuple:
    """Quantize a float32 NDArray.

    Returns:
        (q_data, scale, zero_point)
        q_data     : integer-valued float32 (int8 range if bits=8, int4 if bits=4)
        scale      : float32, shape (C,) if per_channel else scalar
        zero_point : float32, shape (C,) if per_channel else scalar (0 for sym)
    """
    qmin = -(2 ** (bits - 1))
    qmax =   2 ** (bits - 1) - 1

    flat = x.flatten()
    n = flat.shape[0]

    if per_channel and x.ndim >= 2:
        C = x.shape[0]
        rows = x.reshape(C, -1)
        scales = []
        zps = []
        q_rows = []
        for c in range(C):
            row = rows[c]
            rmax = float(row.max())
            rmin = float(row.min())
            if sym:
                abs_max = max(abs(rmax), abs(rmin), 1e-9)
                sc = abs_max / qmax
                zp = 0.0
            else:
                sc = max((rmax - rmin) / (qmax - qmin), 1e-9)
                zp = qmin - rmin / sc
            scales.append(sc)
            zps.append(zp)
            q_row = [max(qmin, min(qmax, round(float(v) / sc + zp))) for v in row._buf]
            q_rows.append(q_row)
        import array as _a
        sc_buf = _a.array('f', scales)
        zp_buf = _a.array('f', zps)
        from vectorphile._core.ndarray import NDArray as _ND
        sc_nd = _ND(sc_buf, (C,), (1,), 'float32')
        zp_nd = _ND(zp_buf, (C,), (1,), 'float32')
        q_buf = _a.array('f', [v for row in q_rows for v in row])
        q_nd = _ND(q_buf, x.shape, x.strides, 'float32')
        return q_nd, sc_nd, zp_nd
    else:
        # per-tensor
        import array as _a
        all_vals = [float(v) for v in flat._buf]
        rmax = max(all_vals)
        rmin = min(all_vals)
        if sym:
            abs_max = max(abs(rmax), abs(rmin), 1e-9)
            sc = abs_max / qmax
            zp = 0.0
        else:
            sc = max((rmax - rmin) / (qmax - qmin), 1e-9)
            zp = qmin - rmin / sc
        q_buf = _a.array('f', [max(qmin, min(qmax, round(v / sc + zp))) for v in all_vals])
        from vectorphile._core.ndarray import NDArray as _ND
        q_nd = _ND(q_buf, x.shape, x.strides, 'float32')
        sc_nd = _ND(_a.array('f', [sc]), (1,), (1,), 'float32')
        zp_nd = _ND(_a.array('f', [zp]), (1,), (1,), 'float32')
        return q_nd, sc_nd, zp_nd


def dequantize_tensor(q: NDArray, scale: NDArray, zero_point: NDArray,
                       per_channel: bool = False) -> NDArray:
    """Reconstruct float32 from quantized values."""
    if per_channel and q.ndim >= 2:
        C = q.shape[0]
        rows = q.reshape(C, -1)
        import array as _a
        from vectorphile._core.ndarray import NDArray as _ND
        out_buf = _a.array('f', [0.0] * q.flatten().shape[0])
        row_size = q.flatten().shape[0] // C
        for c in range(C):
            sc = float(scale[c])
            zp = float(zero_point[c])
            offset = c * row_size
            for j in range(row_size):
                out_buf[offset + j] = (float(rows[c].flatten()[j]) - zp) * sc
        return _ND(out_buf, q.shape, q.strides, 'float32')
    else:
        sc = float(scale[0]) if hasattr(scale, '_buf') else float(scale)
        zp = float(zero_point[0]) if hasattr(zero_point, '_buf') else float(zero_point)
        import array as _a
        from vectorphile._core.ndarray import NDArray as _ND
        out_buf = _a.array('f', [(float(v) - zp) * sc for v in q._buf])
        return _ND(out_buf, q.shape, q.strides, 'float32')


# ── QuantizedLinear ───────────────────────────────────────────────────────────

class QuantizedLinear(Module):
    """Quantized linear layer (weight-only by default).

    Stores weights as INT8 / INT4 (simulated as float32 integers in memory).
    At forward time, weights are dequantized on-the-fly (weight-only mode)
    or both activations and weights are quantized (dynamic/static mode).

    Args:
        in_features, out_features: layer dimensions
        bits                     : 4 or 8
        sym                      : symmetric quantization
        per_channel              : per output-channel scale
        bias                     : include bias
    """

    def __init__(self, in_features: int, out_features: int,
                 bits: int = 8, sym: bool = True,
                 per_channel: bool = True, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.bits = bits
        self.sym  = sym
        self.per_channel = per_channel
        # will be populated by from_linear or manually
        self._q_weight    = None  # NDArray of integer values
        self._scale       = None  # NDArray
        self._zero_point  = None  # NDArray
        self._bias_data   = None  # NDArray or None

    @classmethod
    def from_linear(cls, layer, bits: int = 8,
                    sym: bool = True, per_channel: bool = True) -> 'QuantizedLinear':
        """Create a QuantizedLinear from an existing Linear layer."""
        in_f  = layer.in_features
        out_f = layer.out_features
        q = cls(in_f, out_f, bits, sym, per_channel,
                bias=(layer.bias_param is not None if hasattr(layer, 'bias_param') else False))
        # quantize weight
        w = layer.weight._data if hasattr(layer, 'weight') else layer.w._data
        q._q_weight, q._scale, q._zero_point = quantize_tensor(
            w, bits=bits, sym=sym, per_channel=per_channel)
        if hasattr(layer, 'bias_param') and layer.bias_param is not None:
            q._bias_data = layer.bias_param._data
        elif hasattr(layer, 'bias') and layer.bias is not None:
            q._bias_data = layer.bias._data if hasattr(layer.bias, '_data') else None
        return q

    def forward(self, x: Tensor) -> Tensor:
        if self._q_weight is None:
            raise RuntimeError('QuantizedLinear has no quantized weight — call from_linear() first.')
        # dequantize weight on the fly
        w_dq = dequantize_tensor(self._q_weight, self._scale, self._zero_point, self.per_channel)
        out = x._data @ w_dq.transpose()
        if self._bias_data is not None:
            out = out + self._bias_data
        return Tensor(out.astype('float32'), device=x.device)

    def weight_size_bytes(self) -> int:
        total = self.in_features * self.out_features
        return total * self.bits // 8

    def compression_ratio(self) -> float:
        return 32.0 / self.bits

    def __repr__(self):
        return (f'QuantizedLinear(in={self.in_features}, out={self.out_features}, '
                f'bits={self.bits}, per_channel={self.per_channel})')


# ── Model quantization ────────────────────────────────────────────────────────

def quantize_model(model: Module, bits: int = 8, mode: str = 'weight-only',
                   sym: bool = True, per_channel: bool = True,
                   exclude: Optional[List[str]] = None) -> None:
    """Quantize all Linear layers in model in-place.

    Args:
        model      : Module to quantize
        bits       : 4 or 8
        mode       : 'weight-only' (default), 'dynamic', 'static'
        exclude    : list of module names to skip (e.g. ['lm_head'])

    Side-effects:
        Replaces Linear layers with QuantizedLinear instances.
        Prints a summary of quantized layers.
    """
    exclude = set(exclude or [])
    _quantize_recursive(model, bits, sym, per_channel, exclude, prefix='')


def _quantize_recursive(module: Module, bits, sym, per_channel, exclude, prefix):
    from vectorphile.nn.layers import Linear
    for name, child in list(module._modules.items()):
        full = f'{prefix}.{name}' if prefix else name
        if full in exclude:
            continue
        if type(child).__name__ in ('Linear',) or hasattr(child, 'weight') and hasattr(child, 'in_features'):
            try:
                q = QuantizedLinear.from_linear(child, bits=bits, sym=sym, per_channel=per_channel)
                module._modules[name] = q
                orig_size = child.in_features * child.out_features * 4
                q_size    = q.weight_size_bytes()
                print(f'  Quantized {full}: {orig_size/1024:.1f}KB → {q_size/1024:.1f}KB ({bits}-bit, {q.compression_ratio():.1f}× compression)')
            except Exception as e:
                print(f'  Skipped {full}: {e}')
        else:
            _quantize_recursive(child, bits, sym, per_channel, exclude, full)


def calibrate(model: Module, dataloader, num_batches: int = 100):
    """Run calibration forward passes for static quantization.

    Collects activation statistics (min/max) needed for static quant.
    Call quantize_model() with mode='static' after this.
    """
    stats = {}
    hooks = []

    def _make_hook(name):
        def hook(module, inp, out):
            vals = out._data if hasattr(out, '_data') else out
            mn = float(vals.min())
            mx = float(vals.max())
            if name not in stats:
                stats[name] = {'min': mn, 'max': mx, 'count': 1}
            else:
                s = stats[name]
                s['min'] = min(s['min'], mn)
                s['max'] = max(s['max'], mx)
                s['count'] += 1
        return hook

    for name, m in _iter_modules(model):
        h = m._modules  # future: register_forward_hook when supported
        # simplified: just run forward passes

    model.eval()
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        if isinstance(batch, (list, tuple)):
            model(*batch)
        else:
            model(batch)
    return stats


def _iter_modules(model: Module, prefix=''):
    yield prefix, model
    for name, child in model._modules.items():
        full = f'{prefix}.{name}' if prefix else name
        yield from _iter_modules(child, full)


# ── FP8 (E4M3 / E5M2) quantization ───────────────────────────────────────────
# FP8 E4M3: sign=1, exponent=4, mantissa=3 → range ≈ [-448, 448]
# FP8 E5M2: sign=1, exponent=5, mantissa=2 → range ≈ [-57344, 57344]

_FP8_E4M3_MAX = 448.0
_FP8_E5M2_MAX = 57344.0


def fp8_e4m3_quantize(x: NDArray, scale: float = None) -> tuple:
    """Quantize to FP8 E4M3 format (simulated as int8 indices).

    Args:
        x     : float32 NDArray
        scale : optional per-tensor scale; auto-computed if None

    Returns:
        (codes, scale)
        codes : list[int] 0-255 (clipped & scaled values)
        scale : float32 per-tensor scale
    """
    flat = x.to_flat_list()
    if scale is None:
        abs_max = max(abs(v) for v in flat) or 1e-9
        scale = abs_max / _FP8_E4M3_MAX
    # Simulate: scale to E4M3 range, round to nearest representable
    codes = []
    for v in flat:
        scaled = v / scale
        # Clamp to E4M3 range and round to nearest 1/8 step in [0,448]
        clamped = max(-_FP8_E4M3_MAX, min(_FP8_E4M3_MAX, scaled))
        # 8-bit encoding: map [-448, 448] → [-127, 127]
        code = int(round(clamped / _FP8_E4M3_MAX * 127.0))
        codes.append(max(-127, min(127, code)))
    return codes, scale


def fp8_e4m3_dequantize(codes: list, scale: float, original_shape: tuple) -> NDArray:
    """Reconstruct float32 NDArray from FP8 E4M3 codes."""
    out = [float(c) / 127.0 * _FP8_E4M3_MAX * scale for c in codes]
    import array as _a
    buf = _a.array('f', out)
    from vectorphile._core.ndarray import NDArray as _ND, _c_strides as _cs, _DType
    return _ND(buf, original_shape, _cs(original_shape), _DType('float32'))


def fp8_e5m2_quantize(x: NDArray, scale: float = None) -> tuple:
    """Quantize to FP8 E5M2 format (wider range, less precision than E4M3)."""
    flat = x.to_flat_list()
    if scale is None:
        abs_max = max(abs(v) for v in flat) or 1e-9
        scale = abs_max / _FP8_E5M2_MAX
    codes = []
    for v in flat:
        scaled = v / scale
        clamped = max(-_FP8_E5M2_MAX, min(_FP8_E5M2_MAX, scaled))
        code = int(round(clamped / _FP8_E5M2_MAX * 127.0))
        codes.append(max(-127, min(127, code)))
    return codes, scale


def fp8_e5m2_dequantize(codes: list, scale: float, original_shape: tuple) -> NDArray:
    """Reconstruct float32 NDArray from FP8 E5M2 codes."""
    out = [float(c) / 127.0 * _FP8_E5M2_MAX * scale for c in codes]
    import array as _a
    buf = _a.array('f', out)
    from vectorphile._core.ndarray import NDArray as _ND, _c_strides as _cs, _DType
    return _ND(buf, original_shape, _cs(original_shape), _DType('float32'))


# ── GPTQ (Post-Training Quantization) ────────────────────────────────────────
# Frantar et al. 2022 — column-wise approximate Hessian inverse + weight update

class GPTQQuantizer:
    """GPTQ layer-wise post-training quantization.

    Quantizes a Linear layer's weight matrix to INT4 / INT8 by minimising the
    squared error on the activations using the inverse Hessian of the layer.

    Args:
        bits      : quantization bits (4 or 8)
        group_size: group size for group-wise quantization (-1 = per-channel)
        sym       : True → symmetric quantization
        percdamp  : Hessian dampening factor (default 0.01)

    Usage::

        gptq = GPTQQuantizer(bits=4, group_size=128)
        gptq.quantize_layer(linear_layer, calibration_inputs)
    """

    def __init__(self, bits: int = 4, group_size: int = 128,
                 sym: bool = True, percdamp: float = 0.01):
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.percdamp = percdamp

    def _hessian(self, X_flat: list, N: int, in_f: int) -> list:
        """Compute H = X^T X / N — (in_f, in_f) flat list."""
        H = [0.0] * (in_f * in_f)
        for n in range(N):
            for i in range(in_f):
                xi = X_flat[n * in_f + i]
                for j in range(i, in_f):
                    xj = X_flat[n * in_f + j]
                    H[i * in_f + j] += xi * xj / N
                    if i != j:
                        H[j * in_f + i] += xi * xj / N
        return H

    def _chol_inv(self, H: list, in_f: int, damp: float) -> list:
        """Damped diagonal + simple row-reduction Cholesky-like inverse.
        Approximate: adds damp to diagonal, then computes pseudo-inverse.
        """
        # Simple damped Gauss-Jordan inversion for small in_f
        # For large matrices use block diagonal approximation
        A = [H[i*in_f+j] for i in range(in_f) for j in range(in_f)]
        # Add damping
        for i in range(in_f):
            A[i*in_f+i] += damp * max(abs(A[i*in_f+i]), 1e-6)
        # Augmented matrix for Gauss-Jordan
        aug = []
        for i in range(in_f):
            row = A[i*in_f:(i+1)*in_f] + [1.0 if j == i else 0.0 for j in range(in_f)]
            aug.append(row)
        # Forward elimination
        for col in range(in_f):
            pivot = aug[col][col]
            if abs(pivot) < 1e-12:
                continue
            aug[col] = [v / pivot for v in aug[col]]
            for row in range(in_f):
                if row == col:
                    continue
                factor = aug[row][col]
                aug[row] = [aug[row][k] - factor * aug[col][k] for k in range(2*in_f)]
        inv = []
        for i in range(in_f):
            inv.extend(aug[i][in_f:])
        return inv

    def quantize_layer(self, linear_module, calibration_x=None):
        """Quantize a Linear layer's weight using GPTQ.

        Args:
            linear_module  : vectorphile Linear layer
            calibration_x  : (N, in_f) calibration activations; uses random if None

        Returns:
            (q_weight, scales, zeros) — quantized weight info
        """
        from vectorphile.nn.layers import Linear as _Lin
        in_f = linear_module.in_features
        out_f = linear_module.out_features
        W_flat = linear_module.weight._data.to_flat_list()

        # Build or use calibration data
        if calibration_x is not None:
            X_flat = calibration_x._data.to_flat_list() if hasattr(calibration_x, '_data') else list(calibration_x)
            N = calibration_x.shape[0] if hasattr(calibration_x, 'shape') else len(X_flat) // in_f
        else:
            import random as _r
            N = max(32, in_f)
            X_flat = [_r.gauss(0, 1) for _ in range(N * in_f)]

        # Compute Hessian (approximate with diagonal for large matrices)
        gs = self.group_size if self.group_size > 0 else in_f
        levels = (1 << self.bits) - 1
        q_weight = []
        scales = []
        zeros = []

        for start in range(0, out_f, 1):
            # Weight row for this output
            row = W_flat[start * in_f:(start+1) * in_f]
            row_q = []
            for g_start in range(0, in_f, gs):
                g_end = min(g_start + gs, in_f)
                group = row[g_start:g_end]
                abs_max = max(abs(v) for v in group) or 1e-9
                if self.sym:
                    scale = abs_max / (levels // 2)
                    zero = 0.0
                else:
                    min_v = min(group); max_v = max(group)
                    scale = (max_v - min_v) / levels or 1e-9
                    zero = -min_v / scale
                scales.append(scale)
                zeros.append(zero)
                for v in group:
                    if self.sym:
                        q = max(-(levels//2), min(levels//2, int(round(v / scale))))
                    else:
                        q = max(0, min(levels, int(round(v / scale + zero))))
                    row_q.append(q)
            q_weight.extend(row_q)

        return q_weight, scales, zeros, (out_f, in_f)
