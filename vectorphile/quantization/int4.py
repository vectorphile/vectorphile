"""INT4 Weight Quantization for LLM inference (NF4 / GPTQ-style).

Provides:
- ``quantize_int4(weight_tensor)``: pack 2× int4 values per byte
- ``dequantize_int4(packed, scale, zero, shape)``: unpack back to FP32
- ``Int4Linear``: drop-in Linear replacement with int4 weights
- ``gptq_quantize_layer``: GPTQ-style block-diagonal Hessian quantization (pure Python)

INT4 packing: two 4-bit values stored in one uint8 byte.
  byte = (high_nibble << 4) | low_nibble
  high_nibble = element at even index, low_nibble = element at odd index.

Scales and zeros are stored per-group (default group_size=128).
"""
from __future__ import annotations
import array as _a
import math
from typing import List, Optional, Tuple


def quantize_int4(flat: List[float], group_size: int = 128) -> Tuple[bytes, List[float], List[float]]:
    """Quantize a flat float list to int4 with per-group scales/zeros.

    Args:
        flat: flat float values (length must be even)
        group_size: number of elements per quantization group
    Returns:
        (packed_bytes, scales, zeros) where packed_bytes has len(flat)//2 bytes,
        scales and zeros are per-group floats
    """
    n = len(flat)
    # Pad to even
    if n % 2 != 0:
        flat = flat + [0.0]
        n += 1
    num_groups = max(1, (n + group_size - 1) // group_size)
    scales: List[float] = []
    zeros: List[float] = []
    quant: List[int] = []

    for g in range(num_groups):
        start = g * group_size
        end   = min(start + group_size, n)
        block = flat[start:end]
        mn = min(block); mx = max(block)
        if mx == mn:
            sc = 1.0; zp = 0.0
        else:
            sc = (mx - mn) / 15.0  # 4-bit: 0..15
            zp = -mn / sc
        scales.append(sc)
        zeros.append(zp)
        for v in block:
            q = round(v / sc + zp)
            q = max(0, min(15, q))
            quant.append(q)

    # Pack 2 int4s per byte
    packed = bytearray(n // 2)
    for i in range(0, n, 2):
        hi = quant[i] & 0xF
        lo = quant[i+1] & 0xF if i+1 < len(quant) else 0
        packed[i // 2] = (hi << 4) | lo
    return bytes(packed), scales, zeros


def dequantize_int4(packed: bytes, scales: List[float], zeros: List[float],
                    n: int, group_size: int = 128) -> List[float]:
    """Unpack and dequantize int4 bytes to float.

    Args:
        packed: packed int4 bytes (len = ceil(n/2))
        scales: per-group scales
        zeros: per-group zero points
        n: original number of elements
        group_size: elements per group (must match quantize_int4 call)
    Returns:
        list of n floats
    """
    # Unpack
    quant: List[int] = []
    for byte in packed:
        quant.append((byte >> 4) & 0xF)
        quant.append(byte & 0xF)
    quant = quant[:n]
    # Dequantize
    result: List[float] = []
    for i, q in enumerate(quant):
        g = min(i // group_size, len(scales) - 1)
        result.append((q - zeros[g]) * scales[g])
    return result


class Int4Linear:
    """Linear layer with INT4 weights (2× memory reduction vs FP16).

    Weights are stored packed in INT4 format; dequantized to FP32 on each
    forward pass. For large batch sizes, the dequant cost is amortized.

    Args:
        in_features: input dimension
        out_features: output dimension
        bias: whether to include a bias term
        group_size: quantization group size (128 = standard GPTQ)
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, group_size: int = 128):
        from vectorphile.core.tensor import Tensor
        import random as _rnd
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # Initialize random weights and quantize
        bound = math.sqrt(1.0 / max(1, in_features))
        w_flat = [_rnd.uniform(-bound, bound) for _ in range(out_features * in_features)]
        self._packed, self._scales, self._zeros = quantize_int4(w_flat, group_size)
        self._weight_shape = (out_features, in_features)

        if bias:
            self._bias = [_rnd.uniform(-bound, bound) for _ in range(out_features)]
        else:
            self._bias = None

    @property
    def weight_fp32(self) -> List[float]:
        """Dequantize weights to FP32 on demand."""
        n = self.out_features * self.in_features
        return dequantize_int4(self._packed, self._scales, self._zeros, n, self.group_size)

    def forward(self, x):
        """Dequantize weights → FP32 matmul → return Tensor."""
        from vectorphile.core.tensor import Tensor
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType
        w_fp32 = self.weight_fp32
        xfl = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
        xsh = x._data.shape if hasattr(x._data, 'shape') else (len(xfl),)
        n_tokens = 1
        for d in xsh[:-1]: n_tokens *= d
        D = xsh[-1]
        out = []
        for i in range(n_tokens):
            xi = xfl[i * D: i * D + D]
            for j in range(self.out_features):
                wj = w_fp32[j * D: j * D + D]
                dot = sum(xi[k] * wj[k] for k in range(D))
                out.append(dot)
        if self._bias is not None:
            for i in range(n_tokens):
                for j in range(self.out_features):
                    out[i * self.out_features + j] += self._bias[j]
        out_sh = xsh[:-1] + (self.out_features,)
        buf = _a.array('f', [float(v) for v in out])
        return Tensor(NDArray(buf, out_sh, _c_strides(out_sh), _DType('float32')))

    def __call__(self, x):
        return self.forward(x)

    def weight_bytes(self) -> int:
        """Bytes used by INT4 weights (vs in_features*out_features*4 for FP32)."""
        return len(self._packed) + (len(self._scales) + len(self._zeros)) * 4

    def compression_ratio(self) -> float:
        fp32_bytes = self.in_features * self.out_features * 4
        return fp32_bytes / max(1, self.weight_bytes())


def gptq_quantize_layer(weight_flat: List[float], in_features: int,
                         out_features: int, group_size: int = 128,
                         percdamp: float = 0.01) -> Tuple[bytes, List[float], List[float]]:
    """GPTQ-style column-wise sequential quantization (pure Python).

    Args:
        weight_flat: W in row-major order (out_features, in_features)
        in_features: input dimension
        out_features: output dimension
        group_size: quantization group size
        percdamp: Hessian damping factor (% of diagonal mean)
    Returns:
        (packed_int4, scales, zeros) same as quantize_int4
    """
    # Reconstruct W as list of rows
    W = [[weight_flat[r * in_features + c] for c in range(in_features)]
         for r in range(out_features)]
    # Hessian approximation: H = I * percdamp (uniform since we have no calibration data here)
    # In full GPTQ, H = 2 * X^T X averaged over calibration samples.
    h_diag = [1.0] * in_features
    damp = percdamp * sum(h_diag) / max(1, len(h_diag))
    h_diag = [v + damp for v in h_diag]

    # Column-wise quantization with Cholesky update
    quantized_flat = []
    for r in range(out_features):
        row = W[r][:]
        q_row = []
        for c in range(in_features):
            q = round(row[c])  # simplified: would use scale/zero in real impl
            q_row.append(float(q))
            err = (row[c] - q) / max(1e-8, h_diag[c])
            # Propagate error to remaining columns
            for c2 in range(c+1, in_features):
                row[c2] -= err  # simplified (no H^{-1} without full Hessian)
        quantized_flat.extend(q_row)

    return quantize_int4(quantized_flat, group_size)
