"""NF4 (NormalFloat-4) quantization as used in QLoRA (Dettmers et al. 2023).

The 16 NF4 bins are chosen so that each bin contains an equal fraction of
a standard normal distribution — information-theoretically optimal for
normally-distributed weights.
"""
from __future__ import annotations
import array as _a
from vectorphile._core.ndarray import NDArray


# 16 NF4 quantization levels (symmetric, normalised to [-1, 1])
# These are the 16 quantiles of N(0,1) spaced evenly in probability space.
_NF4_LEVELS = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]

_NF4_LEN = len(_NF4_LEVELS)  # 16


def _nearest_nf4(val: float) -> int:
    """Return the index of the nearest NF4 level."""
    best_i = 0
    best_d = abs(val - _NF4_LEVELS[0])
    for i in range(1, _NF4_LEN):
        d = abs(val - _NF4_LEVELS[i])
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def nf4_quantize(x: NDArray, block_size: int = 64) -> tuple:
    """Quantize a float32 NDArray to NF4.

    Weights are split into blocks of `block_size` values; each block gets its
    own absolute-max scale so the values fall in [-1, 1] before mapping to the
    16 NF4 bins.

    Args:
        x          : float32 NDArray of any shape
        block_size : number of elements per quantization block (default 64)

    Returns:
        (codes, scales, original_shape)
        codes          : list[int] of NF4 indices (0-15) per element
        scales         : list[float] of per-block absolute max
        original_shape : tuple, needed to reconstruct the tensor
    """
    flat = x.flatten()
    n = flat.shape[0]
    codes = []
    scales = []

    i = 0
    while i < n:
        block = [float(flat[j]) for j in range(i, min(i + block_size, n))]
        abs_max = max(abs(v) for v in block) or 1e-9
        scales.append(abs_max)
        for v in block:
            norm = v / abs_max          # → [-1, 1]
            codes.append(_nearest_nf4(norm))
        i += block_size

    return codes, scales, x.shape


def nf4_dequantize(codes: list, scales: list, original_shape: tuple,
                   block_size: int = 64) -> NDArray:
    """Reconstruct float32 NDArray from NF4 codes.

    Args:
        codes          : list[int] of NF4 indices
        scales         : list[float] of per-block absolute max
        original_shape : tuple, the original tensor shape
        block_size     : must match the value used during quantize

    Returns:
        float32 NDArray of shape `original_shape`
    """
    n = len(codes)
    out = []
    block_idx = 0
    for i in range(n):
        if i > 0 and i % block_size == 0:
            block_idx += 1
        abs_max = scales[block_idx]
        out.append(_NF4_LEVELS[codes[i]] * abs_max)

    buf = _a.array('f', out)
    # compute strides for original_shape
    if len(original_shape) == 0:
        strides = (1,)
    else:
        strides = [1] * len(original_shape)
        for i in range(len(original_shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * original_shape[i + 1]
        strides = tuple(strides)
    return NDArray(buf, original_shape, strides, 'float32')


def nf4_storage_bits(num_elements: int) -> int:
    """Packed storage size: 4 bits per code (two codes per byte)."""
    return num_elements * 4


def nf4_compression_ratio(num_elements: int) -> float:
    """Compression ratio vs fp32."""
    fp32_bits  = num_elements * 32
    nf4_bits   = nf4_storage_bits(num_elements)
    return fp32_bits / nf4_bits


def nf4_linear(x: NDArray, codes: list, scales: list, weight_shape: tuple,
               block_size: int = 64, bias: NDArray = None) -> NDArray:
    """Fused NF4 dequant + linear forward pass.

    Dequantizes weight block-by-block and immediately multiplies with the
    corresponding column slice of the input, avoiding the full float32 weight
    materialization. This halves peak memory usage versus dequant-then-matmul.

    Args:
        x           : (N, in_features) float32 input
        codes       : NF4 codes (output of nf4_quantize)
        scales      : per-block scales
        weight_shape: (out_features, in_features) — the original weight shape
        block_size  : must match quantize block_size (default 64)
        bias        : optional (out_features,) bias

    Returns:
        (N, out_features) float32 NDArray
    """
    out_f, in_f = weight_shape
    x_flat = [float(x[i, j]) for i in range(x.shape[0]) for j in range(x.shape[1])] \
        if hasattr(x, '__getitem__') else list(x.flatten())
    N = x.shape[0]

    # Build output buffer
    result = [0.0] * (N * out_f)

    # nf4_quantize flattens row-major and splits into block_size chunks.
    # codes[i] belongs to block i // block_size, scales[i // block_size].
    # We iterate in the same order (row-major) for correct block lookup.
    code_idx = 0
    for row in range(out_f):
        for col in range(in_f):
            blk = code_idx // block_size
            abs_max = scales[blk] if blk < len(scales) else scales[-1]
            w_val = _NF4_LEVELS[codes[code_idx]] * abs_max
            code_idx += 1
            for n in range(N):
                result[n * out_f + row] += x_flat[n * in_f + col] * w_val

    if bias is not None:
        bias_flat = list(bias.flatten()) if hasattr(bias, 'flatten') else list(bias)
        for n in range(N):
            for j in range(out_f):
                result[n * out_f + j] += bias_flat[j]

    buf = _a.array('f', result)
    strides = (out_f, 1)
    return NDArray(buf, (N, out_f), strides, 'float32')
