"""vectorphile.mojo — Mojo-accelerated kernels for maximum throughput.

Architecture
------------
Each kernel is written in Mojo (.mojo source in vectorphile/mojo/kernels/).
At import time we try to load the compiled shared library; if unavailable we
fall back transparently to a NumPy/NDArray implementation so the library never
breaks on systems without Mojo installed.

Accelerated operations
----------------------
- matmul          : tiled SIMD matrix multiply
- flash_attention  : Flash-Attention-2 tiled algorithm
- conv2d_im2col   : im2col + GEMM convolution
- layer_norm      : fused mean/var/scale in one pass
- rms_norm        : fused RMS normalization
- gelu / silu     : element-wise activations (SIMD vectorised)
- softmax         : numerically-stable vectorised softmax

Usage
-----
    from vectorphile.mojo import matmul, flash_attention
    # Uses Mojo if available, otherwise NumPy
    out = matmul(a, b)
"""
from __future__ import annotations
from vectorphile.mojo._dispatch import (
    matmul,
    flash_attention_fwd,
    conv2d,
    layer_norm,
    rms_norm,
    gelu,
    silu,
    softmax,
    batch_norm,
    cross_entropy,
    embedding,
    dropout,
    speculative_verify,
    relu,
    add,
    mul,
    scale,
    transpose,
    adam_step,
    linear_silu,
    linear_gelu,
    qkv_proj,
    rope,
    swiglu,
    geglu,
    online_softmax,
    layer_norm_mojo,
    embedding_lookup_mojo,
    gelu_mojo,
    silu_mojo,
    adam_mojo,
    is_mojo_available,
    mojo_backend_info,
)

__all__ = [
    'matmul', 'flash_attention_fwd', 'conv2d',
    'layer_norm', 'rms_norm', 'gelu', 'silu', 'softmax',
    'batch_norm', 'cross_entropy', 'embedding', 'dropout',
    'relu', 'add', 'mul', 'scale', 'transpose',
    'adam_step', 'linear_silu', 'linear_gelu', 'qkv_proj',
    'rope', 'swiglu', 'geglu',
    'online_softmax', 'layer_norm_mojo', 'embedding_lookup_mojo',
    'gelu_mojo', 'silu_mojo', 'adam_mojo',
    'speculative_verify',
    'is_mojo_available', 'mojo_backend_info',
]
