"""OpenAI Triton fused GPU kernels for maximum throughput.

Falls back to numpy when triton is not installed. Triton kernels compile to
PTX and outperform hand-written CUDA for ops like softmax and layer norm by
minimising memory bandwidth with fused passes.

Usage::

    from vectorphile.cuda.triton_kernels import fused_softmax, fused_layer_norm
    out = fused_softmax(logits)        # GPU if triton available
    out = fused_layer_norm(x, w, b)
"""
from __future__ import annotations
import numpy as np
from typing import Optional

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False


def is_available() -> bool:
    """Return True if triton is installed (requires CUDA GPU + torch)."""
    return _TRITON_AVAILABLE and _TORCH_AVAILABLE


# ── Triton kernels (compiled on first call) ───────────────────────────────────

if _TRITON_AVAILABLE and _TORCH_AVAILABLE:

    @triton.jit
    def _softmax_kernel(
        output_ptr, input_ptr, input_row_stride, output_row_stride,
        n_cols, BLOCK_SIZE: tl.constexpr,
    ):
        """One program per row. Single-pass online softmax with shared accum."""
        row_idx = tl.program_id(0)
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        # Load row (mask out-of-bound)
        row = tl.load(row_start_ptr + col_offsets, mask=col_offsets < n_cols, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=col_offsets < n_cols)

    @triton.jit
    def _layer_norm_kernel(
        x_ptr, w_ptr, b_ptr, out_ptr,
        stride, N: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """One program per row. Fused mean+var+normalize+affine."""
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < N
        x_row = tl.load(x_ptr + row * stride + offsets, mask=mask, other=0.0)
        mean = tl.sum(x_row, axis=0) / N
        diff = tl.where(mask, x_row - mean, 0.0)
        var  = tl.sum(diff * diff, axis=0) / N
        inv_std = 1.0 / tl.sqrt(var + eps)
        x_norm = diff * inv_std
        w = tl.load(w_ptr + offsets, mask=mask, other=1.0)
        b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        out = x_norm * w + b
        tl.store(out_ptr + row * stride + offsets, out, mask=mask)

    @triton.jit
    def _cross_entropy_kernel(
        logits_ptr, labels_ptr, out_ptr,
        n_classes: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Fused log_softmax + nll for one row (one sample)."""
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_classes
        row_start = row * n_classes
        logits = tl.load(logits_ptr + row_start + offsets, mask=mask, other=-float('inf'))
        # Stable log-softmax
        max_val = tl.max(logits, axis=0)
        shifted = logits - max_val
        exp_sum = tl.sum(tl.exp(shifted), axis=0)
        log_sum_exp = tl.log(exp_sum) + max_val
        label = tl.load(labels_ptr + row)
        label_logit = tl.load(logits_ptr + row_start + label)
        loss = log_sum_exp - label_logit
        tl.store(out_ptr + row, loss)

    def _next_power_of_2(n: int) -> int:
        p = 1
        while p < n:
            p <<= 1
        return p


# ── Public API ────────────────────────────────────────────────────────────────

def fused_softmax(x: np.ndarray) -> np.ndarray:
    """Fused softmax over last dimension.

    Uses Triton single-pass kernel when available. Falls back to numpy.

    Args:
        x: (..., N) float32 array
    Returns:
        same shape, softmax applied over last axis
    """
    if not (is_available() and x.ndim >= 1):
        e = np.exp(x - x.max(-1, keepdims=True))
        return (e / (e.sum(-1, keepdims=True) + 1e-9)).astype(np.float32)

    orig_shape = x.shape
    x2d = x.reshape(-1, x.shape[-1])
    rows, cols = x2d.shape
    t = torch.from_numpy(np.ascontiguousarray(x2d, np.float32)).cuda()
    out = torch.empty_like(t)
    BLOCK = _next_power_of_2(cols)
    _softmax_kernel[(rows,)](out, t, t.stride(0), out.stride(0), cols, BLOCK_SIZE=BLOCK)
    return out.cpu().numpy().reshape(orig_shape)


def fused_layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray,
                     eps: float = 1e-5) -> np.ndarray:
    """Fused layer normalisation (mean, var, normalise, affine in one kernel).

    Uses Triton kernel when available. Falls back to numpy.

    Args:
        x: (..., N) float32
        w: (N,) scale
        b: (N,) bias
    Returns:
        same shape as x
    """
    if not is_available():
        mean = x.mean(-1, keepdims=True)
        var  = x.var(-1, keepdims=True)
        return ((x - mean) / np.sqrt(var + eps) * w + b).astype(np.float32)

    orig_shape = x.shape
    N = x.shape[-1]
    x2d = x.reshape(-1, N)
    rows = x2d.shape[0]
    t    = torch.from_numpy(np.ascontiguousarray(x2d, np.float32)).cuda()
    tw   = torch.from_numpy(np.ascontiguousarray(w,   np.float32)).cuda()
    tb   = torch.from_numpy(np.ascontiguousarray(b,   np.float32)).cuda()
    out  = torch.empty_like(t)
    BLOCK = _next_power_of_2(N)
    _layer_norm_kernel[(rows,)](t, tw, tb, out, t.stride(0), N, eps, BLOCK)
    return out.cpu().numpy().reshape(orig_shape)


def fused_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fused log-softmax + NLL cross-entropy loss per sample.

    Uses Triton kernel when available. Falls back to numpy.

    Args:
        logits: (B, C) float32 — unnormalised scores
        labels: (B,) int64 — class indices
    Returns:
        (B,) per-sample losses
    """
    if not is_available():
        # numpy fallback
        x = logits - logits.max(-1, keepdims=True)
        log_probs = x - np.log(np.exp(x).sum(-1, keepdims=True) + 1e-9)
        return -log_probs[np.arange(len(labels)), labels.astype(int)].astype(np.float32)

    B, C = logits.shape
    t_logits = torch.from_numpy(np.ascontiguousarray(logits, np.float32)).cuda()
    t_labels = torch.from_numpy(np.ascontiguousarray(labels, np.int32)).cuda()
    out = torch.empty(B, dtype=torch.float32, device='cuda')
    BLOCK = _next_power_of_2(C)
    _cross_entropy_kernel[(B,)](t_logits, t_labels, out, n_classes=C, BLOCK=BLOCK)
    return out.cpu().numpy()
