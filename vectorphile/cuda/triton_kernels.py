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
import math as _math
from typing import Optional
from vectorphile._core.ndarray import NDArray, array as _nd_array

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

    # ── Autotuning configs ────────────────────────────────────────────────────
    # Each config tries a different BLOCK_SIZE × num_warps combo. Triton picks
    # the fastest on first real call and caches the winner for the GPU arch.
    _SOFTMAX_CONFIGS = [
        triton.Config({'BLOCK_SIZE': bs}, num_warps=nw)
        for bs in [256, 512, 1024, 2048, 4096]
        for nw in [4, 8]
    ]

    _LAYERNORM_CONFIGS = [
        triton.Config({'BLOCK': bs}, num_warps=nw)
        for bs in [128, 256, 512, 1024, 2048]
        for nw in [4, 8, 16]
    ]

    _CROSS_ENTROPY_CONFIGS = [
        triton.Config({'BLOCK': bs}, num_warps=nw)
        for bs in [64, 128, 256, 512, 1024]
        for nw in [4, 8]
    ]

    @triton.autotune(configs=_SOFTMAX_CONFIGS, key=['n_cols'])
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

    @triton.autotune(configs=_LAYERNORM_CONFIGS, key=['N'])
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

    @triton.autotune(configs=_CROSS_ENTROPY_CONFIGS, key=['n_classes'])
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

def fused_softmax(x):
    """Fused softmax over last dimension.

    Uses Triton single-pass kernel when available. Falls back to pure Python.
    """
    xf = x.to_flat_list() if isinstance(x, NDArray) else list(x)
    sh = x.shape if isinstance(x, NDArray) else (len(xf),)
    if not is_available():
        mx = max(xf); e = [_math.exp(v - mx) for v in xf]
        s = sum(e); result = [v / (s + 1e-9) for v in e]
        return _nd_array(result, dtype='float32').reshape(sh)

    orig_shape = sh
    n_cols = sh[-1] if len(sh) > 1 else sh[0]
    x2d_flat = xf
    rows = len(xf) // n_cols
    t = torch.tensor(x2d_flat, dtype=torch.float32).reshape(rows, n_cols).cuda()
    out = torch.empty_like(t)
    BLOCK = _next_power_of_2(n_cols)
    _softmax_kernel[(rows,)](out, t, t.stride(0), out.stride(0), n_cols, BLOCK_SIZE=BLOCK)
    result = out.cpu().reshape(-1).tolist()
    return _nd_array(result, dtype='float32').reshape(orig_shape)


def fused_layer_norm(x, w, b, eps: float = 1e-5):
    """Fused layer normalisation. Uses Triton kernel when available, else pure Python."""
    xf = x.to_flat_list() if isinstance(x, NDArray) else list(x)
    wf = w.to_flat_list() if isinstance(w, NDArray) else list(w)
    bf = b.to_flat_list() if isinstance(b, NDArray) else list(b)
    sh = x.shape if isinstance(x, NDArray) else (len(xf),)
    N = sh[-1] if len(sh) > 1 else sh[0]
    if not is_available():
        rows = len(xf) // N
        result = []
        for r in range(rows):
            row = xf[r*N:(r+1)*N]
            mu = sum(row) / N
            var = sum((v - mu)**2 for v in row) / N
            inv = 1.0 / _math.sqrt(var + eps)
            result.extend([(v - mu) * inv * wf[i] + bf[i] for i, v in enumerate(row)])
        return _nd_array(result, dtype='float32').reshape(sh)

    rows = len(xf) // N
    t  = torch.tensor(xf,  dtype=torch.float32).reshape(rows, N).cuda()
    tw = torch.tensor(wf,  dtype=torch.float32).cuda()
    tb = torch.tensor(bf,  dtype=torch.float32).cuda()
    out = torch.empty_like(t)
    BLOCK = _next_power_of_2(N)
    _layer_norm_kernel[(rows,)](t, tw, tb, out, t.stride(0), N, eps, BLOCK)
    result = out.cpu().reshape(-1).tolist()
    return _nd_array(result, dtype='float32').reshape(sh)


def fused_cross_entropy(logits, labels):
    """Fused log-softmax + NLL cross-entropy loss per sample.

    Uses Triton kernel when available. Falls back to pure Python.
    """
    lf = logits.to_flat_list() if isinstance(logits, NDArray) else list(logits)
    ls = logits.shape if isinstance(logits, NDArray) else (len(lf),)
    lblf = labels.to_flat_list() if isinstance(labels, NDArray) else list(labels)
    B = ls[0] if len(ls) > 1 else 1
    C = ls[1] if len(ls) > 1 else ls[0]
    if not is_available():
        result = []
        for i in range(B):
            row = lf[i*C:(i+1)*C]
            mx = max(row)
            shifted = [v - mx for v in row]
            exp_sum = sum(_math.exp(v) for v in shifted)
            log_probs = [v - _math.log(exp_sum + 1e-9) for v in shifted]
            lbl = int(lblf[i])
            result.append(-log_probs[lbl])
        return _nd_array(result, dtype='float32')

    t_logits = torch.tensor(lf, dtype=torch.float32).reshape(B, C).cuda()
    t_labels = torch.tensor([int(v) for v in lblf], dtype=torch.int32).cuda()
    out = torch.empty(B, dtype=torch.float32, device='cuda')
    BLOCK = _next_power_of_2(C)
    _cross_entropy_kernel[(B,)](t_logits, t_labels, out, n_classes=C, BLOCK=BLOCK)
    return _nd_array(out.cpu().tolist(), dtype='float32')


# ── Additional high-performance Triton kernels ────────────────────────────────

if _TRITON_AVAILABLE and _TORCH_AVAILABLE:

    _RMSNORM_CONFIGS = [
        triton.Config({'BLOCK': bs}, num_warps=nw)
        for bs in [128, 256, 512, 1024, 2048]
        for nw in [4, 8, 16]
    ]

    @triton.autotune(configs=_RMSNORM_CONFIGS, key=['N'])
    @triton.jit
    def _rms_norm_kernel(
        x_ptr, w_ptr, out_ptr,
        stride, N: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Fused RMS norm: single-pass sum-of-squares + normalize + scale."""
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < N
        x = tl.load(x_ptr + row * stride + offsets, mask=mask, other=0.0)
        ss = tl.sum(x * x, axis=0) / N
        rms = tl.sqrt(ss + eps)
        x_norm = x / rms
        w = tl.load(w_ptr + offsets, mask=mask, other=1.0)
        tl.store(out_ptr + row * stride + offsets, x_norm * w, mask=mask)

    _ADAM_CONFIGS = [
        triton.Config({'BLOCK': bs}, num_warps=nw)
        for bs in [512, 1024, 2048, 4096]
        for nw in [4, 8]
    ]

    @triton.autotune(configs=_ADAM_CONFIGS, key=['N'])
    @triton.jit
    def _adam_kernel(
        p_ptr, g_ptr, m_ptr, v_ptr,
        N, lr, beta1, beta2, eps, wd, bc1, bc2,
        BLOCK: tl.constexpr,
    ):
        """Fused Adam update: load p/g/m/v, update m/v, write p — no extra buffers."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        p = tl.load(p_ptr + offsets, mask=mask)
        g = tl.load(g_ptr + offsets, mask=mask)
        m = tl.load(m_ptr + offsets, mask=mask)
        v = tl.load(v_ptr + offsets, mask=mask)
        g = g + wd * p
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        m_hat = m / bc1
        v_hat = v / bc2
        p = p - lr * m_hat / (tl.sqrt(v_hat) + eps)
        tl.store(p_ptr + offsets, p, mask=mask)
        tl.store(m_ptr + offsets, m, mask=mask)
        tl.store(v_ptr + offsets, v, mask=mask)

    _GELU_CONFIGS = [
        triton.Config({'BLOCK': bs}, num_warps=nw)
        for bs in [512, 1024, 2048, 4096]
        for nw in [4, 8]
    ]

    @triton.autotune(configs=_GELU_CONFIGS, key=['N'])
    @triton.jit
    def _gelu_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
        """Fused GELU activation — exact tanh approximation."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask)
        # GELU(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
        x3 = x * x * x
        inner = 0.7978845608 * (x + 0.044715 * x3)
        out = 0.5 * x * (1.0 + tl.math.tanh(inner))
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.autotune(configs=_GELU_CONFIGS, key=['N'])
    @triton.jit
    def _silu_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
        """Fused SiLU/Swish activation: x * sigmoid(x)."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask)
        out = x / (1.0 + tl.math.exp(-x))
        tl.store(out_ptr + offsets, out, mask=mask)


def fused_rms_norm(x, w, eps: float = 1e-6):
    """Fused RMS normalisation. Single-pass Triton kernel when available."""
    xf = x.to_flat_list() if isinstance(x, NDArray) else list(x)
    wf = w.to_flat_list() if isinstance(w, NDArray) else list(w)
    sh = x.shape if isinstance(x, NDArray) else (len(xf),)
    N = sh[-1] if len(sh) > 1 else sh[0]
    if not is_available():
        rows = len(xf) // N
        result = []
        for r in range(rows):
            row = xf[r*N:(r+1)*N]
            rms = _math.sqrt(sum(v*v for v in row) / N + eps)
            result.extend([v / rms * wf[i] for i, v in enumerate(row)])
        return _nd_array(result, dtype='float32').reshape(sh)
    rows = len(xf) // N
    t  = torch.tensor(xf, dtype=torch.float32).reshape(rows, N).cuda()
    tw = torch.tensor(wf, dtype=torch.float32).cuda()
    out = torch.empty_like(t)
    BLOCK = _next_power_of_2(N)
    _rms_norm_kernel[(rows,)](t, tw, out, t.stride(0), N, eps, BLOCK)
    result = out.cpu().reshape(-1).tolist()
    return _nd_array(result, dtype='float32').reshape(sh)


def fused_adam(params, grads, m, v, lr: float, beta1: float, beta2: float,
               eps: float, weight_decay: float, t: int):
    """Fused Adam parameter update. All in one Triton kernel — no extra buffers."""
    import math as _m
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    if not is_available():
        pf = params.to_flat_list() if isinstance(params, NDArray) else list(params)
        gf = grads.to_flat_list()  if isinstance(grads,  NDArray) else list(grads)
        mf = m.to_flat_list()      if isinstance(m,      NDArray) else list(m)
        vf = v.to_flat_list()      if isinstance(v,      NDArray) else list(v)
        sh = params.shape if isinstance(params, NDArray) else (len(pf),)
        out_p, out_m, out_v = [], [], []
        for pi, gi, mi, vi in zip(pf, gf, mf, vf):
            gi = gi + weight_decay * pi
            mi_new = beta1 * mi + (1.0 - beta1) * gi
            vi_new = beta2 * vi + (1.0 - beta2) * gi * gi
            pi_new = pi - lr * (mi_new / bc1) / (_m.sqrt(vi_new / bc2) + eps)
            out_p.append(pi_new); out_m.append(mi_new); out_v.append(vi_new)
        return (_nd_array(out_p,'float32').reshape(sh),
                _nd_array(out_m,'float32').reshape(sh),
                _nd_array(out_v,'float32').reshape(sh))
    pf = params.to_flat_list() if isinstance(params, NDArray) else list(params)
    gf = grads.to_flat_list()  if isinstance(grads,  NDArray) else list(grads)
    mf = m.to_flat_list()      if isinstance(m,      NDArray) else list(m)
    vf = v.to_flat_list()      if isinstance(v,      NDArray) else list(v)
    sh = params.shape if isinstance(params, NDArray) else (len(pf),)
    N = len(pf)
    tp = torch.tensor(pf, dtype=torch.float32).cuda()
    tg = torch.tensor(gf, dtype=torch.float32).cuda()
    tm = torch.tensor(mf, dtype=torch.float32).cuda()
    tv = torch.tensor(vf, dtype=torch.float32).cuda()
    BLOCK = 1024
    grid = lambda meta: ((N + meta['BLOCK'] - 1) // meta['BLOCK'],)
    _adam_kernel[grid](tp, tg, tm, tv, N, lr, beta1, beta2, eps, weight_decay, bc1, bc2, BLOCK=BLOCK)
    return (_nd_array(tp.cpu().tolist(),'float32').reshape(sh),
            _nd_array(tm.cpu().tolist(),'float32').reshape(sh),
            _nd_array(tv.cpu().tolist(),'float32').reshape(sh))


def triton_gelu(x):
    """Element-wise GELU via autotuned Triton kernel."""
    xf = x.to_flat_list() if isinstance(x, NDArray) else list(x)
    sh = x.shape if isinstance(x, NDArray) else (len(xf),)
    if not is_available():
        result = [v * 0.5 * (1.0 + _math.tanh(0.7978845608*(v+0.044715*v**3))) for v in xf]
        return _nd_array(result, 'float32').reshape(sh)
    N = len(xf)
    t = torch.tensor(xf, dtype=torch.float32).cuda()
    out = torch.empty_like(t)
    BLOCK = 1024
    grid = lambda meta: ((N + meta['BLOCK'] - 1) // meta['BLOCK'],)
    _gelu_kernel[grid](t, out, N, BLOCK=BLOCK)
    return _nd_array(out.cpu().tolist(), 'float32').reshape(sh)


def triton_silu(x):
    """Element-wise SiLU via autotuned Triton kernel."""
    xf = x.to_flat_list() if isinstance(x, NDArray) else list(x)
    sh = x.shape if isinstance(x, NDArray) else (len(xf),)
    if not is_available():
        result = [v / (1.0 + _math.exp(-v)) for v in xf]
        return _nd_array(result, 'float32').reshape(sh)
    N = len(xf)
    t = torch.tensor(xf, dtype=torch.float32).cuda()
    out = torch.empty_like(t)
    BLOCK = 1024
    grid = lambda meta: ((N + meta['BLOCK'] - 1) // meta['BLOCK'],)
    _silu_kernel[grid](t, out, N, BLOCK=BLOCK)
    return _nd_array(out.cpu().tolist(), 'float32').reshape(sh)


# ── Flash Decoding (split-KV parallel decode) ─────────────────────────────────
# Tri Dao 2023: parallelise decoding across sequence positions by splitting K/V
# into chunks, running partial softmax per chunk, then merging with log-sum-exp.
# Beats Flash Attention v2 for autoregressive generation (small Q, large KV).

if _TRITON_AVAILABLE and _TORCH_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_KV': bkv, 'BLOCK_D': bd}, num_warps=nw)
            for bkv in [32, 64, 128]
            for bd  in [32, 64]
            for nw  in [4, 8]
        ],
        key=['SEQ_LEN', 'HEAD_DIM'],
    )
    @triton.jit
    def _flash_decode_partial_kernel(
        Q_ptr, K_ptr, V_ptr,  # (batch, heads, 1, D), (batch, heads, S, D), (batch, heads, S, D)
        Out_ptr,              # (batch, heads, num_splits, D)
        Lse_ptr,              # (batch, heads, num_splits) — log-sum-exp per split
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        stride_lb, stride_lh, stride_ls,
        split_idx, split_size,
        SEQ_LEN: tl.constexpr, HEAD_DIM: tl.constexpr,
        BLOCK_KV: tl.constexpr, BLOCK_D: tl.constexpr,
        scale,
    ):
        pid_b  = tl.program_id(0)
        pid_h  = tl.program_id(1)
        # load Q row (single decode token)
        q_off = pid_b * stride_qb + pid_h * stride_qh
        d_range = tl.arange(0, BLOCK_D)
        q = tl.load(Q_ptr + q_off + d_range, mask=d_range < HEAD_DIM, other=0.0)
        # partial attention over [split_idx*split_size, min((split_idx+1)*split_size, SEQ_LEN))
        kv_start = split_idx * split_size
        kv_end   = tl.minimum(kv_start + split_size, SEQ_LEN)
        m_i = -1e9  # running max
        l_i = 0.0   # running sum-exp
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        kv_base_k = pid_b * stride_kb + pid_h * stride_kh
        kv_base_v = pid_b * stride_vb + pid_h * stride_vh
        for kv in tl.range(kv_start, kv_end, BLOCK_KV):
            kv_range = kv + tl.arange(0, BLOCK_KV)
            mask_kv  = kv_range < kv_end
            # load K block → (BLOCK_KV, BLOCK_D)
            k_ptrs = kv_base_k + kv_range[:, None] * stride_ks + d_range[None, :] * stride_kd
            k_block = tl.load(k_ptrs, mask=mask_kv[:, None] & (d_range[None, :] < HEAD_DIM), other=0.0)
            # dot Q·K^T
            attn = tl.sum(q[None, :] * k_block, axis=1) * scale  # (BLOCK_KV,)
            attn = tl.where(mask_kv, attn, -1e9)
            # update online softmax
            m_new = tl.maximum(m_i, tl.max(attn, axis=0))
            alpha  = tl.exp(m_i - m_new)
            l_i    = l_i * alpha + tl.sum(tl.exp(attn - m_new), axis=0)
            # load V block
            v_ptrs = kv_base_v + kv_range[:, None] * stride_vs + d_range[None, :] * stride_vd
            v_block = tl.load(v_ptrs, mask=mask_kv[:, None] & (d_range[None, :] < HEAD_DIM), other=0.0)
            acc = acc * alpha + tl.sum(tl.exp(attn - m_new)[:, None] * v_block, axis=0)
            m_i = m_new
        # write partial output and log-sum-exp
        out_off = pid_b * stride_ob + pid_h * stride_oh + split_idx * stride_os
        tl.store(Out_ptr + out_off + d_range, acc / l_i, mask=d_range < HEAD_DIM)
        lse_off = pid_b * stride_lb + pid_h * stride_lh + split_idx * stride_ls
        tl.store(Lse_ptr + lse_off, m_i + tl.log(l_i))

    @triton.jit
    def _flash_decode_merge_kernel(
        Out_ptr, Lse_ptr, Final_ptr,
        stride_ob, stride_oh, stride_os, stride_od,
        stride_lb, stride_lh, stride_ls,
        stride_fb, stride_fh, stride_fd,
        num_splits, HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        d_range = tl.arange(0, BLOCK_D)
        # find global max LSE across splits
        m_global = -1e9
        for s in tl.range(num_splits):
            lse = tl.load(Lse_ptr + pid_b * stride_lb + pid_h * stride_lh + s * stride_ls)
            m_global = tl.maximum(m_global, lse)
        # weighted sum
        acc   = tl.zeros([BLOCK_D], dtype=tl.float32)
        total = 0.0
        for s in tl.range(num_splits):
            lse  = tl.load(Lse_ptr + pid_b * stride_lb + pid_h * stride_lh + s * stride_ls)
            w    = tl.exp(lse - m_global)
            total = total + w
            o_off = pid_b * stride_ob + pid_h * stride_oh + s * stride_os
            o_row = tl.load(Out_ptr + o_off + d_range, mask=d_range < HEAD_DIM, other=0.0)
            acc   = acc + w * o_row
        out_off = pid_b * stride_fb + pid_h * stride_fh
        tl.store(Final_ptr + out_off + d_range, acc / total, mask=d_range < HEAD_DIM)


def flash_decode(q, k, v, scale: float = None, num_splits: int = 8):
    """Flash Decoding: O(S/num_splits) parallel reduce for single-token decode.

    q : (B, H, 1, D)  — single query token (decode step)
    k : (B, H, S, D)  — full KV cache
    v : (B, H, S, D)
    Returns out : (B, H, 1, D)
    """
    import math as _m
    if isinstance(q, _nd_array.__class__):
        qf = q; kf = k; vf = v
    # pure-Python path (CPU fallback)
    if not is_available():
        def _nd_to_nested(nd, shape):
            flat = nd.to_flat_list() if hasattr(nd, 'to_flat_list') else list(nd)
            return flat, shape
        qfl, qsh = _nd_to_nested(q, q.shape)
        kfl, ksh = _nd_to_nested(k, k.shape)
        vfl, vsh = _nd_to_nested(v, v.shape)
        B, H, _, D = qsh
        S = ksh[2]
        sc = scale or (1.0 / _m.sqrt(D))
        result = []
        for b in range(B):
            for h in range(H):
                q_row = qfl[((b*H+h)*1)*D : ((b*H+h)*1+1)*D]
                attn_scores = []
                for s in range(S):
                    k_row = kfl[((b*H+h)*S+s)*D : ((b*H+h)*S+s+1)*D]
                    score = sum(q_row[d]*k_row[d] for d in range(D)) * sc
                    attn_scores.append(score)
                m = max(attn_scores)
                exps = [_m.exp(a - m) for a in attn_scores]
                denom = sum(exps)
                out_row = [0.0]*D
                for s in range(S):
                    v_row = vfl[((b*H+h)*S+s)*D : ((b*H+h)*S+s+1)*D]
                    w = exps[s] / denom
                    for d in range(D): out_row[d] += w * v_row[d]
                result.extend(out_row)
        return _nd_array(result, 'float32').reshape((B, H, 1, D))

    # Triton path
    qsh = q.shape if hasattr(q, 'shape') else None
    if qsh is None:
        raise ValueError("flash_decode requires NDArray inputs")
    B, H, _, D = qsh
    S = k.shape[2]
    sc = scale or (1.0 / _m.sqrt(D))
    split_size = max(1, (S + num_splits - 1) // num_splits)
    actual_splits = (S + split_size - 1) // split_size

    qfl = q.to_flat_list(); kfl = k.to_flat_list(); vfl = v.to_flat_list()
    tq = torch.tensor(qfl, dtype=torch.float32).reshape(B, H, 1, D).cuda().contiguous()
    tk = torch.tensor(kfl, dtype=torch.float32).reshape(B, H, S, D).cuda().contiguous()
    tv = torch.tensor(vfl, dtype=torch.float32).reshape(B, H, S, D).cuda().contiguous()

    partial_out = torch.empty(B, H, actual_splits, D, device='cuda', dtype=torch.float32)
    partial_lse = torch.empty(B, H, actual_splits,    device='cuda', dtype=torch.float32)

    BLOCK_D = max(32, 1 << (_m.ceil(_m.log2(D)) if D > 1 else 1))
    grid = (B, H)
    for sp in range(actual_splits):
        _flash_decode_partial_kernel[grid](
            tq, tk, tv, partial_out, partial_lse,
            tq.stride(0), tq.stride(1), tq.stride(2), tq.stride(3),
            tk.stride(0), tk.stride(1), tk.stride(2), tk.stride(3),
            tv.stride(0), tv.stride(1), tv.stride(2), tv.stride(3),
            partial_out.stride(0), partial_out.stride(1), partial_out.stride(2), partial_out.stride(3),
            partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
            sp, split_size,
            SEQ_LEN=S, HEAD_DIM=D,
            scale=sc, BLOCK_D=BLOCK_D,
        )

    final = torch.empty(B, H, 1, D, device='cuda', dtype=torch.float32)
    _flash_decode_merge_kernel[grid](
        partial_out, partial_lse, final,
        partial_out.stride(0), partial_out.stride(1), partial_out.stride(2), partial_out.stride(3),
        partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
        final.stride(0), final.stride(1), final.stride(3),
        actual_splits, HEAD_DIM=D, BLOCK_D=BLOCK_D,
    )
    result = final.cpu().reshape(-1).tolist()
    return _nd_array(result, 'float32').reshape((B, H, 1, D))


# ── Persistent element-wise kernel (bias-add + activation, any shape) ─────────
# Persistent kernels keep warps alive across calls: for small-N ops the launch
# overhead dominates; persistent warps reuse CTA state and skip re-launch.

if _TRITON_AVAILABLE and _TORCH_AVAILABLE:

    @triton.jit
    def _persistent_bias_act_kernel(
        X_ptr, Bias_ptr, Out_ptr,
        N, D,          # (tokens, features) — bias shape is (D,)
        activation: tl.constexpr,  # 0=identity, 1=relu, 2=gelu, 3=silu
        BLOCK: tl.constexpr,
    ):
        pid   = tl.program_id(0)
        start = pid * BLOCK
        while start < N * D:
            offsets = start + tl.arange(0, BLOCK)
            mask    = offsets < N * D
            x    = tl.load(X_ptr    + offsets, mask=mask, other=0.0)
            bias = tl.load(Bias_ptr + (offsets % D), mask=mask, other=0.0)
            val  = x + bias
            if activation == 1:
                val = tl.maximum(val, 0.0)
            elif activation == 2:
                val = val * 0.5 * (1.0 + tl.math.tanh(0.7978845608 * (val + 0.044715 * val * val * val)))
            elif activation == 3:
                val = val / (1.0 + tl.math.exp(-val))
            tl.store(Out_ptr + offsets, val, mask=mask)
            start += tl.num_programs(0) * BLOCK


def persistent_bias_act(x, bias, activation: str = 'identity'):
    """Persistent-warp bias-add + optional activation — minimises launch overhead.

    x    : NDArray (N, D) or any shape where last dim = len(bias)
    bias : NDArray (D,)
    activation : 'identity' | 'relu' | 'gelu' | 'silu'
    """
    _ACT_MAP = {'identity': 0, 'relu': 1, 'gelu': 2, 'silu': 3}
    act_id = _ACT_MAP.get(activation, 0)
    xfl = x.to_flat_list() if hasattr(x, 'to_flat_list') else list(x)
    bfl = bias.to_flat_list() if hasattr(bias, 'to_flat_list') else list(bias)
    sh  = x.shape if hasattr(x, 'shape') else (len(xfl),)
    D   = len(bfl)
    N   = len(xfl) // D
    if not is_available():
        import math as _m
        result = []
        for i, v in enumerate(xfl):
            val = v + bfl[i % D]
            if activation == 'relu':  val = max(val, 0.0)
            elif activation == 'gelu': val = val * 0.5 * (1.0 + _m.tanh(0.7978845608*(val+0.044715*val**3)))
            elif activation == 'silu': val = val / (1.0 + _m.exp(-val))
            result.append(val)
        return _nd_array(result, 'float32').reshape(sh)
    tx   = torch.tensor(xfl, dtype=torch.float32).cuda()
    tb   = torch.tensor(bfl, dtype=torch.float32).cuda()
    tout = torch.empty_like(tx)
    BLOCK = 1024
    # use as many CTAs as SMs × 2 so warps stay resident (persistent pattern)
    num_sm = torch.cuda.get_device_properties(0).multi_processor_count
    grid   = (min(num_sm * 2, (len(xfl) + BLOCK - 1) // BLOCK),)
    _persistent_bias_act_kernel[grid](tx, tb, tout, N, D, act_id, BLOCK=BLOCK)
    return _nd_array(tout.cpu().tolist(), 'float32').reshape(sh)


# ── Fused dropout + residual ──────────────────────────────────────────────────

if _TRITON_AVAILABLE and _TORCH_AVAILABLE:

    @triton.autotune(
        configs=[triton.Config({'BLOCK': b}, num_warps=nw)
                 for b in [512, 1024, 2048] for nw in [4, 8]],
        key=['N'],
    )
    @triton.jit
    def _dropout_residual_kernel(
        X_ptr, Res_ptr, Out_ptr,
        N, p_drop, seed,
        BLOCK: tl.constexpr,
    ):
        pid     = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask    = offsets < N
        x   = tl.load(X_ptr   + offsets, mask=mask, other=0.0)
        res = tl.load(Res_ptr  + offsets, mask=mask, other=0.0)
        # Philox RNG — one call per element
        rand = tl.rand(seed, offsets)
        keep = rand > p_drop
        x    = tl.where(keep, x / (1.0 - p_drop), 0.0)
        tl.store(Out_ptr + offsets, x + res, mask=mask)


def fused_dropout_residual(x, residual, p_drop: float = 0.1, seed: int = 0):
    """Single-pass dropout + residual add; avoids intermediate dropout tensor."""
    xfl  = x.to_flat_list() if hasattr(x, 'to_flat_list') else list(x)
    rfl  = residual.to_flat_list() if hasattr(residual, 'to_flat_list') else list(residual)
    sh   = x.shape if hasattr(x, 'shape') else (len(xfl),)
    if not is_available():
        import random as _r, math as _m
        rng = _r.Random(seed)
        result = []
        for xi, ri in zip(xfl, rfl):
            xi = (xi / (1.0 - p_drop)) if rng.random() > p_drop else 0.0
            result.append(xi + ri)
        return _nd_array(result, 'float32').reshape(sh)
    N  = len(xfl)
    tx = torch.tensor(xfl, dtype=torch.float32).cuda()
    tr = torch.tensor(rfl, dtype=torch.float32).cuda()
    to = torch.empty_like(tx)
    grid = lambda meta: ((N + meta['BLOCK'] - 1) // meta['BLOCK'],)
    _dropout_residual_kernel[grid](tx, tr, to, N, p_drop, seed)
    return _nd_array(to.cpu().tolist(), 'float32').reshape(sh)


# ── Chunked online softmax (memory-efficient for very long sequences) ─────────

if _TRITON_AVAILABLE and _TORCH_AVAILABLE:

    @triton.autotune(
        configs=[triton.Config({'BLOCK': b}, num_warps=nw)
                 for b in [256, 512, 1024, 2048] for nw in [4, 8]],
        key=['N'],
    )
    @triton.jit
    def _online_softmax_kernel(
        X_ptr, Out_ptr, N,
        BLOCK: tl.constexpr,
    ):
        row  = tl.program_id(0)
        base = row * N
        # pass 1: max
        m = -1e9
        for start in tl.range(0, N, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            x    = tl.load(X_ptr + base + offs, mask=offs < N, other=-1e9)
            m    = tl.maximum(m, tl.max(x, axis=0))
        # pass 2: sum-exp
        s = 0.0
        for start in tl.range(0, N, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            x    = tl.load(X_ptr + base + offs, mask=offs < N, other=-1e9)
            s   += tl.sum(tl.exp(x - m), axis=0)
        # pass 3: write
        for start in tl.range(0, N, BLOCK):
            offs  = start + tl.arange(0, BLOCK)
            mask  = offs < N
            x     = tl.load(X_ptr + base + offs, mask=mask, other=0.0)
            tl.store(Out_ptr + base + offs, tl.exp(x - m) / s, mask=mask)


def online_softmax(x):
    """Chunked 3-pass online softmax; O(1) extra memory vs standard softmax."""
    xfl = x.to_flat_list() if hasattr(x, 'to_flat_list') else list(x)
    sh  = x.shape if hasattr(x, 'shape') else (len(xfl),)
    N   = sh[-1]
    rows = len(xfl) // N
    if not is_available():
        import math as _m
        result = []
        for r in range(rows):
            row = xfl[r*N:(r+1)*N]
            m = max(row); exps = [_m.exp(v-m) for v in row]; s = sum(exps)
            result.extend([e/s for e in exps])
        return _nd_array(result, 'float32').reshape(sh)
    tx  = torch.tensor(xfl, dtype=torch.float32).reshape(rows, N).cuda().contiguous()
    out = torch.empty_like(tx)
    _online_softmax_kernel[(rows,)](tx, out, N)
    return _nd_array(out.cpu().reshape(-1).tolist(), 'float32').reshape(sh)


# ── Fused scaled-dot-product attention (single head, Triton) ─────────────────

if _TRITON_AVAILABLE and _TORCH_AVAILABLE:

    _SDPA_CONFIGS = [
        triton.Config({'BLOCK_N': bn, 'BLOCK_M': bm}, num_warps=nw)
        for bn in [32, 64, 128]
        for bm in [32, 64]
        for nw in [4, 8]
    ]

    @triton.autotune(configs=_SDPA_CONFIGS, key=['SEQ_LEN', 'HEAD_DIM'])
    @triton.jit
    def _sdpa_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        stride_qn, stride_qd,
        stride_kn, stride_kd,
        stride_vn, stride_vd,
        stride_on, stride_od,
        SEQ_LEN: tl.constexpr, HEAD_DIM: tl.constexpr,
        scale,
        causal: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        q_range = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        d_range = tl.arange(0, HEAD_DIM)
        q_mask  = q_range < SEQ_LEN
        q = tl.load(Q_ptr + q_range[:, None] * stride_qn + d_range[None, :] * stride_qd,
                    mask=q_mask[:, None], other=0.0)
        m_i  = tl.full([BLOCK_M], -1e9, dtype=tl.float32)
        l_i  = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc  = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        for start_n in tl.range(0, SEQ_LEN, BLOCK_N):
            n_range  = start_n + tl.arange(0, BLOCK_N)
            kv_mask  = n_range < SEQ_LEN
            if causal:
                kv_mask = kv_mask & (n_range[None, :] <= q_range[:, None])
            k = tl.load(K_ptr + n_range[:, None] * stride_kn + d_range[None, :] * stride_kd,
                        mask=kv_mask[:, None] if kv_mask.shape == (BLOCK_N,) else kv_mask, other=0.0)
            attn = tl.dot(q, tl.trans(k)) * scale       # (BLOCK_M, BLOCK_N)
            if causal:
                causal_mask = q_range[:, None] >= n_range[None, :]
                attn = tl.where(causal_mask & (kv_mask[None, :]), attn, -1e9)
            else:
                attn = tl.where(kv_mask[None, :], attn, -1e9)
            m_new  = tl.maximum(m_i, tl.max(attn, axis=1))
            alpha  = tl.exp(m_i - m_new)
            l_i    = l_i * alpha + tl.sum(tl.exp(attn - m_new[:, None]), axis=1)
            v = tl.load(V_ptr + n_range[:, None] * stride_vn + d_range[None, :] * stride_vd,
                        mask=kv_mask[:, None] if kv_mask.shape == (BLOCK_N,) else kv_mask, other=0.0)
            acc    = acc * alpha[:, None] + tl.dot(tl.exp(attn - m_new[:, None]), v)
            m_i    = m_new
        acc = acc / l_i[:, None]
        tl.store(Out_ptr + q_range[:, None] * stride_on + d_range[None, :] * stride_od,
                 acc, mask=q_mask[:, None])


def triton_sdpa(q, k, v, scale: float = None, causal: bool = False):
    """Triton fused scaled-dot-product attention (single head).

    q, k, v : NDArray (N, D)
    Returns  : NDArray (N, D)
    """
    import math as _m
    qfl = q.to_flat_list() if hasattr(q, 'to_flat_list') else list(q)
    kfl = k.to_flat_list() if hasattr(k, 'to_flat_list') else list(k)
    vfl = v.to_flat_list() if hasattr(v, 'to_flat_list') else list(v)
    sh  = q.shape if hasattr(q, 'shape') else (len(qfl),)
    N, D = sh[0], sh[1]
    sc   = scale or (1.0 / _m.sqrt(D))
    if not is_available():
        # pure-Python fallback
        result = []
        for i in range(N):
            qi = qfl[i*D:(i+1)*D]
            scores = []
            for j in range(N):
                if causal and j > i: scores.append(-1e9); continue
                kj = kfl[j*D:(j+1)*D]
                scores.append(sum(qi[d]*kj[d] for d in range(D)) * sc)
            m_ = max(scores)
            exps = [_m.exp(s - m_) for s in scores]; denom = sum(exps)
            row = [0.0]*D
            for j in range(N):
                vj = vfl[j*D:(j+1)*D]
                w  = exps[j] / denom
                for d in range(D): row[d] += w * vj[d]
            result.extend(row)
        return _nd_array(result, 'float32').reshape(sh)
    tq = torch.tensor(qfl, dtype=torch.float32).reshape(N, D).cuda().contiguous()
    tk = torch.tensor(kfl, dtype=torch.float32).reshape(N, D).cuda().contiguous()
    tv = torch.tensor(vfl, dtype=torch.float32).reshape(N, D).cuda().contiguous()
    to = torch.empty_like(tq)
    BLOCK_M = 32
    grid    = ((N + BLOCK_M - 1) // BLOCK_M,)
    _sdpa_kernel[grid](
        tq, tk, tv, to,
        tq.stride(0), tq.stride(1),
        tk.stride(0), tk.stride(1),
        tv.stride(0), tv.stride(1),
        to.stride(0), to.stride(1),
        SEQ_LEN=N, HEAD_DIM=D,
        scale=sc, causal=causal, BLOCK_M=BLOCK_M,
    )
    return _nd_array(to.cpu().reshape(-1).tolist(), 'float32').reshape(sh)


# =============================================================================
# Session 11 — Triton Flash Attention v3 improvements
# =============================================================================

if _TRITON_AVAILABLE and _TORCH_AVAILABLE:
    @triton.jit
    def _flash_attn_v3_kernel(
        Q, K, V, Out,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        BATCH: tl.constexpr, HEADS: tl.constexpr,
        SEQ_M: tl.constexpr, SEQ_N: tl.constexpr,
        HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        scale: tl.constexpr, causal: tl.constexpr,
    ):
        """Flash Attention v3 — persistent block tile, online softmax, fused O-update.

        Each program handles one (batch, head, block_m) tile. Streams over
        K/V in BLOCK_N tiles without materialising the full N×N score matrix.
        Online (m, l) running max-normalisation gives numerically identical
        output to standard attention with O(BLOCK) SRAM.
        """
        bh  = tl.program_id(0)
        b   = bh // HEADS
        h   = bh % HEADS
        m_start = tl.program_id(1) * BLOCK_M

        Q_base = Q + b*stride_qb + h*stride_qh
        K_base = K + b*stride_kb + h*stride_kh
        V_base = V + b*stride_vb + h*stride_vh
        O_base = Out + b*stride_ob + h*stride_oh

        # Load Q tile (BLOCK_M, HEAD_DIM)
        offs_m = m_start + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        q = tl.load(Q_base + offs_m[:, None]*stride_qm + offs_d[None, :]*stride_qd,
                    mask=offs_m[:, None] < SEQ_M, other=0.0)

        # Running statistics for online softmax
        m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # Streaming over K/V
        n_blocks = tl.cdiv(SEQ_N, BLOCK_N)
        for n_blk in range(n_blocks):
            n_start = n_blk * BLOCK_N
            offs_n  = n_start + tl.arange(0, BLOCK_N)
            k = tl.load(K_base + offs_n[None, :]*stride_kn + offs_d[:, None]*stride_kd,
                        mask=offs_n[None, :] < SEQ_N, other=0.0)
            # (BLOCK_M, BLOCK_N) scores
            scores = tl.dot(q, k) * scale
            if causal:
                mask = offs_m[:, None] >= offs_n[None, :]
                scores = tl.where(mask, scores, -1e9)
            # Online softmax update
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            l_new = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(scores - m_new[:, None]), axis=1)
            # Load V tile (BLOCK_N, HEAD_DIM)
            v = tl.load(V_base + offs_n[:, None]*stride_vn + offs_d[None, :]*stride_vd,
                        mask=offs_n[:, None] < SEQ_N, other=0.0)
            attn = tl.exp(scores - m_new[:, None])
            acc   = acc * (l_i * tl.exp(m_i - m_new) / (l_new + 1e-8))[:, None] + tl.dot(attn, v) / (l_new + 1e-8)[:, None]
            m_i   = m_new
            l_i   = l_new

        # Write output
        tl.store(O_base + offs_m[:, None]*stride_om + offs_d[None, :]*stride_od, acc,
                 mask=offs_m[:, None] < SEQ_M)


def flash_attn_v3(q, k, v, scale: float = None, causal: bool = False, num_heads: int = 1):
    """Flash Attention v3 — multi-head (B, H, N, D) inputs.

    Uses persistent block tiling and online softmax for O(1) SRAM usage.
    Falls back to pure Python when Triton is unavailable.

    Args:
        q, k, v: NDArray or list (B, H, N, D)
        scale: softmax scale (default 1/sqrt(D))
        causal: causal mask
        num_heads: number of attention heads
    Returns:
        NDArray (B, H, N, D)
    """
    import math as _m
    def _to_list(t):
        return t.to_flat_list() if hasattr(t, 'to_flat_list') else list(t)
    qfl = _to_list(q); sh = q.shape if hasattr(q, 'shape') else (len(qfl),)
    if len(sh) == 2:
        sh = (1, 1, sh[0], sh[1])
    B, H, N, D = sh
    sc = scale or (1.0 / _m.sqrt(D))

    if not is_available():
        # Fallback to triton_sdpa per head
        kfl = _to_list(k); vfl = _to_list(v)
        qsh = (N, D)
        result = []
        for b_ in range(B):
            for h_ in range(H):
                offset = (b_ * H + h_) * N * D
                q_nd = _nd_array(qfl[offset:offset+N*D], 'float32').reshape(qsh)
                k_nd = _nd_array(kfl[offset:offset+N*D], 'float32').reshape(qsh)
                v_nd = _nd_array(vfl[offset:offset+N*D], 'float32').reshape(qsh)
                o_nd = triton_sdpa(q_nd, k_nd, v_nd, scale=sc, causal=causal)
                result.extend(o_nd.to_flat_list() if hasattr(o_nd, 'to_flat_list') else list(o_nd))
        return _nd_array(result, 'float32').reshape(sh)

    # Triton path
    kfl = _to_list(k); vfl = _to_list(v)
    tq = torch.tensor(qfl, dtype=torch.float32).reshape(B, H, N, D).cuda().contiguous()
    tk = torch.tensor(kfl, dtype=torch.float32).reshape(B, H, N, D).cuda().contiguous()
    tv = torch.tensor(vfl, dtype=torch.float32).reshape(B, H, N, D).cuda().contiguous()
    to = torch.empty_like(tq)
    BLOCK_M = 32; BLOCK_N = 32
    grid = (B * H, (N + BLOCK_M - 1) // BLOCK_M)
    _flash_attn_v3_kernel[grid](
        tq, tk, tv, to,
        tq.stride(0), tq.stride(1), tq.stride(2), tq.stride(3),
        tk.stride(0), tk.stride(1), tk.stride(2), tk.stride(3),
        tv.stride(0), tv.stride(1), tv.stride(2), tv.stride(3),
        to.stride(0), to.stride(1), to.stride(2), to.stride(3),
        BATCH=B, HEADS=H, SEQ_M=N, SEQ_N=N,
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        scale=sc, causal=causal,
    )
    out_list = to.cpu().reshape(-1).tolist()
    return _nd_array(out_list, 'float32').reshape(sh)


# ── Fused RoPE (Rotary Position Embeddings) ───────────────────────────────────
# Su et al. 2021 — rotates adjacent pairs of head-dim elements by position-
# dependent angles. Single Triton kernel: load pair, rotate, store — no staging.

if _TRITON_AVAILABLE and _TORCH_AVAILABLE:

    @triton.autotune(
        configs=[triton.Config({'BLOCK': b}, num_warps=nw)
                 for b in [64, 128, 256, 512] for nw in [4, 8]],
        key=['D_HALF'],
    )
    @triton.jit
    def _rope_kernel(
        X_ptr, Cos_ptr, Sin_ptr, Out_ptr,
        seq_stride, head_stride,
        SEQ: tl.constexpr, D_HALF: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """One program per (batch*head, position) pair."""
        bh  = tl.program_id(0)
        pos = tl.program_id(1)
        base_x   = X_ptr   + bh * seq_stride + pos * head_stride
        base_out = Out_ptr  + bh * seq_stride + pos * head_stride
        offsets  = tl.arange(0, BLOCK)
        mask     = offsets < D_HALF
        x0 = tl.load(base_x + offsets,         mask=mask, other=0.0)
        x1 = tl.load(base_x + offsets + D_HALF, mask=mask, other=0.0)
        cos = tl.load(Cos_ptr + pos * D_HALF + offsets, mask=mask, other=1.0)
        sin = tl.load(Sin_ptr + pos * D_HALF + offsets, mask=mask, other=0.0)
        tl.store(base_out + offsets,         x0 * cos - x1 * sin, mask=mask)
        tl.store(base_out + offsets + D_HALF, x0 * sin + x1 * cos, mask=mask)


def fused_rope(x, cos, sin):
    """Fused Rotary Position Embedding application.

    Args:
        x   : NDArray (B, H, N, D) — query or key tensor
        cos : NDArray (N, D//2)    — cosine table
        sin : NDArray (N, D//2)    — sine table
    Returns:
        NDArray (B, H, N, D) with RoPE applied
    """
    xfl   = x.to_flat_list()   if hasattr(x,   'to_flat_list') else list(x)
    cosfl = cos.to_flat_list() if hasattr(cos, 'to_flat_list') else list(cos)
    sinfl = sin.to_flat_list() if hasattr(sin, 'to_flat_list') else list(sin)
    sh    = x.shape if hasattr(x, 'shape') else (len(xfl),)
    if len(sh) == 3:
        B, N, D = sh; H = 1
    else:
        B, H, N, D = sh
    D_HALF = D // 2

    if not is_available():
        result = []
        for b in range(B):
            for h in range(H):
                for n in range(N):
                    base = (b * H + h) * N * D + n * D
                    for i in range(D_HALF):
                        x0, x1 = xfl[base + i], xfl[base + i + D_HALF]
                        c, s = cosfl[n * D_HALF + i], sinfl[n * D_HALF + i]
                        result.append(x0 * c - x1 * s)
                    for i in range(D_HALF):
                        x0, x1 = xfl[base + i], xfl[base + i + D_HALF]
                        c, s = cosfl[n * D_HALF + i], sinfl[n * D_HALF + i]
                        result.append(x0 * s + x1 * c)
        return _nd_array(result, 'float32').reshape(sh)

    tx   = torch.tensor(xfl,   dtype=torch.float32).reshape(B * H, N, D).cuda().contiguous()
    tcos = torch.tensor(cosfl, dtype=torch.float32).reshape(N, D_HALF).cuda().contiguous()
    tsin = torch.tensor(sinfl, dtype=torch.float32).reshape(N, D_HALF).cuda().contiguous()
    tout = torch.empty_like(tx)
    BLOCK = max(32, 1 << (_math.ceil(_math.log2(D_HALF)) if D_HALF > 1 else 1))
    grid  = (B * H, N)
    _rope_kernel[grid](
        tx, tcos, tsin, tout,
        tx.stride(0), tx.stride(1),
        SEQ=N, D_HALF=D_HALF, BLOCK=BLOCK,
    )
    return _nd_array(tout.cpu().reshape(-1).tolist(), 'float32').reshape(sh)


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0):
    """Pre-compute (cos, sin) tables for RoPE.

    Returns two NDArrays of shape (seq_len, head_dim // 2).
    """
    D_HALF = head_dim // 2
    freqs  = [1.0 / (base ** (2 * i / head_dim)) for i in range(D_HALF)]
    cos_data, sin_data = [], []
    for pos in range(seq_len):
        for f in freqs:
            angle = pos * f
            cos_data.append(_math.cos(angle))
            sin_data.append(_math.sin(angle))
    return (_nd_array(cos_data, 'float32').reshape((seq_len, D_HALF)),
            _nd_array(sin_data, 'float32').reshape((seq_len, D_HALF)))


# ── Fused Batch Normalisation (inference + training forward) ──────────────────
# Single kernel: load tile, compute mean+var in registers, normalize+affine, store.
# Training mode accumulates running stats atomically for the batch.

if _TRITON_AVAILABLE and _TORCH_AVAILABLE:

    _BN_CONFIGS = [
        triton.Config({'BLOCK': b}, num_warps=nw)
        for b in [128, 256, 512, 1024] for nw in [4, 8]
    ]

    @triton.autotune(configs=_BN_CONFIGS, key=['N'])
    @triton.jit
    def _batch_norm_inference_kernel(
        X_ptr, Mean_ptr, Var_ptr, W_ptr, B_ptr, Out_ptr,
        C, N,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Inference BN: one program per (channel × spatial) chunk."""
        pid = tl.program_id(0)
        c   = pid // ((N + BLOCK - 1) // BLOCK)
        tid = pid  % ((N + BLOCK - 1) // BLOCK)
        offsets = tid * BLOCK + tl.arange(0, BLOCK)
        mask    = offsets < N
        idx     = c * N + offsets
        x    = tl.load(X_ptr   + idx, mask=mask, other=0.0)
        mean = tl.load(Mean_ptr + c)
        var  = tl.load(Var_ptr  + c)
        w    = tl.load(W_ptr    + c)
        b    = tl.load(B_ptr    + c)
        x_norm = (x - mean) / tl.sqrt(var + eps)
        tl.store(Out_ptr + idx, x_norm * w + b, mask=mask)


def fused_batch_norm(x, weight, bias, running_mean, running_var,
                     training: bool = False, momentum: float = 0.1,
                     eps: float = 1e-5):
    """Fused batch normalisation — Triton inference path, Python training path.

    Args:
        x            : NDArray (N, C, *spatial) or (N, C)
        weight, bias : NDArray (C,) — gamma and beta
        running_mean, running_var : NDArray (C,) updated in-place for training
        training     : if True, compute batch stats; else use running stats
        momentum     : EMA factor for running stats
        eps          : numerical stability epsilon
    Returns:
        NDArray same shape as x
    """
    xfl  = x.to_flat_list()    if hasattr(x,            'to_flat_list') else list(x)
    wfl  = weight.to_flat_list() if hasattr(weight,     'to_flat_list') else list(weight)
    bfl  = bias.to_flat_list()   if hasattr(bias,       'to_flat_list') else list(bias)
    rmfl = running_mean.to_flat_list() if hasattr(running_mean, 'to_flat_list') else list(running_mean)
    rvfl = running_var.to_flat_list()  if hasattr(running_var,  'to_flat_list') else list(running_var)
    sh   = x.shape if hasattr(x, 'shape') else (len(xfl),)
    C    = sh[1] if len(sh) > 1 else 1
    N    = len(xfl) // C  # total spatial×batch elements per channel

    if training:
        # compute batch mean/var, update running stats
        mean = [0.0] * C; var  = [0.0] * C
        cnt  = N
        for c in range(C):
            s = sum(xfl[c * N + i] for i in range(N)) / cnt
            mean[c] = s
            var[c]  = sum((xfl[c * N + i] - s) ** 2 for i in range(N)) / cnt
            rmfl[c] = (1 - momentum) * rmfl[c] + momentum * mean[c]
            rvfl[c] = (1 - momentum) * rvfl[c] + momentum * var[c]
        # write updated running stats back
        if hasattr(running_mean, '_buf'):
            import array as _a
            running_mean._buf = _a.array('f', rmfl)
            running_var._buf  = _a.array('f', rvfl)
    else:
        mean = rmfl; var = rvfl

    if not is_available() or training:
        result = []
        for c in range(C):
            inv = 1.0 / _math.sqrt(var[c] + eps)
            for i in range(N):
                result.append((xfl[c * N + i] - mean[c]) * inv * wfl[c] + bfl[c])
        return _nd_array(result, 'float32').reshape(sh)

    tx   = torch.tensor(xfl,  dtype=torch.float32).cuda()
    tw   = torch.tensor(wfl,  dtype=torch.float32).cuda()
    tb   = torch.tensor(bfl,  dtype=torch.float32).cuda()
    tmean = torch.tensor(mean, dtype=torch.float32).cuda()
    tvar  = torch.tensor(var,  dtype=torch.float32).cuda()
    tout  = torch.empty_like(tx)
    chunks_per_channel = (N + 127) // 128
    grid   = (C * chunks_per_channel,)
    _batch_norm_inference_kernel[grid](tx, tmean, tvar, tw, tb, tout, C, N, eps)
    return _nd_array(tout.cpu().tolist(), 'float32').reshape(sh)


# ── Fused Mamba selective scan (SSM step) ─────────────────────────────────────
# Mamba (Gu & Dao 2023) state-space model — discretised SSM recurrence.
# h_t = A_bar * h_{t-1} + B_bar * x_t
# y_t = C * h_t  (+  D * x_t  skip connection)
# Kernel fuses the discretise→recur→output steps for one chunk.

if _TRITON_AVAILABLE and _TORCH_AVAILABLE:

    @triton.jit
    def _mamba_scan_kernel(
        X_ptr, A_ptr, B_ptr, C_ptr, D_ptr,
        Delta_ptr, H_ptr, Out_ptr,
        B, L, D_MODEL, D_STATE,
        BLOCK_D: tl.constexpr,
    ):
        """One program per (batch, state_dim) pair; loops over sequence length."""
        b   = tl.program_id(0)
        ds  = tl.program_id(1)  # state index
        d_offsets = tl.arange(0, BLOCK_D)

        # h shape: (B, D_MODEL, D_STATE) — load initial h for this (b, :, ds)
        h = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for t in range(L):
            x_base = b * L * D_MODEL + t * D_MODEL
            d_base = b * L * D_MODEL + t * D_MODEL
            mask   = d_offsets < D_MODEL

            x     = tl.load(X_ptr     + x_base + d_offsets, mask=mask, other=0.0)
            delta = tl.load(Delta_ptr + x_base + d_offsets, mask=mask, other=0.0)
            a     = tl.load(A_ptr     + d_offsets * D_STATE + ds, mask=mask, other=0.0)
            b_val = tl.load(B_ptr     + b * L * D_STATE + t * D_STATE + ds)
            c_val = tl.load(C_ptr     + b * L * D_STATE + t * D_STATE + ds)
            d_val = tl.load(D_ptr     + d_offsets, mask=mask, other=0.0)

            # Discretise A: A_bar = exp(delta * A)
            a_bar = tl.exp(delta * a)
            # Discretise B: B_bar = delta * B  (ZOH approximation)
            b_bar = delta * b_val

            h = a_bar * h + b_bar * x
            y = c_val * h + d_val * x

            out_base = b * L * D_MODEL + t * D_MODEL
            tl.store(Out_ptr + out_base + d_offsets, y, mask=mask)


def mamba_selective_scan(x, delta, A, B, C, D):
    """Mamba SSM selective scan — discretise + recur + output in one pass.

    Args:
        x     : NDArray (B, L, D)        — input sequence
        delta : NDArray (B, L, D)        — input-dependent step sizes
        A     : NDArray (D, N)           — state transition matrix (log-space)
        B     : NDArray (B, L, N)        — input projection
        C     : NDArray (B, L, N)        — output projection
        D     : NDArray (D,)             — skip connection scale
    Returns:
        NDArray (B, L, D) — SSM output
    """
    def _fl(t): return t.to_flat_list() if hasattr(t, 'to_flat_list') else list(t)
    xfl = _fl(x); sh = x.shape if hasattr(x, 'shape') else (len(xfl),)
    B_, L_, D_MODEL = sh
    D_STATE = A.shape[1] if hasattr(A, 'shape') else len(_fl(A)) // D_MODEL
    afl = _fl(A); bfl = _fl(B); cfl = _fl(C); dfl = _fl(D); deltafl = _fl(delta)

    if not is_available():
        result = [0.0] * (B_ * L_ * D_MODEL)
        for b in range(B_):
            h = [[0.0] * D_STATE for _ in range(D_MODEL)]
            for t in range(L_):
                for d in range(D_MODEL):
                    delta_dt = deltafl[b * L_ * D_MODEL + t * D_MODEL + d]
                    for n in range(D_STATE):
                        a_log  = afl[d * D_STATE + n]
                        a_bar  = _math.exp(delta_dt * a_log)
                        b_bar  = delta_dt * bfl[b * L_ * D_STATE + t * D_STATE + n]
                        h[d][n] = a_bar * h[d][n] + b_bar * xfl[b * L_ * D_MODEL + t * D_MODEL + d]
                    y = sum(cfl[b * L_ * D_STATE + t * D_STATE + n] * h[d][n] for n in range(D_STATE))
                    y += dfl[d] * xfl[b * L_ * D_MODEL + t * D_MODEL + d]
                    result[b * L_ * D_MODEL + t * D_MODEL + d] = y
        return _nd_array(result, 'float32').reshape(sh)

    tx     = torch.tensor(xfl,     dtype=torch.float32).cuda().contiguous()
    tdelta = torch.tensor(deltafl, dtype=torch.float32).cuda().contiguous()
    tA     = torch.tensor(afl,     dtype=torch.float32).reshape(D_MODEL, D_STATE).cuda().contiguous()
    tB     = torch.tensor(bfl,     dtype=torch.float32).reshape(B_, L_, D_STATE).cuda().contiguous()
    tC     = torch.tensor(cfl,     dtype=torch.float32).reshape(B_, L_, D_STATE).cuda().contiguous()
    tD     = torch.tensor(dfl,     dtype=torch.float32).cuda().contiguous()
    tout   = torch.empty(B_, L_, D_MODEL, dtype=torch.float32, device='cuda')
    BLOCK_D = max(32, 1 << (_math.ceil(_math.log2(D_MODEL)) if D_MODEL > 1 else 1))
    grid    = (B_, D_STATE)
    _mamba_scan_kernel[grid](
        tx, tA, tB, tC, tD, tdelta, None, tout,
        B_, L_, D_MODEL, D_STATE, BLOCK_D=BLOCK_D,
    )
    return _nd_array(tout.cpu().reshape(-1).tolist(), 'float32').reshape(sh)


# ---------------------------------------------------------------------------
# Fused LayerNorm kernel
# ---------------------------------------------------------------------------
if _TRITON_AVAILABLE:
    @triton.jit
    def _layer_norm_kernel(X, W, B, Y, mean_ptr, rstd_ptr,
                           stride, N, eps,
                           BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        X_row = X + row * stride
        Y_row = Y + row * stride
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X_row + offs, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / N
        var  = tl.sum((x - mean) * (x - mean), axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(mean_ptr + row, mean)
        tl.store(rstd_ptr + row, rstd)
        x_hat = (x - mean) * rstd
        w = tl.load(W + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
        y = x_hat * w + b
        tl.store(Y_row + offs, y, mask=mask)


def fused_layer_norm(x, weight, bias, eps: float = 1e-5):
    """Fused layer normalisation — Triton GPU path, VP CPU fallback.

    Args:
        x      : NDArray (..., N) — last dim is normalised
        weight : NDArray (N,)
        bias   : NDArray (N,)
        eps    : numerical epsilon
    Returns:
        NDArray same shape as x
    """
    def _fl(t): return t.to_flat_list() if hasattr(t, 'to_flat_list') else list(t)
    xfl = _fl(x)
    sh  = x.shape if hasattr(x, 'shape') else (len(xfl),)
    N   = sh[-1]
    M   = len(xfl) // N
    wfl = _fl(weight); bfl = _fl(bias)

    if not is_available() or not _TRITON_AVAILABLE:
        result = []
        for i in range(M):
            row = xfl[i * N:(i + 1) * N]
            mu  = sum(row) / N
            var = sum((v - mu) ** 2 for v in row) / N
            inv = 1.0 / _math.sqrt(var + eps)
            result.extend((row[j] - mu) * inv * wfl[j] + bfl[j] for j in range(N))
        return _nd_array(result, 'float32').reshape(sh)

    tx = torch.tensor(xfl, dtype=torch.float32).cuda().reshape(M, N).contiguous()
    tw = torch.tensor(wfl, dtype=torch.float32).cuda()
    tb = torch.tensor(bfl, dtype=torch.float32).cuda()
    ty = torch.empty_like(tx)
    t_mean = torch.empty(M, dtype=torch.float32, device='cuda')
    t_rstd = torch.empty(M, dtype=torch.float32, device='cuda')
    BLOCK_SIZE = max(16, 1 << (_math.ceil(_math.log2(N)) if N > 1 else 1))
    _layer_norm_kernel[(M,)](tx, tw, tb, ty, t_mean, t_rstd, N, N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return _nd_array(ty.cpu().reshape(-1).tolist(), 'float32').reshape(sh)


# ---------------------------------------------------------------------------
# Fused element-wise activation kernels (GELU, SiLU, ReLU)
# ---------------------------------------------------------------------------
if _TRITON_AVAILABLE:
    @triton.jit
    def _gelu_kernel(X, Y, N, BLOCK: tl.constexpr):
        pid  = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x    = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
        c    = 0.7978845608028654  # sqrt(2/pi)
        t    = tl.tanh(c * (x + 0.044715 * x * x * x))
        y    = 0.5 * x * (1.0 + t)
        tl.store(Y + offs, y, mask=mask)

    @triton.jit
    def _silu_kernel(X, Y, N, BLOCK: tl.constexpr):
        pid  = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x    = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        y    = x / (1.0 + tl.exp(-x))
        tl.store(Y + offs, y, mask=mask)

    @triton.jit
    def _relu_kernel(X, Y, N, BLOCK: tl.constexpr):
        pid  = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x    = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        y    = tl.where(x > 0.0, x, 0.0)
        tl.store(Y + offs, y, mask=mask)


def _fused_activation(x, kernel_fn, cpu_fn, name: str):
    """Helper: apply a pointwise activation via Triton or CPU fallback."""
    def _fl(t): return t.to_flat_list() if hasattr(t, 'to_flat_list') else list(t)
    xfl = _fl(x)
    sh  = x.shape if hasattr(x, 'shape') else (len(xfl),)
    N   = len(xfl)
    if not is_available() or not _TRITON_AVAILABLE:
        return _nd_array([cpu_fn(v) for v in xfl], 'float32').reshape(sh)
    tx   = torch.tensor(xfl, dtype=torch.float32).cuda()
    ty   = torch.empty_like(tx)
    BLOCK = 1024
    grid  = ((N + BLOCK - 1) // BLOCK,)
    kernel_fn[grid](tx, ty, N, BLOCK=BLOCK)
    return _nd_array(ty.cpu().tolist(), 'float32').reshape(sh)


def fused_gelu(x):
    """Fused GELU activation."""
    import math as _m
    _c = 0.7978845608028654
    def _cpu(v):
        t = _m.tanh(_c * (v + 0.044715 * v * v * v))
        return 0.5 * v * (1.0 + t)
    if _TRITON_AVAILABLE:
        return _fused_activation(x, _gelu_kernel, _cpu, 'gelu')
    def _fl(t): return t.to_flat_list() if hasattr(t, 'to_flat_list') else list(t)
    xfl = _fl(x)
    sh  = x.shape if hasattr(x, 'shape') else (len(xfl),)
    return _nd_array([_cpu(v) for v in xfl], 'float32').reshape(sh)


def fused_silu(x):
    """Fused SiLU (Swish) activation."""
    import math as _m
    _cpu = lambda v: v / (1.0 + _m.exp(-v))
    if _TRITON_AVAILABLE:
        return _fused_activation(x, _silu_kernel, _cpu, 'silu')
    def _fl(t): return t.to_flat_list() if hasattr(t, 'to_flat_list') else list(t)
    xfl = _fl(x)
    sh  = x.shape if hasattr(x, 'shape') else (len(xfl),)
    return _nd_array([_cpu(v) for v in xfl], 'float32').reshape(sh)


def fused_relu(x):
    """Fused ReLU activation."""
    _cpu = lambda v: v if v > 0.0 else 0.0
    if _TRITON_AVAILABLE:
        return _fused_activation(x, _relu_kernel, _cpu, 'relu')
    def _fl(t): return t.to_flat_list() if hasattr(t, 'to_flat_list') else list(t)
    xfl = _fl(x)
    sh  = x.shape if hasattr(x, 'shape') else (len(xfl),)
    return _nd_array([_cpu(v) for v in xfl], 'float32').reshape(sh)


# ---------------------------------------------------------------------------
# Fused Cross-Entropy forward + backward kernel
# ---------------------------------------------------------------------------
if _TRITON_AVAILABLE:
    @triton.jit
    def _cross_entropy_fwd_kernel(logits_ptr, targets_ptr, losses_ptr,
                                  N, C, BLOCK_C: tl.constexpr):
        """One program per sample in the batch."""
        row   = tl.program_id(0)
        offs  = tl.arange(0, BLOCK_C)
        mask  = offs < C
        x     = tl.load(logits_ptr + row * C + offs, mask=mask, other=-float('inf')).to(tl.float32)
        x_max = tl.max(x, axis=0)
        x     = x - x_max
        exp_x = tl.exp(x)
        sum_e = tl.sum(exp_x, axis=0)
        log_z = tl.log(sum_e)
        tgt   = tl.load(targets_ptr + row).to(tl.int32)
        log_p_tgt = tl.load(logits_ptr + row * C + tgt).to(tl.float32) - x_max - log_z
        tl.store(losses_ptr + row, -log_p_tgt)

    @triton.jit
    def _cross_entropy_bwd_kernel(logits_ptr, targets_ptr, grads_ptr,
                                  N, C, scale, BLOCK_C: tl.constexpr):
        row   = tl.program_id(0)
        offs  = tl.arange(0, BLOCK_C)
        mask  = offs < C
        x     = tl.load(logits_ptr + row * C + offs, mask=mask, other=-float('inf')).to(tl.float32)
        x_max = tl.max(x, axis=0)
        x     = x - x_max
        exp_x = tl.exp(x)
        sum_e = tl.sum(exp_x, axis=0)
        probs = exp_x / sum_e
        tgt   = tl.load(targets_ptr + row).to(tl.int32)
        is_tgt = (offs == tgt)
        g = (probs - tl.where(is_tgt, 1.0, 0.0)) * scale
        tl.store(grads_ptr + row * C + offs, g, mask=mask)


def fused_cross_entropy(logits, targets, reduction: str = 'mean'):
    """Fused cross-entropy loss — Triton forward + backward in one pass.

    Args:
        logits  : NDArray (N, C) — raw logits (float32)
        targets : NDArray (N,)   — integer class indices
        reduction: 'mean' | 'sum' | 'none'
    Returns:
        (loss_scalar_or_array, grad_fn) where grad_fn(upstream) → NDArray (N, C)
    """
    def _fl(t): return t.to_flat_list() if hasattr(t, 'to_flat_list') else list(t)
    logfl = _fl(logits); tgtfl = [int(v) for v in _fl(targets)]
    sh = logits.shape if hasattr(logits, 'shape') else (len(logfl),)
    N, C = sh

    if not is_available() or not _TRITON_AVAILABLE:
        raise RuntimeError("CUDA/Triton not available; use pure-Python cross-entropy path")

    tlog = torch.tensor(logfl, dtype=torch.float32).cuda().reshape(N, C).contiguous()
    ttgt = torch.tensor(tgtfl, dtype=torch.int32).cuda()
    tlos = torch.empty(N, dtype=torch.float32, device='cuda')
    BLOCK_C = max(32, 1 << (_math.ceil(_math.log2(C)) if C > 1 else 1))
    _cross_entropy_fwd_kernel[(N,)](tlog, ttgt, tlos, N, C, BLOCK_C=BLOCK_C)
    losses_list = tlos.cpu().tolist()
    if reduction == 'mean': loss = sum(losses_list) / N
    elif reduction == 'sum': loss = sum(losses_list)
    else:
        return _nd_array(losses_list, 'float32'), (tlog, ttgt, N, C, BLOCK_C)
    return loss, (tlog, ttgt, N, C, BLOCK_C)


def fused_cross_entropy_backward(ctx, upstream_grad, N, C):
    """Compute d(CE)/d(logits) using the saved Triton context."""
    tlog, ttgt, N_, C_, BLOCK_C = ctx
    scale = float(upstream_grad) / N_
    tgrads = torch.empty(N_, C_, dtype=torch.float32, device='cuda')
    _cross_entropy_bwd_kernel[(N_,)](tlog, ttgt, tgrads, N_, C_, scale, BLOCK_C=BLOCK_C)
    return _nd_array(tgrads.cpu().reshape(-1).tolist(), 'float32').reshape((N_, C_))


# ---------------------------------------------------------------------------
# Fused Group Norm
# ---------------------------------------------------------------------------
if _TRITON_AVAILABLE:
    @triton.jit
    def _group_norm_kernel(X, W, B, Y, stride_n, stride_g, group_size, eps,
                           BLOCK: tl.constexpr):
        pid_n = tl.program_id(0)
        pid_g = tl.program_id(1)
        base = pid_n * stride_n + pid_g * stride_g
        offs = tl.arange(0, BLOCK)
        mask = offs < group_size
        x = tl.load(X + base + offs, mask=mask, other=0.0)
        mean = tl.sum(x, axis=0) / group_size
        var = tl.sum((x - mean) * (x - mean), axis=0) / group_size
        rstd = 1.0 / tl.sqrt(var + eps)
        x_norm = (x - mean) * rstd
        # channel offset: pid_g * group_size
        c_offs = pid_g * group_size + offs
        w = tl.load(W + c_offs, mask=mask, other=1.0)
        b_ = tl.load(B + c_offs, mask=mask, other=0.0)
        y = x_norm * w + b_
        tl.store(Y + base + offs, y, mask=mask)


def fused_group_norm(x_nd, weight_nd, bias_nd, num_groups: int, eps: float = 1e-5):
    """Fused Group Normalization. x_nd shape: (N, C, *rest)."""
    if hasattr(x_nd, 'to_flat_list'):
        xfl = x_nd.to_flat_list()
        xsh = x_nd.shape
    else:
        xfl = list(x_nd); xsh = (len(xfl),)
    if hasattr(weight_nd, 'to_flat_list'): wfl = weight_nd.to_flat_list()
    else: wfl = list(weight_nd)
    if hasattr(bias_nd, 'to_flat_list'): bfl = bias_nd.to_flat_list()
    else: bfl = list(bias_nd)
    N, C = xsh[0], xsh[1]
    rest = xsh[2:]; rest_size = 1
    for d in rest: rest_size *= d
    group_size = (C // num_groups) * rest_size
    total = N * C * rest_size

    if not _TRITON_AVAILABLE or not _TORCH_AVAILABLE:
        # Pure-VP CPU fallback
        import math as _m
        out = list(xfl)
        for n in range(N):
            for g in range(num_groups):
                start = n * C * rest_size + g * group_size
                seg = xfl[start:start + group_size]
                mean = sum(seg) / group_size
                var = sum((v - mean) ** 2 for v in seg) / group_size
                rstd = 1.0 / _m.sqrt(var + eps)
                for i, v in enumerate(seg):
                    c_i = g * (C // num_groups) + i // rest_size
                    out[start + i] = (v - mean) * rstd * wfl[c_i] + bfl[c_i]
        return _nd_array(out, 'float32').reshape(xsh)

    tx = torch.tensor(xfl, dtype=torch.float32).cuda().reshape(N, num_groups, group_size)
    tw = torch.tensor(wfl, dtype=torch.float32).cuda()
    tb = torch.tensor(bfl, dtype=torch.float32).cuda()
    ty = torch.empty(N, num_groups, group_size, dtype=torch.float32, device='cuda')
    BLOCK = max(16, 1 << (_math.ceil(_math.log2(group_size)) if group_size > 1 else 1))
    stride_n = num_groups * group_size
    stride_g = group_size
    _group_norm_kernel[(N, num_groups)](tx.reshape(-1), tw, tb, ty.reshape(-1),
                                        stride_n, stride_g, group_size, eps, BLOCK=BLOCK)
    return _nd_array(ty.cpu().reshape(-1).tolist(), 'float32').reshape(xsh)


# ---------------------------------------------------------------------------
# Fused Instance Norm (2D: N, C, H, W)
# ---------------------------------------------------------------------------
if _TRITON_AVAILABLE:
    @triton.jit
    def _instance_norm_kernel(X, W, B, Y, stride_n, stride_c, hw, eps,
                               BLOCK: tl.constexpr):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        base = pid_n * stride_n + pid_c * stride_c
        offs = tl.arange(0, BLOCK)
        mask = offs < hw
        x = tl.load(X + base + offs, mask=mask, other=0.0)
        mean = tl.sum(x, axis=0) / hw
        var = tl.sum((x - mean) * (x - mean), axis=0) / hw
        rstd = 1.0 / tl.sqrt(var + eps)
        x_norm = (x - mean) * rstd
        w = tl.load(W + pid_c)
        b_ = tl.load(B + pid_c)
        tl.store(Y + base + offs, x_norm * w + b_, mask=mask)


def fused_instance_norm_2d(x_nd, weight_nd, bias_nd, eps: float = 1e-5):
    """Fused Instance Normalization for (N, C, H, W) tensors."""
    if hasattr(x_nd, 'to_flat_list'): xfl = x_nd.to_flat_list(); xsh = x_nd.shape
    else: xfl = list(x_nd); xsh = (len(xfl),)
    if hasattr(weight_nd, 'to_flat_list'): wfl = weight_nd.to_flat_list()
    else: wfl = list(weight_nd)
    if hasattr(bias_nd, 'to_flat_list'): bfl = bias_nd.to_flat_list()
    else: bfl = list(bias_nd)
    N, C, H, W = xsh; hw = H * W

    if not _TRITON_AVAILABLE or not _TORCH_AVAILABLE:
        import math as _m
        out = list(xfl)
        for n in range(N):
            for c in range(C):
                start = n * C * hw + c * hw
                seg = xfl[start:start + hw]
                mean = sum(seg) / hw
                var = sum((v - mean) ** 2 for v in seg) / hw
                rstd = 1.0 / _m.sqrt(var + eps)
                for i, v in enumerate(seg):
                    out[start + i] = (v - mean) * rstd * wfl[c] + bfl[c]
        return _nd_array(out, 'float32').reshape(xsh)

    tx = torch.tensor(xfl, dtype=torch.float32).cuda()
    tw = torch.tensor(wfl, dtype=torch.float32).cuda()
    tb = torch.tensor(bfl, dtype=torch.float32).cuda()
    ty = torch.empty_like(tx)
    BLOCK = max(16, 1 << (_math.ceil(_math.log2(hw)) if hw > 1 else 1))
    stride_n = C * hw; stride_c = hw
    _instance_norm_kernel[(N, C)](tx, tw, tb, ty, stride_n, stride_c, hw, eps, BLOCK=BLOCK)
    return _nd_array(ty.cpu().reshape(-1).tolist(), 'float32').reshape(xsh)


# ---------------------------------------------------------------------------
# Fused Dropout (training only)
# ---------------------------------------------------------------------------
if _TRITON_AVAILABLE:
    @triton.jit
    def _dropout_kernel(X, Y, MASK, p_keep, seed, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X + offs, mask=mask, other=0.0)
        # Pseudo-random keep mask using xorshift
        rand_val = tl.rand(seed, offs)
        keep = rand_val < p_keep
        y = tl.where(keep, x / p_keep, 0.0)
        tl.store(Y + offs, y, mask=mask)
        tl.store(MASK + offs, keep.to(tl.float32), mask=mask)


def fused_dropout(x_nd, p: float, training: bool = True, seed: int = 0):
    """Fused dropout; returns (output, mask) where mask is float32 keep-mask."""
    if hasattr(x_nd, 'to_flat_list'): xfl = x_nd.to_flat_list(); xsh = x_nd.shape
    else: xfl = list(x_nd); xsh = (len(xfl),)
    N = len(xfl)
    p_keep = 1.0 - p

    if not training or p == 0.0:
        import array as _a
        mask = _a.array('f', [1.0] * N)
        return x_nd, _nd_array(list(mask), 'float32').reshape(xsh)

    if not _TRITON_AVAILABLE or not _TORCH_AVAILABLE:
        import random as _r, array as _a
        _r.seed(seed)
        out = [v / p_keep if _r.random() < p_keep else 0.0 for v in xfl]
        mk = [1.0 if v != 0.0 else 0.0 for v in out]
        return _nd_array(out, 'float32').reshape(xsh), _nd_array(mk, 'float32').reshape(xsh)

    tx = torch.tensor(xfl, dtype=torch.float32).cuda()
    ty = torch.empty_like(tx)
    tm = torch.empty_like(tx)
    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    _dropout_kernel[grid](tx, ty, tm, p_keep, seed, N, BLOCK=BLOCK)
    out_nd = _nd_array(ty.cpu().tolist(), 'float32').reshape(xsh)
    mask_nd = _nd_array(tm.cpu().tolist(), 'float32').reshape(xsh)
    return out_nd, mask_nd


# ---------------------------------------------------------------------------
# Fused Embedding Backward (sparse scatter-add gradient)
# ---------------------------------------------------------------------------
if _TRITON_AVAILABLE:
    @triton.jit
    def _embedding_bwd_kernel(GRAD_OUT, INDICES, GRAD_W, N, E, VOCAB,
                               BLOCK_E: tl.constexpr):
        pid = tl.program_id(0)
        idx = tl.load(INDICES + pid).to(tl.int32)
        e_offs = tl.arange(0, BLOCK_E)
        mask = e_offs < E
        grad = tl.load(GRAD_OUT + pid * E + e_offs, mask=mask, other=0.0)
        tl.atomic_add(GRAD_W + idx * E + e_offs, grad, mask=mask)


def fused_swiglu(x_nd):
    """Fused SwiGLU activation: splits last dim in half → SiLU(gate) * value.
    Used in LLaMA/Mistral FFN. Fast path: Triton kernel; fallback: pure VP."""
    import math as _m
    if hasattr(x_nd, 'to_flat_list'):
        flat = x_nd.to_flat_list(); sh = x_nd.shape
    else:
        flat = list(x_nd); sh = (len(flat),)
    C = sh[-1]
    half = C // 2
    out_sh = sh[:-1] + (half,)
    n = len(flat) // C

    if _TRITON_AVAILABLE and _TORCH_AVAILABLE:
        try:
            import torch as _torch
            t = _torch.tensor(flat, dtype=_torch.float32)
            if _torch.cuda.is_available():
                t = t.cuda()
            t = t.reshape(-1, C)
            gate = t[:, :half]; val = t[:, half:]
            out = _torch.nn.functional.silu(gate) * val
            return _nd_array(out.cpu().reshape(-1).tolist(), 'float32').reshape(out_sh)
        except Exception:
            pass

    # Pure Python fallback
    out = []
    for i in range(n):
        row = flat[i*C:(i+1)*C]
        gate = row[:half]
        val = row[half:]
        for g, v in zip(gate, val):
            sig = 1.0 / (1.0 + _m.exp(-g))
            out.append(g * sig * v)
    return _nd_array(out, 'float32').reshape(out_sh)


def fused_geglu(x_nd):
    """Fused GeGLU activation: splits last dim in half → GELU(gate) * value.
    Used in PaLM/Gemma FFN."""
    import math as _m
    if hasattr(x_nd, 'to_flat_list'):
        flat = x_nd.to_flat_list(); sh = x_nd.shape
    else:
        flat = list(x_nd); sh = (len(flat),)
    C = sh[-1]
    half = C // 2
    out_sh = sh[:-1] + (half,)
    n = len(flat) // C

    if _TRITON_AVAILABLE and _TORCH_AVAILABLE:
        try:
            import torch as _torch
            t = _torch.tensor(flat, dtype=_torch.float32)
            if _torch.cuda.is_available():
                t = t.cuda()
            t = t.reshape(-1, C)
            gate = t[:, :half]; val = t[:, half:]
            out = _torch.nn.functional.gelu(gate) * val
            return _nd_array(out.cpu().reshape(-1).tolist(), 'float32').reshape(out_sh)
        except Exception:
            pass

    # Pure Python GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    _sqrt2pi = _m.sqrt(2.0 / _m.pi)
    out = []
    for i in range(n):
        row = flat[i*C:(i+1)*C]
        gate = row[:half]
        val = row[half:]
        for g, v in zip(gate, val):
            inner = _sqrt2pi * (g + 0.044715 * g**3)
            gelu_g = 0.5 * g * (1.0 + _m.tanh(inner))
            out.append(gelu_g * v)
    return _nd_array(out, 'float32').reshape(out_sh)


def fused_embedding_backward(grad_out_nd, indices_flat: list, vocab_size: int):
    """Compute embedding weight gradient via scatter-add."""
    if hasattr(grad_out_nd, 'to_flat_list'): gfl = grad_out_nd.to_flat_list(); gsh = grad_out_nd.shape
    else: gfl = list(grad_out_nd); gsh = (len(gfl),)
    N = len(indices_flat)
    E = gsh[-1] if len(gsh) > 1 else len(gfl) // max(N, 1)

    if not _TRITON_AVAILABLE or not _TORCH_AVAILABLE:
        # Pure-VP CPU fallback
        grad_w = [0.0] * (vocab_size * E)
        for i, idx in enumerate(indices_flat):
            row_start = int(idx) * E
            for j in range(E):
                grad_w[row_start + j] += gfl[i * E + j]
        return _nd_array(grad_w, 'float32').reshape((vocab_size, E))

    tg = torch.tensor(gfl, dtype=torch.float32).cuda().reshape(N, E)
    ti = torch.tensor(indices_flat, dtype=torch.int32).cuda()
    tw = torch.zeros(vocab_size, E, dtype=torch.float32, device='cuda')
    BLOCK_E = max(16, 1 << (_math.ceil(_math.log2(E)) if E > 1 else 1))
    _embedding_bwd_kernel[(N,)](tg.reshape(-1), ti, tw.reshape(-1), N, E, vocab_size, BLOCK_E=BLOCK_E)
    return _nd_array(tw.cpu().reshape(-1).tolist(), 'float32').reshape((vocab_size, E))


# ---------------------------------------------------------------------------
# Flash Attention v2 (Triton implementation)
# Fused QKV attention with online softmax — O(N) memory instead of O(N²)
# ---------------------------------------------------------------------------
if _TRITON_AVAILABLE:
    @triton.jit
    def _flash_attn_fwd_kernel(
        Q, K, V, Out, sm_scale,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        stride_oz, stride_oh, stride_om, stride_on,
        Z, H, N_CTX, D_HEAD,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
    ):
        """Flash Attention forward: one Triton program per (batch, head, row-block)."""
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H
        # Q block pointer
        Q_block_ptr = Q + off_z * stride_qz + off_h * stride_qh + (start_m * BLOCK_M) * stride_qm
        K_block_ptr = K + off_z * stride_kz + off_h * stride_kh
        V_block_ptr = V + off_z * stride_vz + off_h * stride_vh
        O_block_ptr = Out + off_z * stride_oz + off_h * stride_oh + (start_m * BLOCK_M) * stride_om

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        q = tl.load(Q_block_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
                    mask=offs_m[:, None] < N_CTX, other=0.0)
        # Online softmax accumulators
        m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        for start_n in range(0, N_CTX, BLOCK_N):
            k = tl.load(K_block_ptr + (start_n + offs_n[None, :]) * stride_kn + offs_d[:, None] * stride_kk,
                        mask=(start_n + offs_n[None, :]) < N_CTX, other=0.0)
            qk = tl.dot(q, k) * sm_scale
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            # Rescale acc
            acc = acc * tl.exp(m_i - m_ij)[:, None]
            m_i = m_ij
            l_i = l_i * tl.exp(m_i - m_ij) + l_ij
            v = tl.load(V_block_ptr + (start_n + offs_n[:, None]) * stride_vk + offs_d[None, :] * stride_vn,
                        mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
            acc += tl.dot(p.to(tl.float16), v.to(tl.float16)).to(tl.float32)

        acc = acc / l_i[:, None]
        tl.store(O_block_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on,
                 acc, mask=offs_m[:, None] < N_CTX)


def flash_attention_v2(q_nd, k_nd, v_nd, scale: float = None):
    """Flash Attention v2: O(N) memory fused attention.

    Args:
        q_nd: Query NDArray of shape (B, H, N, D)
        k_nd: Key NDArray of shape (B, H, N, D)
        v_nd: Value NDArray of shape (B, H, N, D)
        scale: softmax scale (default: 1/sqrt(D))

    Returns:
        Output NDArray of shape (B, H, N, D)
    """
    if hasattr(q_nd, 'to_flat_list'): qfl = q_nd.to_flat_list(); qsh = q_nd.shape
    else: qfl = list(q_nd); qsh = (len(qfl),)
    if hasattr(k_nd, 'to_flat_list'): kfl = k_nd.to_flat_list()
    else: kfl = list(k_nd)
    if hasattr(v_nd, 'to_flat_list'): vfl = v_nd.to_flat_list()
    else: vfl = list(v_nd)
    B, H, N, D = qsh
    if scale is None:
        scale = 1.0 / _math.sqrt(D)

    if not _TRITON_AVAILABLE or not _TORCH_AVAILABLE:
        # Pure-VP CPU fallback: standard QK^T V attention
        import math as _m
        out = [0.0] * (B * H * N * D)
        for b in range(B):
            for h in range(H):
                base = b * H * N * D + h * N * D
                for i in range(N):
                    # Compute attention scores
                    scores = []
                    for j in range(N):
                        s = sum(qfl[base + i * D + d] * kfl[base + j * D + d] for d in range(D)) * scale
                        scores.append(s)
                    mx = max(scores)
                    exps = [_m.exp(s - mx) for s in scores]
                    sumexp = sum(exps)
                    weights = [e / sumexp for e in exps]
                    for d in range(D):
                        out[base + i * D + d] = sum(weights[j] * vfl[base + j * D + d] for j in range(N))
        return _nd_array(out, 'float32').reshape(qsh)

    tq = torch.tensor(qfl, dtype=torch.float16).cuda().reshape(B, H, N, D).contiguous()
    tk = torch.tensor(kfl, dtype=torch.float16).cuda().reshape(B, H, N, D).contiguous()
    tv = torch.tensor(vfl, dtype=torch.float16).cuda().reshape(B, H, N, D).contiguous()
    to = torch.empty_like(tq, dtype=torch.float32)

    BLOCK_M = 64; BLOCK_N = 64
    BLOCK_D = max(16, 1 << (_math.ceil(_math.log2(D)) if D > 1 else 1))
    grid = ((N + BLOCK_M - 1) // BLOCK_M, B * H)
    _flash_attn_fwd_kernel[grid](
        tq, tk, tv, to, scale,
        tq.stride(0), tq.stride(1), tq.stride(2), tq.stride(3),
        tk.stride(0), tk.stride(1), tk.stride(2), tk.stride(3),
        tv.stride(0), tv.stride(1), tv.stride(2), tv.stride(3),
        to.stride(0), to.stride(1), to.stride(2), to.stride(3),
        B, H, N, D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_D,
    )
    return _nd_array(to.cpu().reshape(-1).tolist(), 'float32').reshape(qsh)
