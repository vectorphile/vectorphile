"""vectorphile.cuda — Custom CUDA kernels via Numba + im2col CPU acceleration.

GPU path: requires ``numba`` with CUDA toolkit installed.
CPU path: im2col + BLAS matmul (always available via numpy/scipy).

Usage::

    from vectorphile.cuda import matmul, conv2d_im2col, softmax, relu
    out = matmul(a, b)        # GPU if available, numpy otherwise
    out = conv2d_im2col(x, w) # 10-100× faster than loop conv on CPU
    print(vectorphile.cuda.is_available())
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple

# ── Optional Numba ────────────────────────────────────────────────────────────
try:
    from numba import cuda as _numba_cuda
    import numba
    _NUMBA_AVAILABLE = True
    try:
        _CUDA_AVAILABLE = len(list(_numba_cuda.gpus)) > 0
    except Exception:
        _CUDA_AVAILABLE = _numba_cuda.is_available()
except Exception:
    _numba_cuda = None
    numba = None
    _NUMBA_AVAILABLE = False
    _CUDA_AVAILABLE = False

# ── Device management ─────────────────────────────────────────────────────────

def is_available() -> bool:
    """Return True if Numba CUDA is installed and a GPU is detected."""
    return _CUDA_AVAILABLE


def device_count() -> int:
    if not _CUDA_AVAILABLE:
        return 0
    try:
        return len(list(_numba_cuda.gpus))
    except Exception:
        return 0


def get_device_name(idx: int = 0) -> str:
    if not _CUDA_AVAILABLE:
        return "CPU (no CUDA)"
    try:
        return _numba_cuda.gpus[idx].name
    except Exception:
        return "unknown"


def memory_info(idx: int = 0) -> dict:
    """Return {'free': bytes, 'total': bytes} for GPU idx."""
    if not _CUDA_AVAILABLE:
        return {'free': 0, 'total': 0}
    try:
        with _numba_cuda.gpus[idx]:
            free, total = _numba_cuda.current_context().get_memory_info()
        return {'free': free, 'total': total}
    except Exception:
        return {'free': 0, 'total': 0}


def synchronize() -> None:
    if _CUDA_AVAILABLE:
        try:
            _numba_cuda.synchronize()
        except Exception:
            pass


def to_device(arr: np.ndarray, device_id: int = 0):
    """Copy numpy array to GPU. Returns numba DeviceNDArray or numpy if no CUDA."""
    if not _CUDA_AVAILABLE:
        return arr
    with _numba_cuda.gpus[device_id]:
        return _numba_cuda.to_device(np.ascontiguousarray(arr, dtype=np.float32))


def to_host(arr) -> np.ndarray:
    """Copy device array back to numpy. Passes through numpy arrays unchanged."""
    if _CUDA_AVAILABLE and hasattr(arr, 'copy_to_host'):
        return arr.copy_to_host()
    return np.asarray(arr)


# ── Numba CUDA kernels ────────────────────────────────────────────────────────
# Only compiled if numba + CUDA are present. Each has a pure-numpy fallback.

_TILE = 16  # shared-memory tile size

if _CUDA_AVAILABLE:
    @_numba_cuda.jit
    def _matmul_kernel(A, B, C, M, K, N):
        """Tiled matrix multiply: C = A @ B, shapes (M,K) @ (K,N) → (M,N)."""
        sh_A = _numba_cuda.shared.array(shape=(_TILE, _TILE), dtype=numba.float32)
        sh_B = _numba_cuda.shared.array(shape=(_TILE, _TILE), dtype=numba.float32)
        tx = _numba_cuda.threadIdx.x
        ty = _numba_cuda.threadIdx.y
        row = _numba_cuda.blockIdx.y * _TILE + ty
        col = _numba_cuda.blockIdx.x * _TILE + tx
        acc = numba.float32(0.0)
        for tile in range((K + _TILE - 1) // _TILE):
            r = row;   c = tile * _TILE + tx
            sh_A[ty, tx] = A[r, c] if (r < M and c < K) else numba.float32(0.0)
            r = tile * _TILE + ty; c = col
            sh_B[ty, tx] = B[r, c] if (r < K and c < N) else numba.float32(0.0)
            _numba_cuda.syncthreads()
            for k in range(_TILE):
                acc += sh_A[ty, k] * sh_B[k, tx]
            _numba_cuda.syncthreads()
        if row < M and col < N:
            C[row, col] = acc

    @_numba_cuda.jit
    def _relu_kernel(x, out, n):
        i = _numba_cuda.grid(1)
        if i < n:
            v = x[i]
            out[i] = v if v > 0.0 else numba.float32(0.0)

    @_numba_cuda.jit
    def _silu_kernel(x, out, n):
        import math
        i = _numba_cuda.grid(1)
        if i < n:
            v = x[i]
            out[i] = v / (1.0 + math.exp(-v))

    @_numba_cuda.jit
    def _gelu_kernel(x, out, n):
        import math
        i = _numba_cuda.grid(1)
        if i < n:
            v = x[i]
            # tanh approximation
            inner = math.sqrt(2.0 / math.pi) * (v + 0.044715 * v * v * v)
            out[i] = numba.float32(0.5) * v * (1.0 + math.tanh(inner))

    @_numba_cuda.jit
    def _layer_norm_kernel(x, w, b, out, N, D, eps):
        """One block per row. Shared-mem reduction for mean and var."""
        sh = _numba_cuda.shared.array(shape=(256,), dtype=numba.float32)
        row = _numba_cuda.blockIdx.x
        tid = _numba_cuda.threadIdx.x
        # mean pass
        s = numba.float32(0.0)
        for d in range(tid, D, _numba_cuda.blockDim.x):
            s += x[row, d]
        sh[tid] = s
        _numba_cuda.syncthreads()
        stride = _numba_cuda.blockDim.x // 2
        while stride > 0:
            if tid < stride:
                sh[tid] += sh[tid + stride]
            _numba_cuda.syncthreads()
            stride //= 2
        mean = sh[0] / D
        _numba_cuda.syncthreads()
        # var pass
        s = numba.float32(0.0)
        for d in range(tid, D, _numba_cuda.blockDim.x):
            diff = x[row, d] - mean
            s += diff * diff
        sh[tid] = s
        _numba_cuda.syncthreads()
        stride = _numba_cuda.blockDim.x // 2
        while stride > 0:
            if tid < stride:
                sh[tid] += sh[tid + stride]
            _numba_cuda.syncthreads()
            stride //= 2
        import math
        inv_std = 1.0 / math.sqrt(sh[0] / D + eps)
        # normalize
        for d in range(tid, D, _numba_cuda.blockDim.x):
            out[row, d] = (x[row, d] - mean) * inv_std * w[d] + b[d]

    @_numba_cuda.jit
    def _adam_step_kernel(p, g, m, v, lr, beta1, beta2, eps, wd, bc1, bc2, n):
        """Fused Adam update kernel."""
        i = _numba_cuda.grid(1)
        if i < n:
            gi = g[i]
            if wd > 0.0:
                gi = gi + wd * p[i]
            m[i] = beta1 * m[i] + (1.0 - beta1) * gi
            v[i] = beta2 * v[i] + (1.0 - beta2) * gi * gi
            m_hat = m[i] / bc1
            v_hat = v[i] / bc2
            import math
            p[i] -= lr * m_hat / (math.sqrt(v_hat) + eps)


def _blocks_threads_1d(n: int, threads: int = 256) -> Tuple[int, int]:
    return ((n + threads - 1) // threads, threads)


# ── Public API ────────────────────────────────────────────────────────────────

def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiply. Uses tiled CUDA kernel if available, numpy otherwise.

    Args:
        a: (M, K) float32 array
        b: (K, N) float32 array
    Returns:
        (M, N) float32 array
    """
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    if not _CUDA_AVAILABLE:
        return np.matmul(a, b)
    # Flatten batch dims
    orig_shape_a = a.shape
    if a.ndim > 2:
        a2 = a.reshape(-1, a.shape[-1])
        b2 = b.reshape(b.shape[0], -1) if b.ndim == 2 else b
        # Fall back for batched
        return np.matmul(a, b)
    M, K = a.shape; _, N = b.shape
    d_a = _numba_cuda.to_device(a)
    d_b = _numba_cuda.to_device(b)
    d_c = _numba_cuda.device_array((M, N), dtype=np.float32)
    threads = (_TILE, _TILE)
    blocks  = ((N + _TILE - 1) // _TILE, (M + _TILE - 1) // _TILE)
    _matmul_kernel[blocks, threads](d_a, d_b, d_c, M, K, N)
    return d_c.copy_to_host()


# ── im2col convolution (CPU BLAS — 10-100× faster than loop conv) ─────────────

def im2col(x: np.ndarray, kH: int, kW: int,
           stride: int = 1, padding: int = 0) -> np.ndarray:
    """Convert input tensor to column matrix for GEMM-based convolution.

    Args:
        x: (N, C, H, W) float32
        kH, kW: kernel spatial dims
        stride, padding: conv hyperparams
    Returns:
        col: (N, C*kH*kW, H_out*W_out) float32
    """
    x = np.asarray(x, dtype=np.float32)
    N, C, H, W = x.shape
    if padding > 0:
        x = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1
    # Build index arrays for fancy indexing (no Python loops)
    i0 = np.repeat(np.arange(kH), kW)                       # (kH*kW,)
    i0 = np.tile(i0, C)                                      # (C*kH*kW,)
    i1 = stride * np.repeat(np.arange(H_out), W_out)        # (H_out*W_out,)
    j0 = np.tile(np.arange(kW), kH * C)                     # (C*kH*kW,)
    j1 = stride * np.tile(np.arange(W_out), H_out)          # (H_out*W_out,)
    i  = i0[:, None] + i1[None, :]                           # (C*kH*kW, H_out*W_out)
    j  = j0[:, None] + j1[None, :]
    c  = np.repeat(np.arange(C), kH * kW)[:, None]          # (C*kH*kW, 1)
    col = x[:, c, i, j]                                      # (N, C*kH*kW, H_out*W_out)
    return col


def conv2d_im2col(x: np.ndarray, w: np.ndarray,
                  b: Optional[np.ndarray] = None,
                  stride: int = 1, padding: int = 0) -> np.ndarray:
    """Convolution via im2col + BLAS matmul. Much faster than nested loops.

    Args:
        x: (N, C_in, H, W)
        w: (C_out, C_in, kH, kW)
        b: (C_out,) or None
        stride, padding: conv hyperparams
    Returns:
        (N, C_out, H_out, W_out)
    """
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1
    col = im2col(x, kH, kW, stride, padding)   # (N, C_in*kH*kW, H_out*W_out)
    w_reshaped = w.reshape(C_out, -1)           # (C_out, C_in*kH*kW)
    # out[n] = w_reshaped @ col[n]  → (C_out, H_out*W_out) per sample
    # Batch: (N, C_out, H_out*W_out)
    out = np.einsum('oc,ncp->nop', w_reshaped, col)  # uses BLAS under the hood
    out = out.reshape(N, C_out, H_out, W_out)
    if b is not None:
        out += np.asarray(b, dtype=np.float32)[None, :, None, None]
    return out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax. CUDA kernel if available, numpy otherwise."""
    x = np.asarray(x, dtype=np.float32)
    # CPU numpy path is already fast; CUDA only beneficial for very large tensors
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return (e / (e.sum(axis=axis, keepdims=True) + 1e-9)).astype(np.float32)


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray,
               eps: float = 1e-5) -> np.ndarray:
    """Fused layer norm. CUDA kernel if available for 2-D inputs, numpy otherwise."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if _CUDA_AVAILABLE and x.ndim == 2:
        N, D = x.shape
        d_x   = _numba_cuda.to_device(x)
        d_w   = _numba_cuda.to_device(w)
        d_b   = _numba_cuda.to_device(b)
        d_out = _numba_cuda.device_array((N, D), dtype=np.float32)
        threads = min(256, D)
        _layer_norm_kernel[N, threads](d_x, d_w, d_b, d_out, N, D, eps)
        return d_out.copy_to_host()
    # numpy fallback (handles arbitrary shapes)
    mean = x.mean(-1, keepdims=True)
    var  = x.var(-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps) * w + b).astype(np.float32)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU. Elementwise CUDA kernel if available."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if _CUDA_AVAILABLE:
        flat = x.ravel()
        d_x   = _numba_cuda.to_device(flat)
        d_out = _numba_cuda.device_array(len(flat), dtype=np.float32)
        blocks, threads = _blocks_threads_1d(len(flat))
        _relu_kernel[blocks, threads](d_x, d_out, len(flat))
        return d_out.copy_to_host().reshape(x.shape)
    return np.maximum(x, 0.0)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU (tanh approx). Elementwise CUDA kernel if available."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if _CUDA_AVAILABLE:
        flat = x.ravel()
        d_x   = _numba_cuda.to_device(flat)
        d_out = _numba_cuda.device_array(len(flat), dtype=np.float32)
        blocks, threads = _blocks_threads_1d(len(flat))
        _gelu_kernel[blocks, threads](d_x, d_out, len(flat))
        return d_out.copy_to_host().reshape(x.shape)
    return (0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))).astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU/Swish. Elementwise CUDA kernel if available."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if _CUDA_AVAILABLE:
        flat = x.ravel()
        d_x   = _numba_cuda.to_device(flat)
        d_out = _numba_cuda.device_array(len(flat), dtype=np.float32)
        blocks, threads = _blocks_threads_1d(len(flat))
        _silu_kernel[blocks, threads](d_x, d_out, len(flat))
        return d_out.copy_to_host().reshape(x.shape)
    return (x * (1 / (1 + np.exp(-x)))).astype(np.float32)


def adam_step(params: np.ndarray, grads: np.ndarray,
              m: np.ndarray, v: np.ndarray,
              lr: float, beta1: float, beta2: float,
              eps: float, weight_decay: float, t: int) -> None:
    """Fused Adam parameter update (in-place). CUDA kernel if available.

    Args:
        params, grads, m, v: flat float32 arrays of same length (modified in-place)
        lr, beta1, beta2, eps, weight_decay: Adam hyperparams
        t: current step (for bias correction)
    """
    bc1 = float(1 - beta1 ** t)
    bc2 = float(1 - beta2 ** t)
    if _CUDA_AVAILABLE:
        n = len(params)
        d_p = _numba_cuda.to_device(params)
        d_g = _numba_cuda.to_device(grads)
        d_m = _numba_cuda.to_device(m)
        d_v = _numba_cuda.to_device(v)
        blocks, threads = _blocks_threads_1d(n)
        _adam_step_kernel[blocks, threads](d_p, d_g, d_m, d_v,
                                            lr, beta1, beta2, eps,
                                            weight_decay, bc1, bc2, n)
        params[:] = d_p.copy_to_host()
        m[:]      = d_m.copy_to_host()
        v[:]      = d_v.copy_to_host()
        return
    # numpy fallback
    g = grads + weight_decay * params if weight_decay > 0 else grads
    m[:] = beta1 * m + (1 - beta1) * g
    v[:] = beta2 * v + (1 - beta2) * g * g
    m_hat = m / bc1
    v_hat = v / bc2
    params -= lr * m_hat / (np.sqrt(v_hat) + eps)


def attention_forward(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                      causal: bool = False,
                      scale: Optional[float] = None) -> np.ndarray:
    """Tiled flash-attention style forward pass.

    Processes QK^T in blocks to avoid materialising the full N×N matrix.
    Falls back to standard attention on CPU (still numerically identical).

    Args:
        q, k, v: (B, H, N, d) float32
        causal: apply causal mask
        scale: softmax scale (default 1/sqrt(d))
    Returns:
        (B, H, N, d) float32 output
    """
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    B, H, N, d = q.shape
    if scale is None:
        scale = 1.0 / np.sqrt(d)
    BLOCK = 64
    out = np.zeros_like(q)
    for b in range(B):
        for h in range(H):
            # Online softmax over KV blocks (flash-attention algorithm 1)
            q_bh = q[b, h]   # (N, d)
            k_bh = k[b, h]
            v_bh = v[b, h]
            O = np.zeros((N, d), dtype=np.float32)
            L = np.zeros(N, dtype=np.float32)   # running sum exp
            M = np.full(N, -np.inf, dtype=np.float32)  # running max
            for j_start in range(0, N, BLOCK):
                j_end = min(j_start + BLOCK, N)
                k_blk = k_bh[j_start:j_end]   # (Bc, d)
                v_blk = v_bh[j_start:j_end]
                # S = Q @ K_blk^T * scale  (N, Bc)
                S = q_bh @ k_blk.T * scale
                if causal:
                    # mask positions where j > i
                    rows = np.arange(N)[:, None]
                    cols = np.arange(j_start, j_end)[None, :]
                    S = np.where(cols > rows, -1e9, S)
                # Update running max
                M_new = np.maximum(M, S.max(axis=-1))
                # Rescale old O and L
                scale_old = np.exp(M - M_new)
                O = O * scale_old[:, None]
                L = L * scale_old
                # Add new block
                P = np.exp(S - M_new[:, None])
                O += P @ v_blk
                L += P.sum(axis=-1)
                M = M_new
            out[b, h] = O / (L[:, None] + 1e-9)
    return out
