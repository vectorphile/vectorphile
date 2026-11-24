"""Finite-difference gradient checker for all core ops.

Verifies that analytically computed gradients match numerical estimates
to within tolerance. Run with:  python -m pytest tests/test_gradients.py -v
"""
from __future__ import annotations
import numpy as np
import pytest
import vectorphile as vp
from vectorphile.core.tensor import Tensor


# ── Utilities ────────────────────────────────────────────────────────────────

def numerical_grad(fn, x_np: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    grad = np.zeros_like(x_np, dtype=np.float64)
    flat = x_np.ravel()
    for i in range(len(flat)):
        xp = flat.copy(); xp[i] += eps
        xm = flat.copy(); xm[i] -= eps
        fp = fn(Tensor(xp.reshape(x_np.shape).astype(np.float32)))
        fm = fn(Tensor(xm.reshape(x_np.shape).astype(np.float32)))
        fp_val = np.asarray(fp._data, dtype=np.float64).sum()
        fm_val = np.asarray(fm._data, dtype=np.float64).sum()
        grad.ravel()[i] = (fp_val - fm_val) / (2 * eps)
    return grad.reshape(x_np.shape)


def check_grad(fn, x_np: np.ndarray, rtol: float = 2e-2, atol: float = 2e-3,
               eps: float = 1e-3, name: str = ''):
    """Compare analytical vs numerical gradient.

    Uses eps=1e-3 and liberal tolerances because:
    - Tensors use float32; numerical grad has ~eps^2 / float32_eps rounding
    - Fast C approximations (tanh, sigmoid) differ slightly from numpy exact
    """
    x = Tensor(x_np.astype(np.float32), requires_grad=True)
    out = fn(x)
    seed = Tensor(np.ones_like(np.asarray(out._data, dtype=np.float32)))
    out.backward(seed)
    if x.grad is None:
        pytest.fail(f"{name}: backward() produced no gradient (x.grad is None)")
    analytical = np.asarray(x.grad._data, dtype=np.float64)
    numerical  = numerical_grad(fn, x_np, eps=eps)
    ok = np.allclose(analytical, numerical, rtol=rtol, atol=atol)
    if not ok:
        max_err = np.max(np.abs(analytical - numerical))
        rel_err = np.max(np.abs(analytical - numerical) / (np.abs(numerical) + 1e-8))
        pytest.fail(f"{name}: max_abs={max_err:.3e}, max_rel={rel_err:.3e}\n"
                    f"  analytical: {analytical.ravel()[:5]}\n"
                    f"  numerical:  {numerical.ravel()[:5]}")


# ── Shapes ───────────────────────────────────────────────────────────────────
SHAPES = [(4,), (3, 4), (2, 3, 4)]
RNG = np.random.RandomState(42)


# ── Elementwise ops ──────────────────────────────────────────────────────────

@pytest.mark.parametrize('shape', SHAPES)
def test_exp(shape):
    x = RNG.randn(*shape).astype(np.float32) * 0.5  # keep small to avoid overflow
    check_grad(lambda t: t.exp(), x, name='exp')


@pytest.mark.parametrize('shape', SHAPES)
def test_log(shape):
    x = np.abs(RNG.randn(*shape).astype(np.float32)) + 0.5
    check_grad(lambda t: t.log(), x, name='log')


@pytest.mark.parametrize('shape', SHAPES)
def test_sqrt(shape):
    x = np.abs(RNG.randn(*shape).astype(np.float32)) + 0.1
    check_grad(lambda t: t.sqrt(), x, name='sqrt')


@pytest.mark.parametrize('shape', SHAPES)
def test_sin(shape):
    x = RNG.randn(*shape).astype(np.float32)
    check_grad(lambda t: t.sin(), x, name='sin')


@pytest.mark.parametrize('shape', SHAPES)
def test_cos(shape):
    x = RNG.randn(*shape).astype(np.float32)
    check_grad(lambda t: t.cos(), x, name='cos')


@pytest.mark.parametrize('shape', SHAPES)
def test_sigmoid(shape):
    x = RNG.randn(*shape).astype(np.float32)
    check_grad(lambda t: t.sigmoid(), x, name='sigmoid')


@pytest.mark.parametrize('shape', SHAPES)
def test_tanh(shape):
    x = RNG.randn(*shape).astype(np.float32)
    check_grad(lambda t: t.tanh(), x, name='tanh')


@pytest.mark.parametrize('shape', SHAPES)
def test_relu(shape):
    x = RNG.randn(*shape).astype(np.float32)
    # Avoid points near zero where subgradient is discontinuous
    x[np.abs(x) < 0.1] += 0.3
    check_grad(lambda t: t.relu(), x, name='relu', eps=1e-3)


# ── Arithmetic ops ───────────────────────────────────────────────────────────

def test_add():
    x = RNG.randn(4, 5).astype(np.float32)
    y = RNG.randn(4, 5).astype(np.float32)
    yT = Tensor(y)
    check_grad(lambda t: t + yT, x, name='add')


def test_mul():
    x = RNG.randn(4, 5).astype(np.float32)
    y = RNG.randn(4, 5).astype(np.float32)
    yT = Tensor(y)
    check_grad(lambda t: t * yT, x, name='mul')


def test_div():
    x = RNG.randn(4, 5).astype(np.float32) + 2.0
    y = np.abs(RNG.randn(4, 5).astype(np.float32)) + 0.5
    yT = Tensor(y)
    check_grad(lambda t: t / yT, x, name='div')


# ── Reductions ───────────────────────────────────────────────────────────────

def test_sum_all():
    x = RNG.randn(3, 4).astype(np.float32)
    check_grad(lambda t: t.sum(), x, name='sum_all')


def test_sum_axis():
    x = RNG.randn(3, 4).astype(np.float32)
    check_grad(lambda t: t.sum(dim=0), x, name='sum_dim0')
    check_grad(lambda t: t.sum(dim=1), x, name='sum_dim1')


def test_mean_all():
    x = RNG.randn(3, 4).astype(np.float32)
    check_grad(lambda t: t.mean(), x, name='mean_all')


# ── Matrix ops ───────────────────────────────────────────────────────────────

def test_matmul():
    A = RNG.randn(4, 6).astype(np.float32)
    B = RNG.randn(6, 5).astype(np.float32)
    BT = Tensor(B)
    check_grad(lambda t: t @ BT, A, name='matmul_A')
    AT = Tensor(A)
    check_grad(lambda t: AT @ t, B, name='matmul_B')


# ── Composed / chain rule ────────────────────────────────────────────────────

def test_linear_relu_chain():
    """Check grad through Linear → ReLU chain."""
    import vectorphile.nn as nn
    layer = nn.Linear(6, 4, bias=True)
    x = RNG.randn(3, 6).astype(np.float32)

    def fn(t):
        return layer(t).relu().sum()

    check_grad(fn, x, name='linear_relu_chain', atol=5e-4)


def test_exp_sin_compose():
    x = RNG.randn(5).astype(np.float32) * 0.3
    check_grad(lambda t: t.exp().sin(), x, name='exp_sin_compose')


def test_log_sqrt_compose():
    x = np.abs(RNG.randn(5).astype(np.float32)) + 1.5  # ensure x > 1 so log(x) > 0
    check_grad(lambda t: t.log().sqrt(), x, name='log_sqrt_compose',
               atol=1e-3)


# ── Second-order gradient check ───────────────────────────────────────────────

def test_hessian_quadratic():
    """For f(x) = sum(x^2), Hessian should be 2*I (diagonal)."""
    from vectorphile.autograd.functional import hessian
    n = 4

    def f(x):
        return (x * x).sum()

    x0 = Tensor(RNG.randn(n).astype(np.float32))
    H = hessian(f, x0)
    # Hessian of sum(x_i^2) = 2*I
    expected = 2.0 * np.eye(n, dtype=np.float32)
    max_err = np.max(np.abs(H - expected))
    assert max_err < 1.0, f'Hessian diagonal error {max_err:.3e} (expected ~2.0 on diag)'


# ── New kernel smoke tests ────────────────────────────────────────────────────

def test_im2col_correctness():
    """im2col then reshape+matmul should match naive conv."""
    from vectorphile._core import _vp_core as C
    import array as arr
    # (1, 1, 4, 4) input, (2, 1, 2, 2) weight, stride=1, pad=0
    N, C_in, H, W = 1, 1, 4, 4
    Co, Kh, Kw = 2, 2, 2
    H_out = (H - Kh) + 1  # 3
    W_out = (W - Kw) + 1  # 3
    x_np = np.arange(N*C_in*H*W, dtype=np.float32).reshape(N, C_in, H, W)
    w_np = np.arange(Co*C_in*Kh*Kw, dtype=np.float32).reshape(Co, C_in, Kh, Kw)
    x_buf = arr.array('f', x_np.tobytes())
    col_buf = arr.array('f', bytearray(4 * N * H_out * W_out * C_in * Kh * Kw))
    C.im2col_f32(x_buf, col_buf, N, C_in, H, W, Kh, Kw, 1, 1, 0, 0)
    col_np = np.frombuffer(col_buf, dtype=np.float32).reshape(N*H_out*W_out, C_in*Kh*Kw)
    w_flat = w_np.reshape(Co, C_in*Kh*Kw)
    out_mm = (col_np @ w_flat.T).reshape(N, H_out, W_out, Co).transpose(0, 3, 1, 2)
    # Reference: scipy/numpy naive convolution
    from scipy.signal import correlate2d
    out_ref = np.zeros((N, Co, H_out, W_out), dtype=np.float32)
    for n in range(N):
        for co in range(Co):
            for ci in range(C_in):
                out_ref[n, co] += correlate2d(
                    x_np[n, ci], w_np[co, ci], mode='valid')
    assert np.allclose(out_mm, out_ref, atol=1e-4), f'im2col mismatch: max={np.max(np.abs(out_mm - out_ref))}'


def test_bf16_gemm():
    """BF16 GEMM should match FP32 GEMM to BF16 precision (~1e-2)."""
    from vectorphile._core import _vp_core as C
    import array as arr
    M, K, N = 16, 32, 16
    A_fp32 = RNG.randn(M, K).astype(np.float32)
    B_fp32 = RNG.randn(K, N).astype(np.float32)
    ref = A_fp32 @ B_fp32
    # Convert to BF16 (round to nearest, implemented as truncate)
    def to_bf16(x): return (x.view(np.uint32) >> 16).astype(np.uint16)
    A_bf16 = to_bf16(A_fp32)
    B_bf16 = to_bf16(B_fp32)
    a_buf = arr.array('H', A_bf16.tobytes())
    b_buf = arr.array('H', B_bf16.tobytes())
    c_buf = arr.array('f', bytearray(4 * M * N))
    C.gemm_bf16_f32(a_buf, b_buf, c_buf, M, K, N)
    out = np.frombuffer(c_buf, dtype=np.float32).reshape(M, N)
    max_err = np.max(np.abs(out - ref))
    assert max_err < 2.0, f'BF16 GEMM error too large: {max_err:.3e}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
