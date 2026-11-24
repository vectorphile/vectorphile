"""State Space Models — S4 (HiPPO/LSSL) and Mamba (selective SSM)."""
from __future__ import annotations
import math as _math
import random as _random
from typing import Optional, Tuple, List
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor
from vectorphile._core.ndarray import array as _nd_arr, NDArray


def _nd_randn(shape, scale=1.0):
    n = 1
    for s in shape: n *= s
    flat = [_math.sqrt(-2*_math.log(max(_random.random(),1e-10)))*_math.cos(2*_math.pi*_random.random())*scale
            for _ in range(n)]
    return _nd_arr(flat, dtype='float32').reshape(shape)

def _nd_zeros(shape):
    n = 1
    for s in shape: n *= s
    return _nd_arr([0.0]*n, dtype='float32').reshape(shape)

def _nd_ones(shape):
    n = 1
    for s in shape: n *= s
    return _nd_arr([1.0]*n, dtype='float32').reshape(shape)

def _layer_norm_nd(x: NDArray, w: NDArray, b: NDArray, eps=1e-5) -> NDArray:
    sh = x.shape
    fl = x.to_flat_list()
    last = sh[-1]
    n_rows = len(fl) // last
    wf = w.to_flat_list(); bf = b.to_flat_list()
    result = []
    for r in range(n_rows):
        row = fl[r*last:(r+1)*last]
        mu = sum(row) / last
        var = sum((v-mu)**2 for v in row) / last
        inv = 1.0 / _math.sqrt(var + eps)
        result.extend([(v-mu)*inv*wf[i]+bf[i] for i, v in enumerate(row)])
    return _nd_arr(result, dtype='float32').reshape(sh)

def _silu_nd(x: NDArray) -> NDArray:
    fl = x.to_flat_list()
    return _nd_arr([v / (1 + _math.exp(-v)) for v in fl], dtype='float32').reshape(x.shape)

def _softplus_nd(x: NDArray) -> NDArray:
    fl = x.to_flat_list()
    return _nd_arr([_math.log(1 + _math.exp(v)) for v in fl], dtype='float32').reshape(x.shape)

def _nd_mean_last(x: NDArray) -> NDArray:
    """Mean pooling over last axis."""
    sh = list(x.shape)
    last = sh[-1]
    fl = x.to_flat_list()
    n_rows = len(fl) // last
    result = [sum(fl[r*last:(r+1)*last]) / last for r in range(n_rows)]
    return _nd_arr(result, dtype='float32').reshape(sh[:-1] if len(sh) > 1 else (1,))

def _nd_mean_axis1(x: NDArray) -> NDArray:
    """Mean pooling over axis 1 for 3D (B, L, D)."""
    B, L, D = x.shape
    fl = x.to_flat_list()
    result = []
    for b in range(B):
        for d in range(D):
            result.append(sum(fl[b*L*D + l*D + d] for l in range(L)) / L)
    return _nd_arr(result, dtype='float32').reshape((B, D))


# ── HiPPO matrix construction ─────────────────────────────────────────────────

def hippo_legt_matrix(N: int) -> Tuple[NDArray, NDArray]:
    """Construct HiPPO-LegT (Legendre Time) A, B matrices."""
    A_flat = []
    for n in range(N):
        for k in range(N):
            if n > k:
                A_flat.append(-_math.sqrt((2*n+1) * (2*k+1)))
            elif n == k:
                A_flat.append(-(n+1.0))
            else:
                A_flat.append(0.0)
    B_flat = [_math.sqrt(2*n+1) for n in range(N)]
    A = _nd_arr(A_flat, dtype='float32').reshape((N, N))
    B = _nd_arr(B_flat, dtype='float32').reshape((N, 1))
    return A, B


def _mat_solve(L_mat: list, N: int, b_vec: list) -> list:
    """Solve L_mat @ x = b_vec using Gaussian elimination (N×N system)."""
    # Augmented matrix
    aug = [L_mat[i*N:(i+1)*N] + [b_vec[i]] for i in range(N)]
    for col in range(N):
        # Pivot
        pivot = None
        for row in range(col, N):
            if abs(aug[row][col]) > 1e-12:
                pivot = row; break
        if pivot is None:
            continue
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv_val = aug[col][col]
        aug[col] = [v / piv_val for v in aug[col]]
        for row in range(N):
            if row != col and abs(aug[row][col]) > 1e-12:
                factor = aug[row][col]
                aug[row] = [aug[row][c] - factor * aug[col][c] for c in range(N+1)]
    return [aug[i][N] for i in range(N)]


def discretize_zoh(A: NDArray, B: NDArray, dt: float) -> Tuple[NDArray, NDArray]:
    """Bilinear (Tustin) discretization."""
    N = A.shape[0]
    af = A.to_flat_list()
    bf = B.to_flat_list()
    # left = I - dt/2 * A, right = I + dt/2 * A
    left  = [( 1.0 if i==j else 0.0) - dt/2*af[i*N+j] for i in range(N) for j in range(N)]
    right = [( 1.0 if i==j else 0.0) + dt/2*af[i*N+j] for i in range(N) for j in range(N)]
    # A_bar = left^{-1} @ right  — solve column by column
    A_bar_cols = []
    for c in range(N):
        rhs = [right[i*N+c] for i in range(N)]
        A_bar_cols.append(_mat_solve(left, N, rhs))
    A_bar_flat = [A_bar_cols[c][r] for r in range(N) for c in range(N)]
    A_bar = _nd_arr(A_bar_flat, dtype='float32').reshape((N, N))
    # B_bar = left^{-1} @ (dt * B)
    dtB = [dt * bf[i] for i in range(N)]
    B_bar_flat = _mat_solve(left, N, dtB)
    B_bar = _nd_arr(B_bar_flat, dtype='float32').reshape((N, 1))
    return A_bar, B_bar


# ── S4 ─────────────────────────────────────────────────────────────────────────

class S4Layer(Module):
    """Single S4 (Structured State Space Sequence) layer.

    Gu et al. 2021 — "Efficiently Modeling Long Sequences with Structured State Spaces"

    Models a 1-D sequence channel via the LTI recurrence:
        x'(t) = A x(t) + B u(t)
        y(t)  = C x(t) + D u(t)

    For inference uses sequential recurrence; for training uses convolutional view.

    Args:
        d_model: feature dimension (number of independent SSM channels)
        d_state: state size N (HiPPO matrix dimension)
        dt_min, dt_max: log-uniform range for step size initialization
    """

    def __init__(self, d_model: int, d_state: int = 64,
                 dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Build HiPPO matrices for each channel
        A_base, B_base = hippo_legt_matrix(d_state)
        # Each channel gets its own (A, B) initialized from HiPPO
        self._A_base = A_base  # (N, N) shared init
        self._B_base = B_base  # (N, 1) shared init

        # Trainable parameters
        # C: (d_model, d_state) — output projection per channel
        self.C = Tensor(_nd_randn((d_model, d_state), 0.01))
        self.C.requires_grad = True
        self._parameters['C'] = self.C

        # D (skip connection): (d_model,)
        self.D = Tensor(_nd_ones((d_model,)))
        self.D.requires_grad = True
        self._parameters['D'] = self.D

        # log dt per channel: (d_model,)
        log_min = _math.log(dt_min); log_max = _math.log(dt_max)
        log_dt_flat = [log_min + _random.random() * (log_max - log_min) for _ in range(d_model)]
        self.log_dt = Tensor(_nd_arr(log_dt_flat, dtype='float32').reshape((d_model,)))
        self.log_dt.requires_grad = True
        self._parameters['log_dt'] = self.log_dt

        # B_proj: (d_model, d_state) — B per channel (initialized from HiPPO)
        # B_base shape (d_state, 1); tile to (d_model, d_state)
        B_flat = B_base.to_flat_list()  # length d_state
        B_init_flat = B_flat * d_model
        self.B_proj = Tensor(_nd_arr(B_init_flat, dtype='float32').reshape((d_model, d_state)))
        self.B_proj.requires_grad = True
        self._parameters['B_proj'] = self.B_proj

    def _get_kernel(self, L: int) -> NDArray:
        """Compute SSM convolution kernel K of length L (diagonal A approx)."""
        dt_flat = self.log_dt._data.to_flat_list()
        dt_vals = [_math.exp(v) for v in dt_flat]
        # Diagonal of A_base
        A_flat = self._A_base.to_flat_list()
        N = self.d_state
        A_diag = [A_flat[i*N+i] for i in range(N)]
        B_flat = self._B_base.to_flat_list()
        C_flat = self.C._data.to_flat_list()
        kernels = []
        for i in range(self.d_model):
            dti = dt_vals[i]
            A_bar_diag = [_math.exp(dti * A_diag[s]) for s in range(N)]
            B_bar = [dti * B_flat[s] for s in range(N)]
            C_i = C_flat[i*N:(i+1)*N]
            CB = [C_i[s] * B_bar[s] for s in range(N)]
            # K[l] = sum_s CB[s] * A_bar_diag[s]^l
            K_i = [sum(CB[s] * (A_bar_diag[s] ** l) for s in range(N)) for l in range(L)]
            kernels.extend(K_i)
        return _nd_arr(kernels, dtype='float32').reshape((self.d_model, L))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, length, d_model)
        Returns:
            y: (batch, length, d_model)
        """
        B_sz, L, _ = x._data.shape
        u = x._data  # (B, L, d_model)

        # Compute kernel K: (d_model, L)
        K = self._get_kernel(L)
        Kf = K.to_flat_list()
        uf = u.to_flat_list()
        Df = self.D._data.to_flat_list()

        # Causal direct convolution: y[b,t,i] = sum_{l=0}^{t} K[i,l] * u[b,t-l,i]
        y_flat = []
        for b in range(B_sz):
            for t in range(L):
                for i in range(self.d_model):
                    s = sum(Kf[i*L + l] * uf[b*L*self.d_model + (t-l)*self.d_model + i]
                            for l in range(t + 1))
                    # Skip connection
                    y_flat.append(s + uf[b*L*self.d_model + t*self.d_model + i] * Df[i])
        return Tensor(_nd_arr(y_flat, dtype='float32').reshape((B_sz, L, self.d_model)))


class S4(Module):
    """Full S4 model: stack of S4 layers with LayerNorm and residual connections.

    Args:
        d_model: model dimension
        d_state: SSM state size
        num_layers: depth
        dropout: dropout probability
        num_classes: if set, adds a linear classification head
    """

    def __init__(self, d_model: int = 256, d_state: int = 64,
                 num_layers: int = 6, dropout: float = 0.1,
                 num_classes: Optional[int] = None):
        super().__init__()
        self.layers = []
        self.norms = []
        for i in range(num_layers):
            layer = S4Layer(d_model, d_state)
            self.layers.append(layer)
            self._modules[f's4_{i}'] = layer
            # LayerNorm params
            w = Tensor(_nd_ones((d_model,)))
            b = Tensor(_nd_zeros((d_model,)))
            w.requires_grad = True
            b.requires_grad = True
            self._parameters[f'ln_w_{i}'] = w
            self._parameters[f'ln_b_{i}'] = b
            self.norms.append((w, b))
        self.dropout = dropout
        self.head = None
        if num_classes is not None:
            self.head_w = Tensor(_nd_randn((d_model, num_classes), 0.02))
            self.head_b = Tensor(_nd_zeros((num_classes,)))
            self.head_w.requires_grad = True
            self.head_b.requires_grad = True
            self._parameters['head_w'] = self.head_w
            self._parameters['head_b'] = self.head_b

    def forward(self, x: Tensor) -> Tensor:
        d = x._data
        for (layer, (ln_w, ln_b)) in zip(self.layers, self.norms):
            residual = d
            d = _layer_norm_nd(d, ln_w._data, ln_b._data)
            d = layer(Tensor(d))._data + residual
            if self.dropout > 0 and self.training:
                df = d.to_flat_list()
                df = [v * (1.0 if _random.random() < (1-self.dropout) else 0.0) / (1-self.dropout) for v in df]
                d = _nd_arr(df, dtype='float32').reshape(d.shape)
        if hasattr(self, 'head_w'):
            d = _nd_mean_axis1(d)  # (B, d_model)
            d = d @ self.head_w._data + self.head_b._data
        return Tensor(d)


# ── Mamba ─────────────────────────────────────────────────────────────────────

class MambaBlock(Module):
    """Mamba selective SSM block — Gu & Dao 2023.

    Key innovation: B, C, Δ (dt) are all input-dependent (selective),
    allowing the model to decide what to remember at each step.

    Simplified implementation without hardware-aware parallel scan
    (uses sequential recurrence — exact but O(L*N) per channel).

    Args:
        d_model: model dimension
        d_state: SSM state size (N)
        d_conv: local conv kernel size (Mamba default: 4)
        expand: inner dimension multiplier (Mamba default: 2)
        dt_rank: rank of Δ projection (Mamba default: ceil(d_model/16))
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2,
                 dt_rank: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv  = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank or max(1, int(_math.ceil(d_model / 16)))

        d_in = self.d_inner

        scale = 0.02
        self.in_proj_w = Tensor(_nd_randn((d_model, d_in * 2), scale))
        self.in_proj_w.requires_grad = True
        self._parameters['in_proj_w'] = self.in_proj_w

        self.conv_w = Tensor(_nd_randn((d_in, 1, d_conv), scale))
        self.conv_w.requires_grad = True
        self.conv_b = Tensor(_nd_zeros((d_in,)))
        self.conv_b.requires_grad = True
        self._parameters['conv_w'] = self.conv_w
        self._parameters['conv_b'] = self.conv_b

        self.x_proj_w = Tensor(_nd_randn((d_in, self.dt_rank + 2 * d_state), scale))
        self.x_proj_w.requires_grad = True
        self._parameters['x_proj_w'] = self.x_proj_w

        self.dt_proj_w = Tensor(_nd_randn((self.dt_rank, d_in), scale))
        self.dt_proj_b = Tensor(_nd_zeros((d_in,)))
        self.dt_proj_w.requires_grad = True
        self.dt_proj_b.requires_grad = True
        self._parameters['dt_proj_w'] = self.dt_proj_w
        self._parameters['dt_proj_b'] = self.dt_proj_b

        # A_log: log of indices 1..d_state tiled for d_inner channels
        A_log_flat = [_math.log(float(s+1)) for _ in range(d_in) for s in range(d_state)]
        self.A_log = Tensor(_nd_arr(A_log_flat, dtype='float32').reshape((d_in, d_state)))
        self.A_log.requires_grad = True
        self._parameters['A_log'] = self.A_log

        self.D_ssm = Tensor(_nd_ones((d_in,)))
        self.D_ssm.requires_grad = True
        self._parameters['D_ssm'] = self.D_ssm

        self.out_proj_w = Tensor(_nd_randn((d_in, d_model), scale))
        self.out_proj_w.requires_grad = True
        self._parameters['out_proj_w'] = self.out_proj_w

        self.ln_w = Tensor(_nd_ones((d_model,)))
        self.ln_b = Tensor(_nd_zeros((d_model,)))
        self.ln_w.requires_grad = True
        self.ln_b.requires_grad = True
        self._parameters['ln_w'] = self.ln_w
        self._parameters['ln_b'] = self.ln_b

    def _depthwise_conv1d(self, x: NDArray) -> NDArray:
        """x: (B, L, d_inner) → (B, L, d_inner) via causal depthwise conv."""
        B, L, C = x.shape
        k = self.d_conv
        xf = x.to_flat_list()
        wf = self.conv_w._data.to_flat_list()  # (C, 1, k) → flat: index as [c*k + ki]
        bf = self.conv_b._data.to_flat_list()
        out_flat = []
        for b in range(B):
            for t in range(L):
                for c in range(C):
                    s = 0.0
                    for ki in range(k):
                        src_t = t - (k - 1 - ki)
                        if 0 <= src_t < L:
                            s += xf[b*L*C + src_t*C + c] * wf[c*k + ki]
                    out_flat.append(s + bf[c])
        return _nd_arr(out_flat, dtype='float32').reshape((B, L, C))

    def _ssm_scan(self, x: NDArray, dt: NDArray, A: NDArray, B: NDArray, C: NDArray) -> NDArray:
        """Sequential SSM recurrence."""
        B_sz, L, d_in = x.shape
        d_state = self.d_state
        xf = x.to_flat_list()
        dtf = dt.to_flat_list()
        Af = A.to_flat_list()   # (d_in, d_state)
        Bf = B.to_flat_list()   # (B, L, d_state)
        Cf = C.to_flat_list()   # (B, L, d_state)
        # h[b, i, s] — initial zeros
        h = [0.0] * (B_sz * d_in * d_state)
        ys_flat = []
        for t in range(L):
            for b in range(B_sz):
                for i in range(d_in):
                    dt_ti = dtf[b*L*d_in + t*d_in + i]
                    u_ti  = xf[b*L*d_in + t*d_in + i]
                    for s in range(d_state):
                        A_bar = _math.exp(dt_ti * Af[i*d_state + s])
                        B_bar = dt_ti * Bf[b*L*d_state + t*d_state + s]
                        idx = b*d_in*d_state + i*d_state + s
                        h[idx] = A_bar * h[idx] + B_bar * u_ti
                # y_t[b, i] = sum_s h[b,i,s] * C[b,t,s]
                for i in range(d_in):
                    y_ti = sum(h[b*d_in*d_state + i*d_state + s] * Cf[b*L*d_state + t*d_state + s]
                               for s in range(d_state))
                    ys_flat.append(y_ti)
        # ys_flat order: t-major, b-inner — need to reorder to (B, L, d_in)
        # current order: for t: for b: for i: → index t*B_sz*d_in + b*d_in + i
        result = [0.0] * (B_sz * L * d_in)
        for t in range(L):
            for b in range(B_sz):
                for i in range(d_in):
                    result[b*L*d_in + t*d_in + i] = ys_flat[t*B_sz*d_in + b*d_in + i]
        return _nd_arr(result, dtype='float32').reshape((B_sz, L, d_in))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)
        """
        d = x._data
        B_sz, L, _ = d.shape

        # Pre-norm
        d_norm = _layer_norm_nd(d, self.ln_w._data, self.ln_b._data)

        # Input projection → split into x_ssm and gate z
        xz = d_norm @ self.in_proj_w._data   # (B, L, 2*d_inner)
        # Split along last axis
        xzf = xz.to_flat_list()
        d2 = self.d_inner * 2
        x_ssm_flat = [xzf[b*L*d2 + t*d2 + i] for b in range(B_sz) for t in range(L) for i in range(self.d_inner)]
        z_flat      = [xzf[b*L*d2 + t*d2 + self.d_inner + i] for b in range(B_sz) for t in range(L) for i in range(self.d_inner)]
        x_ssm = _nd_arr(x_ssm_flat, dtype='float32').reshape((B_sz, L, self.d_inner))
        z     = _nd_arr(z_flat,     dtype='float32').reshape((B_sz, L, self.d_inner))

        # Depthwise conv + SiLU
        x_ssm = _silu_nd(self._depthwise_conv1d(x_ssm))

        # Selective projections
        proj = x_ssm @ self.x_proj_w._data    # (B, L, dt_rank + 2*d_state)
        pf = proj.to_flat_list()
        R = self.dt_rank; DS = self.d_state; total = R + 2*DS
        dt_raw_flat = [pf[b*L*total + t*total + i] for b in range(B_sz) for t in range(L) for i in range(R)]
        B_sel_flat  = [pf[b*L*total + t*total + R + i] for b in range(B_sz) for t in range(L) for i in range(DS)]
        C_sel_flat  = [pf[b*L*total + t*total + R + DS + i] for b in range(B_sz) for t in range(L) for i in range(DS)]
        dt_raw = _nd_arr(dt_raw_flat, dtype='float32').reshape((B_sz, L, R))
        B_sel  = _nd_arr(B_sel_flat,  dtype='float32').reshape((B_sz, L, DS))
        C_sel  = _nd_arr(C_sel_flat,  dtype='float32').reshape((B_sz, L, DS))

        # dt: rank → d_inner via learned projection + softplus
        dt = _softplus_nd(dt_raw @ self.dt_proj_w._data + self.dt_proj_b._data)

        # A: (-exp(A_log)) keeps A negative-definite
        Alf = self.A_log._data.to_flat_list()
        A_nd = _nd_arr([-_math.exp(v) for v in Alf], dtype='float32').reshape((self.d_inner, self.d_state))

        # SSM scan
        y = self._ssm_scan(x_ssm, dt, A_nd, B_sel, C_sel)

        # D skip connection + gated output
        Df = self.D_ssm._data.to_flat_list()
        yf = y.to_flat_list()
        xsf = x_ssm.to_flat_list()
        zf  = _silu_nd(z).to_flat_list()
        out_ssm = [(yf[i] + xsf[i] * Df[i % self.d_inner]) * zf[i] for i in range(len(yf))]
        y_gated = _nd_arr(out_ssm, dtype='float32').reshape((B_sz, L, self.d_inner))

        # Output projection + residual
        out = y_gated @ self.out_proj_w._data + d
        return Tensor(out)


class Mamba(Module):
    """Mamba — Gu & Dao 2023, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces".

    Pure-numpy implementation of the selective SSM architecture.
    Stacks multiple MambaBlocks with optional classification/LM head.

    Args:
        d_model: model dimension
        d_state: SSM state size (default 16)
        d_conv: depthwise conv size (default 4)
        expand: inner dim multiplier (default 2)
        num_layers: number of blocks
        vocab_size: if set, adds embedding + LM head (language model mode)
        num_classes: if set, adds classification head (classifier mode)
    """

    def __init__(self, d_model: int = 256, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, num_layers: int = 6,
                 vocab_size: Optional[int] = None,
                 num_classes: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        if vocab_size is not None:
            self.embedding = Tensor(_nd_randn((vocab_size, d_model), 0.02))
            self.embedding.requires_grad = True
            self._parameters['embedding'] = self.embedding

        self.blocks = []
        for i in range(num_layers):
            block = MambaBlock(d_model, d_state, d_conv, expand)
            self.blocks.append(block)
            self._modules[f'mamba_{i}'] = block

        # Final norm
        self.final_ln_w = Tensor(_nd_ones((d_model,)))
        self.final_ln_b = Tensor(_nd_zeros((d_model,)))
        self.final_ln_w.requires_grad = True
        self.final_ln_b.requires_grad = True
        self._parameters['final_ln_w'] = self.final_ln_w
        self._parameters['final_ln_b'] = self.final_ln_b

        if vocab_size is not None:
            # Tie weights with embedding
            self._lm_head_tied = True
        elif num_classes is not None:
            self.cls_w = Tensor(_nd_randn((d_model, num_classes), 0.02))
            self.cls_b = Tensor(_nd_zeros((num_classes,)))
            self.cls_w.requires_grad = True
            self.cls_b.requires_grad = True
            self._parameters['cls_w'] = self.cls_w
            self._parameters['cls_b'] = self.cls_b

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, L) int token ids if vocab_size set, else (B, L, d_model) floats
        Returns:
            (B, L, vocab_size) logits or (B, num_classes) or (B, L, d_model)
        """
        if self.vocab_size is not None:
            # Token embedding lookup: x._data contains int token ids
            ids = [int(v) for v in x._data.to_flat_list()]
            B = x._data.shape[0]
            L = x._data.shape[1]
            emb_f = self.embedding._data.to_flat_list()
            dm = self.d_model
            d_flat = [emb_f[ids[b*L+t]*dm + di] for b in range(B) for t in range(L) for di in range(dm)]
            d = _nd_arr(d_flat, dtype='float32').reshape((B, L, dm))
        else:
            d = x._data

        for block in self.blocks:
            d = block(Tensor(d))._data

        # Final layer norm
        d = _layer_norm_nd(d, self.final_ln_w._data, self.final_ln_b._data)

        if self.vocab_size is not None:
            # Tied LM head: d @ embedding^T
            B, L, dm = d.shape
            VS = self.vocab_size
            df = d.to_flat_list()
            ef = self.embedding._data.to_flat_list()
            logit_flat = []
            for b in range(B):
                for t in range(L):
                    for v in range(VS):
                        logit_flat.append(sum(df[b*L*dm + t*dm + di] * ef[v*dm + di] for di in range(dm)))
            d = _nd_arr(logit_flat, dtype='float32').reshape((B, L, VS))
        elif hasattr(self, 'cls_w'):
            d = _nd_mean_axis1(d) @ self.cls_w._data + self.cls_b._data

        return Tensor(d)
