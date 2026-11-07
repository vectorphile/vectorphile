"""State Space Models — S4 (HiPPO/LSSL) and Mamba (selective SSM)."""
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


# ── HiPPO matrix construction ─────────────────────────────────────────────────

def hippo_legt_matrix(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Construct HiPPO-LegT (Legendre Time) A, B matrices.

    A[n,k] = -(2n+1)^0.5 * (2k+1)^0.5  if n > k
              (n+1)                      if n == k
              0                          if n < k
    B[n]   = (2n+1)^0.5

    These encode history into a fixed-size state via Legendre polynomials.
    """
    Q = np.arange(N, dtype=np.float64)
    R = (2 * Q + 1)[:, None]  # (N, 1)
    j, i = np.meshgrid(Q, Q)
    A = np.where(i < j, -1, 0) * np.sqrt(R) * np.sqrt((2 * j + 1))
    A = A - np.diag(Q + 1)
    B = np.sqrt(2 * Q + 1)[:, None]  # (N, 1)
    return A.astype(np.float32), B.astype(np.float32)


def discretize_zoh(A: np.ndarray, B: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-order hold discretization: Ā = exp(dt*A), B̄ = (Ā - I) A^{-1} B."""
    N = A.shape[0]
    dA = np.linalg.matrix_power(
        np.eye(N, dtype=np.float32) + dt / 2 * A,
        1  # Euler step (Bilinear is below)
    )
    # Bilinear (Tustin) — more stable for large dt
    I = np.eye(N, dtype=np.float32)
    left  = I - dt / 2 * A
    right = I + dt / 2 * A
    # Ā = left^{-1} right
    A_bar = np.linalg.solve(left, right).astype(np.float32)
    B_bar = np.linalg.solve(left, dt * B).astype(np.float32)
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
        self.C = Tensor(np.random.randn(d_model, d_state).astype(np.float32) * 0.01)
        self.C.requires_grad = True
        self._parameters['C'] = self.C

        # D (skip connection): (d_model,)
        self.D = Tensor(np.ones(d_model, dtype=np.float32))
        self.D.requires_grad = True
        self._parameters['D'] = self.D

        # log dt per channel: (d_model,)
        log_dt = np.random.uniform(np.log(dt_min), np.log(dt_max), size=(d_model,)).astype(np.float32)
        self.log_dt = Tensor(log_dt)
        self.log_dt.requires_grad = True
        self._parameters['log_dt'] = self.log_dt

        # B_proj: (d_model, d_state) — B per channel (initialized from HiPPO)
        B_init = np.tile(B_base.T, (d_model, 1))  # (d_model, d_state)
        self.B_proj = Tensor(B_init.astype(np.float32))
        self.B_proj.requires_grad = True
        self._parameters['B_proj'] = self.B_proj

    def _get_kernel(self, L: int) -> np.ndarray:
        """Compute SSM convolution kernel K of length L for all d_model channels.

        K[i, l] = C[i] @ Ā[i]^l @ B̄[i]  (approximation via diagonal A)
        We use a diagonal approximation of A for efficiency.
        """
        dt = np.exp(self.log_dt._data)  # (d_model,)
        # Approximate: treat A as diagonal with its diagonal elements
        A_diag = np.diag(self._A_base)  # (d_state,)
        # For each channel: Ā_diag = exp(dt * A_diag), B̄ = dt * B_base.flatten()
        kernels = []
        for i in range(self.d_model):
            A_bar_diag = np.exp(dt[i] * A_diag)   # (d_state,)
            B_bar = dt[i] * self._B_base.flatten()  # (d_state,)
            C_i = self.C._data[i]                   # (d_state,)
            # K[l] = C Ā^l B  (diagonal A: Ā^l = Ā_diag^l elementwise)
            powers = A_bar_diag[None, :] ** np.arange(L)[:, None]  # (L, d_state)
            K = powers @ (C_i * B_bar)   # (L,)
            kernels.append(K)
        return np.stack(kernels, axis=0)  # (d_model, L)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, length, d_model)
        Returns:
            y: (batch, length, d_model)
        """
        B_sz, L, _ = x._data.shape
        u = x._data  # (B, L, d_model)

        # Compute kernel then convolve via FFT
        K = self._get_kernel(L)  # (d_model, L)

        # Convolve: for each channel convolve u[..., i] with K[i]
        # Using FFT convolution
        K_pad = np.zeros((self.d_model, 2 * L), dtype=np.float32)
        K_pad[:, :L] = K
        K_f = np.fft.rfft(K_pad, axis=-1)   # (d_model, L+1)

        u_T = u.transpose(0, 2, 1)  # (B, d_model, L)
        u_pad = np.zeros((B_sz, self.d_model, 2 * L), dtype=np.float32)
        u_pad[:, :, :L] = u_T
        u_f = np.fft.rfft(u_pad, axis=-1)   # (B, d_model, L+1)

        y_f = u_f * K_f[None, :, :]         # (B, d_model, L+1)
        y = np.fft.irfft(y_f, axis=-1)[:, :, :L]  # (B, d_model, L)
        y = y.transpose(0, 2, 1)             # (B, L, d_model)

        # Skip connection
        y = y + u * self.D._data[None, None, :]
        return Tensor(y.astype(np.float32))


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
            w = Tensor(np.ones(d_model, dtype=np.float32))
            b = Tensor(np.zeros(d_model, dtype=np.float32))
            w.requires_grad = True
            b.requires_grad = True
            self._parameters[f'ln_w_{i}'] = w
            self._parameters[f'ln_b_{i}'] = b
            self.norms.append((w, b))
        self.dropout = dropout
        self.head = None
        if num_classes is not None:
            self.head_w = Tensor(np.random.randn(d_model, num_classes).astype(np.float32) * 0.02)
            self.head_b = Tensor(np.zeros(num_classes, dtype=np.float32))
            self.head_w.requires_grad = True
            self.head_b.requires_grad = True
            self._parameters['head_w'] = self.head_w
            self._parameters['head_b'] = self.head_b

    def _layer_norm(self, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        mean = x.mean(-1, keepdims=True)
        var  = x.var(-1, keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5) * w + b

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, length, d_model)
        Returns:
            (batch, length, d_model) or (batch, num_classes) if head set
        """
        d = x._data
        for (layer, (ln_w, ln_b)) in zip(self.layers, self.norms):
            residual = d
            d = self._layer_norm(d, ln_w._data, ln_b._data)
            d = layer(Tensor(d))._data + residual
            if self.dropout > 0 and self.training:
                mask = np.random.binomial(1, 1 - self.dropout, d.shape).astype(np.float32)
                d = d * mask / (1 - self.dropout)
        if hasattr(self, 'head_w'):
            d = d.mean(axis=1)   # (B, d_model) — mean pooling over sequence
            d = d @ self.head_w._data + self.head_b._data
        return Tensor(d.astype(np.float32))


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
        self.dt_rank = dt_rank or max(1, int(np.ceil(d_model / 16)))

        d_in = self.d_inner

        # Input projection: x → (z, x_ssm) each of d_inner
        scale = 0.02
        self.in_proj_w = Tensor(np.random.randn(d_model, d_in * 2).astype(np.float32) * scale)
        self.in_proj_w.requires_grad = True
        self._parameters['in_proj_w'] = self.in_proj_w

        # Depthwise conv1d over sequence (d_inner channels, kernel d_conv)
        self.conv_w = Tensor(np.random.randn(d_in, 1, d_conv).astype(np.float32) * scale)
        self.conv_w.requires_grad = True
        self.conv_b = Tensor(np.zeros(d_in, dtype=np.float32))
        self.conv_b.requires_grad = True
        self._parameters['conv_w'] = self.conv_w
        self._parameters['conv_b'] = self.conv_b

        # Selective projections: x_ssm → (B, C, dt)
        # dt: (d_in,) → (dt_rank,), B: (d_in,) → (d_state,), C: (d_in,) → (d_state,)
        self.x_proj_w = Tensor(np.random.randn(d_in, self.dt_rank + 2 * d_state).astype(np.float32) * scale)
        self.x_proj_w.requires_grad = True
        self._parameters['x_proj_w'] = self.x_proj_w

        # dt_proj: dt_rank → d_inner (learned expansion + softplus)
        self.dt_proj_w = Tensor(np.random.randn(self.dt_rank, d_in).astype(np.float32) * scale)
        self.dt_proj_b = Tensor(np.zeros(d_in, dtype=np.float32))
        self.dt_proj_w.requires_grad = True
        self.dt_proj_b.requires_grad = True
        self._parameters['dt_proj_w'] = self.dt_proj_w
        self._parameters['dt_proj_b'] = self.dt_proj_b

        # A: log of negative diagonal — (d_inner, d_state), init HiPPO-inspired
        A_log = np.log(np.tile(np.arange(1, d_state + 1, dtype=np.float32), (d_in, 1)))
        self.A_log = Tensor(A_log)
        self.A_log.requires_grad = True
        self._parameters['A_log'] = self.A_log

        # D: skip connection scale (d_inner,)
        self.D_ssm = Tensor(np.ones(d_in, dtype=np.float32))
        self.D_ssm.requires_grad = True
        self._parameters['D_ssm'] = self.D_ssm

        # Output projection
        self.out_proj_w = Tensor(np.random.randn(d_in, d_model).astype(np.float32) * scale)
        self.out_proj_w.requires_grad = True
        self._parameters['out_proj_w'] = self.out_proj_w

        # Layer norm
        self.ln_w = Tensor(np.ones(d_model, dtype=np.float32))
        self.ln_b = Tensor(np.zeros(d_model, dtype=np.float32))
        self.ln_w.requires_grad = True
        self.ln_b.requires_grad = True
        self._parameters['ln_w'] = self.ln_w
        self._parameters['ln_b'] = self.ln_b

    def _silu(self, x: np.ndarray) -> np.ndarray:
        return x * (1 / (1 + np.exp(-x)))

    def _softplus(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(x))

    def _depthwise_conv1d(self, x: np.ndarray) -> np.ndarray:
        """x: (B, L, d_inner) → (B, L, d_inner) via causal depthwise conv."""
        B, L, C = x.shape
        k = self.d_conv
        # Pad left for causality
        x_T = x.transpose(0, 2, 1)  # (B, C, L)
        x_pad = np.pad(x_T, ((0,0),(0,0),(k-1,0)))  # (B, C, L+k-1)
        out = np.zeros_like(x_T)
        w = self.conv_w._data[:, 0, :]  # (C, k)
        for i in range(L):
            out[:, :, i] = (x_pad[:, :, i:i+k] * w[None, :, :]).sum(-1)
        out = out + self.conv_b._data[None, :, None]
        return out.transpose(0, 2, 1)  # (B, L, C)

    def _ssm_scan(self, x: np.ndarray, dt: np.ndarray,
                  A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Sequential SSM recurrence.

        Args:
            x:  (B, L, d_inner)
            dt: (B, L, d_inner)   — input-dependent Δ
            A:  (d_inner, d_state) — fixed (log-negative diagonal)
            B:  (B, L, d_state)   — input-dependent
            C:  (B, L, d_state)   — input-dependent
        Returns:
            y: (B, L, d_inner)
        """
        B_sz, L, d_in = x.shape
        d_state = self.d_state
        h = np.zeros((B_sz, d_in, d_state), dtype=np.float32)
        ys = []
        for t in range(L):
            dt_t = dt[:, t, :]           # (B, d_in)
            # Ā = exp(Δ ⊙ A) — element-wise for diagonal A
            A_bar = np.exp(dt_t[:, :, None] * A[None, :, :])   # (B, d_in, d_state)
            # B̄ = Δ ⊙ B  (outer product over d_in and d_state via broadcasting)
            B_bar = dt_t[:, :, None] * B[:, t, None, :]         # (B, d_in, d_state)
            # h = Ā * h + B̄ * u
            u_t = x[:, t, :, None]       # (B, d_in, 1)
            h = A_bar * h + B_bar * u_t
            # y = C h
            y_t = (h * C[:, t, None, :]).sum(-1)  # (B, d_in)
            ys.append(y_t)
        return np.stack(ys, axis=1)  # (B, L, d_in)

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
        mean = d.mean(-1, keepdims=True)
        var  = d.var(-1, keepdims=True)
        d_norm = (d - mean) / np.sqrt(var + 1e-5) * self.ln_w._data + self.ln_b._data

        # Input projection → split into x_ssm and gate z
        xz = d_norm @ self.in_proj_w._data   # (B, L, 2*d_inner)
        x_ssm = xz[:, :, :self.d_inner]       # (B, L, d_inner)
        z     = xz[:, :, self.d_inner:]        # (B, L, d_inner) — gate

        # Depthwise conv + SiLU
        x_ssm = self._silu(self._depthwise_conv1d(x_ssm))

        # Selective projections
        proj = x_ssm @ self.x_proj_w._data    # (B, L, dt_rank + 2*d_state)
        dt_raw = proj[:, :, :self.dt_rank]
        B_sel  = proj[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C_sel  = proj[:, :, self.dt_rank + self.d_state:]

        # dt: rank → d_inner via learned projection + softplus
        dt = self._softplus(dt_raw @ self.dt_proj_w._data + self.dt_proj_b._data)  # (B, L, d_inner)

        # A: (-exp(A_log)) keeps A negative-definite
        A = -np.exp(self.A_log._data)   # (d_inner, d_state)

        # SSM scan
        y = self._ssm_scan(x_ssm, dt, A, B_sel, C_sel)   # (B, L, d_inner)

        # D skip connection
        y = y + x_ssm * self.D_ssm._data[None, None, :]

        # Gated output
        y = y * self._silu(z)

        # Output projection + residual
        out = y @ self.out_proj_w._data + d   # (B, L, d_model)
        return Tensor(out.astype(np.float32))


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
            self.embedding = Tensor(np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02)
            self.embedding.requires_grad = True
            self._parameters['embedding'] = self.embedding

        self.blocks = []
        for i in range(num_layers):
            block = MambaBlock(d_model, d_state, d_conv, expand)
            self.blocks.append(block)
            self._modules[f'mamba_{i}'] = block

        # Final norm
        self.final_ln_w = Tensor(np.ones(d_model, dtype=np.float32))
        self.final_ln_b = Tensor(np.zeros(d_model, dtype=np.float32))
        self.final_ln_w.requires_grad = True
        self.final_ln_b.requires_grad = True
        self._parameters['final_ln_w'] = self.final_ln_w
        self._parameters['final_ln_b'] = self.final_ln_b

        if vocab_size is not None:
            # Tie weights with embedding
            self._lm_head_tied = True
        elif num_classes is not None:
            self.cls_w = Tensor(np.random.randn(d_model, num_classes).astype(np.float32) * 0.02)
            self.cls_b = Tensor(np.zeros(num_classes, dtype=np.float32))
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
            ids = x._data.astype(int)
            d = self.embedding._data[ids]   # (B, L, d_model)
        else:
            d = x._data

        for block in self.blocks:
            d = block(Tensor(d))._data

        # Final layer norm
        mean = d.mean(-1, keepdims=True)
        var  = d.var(-1, keepdims=True)
        d = (d - mean) / np.sqrt(var + 1e-5) * self.final_ln_w._data + self.final_ln_b._data

        if self.vocab_size is not None:
            # Tied LM head
            d = d @ self.embedding._data.T    # (B, L, vocab_size)
        elif hasattr(self, 'cls_w'):
            d = d.mean(1) @ self.cls_w._data + self.cls_b._data  # (B, num_classes)

        return Tensor(d.astype(np.float32))
